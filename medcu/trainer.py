"""MEDCUTrainer: a thin transformers.Trainer subclass wiring the MEDCU loss.

Total loss = gamma * L_forget(omega-weighted negative entropy) + alpha * L_retain(cross-entropy).
Reference-free. Hidden states come from `hidden_layer` (default penultimate, -2).
"""
from __future__ import annotations

from transformers import Trainer

from .method import medcu_forget_loss


class MEDCUTrainer(Trainer):
    def __init__(self, *args, gamma: float = 1.0, alpha: float = 1.0,
                 hidden_layer: int = -2, rank_k: int = 64, max_retain_tokens: int = 4096,
                 quantile_low: float = 0.1, quantile_high: float = 0.9, weight_floor: float = 0.1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.hidden_layer = hidden_layer
        self.rank_k = rank_k
        self.max_retain_tokens = max_retain_tokens
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high
        self.weight_floor = weight_floor
        self._last = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        f = {k: inputs["forget"][k] for k in ("input_ids", "attention_mask", "labels")}
        r = {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}

        forget_out = model(**f, output_hidden_states=True)
        retain_out = model(**r, output_hidden_states=True)  # retain_out.loss = retain cross-entropy

        forget_loss, info = medcu_forget_loss(
            forget_logits=forget_out.logits,
            forget_hidden=forget_out.hidden_states[self.hidden_layer].detach(),
            forget_labels=f["labels"],
            retain_hidden=retain_out.hidden_states[self.hidden_layer].detach(),
            retain_labels=r["labels"],
            rank_k=self.rank_k, max_retain_tokens=self.max_retain_tokens,
            quantile_low=self.quantile_low, quantile_high=self.quantile_high,
            weight_floor=self.weight_floor,
        )
        retain_loss = retain_out.loss
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        self._last = {"forget_loss": float(forget_loss.detach()),
                      "retain_loss": float(retain_loss.detach()), **info}
        return (loss, forget_out) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if "loss" in logs and self._last:
            logs.update({f"train/{k}": v for k, v in self._last.items()})
        return super().log(logs, *args, **kwargs)
