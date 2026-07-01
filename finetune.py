#!/usr/bin/env python3
"""Fine-tune a base instruction-tuned LLM on the Med-Unlearn QA to build the
target model that MEDCU then unlearns from.

Plain supervised fine-tuning: every QA example is chat-templated, the prompt
tokens are masked, and the loss is the next-token cross-entropy on the answer
tokens only (the same data pipeline as MEDCU training). Train on the union of the
forget and retain sets so the target model memorises both before unlearning.

Run from the package root:  python finetune.py --model <base> --data forget.jsonl retain.jsonl ...
"""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import ConcatDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from medcu.data import QADataset, IGNORE_INDEX


class SFTCollator:
    """Pad input_ids (pad_token_id) and labels (IGNORE_INDEX); build the attention mask."""

    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id

    def __call__(self, items):
        ids = torch.nn.utils.rnn.pad_sequence(
            [it["input_ids"] for it in items], batch_first=True, padding_value=self.pad)
        labels = torch.nn.utils.rnn.pad_sequence(
            [it["labels"] for it in items], batch_first=True, padding_value=IGNORE_INDEX)
        return {"input_ids": ids, "attention_mask": ids.ne(self.pad), "labels": labels}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Supervised fine-tuning to build the MEDCU unlearning target model.")
    ap.add_argument("--model", required=True, help="HF id or path of the base instruct model")
    ap.add_argument("--data", required=True, nargs="+",
                    help="one or more QA jsonl files, e.g. forget.jsonl retain.jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--optim", default="adamw_torch")
    ap.add_argument("--lr_scheduler", default="linear")
    ap.add_argument("--warmup_ratio", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--system_prompt", default=None)
    ap.add_argument("--date_string", default=None)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    parts = [QADataset(p, tok, args.system_prompt, args.date_string, args.max_length)
             for p in args.data]
    train_ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.grad_ckpt,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=SFTCollator(tok), processing_class=tok)
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"[MEDCU] saved fine-tuned target model to {args.output_dir}")


if __name__ == "__main__":
    main()
