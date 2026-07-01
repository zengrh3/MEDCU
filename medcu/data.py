"""Data pipeline for MEDCU: QA jsonl -> chat-templated, prompt-masked tensors,
paired into {forget, retain} mini-batches. Reimplements the framework's
preprocess_chat_instance / ForgetRetainDataset / supervised collator faithfully.

Each jsonl line is a record with `question` and `answer` fields (the public
Med-Unlearn files have exactly these, plus metadata that is ignored here).
"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _to_ids(x) -> List[int]:
    """Normalise apply_chat_template output (list / nested list / tensor / BatchEncoding)
    to a flat list of token ids."""
    if hasattr(x, "input_ids"):          # BatchEncoding
        x = x.input_ids
    elif isinstance(x, dict):
        x = x["input_ids"]
    if hasattr(x, "tolist"):             # tensor / np array
        x = x.tolist()
    if len(x) and isinstance(x[0], (list, tuple)):  # batched -> first row
        x = x[0]
    return list(x)


def preprocess_chat_instance(tokenizer, question: str, answer: str,
                             system_prompt: str | None = None,
                             date_string: str | None = None,
                             max_length: int = 512) -> Dict[str, torch.Tensor]:
    """Apply the model chat template; loss is computed only on the answer tokens
    (prompt tokens are set to IGNORE_INDEX). Mirrors the framework exactly."""
    chat = []
    if system_prompt:
        chat.append({"role": "system", "content": system_prompt})
    chat.append({"role": "user", "content": question})
    chat.append({"role": "assistant", "content": answer})
    date_info = {"date_string": date_string} if date_string is not None else {}

    chat_ids = _to_ids(tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False, **date_info))
    prompt_ids = _to_ids(tokenizer.apply_chat_template(chat[:-1], tokenize=True, add_generation_prompt=True, **date_info))

    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids = chat_ids + [tokenizer.eos_token_id]
    chat_ids = chat_ids[:max_length]
    n_prompt = min(len(prompt_ids), len(chat_ids))
    labels = [IGNORE_INDEX] * n_prompt + chat_ids[n_prompt:]
    return {
        "input_ids": torch.tensor(chat_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor([1] * len(chat_ids)),
    }


class QADataset(Dataset):
    def __init__(self, path, tokenizer, system_prompt=None, date_string=None, max_length=512):
        self.rows = read_jsonl(path)
        self.tok = tokenizer
        self.system_prompt = system_prompt
        self.date_string = date_string
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return preprocess_chat_instance(self.tok, r["question"], r["answer"],
                                        self.system_prompt, self.date_string, self.max_length)


class ForgetRetainDataset(Dataset):
    """Anchors on the forget set; for each forget example samples a random retain
    example (align_pairs=False behaviour). Yields {"forget": item, "retain": item}."""

    def __init__(self, forget: QADataset, retain: QADataset, seed: int = 0):
        self.forget = forget
        self.retain = retain
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, i):
        f = self.forget[i]
        r = self.retain[self.rng.randint(0, len(self.retain) - 1)]
        return {"forget": f, "retain": r}


class ForgetRetainCollator:
    """Pads input_ids (pad_token_id), labels (IGNORE_INDEX), builds attention_mask,
    nested under the 'forget' and 'retain' keys."""

    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id

    def _pad(self, seqs: Sequence[torch.Tensor], value: int) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=value)

    def _collate_side(self, items: List[dict]) -> Dict[str, torch.Tensor]:
        input_ids = self._pad([it["input_ids"] for it in items], self.pad)
        labels = self._pad([it["labels"] for it in items], IGNORE_INDEX)
        attn = input_ids.ne(self.pad)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    def __call__(self, batch: List[dict]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "forget": self._collate_side([b["forget"] for b in batch]),
            "retain": self._collate_side([b["retain"] for b in batch]),
        }
