#!/usr/bin/env python3
"""Train MEDCU to unlearn a forget set while preserving a retain set.

Use via the CLI (`medcu train ...`) or `python -m medcu.train ...`. See README.md.
Hyperparameters default to the paper values (alpha=gamma=1, rank_k=64, q=0.1/0.9,
weight_floor=0.1, penultimate layer).
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from .trainer import MEDCUTrainer
from .data import QADataset, ForgetRetainDataset, ForgetRetainCollator


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="medcu train", description="Unlearn a model with MEDCU.")
    ap.add_argument("--model", required=True, help="HF id or path of the fine-tuned target model")
    ap.add_argument("--forget", required=True)
    ap.add_argument("--retain", required=True)
    ap.add_argument("--output_dir", required=True)
    # optimisation
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--optim", default="adamw_torch")
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--lr_scheduler", default="constant")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # MEDCU method
    ap.add_argument("--gamma", type=float, default=1.0, help="forget weight")
    ap.add_argument("--alpha", type=float, default=1.0, help="retain weight (lambda)")
    ap.add_argument("--rank_k", type=int, default=64)
    ap.add_argument("--quantile_low", type=float, default=0.1)
    ap.add_argument("--quantile_high", type=float, default=0.9)
    ap.add_argument("--weight_floor", type=float, default=0.1, help="rho: omega lower bound")
    ap.add_argument("--hidden_layer", type=int, default=-2)
    ap.add_argument("--max_retain_tokens", type=int, default=4096)
    # chat template
    ap.add_argument("--system_prompt", default=None)
    ap.add_argument("--date_string", default=None)
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)

    forget = QADataset(args.forget, tok, args.system_prompt, args.date_string, args.max_length)
    retain = QADataset(args.retain, tok, args.system_prompt, args.date_string, args.max_length)
    train_ds = ForgetRetainDataset(forget, retain, seed=args.seed)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim=args.optim,
        max_grad_norm=args.grad_clip,
        lr_scheduler_type=args.lr_scheduler,
        gradient_checkpointing=args.grad_ckpt,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,   # nested forget/retain dicts
    )

    trainer = MEDCUTrainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=ForgetRetainCollator(tok), processing_class=tok,
        gamma=args.gamma, alpha=args.alpha, hidden_layer=args.hidden_layer,
        rank_k=args.rank_k, max_retain_tokens=args.max_retain_tokens,
        quantile_low=args.quantile_low, quantile_high=args.quantile_high,
        weight_floor=args.weight_floor,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"[MEDCU] saved unlearned model to {args.output_dir}")


if __name__ == "__main__":
    main()
