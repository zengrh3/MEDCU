#!/usr/bin/env python3
"""Evaluate an unlearned model: generate answers on a QA jsonl and score them.

Use via the CLI (`medcu evaluate ...`) or `python -m medcu.evaluate ...`.
Metrics: ROUGE-L (no API) and optional LLM-judge correctness 1-5 (--judge_model).
Run on forget and retain separately; FRGap = retain ROUGE-L - forget ROUGE-L.
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import read_jsonl


def rouge_l_recall(pred: str, ref: str) -> float:
    """ROUGE-L recall via LCS over whitespace tokens (no external dependency)."""
    a, b = ref.split(), pred.split()
    if not a:
        return 0.0
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ai = a[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, n + 1):
            row[j] = prev[j - 1] + 1 if ai == b[j - 1] else (prev[j] if prev[j] >= row[j - 1] else row[j - 1])
    return dp[m][n] / m


@torch.no_grad()
def generate(model, tok, question, system_prompt, date_string, max_new_tokens, do_sample, temperature):
    chat = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
           [{"role": "user", "content": question}]
    date_info = {"date_string": date_string} if date_string is not None else {}
    ids = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", **date_info)
    ids = ids.to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
                         temperature=(temperature if do_sample else None),
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="medcu evaluate", description="Score an unlearned model.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="forget", choices=["forget", "retain"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--system_prompt", default=None)
    ap.add_argument("--date_string", default=None)
    ap.add_argument("--judge_model", default=None, help="e.g. openai/gpt-4o-mini (needs OPENAI_API_KEY)")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    rows = read_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]

    client = jmodel = None
    if args.judge_model:
        from .judge import make_client
        client, jmodel = make_client(), args.judge_model

    recs, rouges, corrs = [], [], []
    for r in rows:
        q, ref = r["question"], r["answer"]
        gen = generate(model, tok, q, args.system_prompt, args.date_string,
                       args.max_new_tokens, args.do_sample, args.temperature)
        rl = rouge_l_recall(gen, ref)
        rouges.append(rl)
        rec = {"question": q, "reference": ref, "generation": gen, "rougeL": rl}
        if client is not None:
            from .judge import judge_one
            c = judge_one(client, jmodel, q, ref, gen)
            rec["correctness"] = c
            if c is not None:
                corrs.append(c)
        recs.append(rec)

    summary = {"split": args.split, "n": len(recs),
               "rougeL_mean": sum(rouges) / max(len(rouges), 1)}
    if corrs:
        summary["correctness_mean"] = sum(corrs) / len(corrs)
        summary["judge_model"] = args.judge_model
    json.dump({"summary": summary, "samples": recs}, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[{args.split}] n={summary['n']} ROUGE-L={summary['rougeL_mean']*100:.1f}"
          + (f"  correctness={summary['correctness_mean']:.2f}" if corrs else ""))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
