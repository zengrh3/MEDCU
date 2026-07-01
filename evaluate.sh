#!/usr/bin/env bash
# Evaluate an unlearned MEDCU model: forget + retain ROUGE-L -> FRGap.
# Usage:  bash evaluate.sh <model_dir> <independent|coupled> [judge_model]
#   e.g.  bash evaluate.sh runs/medcu_coupled coupled
#         bash evaluate.sh runs/medcu_coupled coupled openai/gpt-4o-mini   # + LLM-correctness judge
set -euo pipefail

MODEL=${1:?usage: evaluate.sh <model_dir> <independent|coupled> [judge_model]}
SETTING=${2:-coupled}
JUDGE=${3:-}
DATA="data/Med-Unlearn/${SETTING}"
JARG=(); [ -n "$JUDGE" ] && JARG=(--judge_model "$JUDGE")

medcu evaluate --model "$MODEL" --split forget --data "$DATA/forget.jsonl" --do_sample \
  --out "$MODEL/eval_forget.json" "${JARG[@]}"
medcu evaluate --model "$MODEL" --split retain --data "$DATA/retain.jsonl" --do_sample \
  --out "$MODEL/eval_retain.json" "${JARG[@]}"

# FRGap = retain ROUGE-L - forget ROUGE-L  (higher is better)
python - "$MODEL/eval_forget.json" "$MODEL/eval_retain.json" <<'PY'
import json, sys
f = json.load(open(sys.argv[1]))["summary"]["rougeL_mean"] * 100
r = json.load(open(sys.argv[2]))["summary"]["rougeL_mean"] * 100
print(f"forget ROUGE-L={f:.1f}  retain ROUGE-L={r:.1f}  FRGap={r - f:.1f}")
PY
