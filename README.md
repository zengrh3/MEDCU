# Unlearning in Medical Language Models is Challenged by Shared Clinical Context

## Overview

This repository contains the reference implementation of **MEDCU**, our method for contextual medical
unlearning. It includes the MEDCU package, the benchmark datasets, and the source data behind the
paper's figures.

## Dataset

- **Med-Unlearn** (`data/Med-Unlearn/`) — synthetic medical QA generated for this study; **included** in
  this repository under CC BY 4.0. Each `*.jsonl` line is a JSON object with `question` and `answer`
  fields. Provided in `independent/` and `coupled/` splits, each with forget/retain sets and their
  paraphrased rewrites (`*_rewrite.jsonl`).
- **MIMIC-Unlearn** (`data/MIMIC-Unlearn/`) — grounded radiology QA derived from MIMIC-CXR, which is
  credentialed-access data under the PhysioNet Data Use Agreement and **cannot be redistributed**, so it
  is not included. Obtain access at https://physionet.org/content/mimic-cxr/, then regenerate the QA
  locally with the provided scripts (see *How to Use*).

## Requirements

- `python3` (tested on Python 3.11, PyTorch 2.13.0, `transformers` 5.8.0), a single 64 GB GPU
- `pip install -e .` &nbsp;— installs the package and the `medcu` command
- `pip install -e ".[judge]"` &nbsp;— optional, adds the OpenAI dependency for the LLM-correctness judge

## Directory Structure

- **`medcu/`**
  - The MEDCU package: the unlearning loss (`method.py`), the trainer (`trainer.py`), the data pipeline
    (`data.py`), the optional LLM judge (`judge.py`), and the `medcu train` / `medcu evaluate` entry points.
- **`finetune.py`**
  - Standalone script that fine-tunes a base model into the unlearning target.
- **`evaluate.sh`**
  - One-line driver that runs forget + retain evaluation and reports FRGap.
- **`data/Med-Unlearn/`**
  - The public synthetic benchmark, in `independent/` and `coupled/` splits.
- **`data/MIMIC-Unlearn/`**
  - Local generation scripts for the restricted MIMIC benchmark (`build_pair_pool.py`, `generate_qa.py`).
    No MIMIC data are shipped.
- **`results/`**
  - Source data for the paper's figures: one CSV per figure (named `main-figure-*.csv` / `supplementary-figure-*.csv`).
- **`configs/medcu_default.yaml`**
  - The default hyperparameters used in the paper.
- **`assets/idk_responses.txt`**
  - The "I don't know" responses used by the entropy-based IDK-redirect evaluation.

## How to Use

**Fine-tune**: build the unlearning target model by supervised fine-tuning a base instruction-tuned
LLM on the union of the forget and retain sets, so it memorises both before unlearning.

```
python finetune.py \
    --model <base-instruct-model> \
    --data data/Med-Unlearn/coupled/forget.jsonl data/Med-Unlearn/coupled/retain.jsonl \
    --output_dir runs/finetuned_coupled \
    --epochs 5 --lr 1e-5
```

**Unlearn**: train MEDCU to forget the forget set while preserving the retain set. `--model` is the
fine-tuned target produced above.

```
medcu train \
    --model runs/finetuned_coupled \
    --forget data/Med-Unlearn/coupled/forget.jsonl \
    --retain data/Med-Unlearn/coupled/retain.jsonl \
    --output_dir runs/medcu_coupled \
    --epochs 3 --lr 1e-5
```

Default hyperparameters (`alpha=gamma=1`, `rank_k=64`, `q=0.1/0.9`, `weight_floor=0.1`, penultimate
layer) are in `configs/medcu_default.yaml`.

**Evaluate**: score the unlearned model on the forget and retain sets and report
FRGap = retain ROUGE-L − forget ROUGE-L (higher is better). Pass a judge model as the third argument
to also compute the LLM-correctness metric.

```
bash evaluate.sh runs/medcu_coupled coupled
bash evaluate.sh runs/medcu_coupled coupled openai/gpt-4o-mini
```

**Regenerate MIMIC-Unlearn** (after obtaining MIMIC-CXR access): build the forget/retain pair pool, then
generate the grounded QA locally, producing the same layout as Med-Unlearn under
`data/MIMIC-Unlearn/{independent,coupled}/`. Run each script with `--help` for backend and model options.

```
python data/MIMIC-Unlearn/build_pair_pool.py
python data/MIMIC-Unlearn/generate_qa.py --backend hf-batched --model-path <local-model>
```

## Demo

**Prepare demo data**: install the package and sample 200 forget/retain examples from the coupled split.

```
pip install -e .
mkdir -p demo
head -n 200 data/Med-Unlearn/coupled/forget.jsonl > demo/forget.jsonl
head -n 200 data/Med-Unlearn/coupled/retain.jsonl > demo/retain.jsonl
```

**Fine-tune**: train a small base model on the demo forget/retain data.

```
python finetune.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --data demo/forget.jsonl demo/retain.jsonl \
    --output_dir demo/finetuned \
    --epochs 5 --lr 1e-5
```

**Unlearn**: run MEDCU on the fine-tuned demo model.

```
medcu train \
    --model demo/finetuned \
    --forget demo/forget.jsonl \
    --retain demo/retain.jsonl \
    --output_dir demo/medcu \
    --epochs 3 --lr 1e-5
```

**Evaluate**: score the forget and retain demo splits.

```
medcu evaluate \
    --model demo/medcu \
    --split forget \
    --data demo/forget.jsonl \
    --out demo/eval_forget.json
medcu evaluate \
    --model demo/medcu \
    --split retain \
    --data demo/retain.jsonl \
    --out demo/eval_retain.json
```

**Expected output**:

```
[forget] n=200 ROUGE-L=0.5
[retain] n=200 ROUGE-L=28.9
```

**Runtime**: on a single 64 GB GPU, expect ~10 min fine-tuning, ~5 min unlearning, and ~10 min
evaluation, for **~25 min total**. The full-scale runs in the paper (2000 pairs, up to 8B models)
take several hours per model per setting.

## Results

We release the numerical source data behind **every figure** in the paper under `results/` — one
machine-readable CSV per figure, named by its figure number (`main-figure-*.csv` /
`supplementary-figure-*.csv`). Each file contains exactly the values plotted in the corresponding
figure, so all reported results can be inspected and reproduced directly.

## License

Code is released under the MIT License (`LICENSE`); the Med-Unlearn data are released under CC BY 4.0.
