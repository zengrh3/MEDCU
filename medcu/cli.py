#!/usr/bin/env python3
"""Unified CLI:  medcu <command> [args]

Commands:
    train       Unlearn a model with MEDCU            (medcu.train)
    evaluate    Score an unlearned model (ROUGE / LLM-judge)   (medcu.evaluate)

Installed as the `medcu` console script (see pyproject.toml), or run as
`python -m medcu.cli <command> ...`.
"""
from __future__ import annotations

import sys

USAGE = "usage: medcu {train|evaluate} [args]   (try: medcu train -h)"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        from . import train
        return train.main(rest)
    if cmd == "evaluate":
        from . import evaluate
        return evaluate.main(rest)
    print(f"unknown command: {cmd}\n{USAGE}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
