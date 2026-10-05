#!/usr/bin/env python3
"""Evaluate the released Qwen3.5 SFT adapter on its group-safe holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_jev.train.sft import evaluate_checkpoint

DEFAULT_DATA = Path("/data/vision-jev/manifests/public-117k-training.jsonl")
DEFAULT_RUN = Path("/data/vision-jev/runs/qwen35-08b-sft-main-117k")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/sft_main.json"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN / "checkpoint-last")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "evaluation.jsonl")
    parser.add_argument(
        "--maximum", type=int, default=0, help="maximum rows; 0 evaluates the full holdout"
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--model-root", type=Path, default=Path("/data/vision-jev/models")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = evaluate_checkpoint(
        args.config,
        args.data,
        args.checkpoint,
        args.output,
        model_root=args.model_root,
        maximum=args.maximum or None,
        progress_every=args.progress_every,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
