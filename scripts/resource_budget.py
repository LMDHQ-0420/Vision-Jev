"""Compute planning hours from measured throughput; output is never a benchmark claim."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-rate", type=float, help="measured complete questions per second")
    parser.add_argument("--rl-rate", type=float, help="measured environment transitions per second")
    parser.add_argument("--sft-questions", type=int, default=120_000)
    parser.add_argument("--sft-epochs", type=int, default=2)
    parser.add_argument("--rl-transitions", type=int, default=200_000)
    args = parser.parse_args()
    if args.sft_rate is not None and args.sft_rate <= 0:
        parser.error("--sft-rate must be positive")
    if args.rl_rate is not None and args.rl_rate <= 0:
        parser.error("--rl-rate must be positive")
    result = {"evidence_status": "planning_from_user_supplied_rate"}
    if args.sft_rate:
        result["sft_hours_with_20pct_overhead"] = (
            args.sft_questions * args.sft_epochs / args.sft_rate / 3600 * 1.2
        )
    if args.rl_rate:
        result["r0_hours_with_30pct_overhead"] = args.rl_transitions / args.rl_rate / 3600 * 1.3
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
