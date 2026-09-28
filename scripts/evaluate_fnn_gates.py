#!/usr/bin/env python
"""Run the isolated validation-to-test hidden-gate audit for FNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hh_self_discovery.fnn_evaluation import evaluate_fnn_gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--evaluation-data")
    parser.add_argument("--windows-per-mode", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9595)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_fnn_gates(
        args.run_dir,
        args.data,
        args.evaluation_data,
        args.windows_per_mode,
        args.seed,
        args.ridge_alpha,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
