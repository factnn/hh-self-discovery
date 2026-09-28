#!/usr/bin/env python
"""Run observable-only STLSQ audits on a trained DDAE coordinate chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hh_self_discovery.ddae_sparse_audit import audit_ddae_sparse_field


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--ood-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=26000)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.0, 0.003, 0.01, 0.03, 0.1, 0.3),
    )
    args = parser.parse_args()
    result = audit_ddae_sparse_field(
        args.run_dir,
        args.data,
        args.ood_data,
        thresholds=tuple(args.thresholds),
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
