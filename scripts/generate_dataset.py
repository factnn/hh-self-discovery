"""CLI for deterministic pilot-dataset generation."""

from __future__ import annotations

import argparse
from collections import Counter

from hh_self_discovery.dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/generated/pilot")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--dt-ms", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    records = generate_dataset(args.output, args.count, args.dt_ms, args.seed, args.workers)
    print(f"generated {len(records)} trajectories in {args.output}")
    print("splits:", dict(Counter(record["split"] for record in records)))
    print("families:", dict(Counter(record["family"] for record in records)))


if __name__ == "__main__":
    main()
