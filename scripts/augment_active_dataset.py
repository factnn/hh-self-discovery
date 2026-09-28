"""Append active-search voltage protocols to a copied base dataset."""

import argparse
from collections import Counter

from hh_self_discovery.active_dataset import append_active_protocols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--count-per-family", type=int, default=200)
    parser.add_argument("--dt-ms", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=8100)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    records = append_active_protocols(
        args.data, args.count_per_family, args.dt_ms, args.seed, args.workers
    )
    print("added", len(records))
    print("families", dict(Counter(item["family"] for item in records)))
    print("splits", dict(Counter(item["split"] for item in records)))


if __name__ == "__main__":
    main()
