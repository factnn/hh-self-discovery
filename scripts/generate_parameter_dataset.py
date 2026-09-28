"""Generate a held-out HH parameter-mismatch dataset."""

import argparse

from hh_self_discovery.parameter_dataset import generate_parameter_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/generated/parameter_ood_10pct")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--dt-ms", type=float, default=0.02)
    parser.add_argument("--variation", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    records = generate_parameter_dataset(
        args.output, args.count, args.dt_ms, args.variation, args.seed, args.workers
    )
    print(f"generated {len(records)} parameter-OOD trajectories in {args.output}")


if __name__ == "__main__":
    main()
