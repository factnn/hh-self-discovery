"""Generate intervention families held out from all model training."""

import argparse

from hh_self_discovery.ood_dataset import generate_ood_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/generated/ood_protocols")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--dt-ms", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    records = generate_ood_dataset(args.output, args.count, args.dt_ms, args.seed, args.workers)
    print(f"generated {len(records)} OOD trajectories in {args.output}")


if __name__ == "__main__":
    main()
