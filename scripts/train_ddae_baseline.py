#!/usr/bin/env python
"""Train the official DDAE or its controlled two-stage HH adapter."""

from __future__ import annotations

import argparse

from hh_self_discovery.ddae_baseline import train_ddae_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-size", type=int, required=True)
    parser.add_argument(
        "--variant",
        choices=("autonomous", "controlled_two_stage"),
        default="controlled_two_stage",
    )
    parser.add_argument("--history-steps", type=int, default=128)
    parser.add_argument("--train-windows-per-mode", type=int, default=2000)
    parser.add_argument("--validation-windows-per-mode", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ridge-alpha", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()
    train_ddae_baseline(
        data_root=args.data,
        output_dir=args.output,
        latent_size=args.latent_size,
        variant=args.variant,
        history_steps=args.history_steps,
        train_windows_per_mode=args.train_windows_per_mode,
        validation_windows_per_mode=args.validation_windows_per_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
