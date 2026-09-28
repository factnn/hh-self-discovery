#!/usr/bin/env python
"""Train the official FNN embedding on isolated HH history windows."""

from __future__ import annotations

import argparse

from hh_self_discovery.fnn_baseline import train_fnn_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-size", type=int, required=True)
    parser.add_argument("--history-steps", type=int, default=750)
    parser.add_argument("--train-windows-per-mode", type=int, default=2000)
    parser.add_argument("--validation-windows-per-mode", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fnn-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()
    train_fnn_baseline(
        data_root=args.data,
        output_dir=args.output,
        latent_size=args.latent_size,
        history_steps=args.history_steps,
        train_windows_per_mode=args.train_windows_per_mode,
        validation_windows_per_mode=args.validation_windows_per_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        regularization_strength=args.fnn_strength,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
