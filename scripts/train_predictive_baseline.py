#!/usr/bin/env python
"""Train a unified predictive latent baseline."""

from __future__ import annotations

import argparse

from hh_self_discovery.baseline_training import train_predictive_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-type",
        required=True,
        choices=("gru_bottleneck", "s4d_bottleneck", "controlled_latent_ode"),
    )
    parser.add_argument("--latent-size", type=int, required=True)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--dynamics-width", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-windows-per-mode", type=int, default=4000)
    parser.add_argument("--validation-windows-per-mode", type=int, default=500)
    parser.add_argument("--history-steps", type=int, default=750)
    parser.add_argument("--prediction-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--curriculum-fraction", type=float, default=0.4)
    parser.add_argument("--current-clamp-loss-weight", type=float, default=3.0)
    parser.add_argument("--voltage-transient-loss-weight", type=float, default=1.0)
    parser.add_argument("--voltage-transient-steps", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train_predictive_baseline(
        data_root=args.data,
        output_dir=args.output,
        model_type=args.model_type,
        latent_size=args.latent_size,
        encoder_width=args.encoder_width,
        dynamics_width=args.dynamics_width,
        epochs=args.epochs,
        train_windows_per_mode=args.train_windows_per_mode,
        validation_windows_per_mode=args.validation_windows_per_mode,
        history_steps=args.history_steps,
        prediction_steps=args.prediction_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        curriculum_fraction=args.curriculum_fraction,
        current_clamp_loss_weight=args.current_clamp_loss_weight,
        voltage_transient_loss_weight=args.voltage_transient_loss_weight,
        voltage_transient_steps=args.voltage_transient_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
