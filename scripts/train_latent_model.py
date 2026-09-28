"""Train a jointly constrained structured latent HH model."""

import argparse
import json

from hh_self_discovery.latent_training import train_structured_latent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/generated/pilot_stratified")
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-size", type=int, required=True)
    parser.add_argument("--encoder-size", type=int, default=64)
    parser.add_argument("--branch-count", type=int, default=3)
    parser.add_argument("--kinetics-width", type=int)
    parser.add_argument("--dynamics-mode", choices=("relaxation", "free"), default="relaxation",
                        help="relaxation law (default) or a same-capacity free vector field")
    parser.add_argument("--free-width", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-windows-per-mode", type=int, default=5000)
    parser.add_argument("--validation-windows-per-mode", type=int, default=1000)
    parser.add_argument("--history-steps", type=int, default=500)
    parser.add_argument("--prediction-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--complexity-scale", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--curriculum-fraction", type=float, default=1.0)
    parser.add_argument("--multi-horizon-loss", action="store_true")
    parser.add_argument("--spike-loss-weight", type=float, default=0.0)
    parser.add_argument("--spike-loss-mode", choices=("global", "conditional"), default="global")
    parser.add_argument("--current-clamp-loss-weight", type=float, default=1.0)
    parser.add_argument("--voltage-transient-loss-weight", type=float, default=0.0)
    parser.add_argument("--voltage-transient-steps", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--current-event-profile",
        choices=("default", "spike_focused"),
        default="default",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = train_structured_latent(
        data_root=args.data,
        output_dir=args.output,
        latent_size=args.latent_size,
        encoder_size=args.encoder_size,
        branch_count=args.branch_count,
        kinetics_width=args.kinetics_width,
        dynamics_mode=args.dynamics_mode,
        free_width=args.free_width,
        epochs=args.epochs,
        train_windows_per_mode=args.train_windows_per_mode,
        validation_windows_per_mode=args.validation_windows_per_mode,
        history_steps=args.history_steps,
        prediction_steps=args.prediction_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        complexity_scale=args.complexity_scale,
        weight_decay=args.weight_decay,
        device_name=args.device,
        curriculum_fraction=args.curriculum_fraction,
        use_multi_horizon_loss=args.multi_horizon_loss,
        spike_loss_weight=args.spike_loss_weight,
        spike_loss_mode=args.spike_loss_mode,
        current_clamp_loss_weight=args.current_clamp_loss_weight,
        voltage_transient_loss_weight=args.voltage_transient_loss_weight,
        voltage_transient_steps=args.voltage_transient_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        current_event_profile=args.current_event_profile,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
