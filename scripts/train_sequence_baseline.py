"""Train a plain RNN or GRU as an autoregressive black-box baseline."""

import argparse
import json

from hh_self_discovery.training import train_sequence_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/generated/pilot_stratified")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["current_clamp", "voltage_clamp"], required=True)
    parser.add_argument("--cell", choices=["mlp", "rnn", "gru"], default="rnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-windows", type=int, default=10000)
    parser.add_argument("--validation-windows", type=int, default=2500)
    parser.add_argument("--history-steps", type=int, default=500)
    parser.add_argument("--prediction-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = train_sequence_baseline(
        data_root=args.data,
        output_dir=args.output,
        control_mode=args.mode,
        cell_type=args.cell,
        epochs=args.epochs,
        train_windows=args.train_windows,
        validation_windows=args.validation_windows,
        history_steps=args.history_steps,
        prediction_steps=args.prediction_steps,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        learning_rate=args.lr,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
