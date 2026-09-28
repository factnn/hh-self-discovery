"""Evaluate and save the constant-last-response lower bound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hh_self_discovery.baselines import evaluate_persistence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/generated/pilot_stratified")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--windows", type=int, default=5000)
    parser.add_argument("--history-steps", type=int, default=500)
    parser.add_argument("--prediction-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["current_clamp", "voltage_clamp"])
    parser.add_argument("--allow-missing-events", action="store_true")
    parser.add_argument("--output", default="runs/baselines/persistence_validation.json")
    args = parser.parse_args()
    metrics = evaluate_persistence(
        args.data,
        args.split,
        args.windows,
        args.history_steps,
        args.prediction_steps,
        args.seed,
        args.mode,
        not args.allow_missing_events,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
