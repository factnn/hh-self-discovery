"""Evaluate a saved RNN/GRU/MLP checkpoint on ID or OOD data."""

import argparse
import json
from pathlib import Path

from hh_self_discovery.sequence_evaluation import evaluate_sequence_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--windows", type=int, default=500)
    parser.add_argument("--prediction-steps", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_sequence_checkpoint(
        args.checkpoint, args.data, args.split, args.windows,
        args.prediction_steps, args.seed, args.device
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
