"""Evaluate a locked latent model using only observable I/V."""

import argparse
import json
from pathlib import Path

from hh_self_discovery.latent_evaluation import evaluate_locked_latent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/generated/pilot_stratified")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--windows-per-mode", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--response-noise-fraction", type=float, default=0.0)
    parser.add_argument("--prediction-steps", type=int)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--history-observation-stride", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_locked_latent(
        args.checkpoint,
        args.data,
        args.split,
        args.windows_per_mode,
        args.seed,
        args.response_noise_fraction,
        args.prediction_steps,
        not args.skip_ablations,
        args.history_observation_stride,
        args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
