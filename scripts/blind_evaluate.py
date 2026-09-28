"""Run the one-way hidden-gate evaluation after model selection is locked."""

import argparse
import json
from pathlib import Path

from hh_self_discovery.blind_evaluation import blind_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/generated/pilot_stratified")
    parser.add_argument("--evaluation-data")
    parser.add_argument("--mapping-split", default="validation")
    parser.add_argument("--evaluation-split", default="test")
    parser.add_argument("--windows-per-mode", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--ridge-alpha-grid", type=float, nargs="+")
    parser.add_argument("--ridge-cv-folds", type=int, default=1)
    parser.add_argument("--evaluation-response-noise-fraction", type=float, default=0.0)
    parser.add_argument("--evaluation-history-observation-stride", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = blind_evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data,
        mapping_split=args.mapping_split,
        evaluation_split=args.evaluation_split,
        windows_per_mode=args.windows_per_mode,
        seed=args.seed,
        device_name=args.device,
        evaluation_data_root=args.evaluation_data,
        ridge_alpha=args.ridge_alpha,
        ridge_alpha_grid=tuple(args.ridge_alpha_grid) if args.ridge_alpha_grid else None,
        ridge_cv_folds=args.ridge_cv_folds,
        evaluation_response_noise_fraction=args.evaluation_response_noise_fraction,
        evaluation_history_observation_stride=args.evaluation_history_observation_stride,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
