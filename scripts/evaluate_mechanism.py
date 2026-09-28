"""Compare locked latent kinetics with canonical HH kinetics."""

import argparse
import json
from pathlib import Path

from hh_self_discovery.mechanism_evaluation import (
    evaluate_mechanism,
    evaluate_phase_portrait,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--blind-result", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-figure", required=True)
    parser.add_argument("--data")
    parser.add_argument("--output-phase-figure")
    args = parser.parse_args()
    result = evaluate_mechanism(
        args.checkpoint, args.blind_result, args.output_figure
    )
    if args.data and args.output_phase_figure:
        result["phase_portrait"] = evaluate_phase_portrait(
            args.checkpoint, args.data, args.output_phase_figure
        )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
