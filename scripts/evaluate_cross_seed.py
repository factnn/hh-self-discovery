"""Audit whether independently trained latent models differ only by coordinates."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from hh_self_discovery.alignment import (
    affine_positive_control,
    align_pair,
    collect_latent_rollouts,
    shared_references,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--data", default="data/generated/full_stratified")
    parser.add_argument("--windows-per-mode", type=int, default=300)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7070)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="runs/phase2/cross_seed_audit.json")
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.checkpoints):
        parser.error("--labels must have the same length as --checkpoints")
    labels = args.labels or [Path(path).parent.name for path in args.checkpoints]
    if len(set(labels)) != len(labels):
        parser.error("checkpoint labels must be unique")

    first = torch.load(args.checkpoints[0], map_location="cpu")
    history_steps = int(first["history_steps"])
    prediction_steps = int(first["prediction_steps"])
    validation_refs = shared_references(
        args.data,
        "validation",
        history_steps,
        prediction_steps,
        args.windows_per_mode,
        args.seed,
    )
    test_refs = shared_references(
        args.data,
        "test",
        history_steps,
        prediction_steps,
        args.windows_per_mode,
        args.seed + 10000,
    )
    validation, test = {}, {}
    for label, checkpoint in zip(labels, args.checkpoints):
        metadata = torch.load(checkpoint, map_location="cpu")
        if (
            metadata["history_steps"] != history_steps
            or metadata["prediction_steps"] != prediction_steps
            or metadata["latent_size"] != first["latent_size"]
        ):
            raise ValueError("all checkpoints must share history, prediction, and latent sizes")
        validation[label] = collect_latent_rollouts(
            checkpoint,
            args.data,
            "validation",
            validation_refs,
            args.device,
            args.stride,
        )
        test[label] = collect_latent_rollouts(
            checkpoint,
            args.data,
            "test",
            test_refs,
            args.device,
            args.stride,
        )

    pairs = {}
    for source, target in itertools.combinations(labels, 2):
        key = f"{source}_to_{target}"
        pairs[key] = align_pair(
            validation[source], validation[target], test[source], test[target],
            bootstrap_seed=args.seed + 100 * len(pairs),
        )
        best = max(pairs[key]["maps"].values(), key=lambda item: item["mean_test_r2"])
        pairs[key]["supports_low_complexity_conjugacy"] = bool(
            best["mean_test_r2"] >= 0.9
            and best["mean_cycle_r2"] >= 0.8
            and best["vector_field_relative_rmse"] <= 0.1
        )
        pairs[key]["candidate_same_output_different_dynamics"] = bool(
            pairs[key]["output_equivalence"]["all_modes_equivalent"]
            and not pairs[key]["supports_low_complexity_conjugacy"]
        )

    reference = labels[0]
    result = {
        "checkpoints": dict(zip(labels, args.checkpoints)),
        "data": args.data,
        "windows_per_mode": args.windows_per_mode,
        "stride": args.stride,
        "seed": args.seed,
        "training_fields_used": ["current", "voltage", "control_mode", "event_labels"],
        "evaluation_only_gates_read": False,
        "preregistered_h1_thresholds": {
            "mean_test_r2_min": 0.9,
            "mean_cycle_r2_min": 0.8,
            "vector_field_relative_rmse_max": 0.1,
        },
        "preregistered_output_equivalence_bound": 0.1,
        "positive_control": affine_positive_control(
            validation[reference]["state"],
            test[reference]["state"],
            validation[reference]["field"],
            test[reference]["field"],
            args.seed,
        ),
        "pairs": pairs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
