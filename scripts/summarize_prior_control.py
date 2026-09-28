#!/usr/bin/env python3
"""Compare the dimension scan under the relaxation law and the free-dynamics control.

Reads the test evaluations of the locked ``latentdim{K}_kw24_h750_s{seed}``
checkpoints (relaxation law) and of the ``freeform_latentdim{K}_kw24_h750_s{seed}``
control (free vector field, comparable parameter count), and reports the mean and
cross-seed spread of the response NRMSE per latent dimension.

    python scripts/summarize_prior_control.py --output runs/phase2/prior_control_summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "phase2"


def collect(prefix: str, template: str, seeds: tuple[int, ...], dimensions: tuple[int, ...]) -> dict:
    report = {}
    for k in dimensions:
        values, sources = [], []
        for seed in seeds:
            path = RUNS / template.format(prefix=prefix, k=k, seed=seed)
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            values.append(payload["baseline"]["mean_normalized_rmse"])
            sources.append(path.name)
        if values:
            report[str(k)] = {
                "values": values,
                "mean": st.mean(values),
                "sample_sd": st.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
                "sources": sources,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[51, 52, 53, 54, 55])
    parser.add_argument("--control-seeds", type=int, nargs="+", default=[51, 52])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    summary = {
        "definition": (
            "Response NRMSE (mean over clamp modes) on held-out test windows, from the "
            "locked dimension scan and from the same-capacity free-dynamics control."
        ),
        "relaxation_law": collect(
            "latentdim", "latentdim{k}_kw24_h750_s{seed}_test20ms.json",
            tuple(args.seeds), tuple(args.dimensions),
        ),
        "free_dynamics_control": collect(
            "freeform_latentdim", "freeform_latentdim{k}_kw24_h750_s{seed}_test20ms.json",
            tuple(args.control_seeds), tuple(args.dimensions),
        ),
    }
    text = json.dumps(summary, indent=1, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    for family in ("relaxation_law", "free_dynamics_control"):
        print(f"== {family}")
        for k, entry in sorted(summary[family].items(), key=lambda item: int(item[0])):
            print("   K=%s  n=%d  NRMSE %.4f +- %.4f" % (
                k, entry["n"], entry["mean"], entry["sample_sd"]))


if __name__ == "__main__":
    main()
