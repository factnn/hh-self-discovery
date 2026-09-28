#!/usr/bin/env python3
"""Aggregate the one-step gate-flow audits into the reported cross-architecture table.

Reads ``runs/phase2/*_discrete_flow.json`` (written by
``scripts/audit_discrete_gate_flow.py``) and reports, per model family, the
decoded gate state and the transported-field score that uses each model's own
one-step latent velocity, on identical windows.

    python scripts/summarize_discrete_flow.py --output runs/phase2/discrete_flow_summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "phase2"

FAMILIES = {
    "structured_k3": "latentdim3_kw24_h750_s5?_discrete_flow.json",
    "s4d_k5": "s4d_k5_s5?_discrete_flow.json",
    "gru_k5": "gru_k5_s5?_discrete_flow.json",
    "latent_ode_k2": "latent_ode_k2_s5?_discrete_flow.json",
}
GATES = ("n", "m", "h")


def mean_sd(values: list[float]) -> dict:
    return {
        "mean": st.mean(values),
        "sample_sd": st.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def summarize(runs_dir: Path, stride: str) -> dict:
    out = {}
    for family, pattern in FAMILIES.items():
        files = sorted(glob.glob(str(runs_dir / pattern)))
        if not files:
            continue
        blocks = [json.load(open(f))["blocks"][stride] for f in files]
        entry = {
            "seeds": len(files),
            "support_pairs": blocks[0]["field_from_discrete_velocity"]["count"],
            "dt_ms": blocks[0]["dt_ms"],
            "state_r2": {g: mean_sd([b["state_r2_on_smooth_support"][g] for b in blocks]) for g in GATES},
            "field_r2": {g: mean_sd([b["field_from_discrete_velocity"]["by_gate"][g]["r2"] for b in blocks]) for g in GATES},
            "field_correlation": {g: mean_sd([b["field_from_discrete_velocity"]["by_gate"][g]["correlation"] for b in blocks]) for g in GATES},
            "field_slope": {g: mean_sd([b["field_from_discrete_velocity"]["by_gate"][g]["slope"] for b in blocks]) for g in GATES},
            "sources": [str(Path(f).relative_to(ROOT)) for f in files],
        }
        if "step_flow_increment" in blocks[0]:
            entry["step_flow_state_r2"] = {
                g: mean_sd([b["step_flow_state"]["by_gate"][g]["r2"] for b in blocks])
                for g in GATES
            }
            entry["step_flow_increment_r2"] = {
                g: mean_sd([b["step_flow_increment"]["by_gate"][g]["r2"] for b in blocks])
                for g in GATES
            }
            entry["step_flow_increment_slope"] = {
                g: mean_sd([b["step_flow_increment"]["by_gate"][g]["slope"] for b in blocks])
                for g in GATES
            }
            entry["step_flow_persistence_r2"] = {
                g: mean_sd([b["step_flow_persistence"]["by_gate"][g]["r2"] for b in blocks])
                for g in GATES
            }
        if "field_from_analytic_velocity" in blocks[0]:
            entry["field_r2_analytic_velocity"] = {
                g: mean_sd([b["field_from_analytic_velocity"]["by_gate"][g]["r2"] for b in blocks])
                for g in GATES
            }
        out[family] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=str(RUNS))
    parser.add_argument("--stride", default="stride_5")
    parser.add_argument("--native-stride", default="stride_1",
                        help="second block reported as a robustness check at the native interval")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    reported = summarize(Path(args.runs), args.stride)
    summary = {
        "definition": (
            "Decoded gate state and transported field on identical held-out smooth "
            "voltage-clamp windows. The latent velocity is estimated from each model's "
            "own one-step map, dz/dt = (z_{t+1} - z_t)/dt, transported through the same "
            "cubic ridge chart, and compared with the canonical HH gate field. The "
            "step_flow_* blocks repeat the comparison without any derivative: the "
            "model's next decoded state against the exact HH one-step relaxation from "
            "the same decoded state."
        ),
        "stride": args.stride,
        "evaluation_interval_ms": 0.02 * int(args.stride.split("_")[1]),
        "families": reported,
    }
    if args.native_stride:
        native = summarize(Path(args.runs), args.native_stride)
        summary["native_interval_robustness"] = {
            "interval_ms": 0.02 * int(args.native_stride.split("_")[1]),
            "note": ("Same audit at the native sampling interval; the ordering is "
                     "unchanged and the fast-m gap is wider. The 0.1-ms evaluation "
                     "grid is the one reported in the manuscript because every other "
                     "audit uses it."),
            "families": native,
        }
    text = json.dumps(summary, indent=1, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(text)


if __name__ == "__main__":
    main()
