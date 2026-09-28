"""Relate observable/latent-only checkpoint metrics to post-hoc physical charts."""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from hh_self_discovery.blind_evaluation import load_latent_checkpoint


CHART_PATTERNS = (
    "*_vc_calibrated_seed*.json",
    "*_vc_crossmain_cal_seed*.json",
    "*_vc_fullseed_cal_seed*.json",
    "*_vc_hierarchical_seed*.json",
)
TRAJECTORY_CHART_PATTERNS = ("*_vc_trajcal_seed*.json",)


def checkpoint_key(path):
    match = re.search(r"(latentdim[456]_kw24_h750_s\d+)", str(path))
    return match.group(1) if match else None


def chart_targets(root, patterns=CHART_PATTERNS):
    grouped = {}
    for pattern in patterns:
        for filename in glob.glob(str(root / pattern)):
            key = checkpoint_key(filename)
            if key is None:
                continue
            data = json.loads(Path(filename).read_text())
            test = data["results"]["field_weight_1"]["test"]
            decomposition = test["field_error_decomposition"]["m"]
            grouped.setdefault(key, []).append({
                "state_amplification": decomposition["state_amplification_rmse"],
                "coordinate_residual": decomposition["coordinate_residual_rmse"],
                "total_rmse": decomposition["total_rmse"],
                "error_correlation": decomposition["error_correlation"],
            })
    result = {}
    for key, values in grouped.items():
        result[key] = {
            name + "_mean": float(np.mean([value[name] for value in values]))
            for name in values[0]
        }
        result[key]["total_rmse_within_std"] = float(
            np.std([value["total_rmse"] for value in values], ddof=1)
            if len(values) > 1 else 0.0
        )
        result[key]["readouts"] = len(values)
    return result


def proxy_record(root, key):
    checkpoint_path = root / key / "best.pt"
    training_path = root / key / "training.json"
    prediction_path = root / f"{key}_test20ms.json"
    intrinsic_candidates = sorted(root.glob(f"{key}_intrinsic_dimension*.json"))
    if not all(path.exists() for path in (checkpoint_path, training_path, prediction_path)):
        return None
    training = json.loads(training_path.read_text())
    prediction = json.loads(prediction_path.read_text())["baseline"]
    intrinsic = json.loads(intrinsic_candidates[0].read_text()) if intrinsic_candidates else {}
    model, _ = load_latent_checkpoint(str(checkpoint_path), torch.device("cpu"))
    voltage = torch.linspace(-100.0, 50.0, 601)
    with torch.no_grad():
        steady, tau = model.gate_kinetics(voltage)
    tau_values = tau.numpy().reshape(-1)
    steady_values = steady.numpy()
    return {
        "prediction_mean_nrmse": prediction["mean_normalized_rmse"],
        "prediction_current_nrmse": prediction["current_clamp"]["normalized_rmse"],
        "prediction_voltage_nrmse": prediction["voltage_clamp"]["normalized_rmse"],
        "best_selection_score": training["best_selection_score"],
        "epochs_completed": training["epochs_completed"],
        "intrinsic_participation_ratio": intrinsic.get("participation_ratio"),
        "intrinsic_effective_rank": intrinsic.get("entropy_effective_rank"),
        "intrinsic_pca95": intrinsic.get("pca_dimension_95"),
        "tau_min": float(np.min(tau_values)),
        "tau_median": float(np.median(tau_values)),
        "tau_max": float(np.max(tau_values)),
        "tau_stiffness_ratio": float(np.max(tau_values) / np.min(tau_values)),
        "fast_tau_fraction_lt_1ms": float(np.mean(tau_values < 1.0)),
        "steady_curve_variation": float(np.mean(np.std(steady_values, axis=0))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/phase2")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--trajectory-only", action="store_true",
        help="Use only whole-trajectory calibration readouts (E83-E84).",
    )
    args = parser.parse_args()
    root = Path(args.runs)
    patterns = TRAJECTORY_CHART_PATTERNS if args.trajectory_only else CHART_PATTERNS
    targets = chart_targets(root, patterns)
    records = []
    for key, target in sorted(targets.items()):
        proxies = proxy_record(root, key)
        if proxies is not None:
            records.append({"checkpoint": key, "proxies": proxies, "targets": target})
    correlations = {}
    if records:
        proxy_names = records[0]["proxies"]
        target_names = [name for name in records[0]["targets"] if name != "readouts"]
        for proxy in proxy_names:
            values = [record["proxies"][proxy] for record in records]
            if any(value is None for value in values):
                continue
            correlations[proxy] = {}
            for target in target_names:
                statistic, pvalue = spearmanr(
                    values, [record["targets"][target] for record in records]
                )
                correlations[proxy][target] = {
                    "spearman": float(statistic), "pvalue": float(pvalue)
                }
    output = {"records": records, "correlations": correlations, "count": len(records)}
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
