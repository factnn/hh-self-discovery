"""Test whether decoded gate trajectories require nonlinear powers of gate state."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from hh_self_discovery.blind_evaluation import TRUE_GATES, _collect_windows, load_latent_checkpoint
from hh_self_discovery.simulator import gate_rates


def chart(alpha):
    return make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False), Ridge(alpha=alpha))


def unpack(items, readout, steps, dt):
    voltage, true_x, mapped_x = [], [], []
    for item in items:
        count = len(item["voltage"]) // steps
        v = item["voltage"].reshape(count, steps)
        true = item["gates"].reshape(count, steps, 3)
        mapped = readout.predict(item["latents"]).reshape(count, steps, 3)
        voltage.append(v[:, 1:-1].reshape(-1))
        true_x.append(np.column_stack([true[:, 1:-1, g].reshape(-1) for g in range(3)]))
        mapped_x.append(np.column_stack([mapped[:, 1:-1, g].reshape(-1) for g in range(3)]))
    def derivative(xs):
        return np.concatenate([np.gradient(x.reshape(-1, steps, 3), dt, axis=1)[:, 1:-1].reshape(-1, 3) for x in xs])
    voltage_array, true_array = np.concatenate(voltage), np.concatenate(true_x)
    true_derivative = np.empty_like(true_array)
    for index, (v, x) in enumerate(zip(voltage_array, true_array)):
        an, bn, am, bm, ah, bh = gate_rates(float(v))
        alpha = np.asarray((an, am, ah)); beta = np.asarray((bn, bm, bh))
        true_derivative[index] = alpha * (1.0 - x) - beta * x
    return voltage_array, true_array, np.concatenate(mapped_x), true_derivative, derivative([readout.predict(i["latents"]) for i in items])


def design(basis, x, degree):
    return np.column_stack([basis * (x[:, None] ** power) for power in range(degree + 1)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--blind-result", required=True)
    p.add_argument("--data", required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--windows-per-mode", type=int, default=500); p.add_argument("--seed", type=int, default=12100)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--output", required=True); a = p.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model, ckpt = load_latent_checkpoint(a.checkpoint, device)
    mapping = _collect_windows(model, ckpt, Path(a.data), "validation", device, a.windows_per_mode, a.seed, stride=a.stride)
    test = _collect_windows(model, ckpt, Path(a.data), "test", device, a.windows_per_mode, a.seed + 10000, stride=a.stride)
    blind = json.load(open(a.blind_result)); readout = chart(blind["ridge_alpha"])
    readout.fit(np.concatenate([x["latents"] for x in mapping]), np.concatenate([x["gates"] for x in mapping]))
    steps = len(range(0, ckpt["prediction_steps"], a.stride)); dt = ckpt["dt_ms"] * a.stride
    train = unpack(mapping, readout, steps, dt); evaluation = unpack(test, readout, steps, dt)
    spline = SplineTransformer(n_knots=12, degree=3, include_bias=True).fit(train[0][:, None])
    btrain, btest = spline.transform(train[0][:, None]), spline.transform(evaluation[0][:, None])
    result = {}
    voltage_grid = np.linspace(-100.0, 50.0, 301)
    support_low, support_high = np.quantile(train[0], [0.01, 0.99])
    support_mask = (voltage_grid >= support_low) & (voltage_grid <= support_high)
    grid_basis = spline.transform(voltage_grid[:, None])
    true_steady, true_tau = [], []
    for voltage_value in voltage_grid:
        an, bn, am, bm, ah, bh = gate_rates(float(voltage_value))
        alpha = np.asarray((an, am, ah)); beta = np.asarray((bn, bm, bh))
        true_steady.append(alpha / (alpha + beta)); true_tau.append(1.0 / (alpha + beta))
    true_steady, true_tau = np.asarray(true_steady), np.asarray(true_tau)
    affine_physics = {}
    for source, xi, dxi, xo, dxo in (("true", train[1], train[3], evaluation[1], evaluation[3]), ("mapped", train[2], train[4], evaluation[2], evaluation[4])):
        result[source] = {}
        for g, gate in enumerate(TRUE_GATES):
            result[source][gate] = {}
            for degree in (1, 2, 3):
                estimator = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 9)))
                estimator.fit(design(btrain, xi[:, g], degree), dxi[:, g])
                pred = estimator.predict(design(btest, xo[:, g], degree))
                result[source][gate][f"degree_{degree}"] = float(r2_score(dxo[:, g], pred))
                if source == "mapped" and degree == 1:
                    test_intercept = estimator.predict(design(btest, np.zeros(len(xo)), 1))
                    test_unit = estimator.predict(design(btest, np.ones(len(xo)), 1))
                    test_slope = test_unit - test_intercept
                    projected_slope = np.minimum(test_slope, -1e-6)
                    test_equilibrium = np.clip(-test_intercept / np.where(np.abs(test_slope) > 1e-8, test_slope, -1e-8), 0.0, 1.0)
                    projected_prediction = projected_slope * (xo[:, g] - test_equilibrium)
                    intercept_curve = estimator.predict(design(grid_basis, np.zeros(len(voltage_grid)), 1))
                    unit_curve = estimator.predict(design(grid_basis, np.ones(len(voltage_grid)), 1))
                    slope_curve = unit_curve - intercept_curve
                    stable = slope_curve < 0
                    inferred_steady = -intercept_curve / slope_curve
                    inferred_tau = -1.0 / slope_curve
                    valid_tau = stable & np.isfinite(inferred_tau) & (inferred_tau > 0)
                    affine_physics[gate] = {
                        "stable_fraction": float(np.mean(stable)),
                        "bounded_steady_fraction": float(np.mean((inferred_steady >= 0) & (inferred_steady <= 1))),
                        "steady_spearman": float(spearmanr(inferred_steady, true_steady[:, g]).statistic),
                        "tau_log_spearman": float(spearmanr(np.log(inferred_tau[valid_tau]), np.log(true_tau[valid_tau, g])).statistic) if valid_tau.sum() > 2 else float("nan"),
                        "tau_log_rmse": float(np.sqrt(np.mean((np.log(inferred_tau[valid_tau]) - np.log(true_tau[valid_tau, g])) ** 2))) if valid_tau.any() else float("nan"),
                        "support_voltage_mV": [float(support_low), float(support_high)],
                        "support_stable_fraction": float(np.mean(stable[support_mask])),
                        "support_bounded_steady_fraction": float(np.mean(((inferred_steady >= 0) & (inferred_steady <= 1))[support_mask])),
                        "support_steady_spearman": float(spearmanr(inferred_steady[support_mask], true_steady[support_mask, g]).statistic),
                        "support_tau_log_spearman": float(spearmanr(np.log(inferred_tau[support_mask & valid_tau]), np.log(true_tau[support_mask & valid_tau, g])).statistic) if (support_mask & valid_tau).sum() > 2 else float("nan"),
                        "support_tau_log_rmse": float(np.sqrt(np.mean((np.log(inferred_tau[support_mask & valid_tau]) - np.log(true_tau[support_mask & valid_tau, g])) ** 2))) if (support_mask & valid_tau).any() else float("nan"),
                        "projected_physical_test_r2": float(r2_score(dxo[:, g], projected_prediction)),
                        "projection_r2_regret": float(r2_score(dxo[:, g], pred) - r2_score(dxo[:, g], projected_prediction)),
                    }
    output = {"checkpoint": a.checkpoint, "blind_result": a.blind_result, "results": result, "mapped_affine_physics": affine_physics}
    Path(a.output).write_text(json.dumps(output, indent=2, sort_keys=True)); print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
