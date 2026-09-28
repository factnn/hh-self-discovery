"""Observable-only sparse polynomial field audit for DDAE coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from .ddae_baseline import (
    MODES,
    _delay_arrays,
    _latent_and_velocity,
    _official_params,
)
from .sampling import BalancedWindowSampler, ObservedDataset
from .training import ModeNormalizers, Standardizer, fit_mode_normalizers


def _normalizers(payload, train_data, mode):
    values = payload["modes"][mode]["normalizer"]
    if values:
        return ModeNormalizers(
            Standardizer(**values["command"]),
            Standardizer(**values["response"]),
        )
    return fit_mode_normalizers(train_data, mode)


def _collect(
    model,
    data,
    mode,
    normalizer,
    history_steps,
    windows,
    seed,
    dt_ms,
    batch_size,
):
    sampler = BalancedWindowSampler(data, history_steps, prediction_steps=1)
    references = sampler.sample_mode_epoch(
        mode, windows, seed, strict=False
    )
    response, derivative, command = _delay_arrays(
        sampler, references, normalizer, dt_ms
    )
    latent, velocity = _latent_and_velocity(
        model, response, derivative, batch_size
    )
    return latent, velocity, command


def _stlsq(theta, target, threshold, alpha, iterations=20):
    coefficients = Ridge(alpha=alpha, fit_intercept=False).fit(
        theta, target
    ).coef_.T
    for _ in range(iterations):
        previous = coefficients.copy()
        coefficients[np.abs(coefficients) < threshold] = 0.0
        for output_index in range(target.shape[1]):
            active = coefficients[:, output_index] != 0.0
            if not np.any(active):
                active[np.argmax(np.abs(previous[:, output_index]))] = True
            fit = Ridge(alpha=alpha, fit_intercept=False).fit(
                theta[:, active], target[:, output_index]
            )
            coefficients[:, output_index] = 0.0
            coefficients[active, output_index] = fit.coef_
        if np.array_equal(
            coefficients != 0.0, previous != 0.0
        ):
            break
    return coefficients


def _normalized_mse(target, prediction, scale):
    return float(np.mean(np.square((prediction - target) / scale)))


def audit_ddae_sparse_field(
    run_dir: str | Path,
    data_root: str | Path,
    ood_data_root: str | Path,
    train_windows_per_mode: int = 2000,
    validation_windows_per_mode: int = 1000,
    test_windows_per_mode: int = 1000,
    ood_windows_per_mode: int = 300,
    thresholds=(0.0, 0.003, 0.01, 0.03, 0.1, 0.3),
    tolerance_fraction: float = 0.05,
    ridge_alpha: float = 1e-4,
    seed: int = 26000,
    batch_size: int = 128,
) -> dict:
    """Select STLSQ sparsity on observable validation and score held-out fields."""
    import tensorflow as tf
    from aesindy.net_config import Sindy_Autoencoder

    run_dir = Path(run_dir)
    data_root = Path(data_root)
    ood_data_root = Path(ood_data_root)
    payload = json.loads((run_dir / "training.json").read_text())
    config = payload["config"]
    variant = payload["variant"]
    train_data = ObservedDataset(data_root, "train")
    validation_data = ObservedDataset(data_root, "validation")
    test_data = ObservedDataset(data_root, "test")
    ood_data = ObservedDataset(ood_data_root, "test")
    dt_ms = float(train_data.records[0]["dt_ms"])
    by_mode = {}

    for mode_index, mode in enumerate(MODES):
        params = _official_params(
            config["latent_size"], config["history_steps"], dt_ms, variant
        )
        model = Sindy_Autoencoder(params)
        model([tf.zeros((2, config["history_steps"]), dtype=tf.float32)])
        status = model.load_weights(str(run_dir / mode / "weights"))
        status.expect_partial()
        normalizer = _normalizers(payload, train_data, mode)
        collected = {}
        for split_index, (name, data, windows) in enumerate(
            (
                ("train", train_data, train_windows_per_mode),
                ("validation", validation_data, validation_windows_per_mode),
                ("test", test_data, test_windows_per_mode),
                ("ood", ood_data, ood_windows_per_mode),
            )
        ):
            collected[name] = _collect(
                model,
                data,
                mode,
                normalizer,
                config["history_steps"],
                windows,
                seed + 1000 * split_index + mode_index,
                dt_ms,
                batch_size,
            )

        polynomial = PolynomialFeatures(2, include_bias=True)
        train_z, train_dz, train_u = collected["train"]
        validation_z, validation_dz, validation_u = collected["validation"]
        if variant == "controlled_two_stage":
            train_features = np.column_stack((train_z, train_u))
            validation_features = np.column_stack((validation_z, validation_u))
        else:
            train_features = train_z
            validation_features = validation_z
        theta_train = polynomial.fit_transform(train_features)
        theta_validation = polynomial.transform(validation_features)
        column_scale = np.sqrt(np.mean(np.square(theta_train), axis=0))
        column_scale = np.maximum(column_scale, 1e-8)
        x_train = theta_train / column_scale
        x_validation = theta_validation / column_scale
        velocity_scale = np.maximum(np.std(train_dz, axis=0), 1e-8)

        candidates = []
        for threshold in thresholds:
            coefficients = _stlsq(
                x_train, train_dz, float(threshold), ridge_alpha
            )
            prediction = x_validation @ coefficients
            candidates.append(
                {
                    "threshold": float(threshold),
                    "normalized_mse": _normalized_mse(
                        validation_dz, prediction, velocity_scale
                    ),
                    "r2": float(
                        r2_score(
                            validation_dz,
                            prediction,
                            multioutput="variance_weighted",
                        )
                    ),
                    "active_terms": int(np.count_nonzero(coefficients)),
                }
            )
        best_mse = min(item["normalized_mse"] for item in candidates)
        eligible = [
            item
            for item in candidates
            if item["normalized_mse"]
            <= best_mse * (1.0 + tolerance_fraction)
        ]
        selected = min(
            eligible,
            key=lambda item: (
                item["active_terms"],
                item["normalized_mse"],
            ),
        )

        combined_features = np.concatenate(
            (train_features, validation_features), axis=0
        )
        combined_velocity = np.concatenate(
            (train_dz, validation_dz), axis=0
        )
        x_combined = polynomial.transform(combined_features) / column_scale
        coefficients = _stlsq(
            x_combined,
            combined_velocity,
            selected["threshold"],
            ridge_alpha,
        )
        scores = {}
        for name in ("test", "ood"):
            latent, velocity, command = collected[name]
            features = (
                np.column_stack((latent, command))
                if variant == "controlled_two_stage"
                else latent
            )
            theta = polynomial.transform(features) / column_scale
            prediction = theta @ coefficients
            scores[name] = {
                "r2": float(
                    r2_score(
                        velocity,
                        prediction,
                        multioutput="variance_weighted",
                    )
                ),
                "normalized_mse": _normalized_mse(
                    velocity, prediction, velocity_scale
                ),
            }
        by_mode[mode] = {
            "candidates": candidates,
            "selected_threshold": selected["threshold"],
            "selected_validation_r2": selected["r2"],
            "active_terms": int(np.count_nonzero(coefficients)),
            "available_terms": int(coefficients.size),
            "active_fraction": float(
                np.count_nonzero(coefficients) / coefficients.size
            ),
            "feature_names": polynomial.get_feature_names_out().tolist(),
            "test": scores["test"],
            "ood": scores["ood"],
        }
        tf.keras.backend.clear_session()

    return {
        "run_dir": str(run_dir),
        "variant": variant,
        "latent_size": config["latent_size"],
        "selection": (
            "sparsest threshold within 5% of minimum observable validation "
            "normalized MSE"
        ),
        "thresholds": [float(value) for value in thresholds],
        "tolerance_fraction": tolerance_fraction,
        "ridge_alpha": ridge_alpha,
        "seed": seed,
        "modes": by_mode,
        "mean_test_r2": float(
            np.mean([by_mode[mode]["test"]["r2"] for mode in MODES])
        ),
        "mean_ood_r2": float(
            np.mean([by_mode[mode]["ood"]["r2"] for mode in MODES])
        ),
        "mean_active_fraction": float(
            np.mean([by_mode[mode]["active_fraction"] for mode in MODES])
        ),
    }
