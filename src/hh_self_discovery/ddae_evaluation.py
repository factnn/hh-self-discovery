"""Hidden-gate audit for the official deep delay autoencoder adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .ddae_baseline import MODES, _official_params
from .sampling import BalancedWindowSampler, ObservedDataset
from .training import ModeNormalizers, Standardizer, WindowTorchDataset


GATES = ("n", "m", "h")


def _normalizers(payload, mode):
    values = payload["modes"][mode]["normalizer"]
    return ModeNormalizers(
        Standardizer(**values["command"]),
        Standardizer(**values["response"]),
    )


def _collect(
    model,
    data_root,
    split,
    mode,
    history_steps,
    windows,
    seed,
    normalizers,
):
    data = ObservedDataset(data_root, split)
    sampler = BalancedWindowSampler(data, history_steps, prediction_steps=1)
    references = sampler.sample_mode_epoch(mode, windows, seed, strict=False)
    dataset = WindowTorchDataset(sampler, references, normalizers)
    histories = np.stack(
        [dataset[index]["history"].numpy() for index in range(len(references))]
    )
    latents = model.encoder.predict(
        histories[:, :, 1], batch_size=128, verbose=0
    )
    records = {record["trajectory_id"]: record for record in data.records}
    observed_cache, hidden_cache = {}, {}
    voltages, gates = [], []
    for reference in references:
        trajectory_id = reference.trajectory_id
        record = records[trajectory_id]
        if trajectory_id not in observed_cache:
            with np.load(data_root / record["observed_path"]) as archive:
                observed_cache[trajectory_id] = archive["voltage_mV"].copy()
            with np.load(data_root / record["evaluation_path"]) as archive:
                hidden_cache[trajectory_id] = {
                    gate: archive[gate].copy() for gate in GATES
                }
        voltages.append(observed_cache[trajectory_id][reference.center])
        gates.append(
            [hidden_cache[trajectory_id][gate][reference.center] for gate in GATES]
        )
    return latents, np.asarray(voltages)[:, None], np.asarray(gates)


def _score(train_features, train_gates, test_features, test_gates, ridge_alpha):
    readout = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3, include_bias=False),
        Ridge(alpha=ridge_alpha),
    ).fit(train_features, train_gates)
    values = r2_score(
        test_gates, readout.predict(test_features), multioutput="raw_values"
    )
    return {gate: float(value) for gate, value in zip(GATES, values)}


def evaluate_ddae_gates(
    run_dir: str | Path,
    data_root: str | Path,
    evaluation_data_root: str | Path | None = None,
    windows_per_mode: int = 1000,
    seed: int = 9595,
    ridge_alpha: float = 1e-3,
) -> dict:
    """Fit charts on validation trajectories and score isolated test trajectories."""
    import tensorflow as tf
    from aesindy.net_config import Sindy_Autoencoder

    run_dir = Path(run_dir)
    data_root = Path(data_root)
    evaluation_root = (
        Path(evaluation_data_root) if evaluation_data_root is not None else data_root
    )
    payload = json.loads((run_dir / "training.json").read_text())
    config = payload["config"]
    by_mode = {}
    for mode_index, mode in enumerate(MODES):
        params = _official_params(
            config["latent_size"],
            config["history_steps"],
            float(ObservedDataset(data_root, "train").records[0]["dt_ms"]),
            payload["variant"],
        )
        model = Sindy_Autoencoder(params)
        model([tf.zeros((2, config["history_steps"]), dtype=tf.float32)])
        status = model.load_weights(str(run_dir / mode / "weights"))
        status.expect_partial()  # Optimizer slots are irrelevant for inference.
        normalizers = _normalizers(payload, mode)
        train_z, train_v, train_gates = _collect(
            model,
            data_root,
            "validation",
            mode,
            config["history_steps"],
            windows_per_mode,
            seed + mode_index,
            normalizers,
        )
        test_z, test_v, test_gates = _collect(
            model,
            evaluation_root,
            "test",
            mode,
            config["history_steps"],
            windows_per_mode,
            seed + 10000 + mode_index,
            normalizers,
        )
        feature_pairs = {
            "voltage_only": (train_v, test_v),
            "latent_only": (train_z, test_z),
            "latent_plus_voltage": (
                np.concatenate((train_z, train_v), axis=1),
                np.concatenate((test_z, test_v), axis=1),
            ),
        }
        scores = {
            name: _score(x_train, train_gates, x_test, test_gates, ridge_alpha)
            for name, (x_train, x_test) in feature_pairs.items()
        }
        scores["increment_over_voltage"] = {
            gate: scores["latent_plus_voltage"][gate]
            - scores["voltage_only"][gate]
            for gate in GATES
        }
        by_mode[mode] = scores
        tf.keras.backend.clear_session()

    mean_scores = {
        feature: {
            gate: float(
                np.mean([by_mode[mode][feature][gate] for mode in MODES])
            )
            for gate in GATES
        }
        for feature in (
            "voltage_only",
            "latent_only",
            "latent_plus_voltage",
            "increment_over_voltage",
        )
    }
    return {
        "run_dir": str(run_dir),
        "mapping_data_root": str(data_root),
        "evaluation_data_root": str(evaluation_root),
        "variant": payload["variant"],
        "latent_size": config["latent_size"],
        "windows_per_mode": windows_per_mode,
        "seed": seed,
        "ridge_alpha": ridge_alpha,
        "readout_scope": "mode-specific, matching the official scalar-response delay charts",
        "modes": by_mode,
        "mean": mean_scores,
    }
