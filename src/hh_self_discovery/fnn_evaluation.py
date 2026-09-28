"""Hidden-gate audit for the representation-only official FNN baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .sampling import BalancedWindowSampler, ObservedDataset
from .training import ModeNormalizers, Standardizer, WindowTorchDataset


GATES = ("n", "m", "h")
MODES = ("current_clamp", "voltage_clamp")


def _normalizers(payload: dict, mode: str) -> ModeNormalizers:
    values = payload["modes"][mode]["normalizer"]
    return ModeNormalizers(
        Standardizer(**values["command"]),
        Standardizer(**values["response"]),
    )


def _collect(
    embedding,
    data_root: Path,
    split: str,
    mode: str,
    history_steps: int,
    windows: int,
    seed: int,
    normalizers: ModeNormalizers,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = ObservedDataset(data_root, split)
    sampler = BalancedWindowSampler(data, history_steps, prediction_steps=1)
    references = sampler.sample_mode_epoch(mode, windows, seed, strict=False)
    windows_dataset = WindowTorchDataset(sampler, references, normalizers)
    histories = np.stack(
        [windows_dataset[index]["history"].numpy() for index in range(len(references))]
    )
    latents = embedding.model.encoder.predict(histories, batch_size=128, verbose=0)
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


def _score(
    train_features: np.ndarray,
    train_gates: np.ndarray,
    test_features: np.ndarray,
    test_gates: np.ndarray,
    ridge_alpha: float,
) -> dict[str, float]:
    readout = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=3, include_bias=False),
        Ridge(alpha=ridge_alpha),
    ).fit(train_features, train_gates)
    values = r2_score(
        test_gates, readout.predict(test_features), multioutput="raw_values"
    )
    return {gate: float(value) for gate, value in zip(GATES, values)}


def evaluate_fnn_gates(
    run_dir: str | Path,
    data_root: str | Path,
    evaluation_data_root: str | Path | None = None,
    windows_per_mode: int = 1000,
    seed: int = 9595,
    ridge_alpha: float = 1e-3,
) -> dict:
    """Fit readouts only on validation trajectories and score isolated test trajectories."""
    import tensorflow as tf
    from models import LSTMEmbedding
    from regularizers import FNN

    run_dir = Path(run_dir)
    data_root = Path(data_root)
    evaluation_root = (
        Path(evaluation_data_root) if evaluation_data_root is not None else data_root
    )
    payload = json.loads((run_dir / "training.json").read_text())
    config = payload["config"]
    by_mode = {}
    for mode_index, mode in enumerate(MODES):
        embedding = LSTMEmbedding(
            config["latent_size"],
            time_window=config["history_steps"],
            n_features=2,
            latent_regularizer=FNN(config["regularization_strength"]),
            random_state=config["seed"] + mode_index,
        )
        embedding.model(
            tf.zeros((2, config["history_steps"], 2), dtype=tf.float32),
            training=False,
        )
        embedding.model.load_weights(str(run_dir / mode / "weights"))
        normalizers = _normalizers(payload, mode)
        train_z, train_v, train_gates = _collect(
            embedding,
            data_root,
            "validation",
            mode,
            config["history_steps"],
            windows_per_mode,
            seed + mode_index,
            normalizers,
        )
        test_z, test_v, test_gates = _collect(
            embedding,
            evaluation_root,
            "test",
            mode,
            config["history_steps"],
            windows_per_mode,
            seed + 10000 + mode_index,
            normalizers,
        )
        features = {
            "voltage_only": (train_v, test_v),
            "latent_only": (train_z, test_z),
            "latent_plus_voltage": (
                np.concatenate((train_z, train_v), axis=1),
                np.concatenate((test_z, test_v), axis=1),
            ),
        }
        scores = {
            name: _score(x_train, train_gates, x_test, test_gates, ridge_alpha)
            for name, (x_train, x_test) in features.items()
        }
        scores["increment_over_voltage"] = {
            gate: scores["latent_plus_voltage"][gate] - scores["voltage_only"][gate]
            for gate in GATES
        }
        by_mode[mode] = scores
        tf.keras.backend.clear_session()

    mean_scores = {}
    for feature in (
        "voltage_only",
        "latent_only",
        "latent_plus_voltage",
        "increment_over_voltage",
    ):
        mean_scores[feature] = {
            gate: float(np.mean([by_mode[mode][feature][gate] for mode in MODES]))
            for gate in GATES
        }
    return {
        "run_dir": str(run_dir),
        "mapping_data_root": str(data_root),
        "evaluation_data_root": str(evaluation_root),
        "latent_size": config["latent_size"],
        "windows_per_mode": windows_per_mode,
        "seed": seed,
        "ridge_alpha": ridge_alpha,
        "readout_scope": "mode-specific because the official FNN baseline has no shared controlled chart",
        "modes": by_mode,
        "mean": mean_scores,
    }
