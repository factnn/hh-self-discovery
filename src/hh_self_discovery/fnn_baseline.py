"""Trajectory-isolated adapter for the official false-nearest-neighbor embedding."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .sampling import BalancedWindowSampler, ObservedDataset
from .training import WindowTorchDataset, fit_mode_normalizers


MODES = ("current_clamp", "voltage_clamp")
UPSTREAM_COMMIT = "4fdeb15639c755150dee5fda65c47f78a4e136b3"


def _window_array(
    sampler: BalancedWindowSampler,
    references: list,
    normalizer,
) -> np.ndarray:
    dataset = WindowTorchDataset(sampler, references, normalizer)
    return np.stack([dataset[index]["history"].numpy() for index in range(len(dataset))])


def train_fnn_baseline(
    data_root: str,
    output_dir: str,
    latent_size: int,
    history_steps: int = 750,
    train_windows_per_mode: int = 2000,
    validation_windows_per_mode: int = 200,
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    regularization_strength: float = 1.0,
    seed: int = 51,
) -> dict:
    """Fit the published LSTM embedding without joining trajectory boundaries."""
    import tensorflow as tf
    from models import LSTMEmbedding
    from regularizers import FNN

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    train_data = ObservedDataset(data_root, "train")
    validation_data = ObservedDataset(data_root, "validation")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {}

    for mode_index, mode in enumerate(MODES):
        normalizer = fit_mode_normalizers(train_data, mode)
        train_sampler = BalancedWindowSampler(
            train_data, history_steps=history_steps, prediction_steps=1
        )
        validation_sampler = BalancedWindowSampler(
            validation_data, history_steps=history_steps, prediction_steps=1
        )
        train_refs = train_sampler.sample_mode_epoch(
            mode, train_windows_per_mode, seed + mode_index
        )
        validation_refs = validation_sampler.sample_mode_epoch(
            mode, validation_windows_per_mode, seed + 10000 + mode_index
        )
        train_windows = _window_array(train_sampler, train_refs, normalizer)
        validation_windows = _window_array(
            validation_sampler, validation_refs, normalizer
        )

        embedding = LSTMEmbedding(
            latent_size,
            time_window=history_steps,
            n_features=2,
            latent_regularizer=FNN(regularization_strength),
            random_state=seed + mode_index,
        )
        embedding.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )
        history = embedding.model.fit(
            train_windows,
            train_windows,
            validation_data=(validation_windows, validation_windows),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
        )
        reconstruction = embedding.model.predict(
            validation_windows, batch_size=batch_size, verbose=0
        )
        latents = embedding.model.encoder.predict(
            validation_windows, batch_size=batch_size, verbose=0
        )
        mode_dir = output / mode
        mode_dir.mkdir(exist_ok=True)
        embedding.model.save_weights(mode_dir / "weights")
        np.save(mode_dir / "validation_latents.npy", latents)
        mode_result = {
            "reconstruction_mse": float(
                np.mean(np.square(reconstruction - validation_windows))
            ),
            "latent_std": np.std(latents, axis=0).tolist(),
            "parameters": int(embedding.model.count_params()),
            "history": {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            "normalizer": {
                "command": asdict(normalizer.command),
                "response": asdict(normalizer.response),
            },
        }
        results[mode] = mode_result
        tf.keras.backend.clear_session()

    payload = {
        "method": "official_fnn_lstm_embedding",
        "upstream_commit": UPSTREAM_COMMIT,
        "config": {
            "data_root": data_root,
            "latent_size": latent_size,
            "history_steps": history_steps,
            "train_windows_per_mode": train_windows_per_mode,
            "validation_windows_per_mode": validation_windows_per_mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "regularization_strength": regularization_strength,
            "seed": seed,
        },
        "modes": results,
        "scope": "representation-only; observable rollout and generator are N/A",
    }
    (output / "training.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload
