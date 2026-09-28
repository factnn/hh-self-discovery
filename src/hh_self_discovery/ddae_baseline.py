"""Trajectory-isolated HH adapter for the official deep delay autoencoder."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from .sampling import BalancedWindowSampler, ObservedDataset
from .training import WindowTorchDataset, fit_mode_normalizers


MODES = ("current_clamp", "voltage_clamp")
UPSTREAM_COMMIT = "1f43e8da8340a43d7370d00f7b1ed21985609671"


def _delay_arrays(sampler, references, normalizer, dt_ms):
    dataset = WindowTorchDataset(sampler, references, normalizer)
    histories = np.stack(
        [dataset[index]["history"].numpy() for index in range(len(dataset))]
    )
    command = histories[:, -1, 0].astype(np.float32)
    response = histories[:, :, 1].astype(np.float32)
    derivative = np.gradient(response, dt_ms, axis=1).astype(np.float32)
    return response, derivative, command


def _official_params(latent_size, history_steps, dt_ms, variant):
    from aesindy.sindy_utils import library_size

    autonomous = variant == "autonomous"
    return {
        "model": "hh_observed",
        "case": variant,
        "latent_dim": latent_size,
        "input_dim": history_steps,
        "widths": [history_steps // 2, history_steps // 4],
        "activation": "elu",
        "library_dim": library_size(latent_size, 2, False, True),
        "poly_order": 2,
        "include_sine": False,
        "exact_features": False,
        "coefficient_initialization": "random_normal",
        "coefficient_threshold": 1e-6,
        "threshold_frequency": 100,
        "fixed_coefficient_mask": False,
        "fix_coefs": False,
        "trainable_auto": True,
        "sindy_pert": 0.0,
        "actual_coefficients": None,
        "use_bias": True,
        "loss_weight_layer_l2": 0.0,
        "loss_weight_layer_l1": 0.0,
        "sparse_weighting": None,
        "dt": dt_ms,
        "max_epochs": 1,
        "ode_net": False,
        "ode_net_widths": [1.5, 3],
        "loss_weight_rec": 1.0,
        "loss_weight_sindy_z": 1e-4 if autonomous else 0.0,
        "loss_weight_sindy_x": 1e-3 if autonomous else 0.0,
        "loss_weight_sindy_regularization": 1e-5 if autonomous else 0.0,
        "loss_weight_integral": 0.0,
        "loss_weight_x0": 0.0,
        "print_frequency": 10,
    }


def _latent_and_velocity(model, response, derivative, batch_size):
    import tensorflow as tf

    latent_batches = []
    velocity_batches = []
    for start in range(0, len(response), batch_size):
        x = tf.convert_to_tensor(response[start : start + batch_size])
        dx = tf.convert_to_tensor(derivative[start : start + batch_size])
        with tf.GradientTape() as tape:
            tape.watch(x)
            z = model.encoder(x, training=False)
        jacobian = tape.batch_jacobian(z, x)
        dz = tf.matmul(jacobian, tf.expand_dims(dx, 2))[:, :, 0]
        latent_batches.append(z.numpy())
        velocity_batches.append(dz.numpy())
    return np.concatenate(latent_batches), np.concatenate(velocity_batches)


def train_ddae_baseline(
    data_root: str,
    output_dir: str,
    latent_size: int,
    variant: str = "controlled_two_stage",
    history_steps: int = 128,
    train_windows_per_mode: int = 2000,
    validation_windows_per_mode: int = 200,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    ridge_alpha: float = 1e-4,
    seed: int = 51,
) -> dict:
    """Train official delay encoders; optionally fit dz=f(z, command) afterward."""
    if variant not in {"autonomous", "controlled_two_stage"}:
        raise ValueError("variant must be autonomous or controlled_two_stage")
    import tensorflow as tf
    from aesindy.net_config import Sindy_Autoencoder

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_data = ObservedDataset(data_root, "train")
    validation_data = ObservedDataset(data_root, "validation")
    dt_ms = float(train_data.records[0]["dt_ms"])
    results = {}

    for mode_index, mode in enumerate(MODES):
        normalizer = fit_mode_normalizers(train_data, mode)
        train_sampler = BalancedWindowSampler(train_data, history_steps, 1)
        validation_sampler = BalancedWindowSampler(validation_data, history_steps, 1)
        train_refs = train_sampler.sample_mode_epoch(
            mode, train_windows_per_mode, seed + mode_index
        )
        validation_refs = validation_sampler.sample_mode_epoch(
            mode, validation_windows_per_mode, seed + 10000 + mode_index
        )
        train_x, train_dx, train_u = _delay_arrays(
            train_sampler, train_refs, normalizer, dt_ms
        )
        validation_x, validation_dx, validation_u = _delay_arrays(
            validation_sampler, validation_refs, normalizer, dt_ms
        )
        params = _official_params(latent_size, history_steps, dt_ms, variant)
        params["max_epochs"] = epochs
        model = Sindy_Autoencoder(params)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_total_loss", patience=15, restore_best_weights=True
            )
        ]
        history = model.fit(
            x=[train_x, train_dx],
            y=[train_x, train_dx],
            validation_data=(
                [validation_x, validation_dx],
                [validation_x, validation_dx],
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
            shuffle=True,
        )
        train_z, train_dz = _latent_and_velocity(
            model, train_x, train_dx, batch_size
        )
        validation_z, validation_dz = _latent_and_velocity(
            model, validation_x, validation_dx, batch_size
        )
        if variant == "controlled_two_stage":
            train_features = np.column_stack([train_z, train_u])
            validation_features = np.column_stack([validation_z, validation_u])
        else:
            train_features = train_z
            validation_features = validation_z
        library = PolynomialFeatures(2, include_bias=True)
        theta_train = library.fit_transform(train_features)
        theta_validation = library.transform(validation_features)
        regressor = Ridge(alpha=ridge_alpha, fit_intercept=False)
        regressor.fit(theta_train, train_dz)
        predicted_dz = regressor.predict(theta_validation)
        reconstruction = model.decoder(validation_z, training=False).numpy()

        mode_dir = output / mode
        mode_dir.mkdir(exist_ok=True)
        model.save_weights(mode_dir / "weights")
        np.savez_compressed(
            mode_dir / "controlled_sindy.npz",
            coefficients=regressor.coef_,
            powers=library.powers_,
        )
        results[mode] = {
            "reconstruction_mse": float(
                np.mean(np.square(reconstruction - validation_x))
            ),
            "velocity_r2": float(
                r2_score(validation_dz, predicted_dz, multioutput="variance_weighted")
            ),
            "latent_std": np.std(validation_z, axis=0).tolist(),
            # The upstream subclass trains encoder/decoder through a custom
            # train_step, so the top-level Keras container is never marked built.
            "parameters": int(
                sum(np.prod(variable.shape) for variable in model.trainable_variables)
            ),
            "history": {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            "normalizer": {
                "command": asdict(normalizer.command),
                "response": asdict(normalizer.response),
            },
            "feature_names": library.get_feature_names_out().tolist(),
        }
        tf.keras.backend.clear_session()

    payload = {
        "method": "official_deep_delay_autoencoder",
        "upstream_commit": UPSTREAM_COMMIT,
        "variant": variant,
        "config": {
            "data_root": data_root,
            "latent_size": latent_size,
            "history_steps": history_steps,
            "train_windows_per_mode": train_windows_per_mode,
            "validation_windows_per_mode": validation_windows_per_mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "ridge_alpha": ridge_alpha,
            "seed": seed,
        },
        "modes": results,
        "scope": (
            "official autonomous delay AE+SINDy"
            if variant == "autonomous"
            else "official delay encoder plus observable-command controlled SINDy"
        ),
    }
    (output / "training.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload
