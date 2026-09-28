"""Metrics that expose flat-line failure instead of hiding it in global MSE."""

from __future__ import annotations

import numpy as np


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    activity_score: np.ndarray | None = None,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must have identical [window, time] shapes")
    error = prediction - target
    mse = float(np.mean(error**2))
    target_std = max(float(np.std(target)), 1e-12)
    result = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "relative_rmse": float(np.sqrt(mse) / target_std),
    }
    if target.shape[1] > 1:
        derivative_error = np.diff(prediction, axis=1) - np.diff(target, axis=1)
        result["derivative_rmse"] = float(np.sqrt(np.mean(derivative_error**2)))
    if activity_score is not None:
        activity = np.asarray(activity_score, dtype=np.float64)
        if activity.shape != target.shape:
            raise ValueError("activity_score must match target shape")
        weights = 1.0 + 4.0 * np.clip(activity, 0.0, 1.0)
        result["activity_weighted_rmse"] = float(np.sqrt(np.sum(weights * error**2) / np.sum(weights)))
    return result


def waveform_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    control_mode: str,
    dt_ms: float,
) -> dict[str, float | int | None]:
    """Summarize event morphology that a flat prediction cannot hide."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 2:
        raise ValueError("target and prediction must have identical [window, time] shapes")
    if dt_ms <= 0.0:
        raise ValueError("dt_ms must be positive")
    if control_mode == "voltage_clamp":
        target_peak_index = np.argmax(np.abs(target), axis=1)
        prediction_peak_index = np.argmax(np.abs(prediction), axis=1)
        target_peak = np.take_along_axis(target, target_peak_index[:, None], axis=1)[:, 0]
        prediction_peak = np.take_along_axis(
            prediction, prediction_peak_index[:, None], axis=1
        )[:, 0]
        return {
            "peak_current_mae_uA_cm2": float(np.mean(np.abs(prediction_peak - target_peak))),
            "peak_timing_mae_ms": float(
                np.mean(np.abs(prediction_peak_index - target_peak_index)) * dt_ms
            ),
        }
    if control_mode != "current_clamp":
        raise ValueError(f"unknown control_mode: {control_mode}")

    target_crossings = (target[:, :-1] < 0.0) & (target[:, 1:] >= 0.0)
    prediction_crossings = (prediction[:, :-1] < 0.0) & (prediction[:, 1:] >= 0.0)
    target_counts = target_crossings.sum(axis=1)
    prediction_counts = prediction_crossings.sum(axis=1)
    both_spike = (target_counts > 0) & (prediction_counts > 0)
    target_peak_index = np.argmax(target, axis=1)
    prediction_peak_index = np.argmax(prediction, axis=1)
    result: dict[str, float | int | None] = {
        "spike_count_mae": float(np.mean(np.abs(prediction_counts - target_counts))),
        "spike_count_exact_fraction": float(np.mean(prediction_counts == target_counts)),
        "peak_voltage_mae_mV": float(
            np.mean(np.abs(prediction.max(axis=1) - target.max(axis=1)))
        ),
        "matched_spike_windows": int(both_spike.sum()),
        "first_spike_timing_mae_ms": None,
        "repolarization_timing_mae_ms": None,
    }
    if both_spike.any():
        target_first = np.argmax(target_crossings[both_spike], axis=1) + 1
        prediction_first = np.argmax(prediction_crossings[both_spike], axis=1) + 1
        result["first_spike_timing_mae_ms"] = float(
            np.mean(np.abs(prediction_first - target_first)) * dt_ms
        )
    repolarization_errors = []
    for index in range(len(target)):
        target_after = np.flatnonzero(target[index, target_peak_index[index] :] <= -50.0)
        prediction_after = np.flatnonzero(
            prediction[index, prediction_peak_index[index] :] <= -50.0
        )
        if len(target_after) and len(prediction_after):
            target_time = target_peak_index[index] + target_after[0]
            prediction_time = prediction_peak_index[index] + prediction_after[0]
            repolarization_errors.append(abs(prediction_time - target_time) * dt_ms)
    if repolarization_errors:
        result["repolarization_timing_mae_ms"] = float(np.mean(repolarization_errors))
    return result
