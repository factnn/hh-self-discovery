"""Observable-only activity scores used for stratified temporal sampling."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class EventClass(IntEnum):
    STEADY = 0
    COMMAND = 1
    FAST_RESPONSE = 2
    ACTIVE = 3
    RECOVERY = 4


def _positive_percentile(values: np.ndarray, percentile: float, fallback: float = 1.0) -> float:
    positive = np.asarray(values)[np.asarray(values) > 1e-12]
    if positive.size == 0:
        return fallback
    return max(float(np.percentile(positive, percentile)), 1e-12)


def observable_activity(
    time_ms: np.ndarray,
    current_uA_cm2: np.ndarray,
    voltage_mV: np.ndarray,
    control_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return activity scores and coarse event labels without using hidden gates.

    For current clamp, voltage is the response and current is the command. For
    voltage clamp those roles are reversed. Labels are only sampling aids; the
    continuous score remains the preferred weighting signal.
    """
    time = np.asarray(time_ms, dtype=np.float64)
    current = np.asarray(current_uA_cm2, dtype=np.float64)
    voltage = np.asarray(voltage_mV, dtype=np.float64)
    if not (time.shape == current.shape == voltage.shape):
        raise ValueError("time, current, and voltage must have identical shapes")
    if control_mode == "current_clamp":
        command, response = current, voltage
    elif control_mode == "voltage_clamp":
        command, response = voltage, current
    else:
        raise ValueError(f"unknown control mode: {control_mode}")

    d_command = np.abs(np.gradient(command, time))
    d_response = np.abs(np.gradient(response, time))
    d2_response = np.abs(np.gradient(np.gradient(response, time), time))
    command_norm = np.clip(d_command / _positive_percentile(d_command, 75.0), 0.0, 4.0)
    response_norm = np.clip(d_response / _positive_percentile(d_response, 85.0), 0.0, 4.0)
    curvature_norm = np.clip(d2_response / _positive_percentile(d2_response, 85.0), 0.0, 4.0)
    score = (0.35 * command_norm + 0.45 * response_norm + 0.20 * curvature_norm) / 4.0
    score = np.clip(score, 0.0, 1.0)

    labels = np.full(time.shape, EventClass.STEADY, dtype=np.uint8)
    displaced = np.abs(response - response[0]) > 0.05 * max(float(np.ptp(response)), 1e-6)
    labels[displaced] = EventClass.RECOVERY
    labels[score >= 0.12] = EventClass.ACTIVE
    labels[response_norm >= 0.8] = EventClass.FAST_RESPONSE
    command_delta = np.abs(np.diff(command, prepend=command[0]))
    jump_threshold = max(0.02 * float(np.ptp(command)), 1e-9)
    command_points = np.flatnonzero(command_delta >= jump_threshold)
    radius = max(1, int(round(0.2 / np.median(np.diff(time)))))
    for point in command_points:
        labels[max(0, point - radius) : min(len(labels), point + radius + 1)] = EventClass.COMMAND
    return score.astype(np.float32), labels
