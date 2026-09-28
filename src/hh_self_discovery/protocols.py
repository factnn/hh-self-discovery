"""Deterministic stimulation protocols expressed in milliseconds and millivolts."""

from __future__ import annotations

import numpy as np


def time_grid(duration_ms: float, dt_ms: float) -> np.ndarray:
    """Return an inclusive, uniformly sampled time grid."""
    if duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("duration_ms and dt_ms must be positive")
    steps = int(round(duration_ms / dt_ms))
    if not np.isclose(steps * dt_ms, duration_ms, rtol=0.0, atol=1e-10):
        raise ValueError("duration_ms must be an integer multiple of dt_ms")
    return np.linspace(0.0, duration_ms, steps + 1, dtype=np.float64)


def current_step(
    time_ms: np.ndarray,
    amplitude_uA_cm2: float,
    start_ms: float,
    stop_ms: float,
    baseline_uA_cm2: float = 0.0,
) -> np.ndarray:
    """Create a rectangular current-clamp command."""
    if stop_ms <= start_ms:
        raise ValueError("stop_ms must be greater than start_ms")
    command = np.full_like(time_ms, baseline_uA_cm2, dtype=np.float64)
    active = (time_ms >= start_ms) & (time_ms < stop_ms)
    command[active] += amplitude_uA_cm2
    return command


def voltage_step(
    time_ms: np.ndarray,
    holding_mV: float,
    step_mV: float,
    start_ms: float,
    stop_ms: float,
    tail_mV: float | None = None,
) -> np.ndarray:
    """Create a voltage-clamp step followed by a holding or tail voltage."""
    if stop_ms <= start_ms:
        raise ValueError("stop_ms must be greater than start_ms")
    command = np.full_like(time_ms, holding_mV, dtype=np.float64)
    command[(time_ms >= start_ms) & (time_ms < stop_ms)] = step_mV
    if tail_mV is not None:
        command[time_ms >= stop_ms] = tail_mV
    return command


def chirp_current(
    time_ms: np.ndarray,
    amplitude_uA_cm2: float,
    start_frequency_hz: float,
    stop_frequency_hz: float,
    offset_uA_cm2: float = 0.0,
) -> np.ndarray:
    """Create a linear-frequency chirp current over the full time interval."""
    if start_frequency_hz < 0.0 or stop_frequency_hz <= 0.0:
        raise ValueError("chirp frequencies must be non-negative with a positive stop frequency")
    time_s = (np.asarray(time_ms, dtype=np.float64) - float(time_ms[0])) / 1000.0
    duration_s = float(time_s[-1])
    slope = (stop_frequency_hz - start_frequency_hz) / duration_s
    phase = 2.0 * np.pi * (start_frequency_hz * time_s + 0.5 * slope * time_s**2)
    return offset_uA_cm2 + amplitude_uA_cm2 * np.sin(phase)


def prbs_current(
    time_ms: np.ndarray,
    rng: np.random.Generator,
    block_ms: float,
    low_uA_cm2: float,
    high_uA_cm2: float,
) -> np.ndarray:
    """Create a reproducible piecewise-constant random current command."""
    if block_ms <= 0.0 or high_uA_cm2 <= low_uA_cm2:
        raise ValueError("invalid PRBS block duration or amplitude bounds")
    block_index = np.floor((time_ms - time_ms[0]) / block_ms).astype(int)
    levels = rng.uniform(low_uA_cm2, high_uA_cm2, int(block_index.max()) + 1)
    return levels[block_index]


def smooth_random_current(
    time_ms: np.ndarray,
    rng: np.random.Generator,
    correlation_ms: float = 3.0,
    mean_uA_cm2: float = 5.0,
    std_uA_cm2: float = 3.0,
) -> np.ndarray:
    """Create a reproducible low-pass random current without command jumps."""
    if correlation_ms <= 0.0 or std_uA_cm2 < 0.0:
        raise ValueError("correlation_ms must be positive and std must be non-negative")
    dt_ms = float(np.median(np.diff(time_ms)))
    alpha = np.exp(-dt_ms / correlation_ms)
    values = np.empty_like(time_ms, dtype=np.float64)
    values[0] = mean_uA_cm2
    innovation_scale = std_uA_cm2 * np.sqrt(1.0 - alpha * alpha)
    for index in range(1, len(values)):
        values[index] = (
            mean_uA_cm2
            + alpha * (values[index - 1] - mean_uA_cm2)
            + innovation_scale * rng.normal()
        )
    return values


def prepulse_voltage(
    time_ms: np.ndarray,
    holding_mV: float,
    prepulse_mV: float,
    test_mV: float,
    prepulse_start_ms: float,
    test_start_ms: float,
    test_stop_ms: float,
    tail_mV: float | None = None,
) -> np.ndarray:
    """Create holding, prepulse, test-pulse, and optional tail phases."""
    if not prepulse_start_ms < test_start_ms < test_stop_ms:
        raise ValueError("prepulse and test times must be strictly ordered")
    command = np.full_like(time_ms, holding_mV, dtype=np.float64)
    command[(time_ms >= prepulse_start_ms) & (time_ms < test_start_ms)] = prepulse_mV
    command[(time_ms >= test_start_ms) & (time_ms < test_stop_ms)] = test_mV
    if tail_mV is not None:
        command[time_ms >= test_stop_ms] = tail_mV
    return command
