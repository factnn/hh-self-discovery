"""Held-out intervention families for out-of-distribution evaluation."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .dataset import _write_trajectory
from .protocols import time_grid
from .simulator import simulate_current_clamp, simulate_voltage_clamp


OOD_FAMILIES = ("current_ramp", "current_multisine", "voltage_ramp", "voltage_tail")


def sample_ood_protocol(family: str, rng: np.random.Generator, dt_ms: float):
    if family == "current_ramp":
        time = time_grid(120.0, dt_ms)
        peak = float(rng.uniform(8.0, 18.0))
        command = np.zeros_like(time)
        rising = (time >= 15.0) & (time < 55.0)
        falling = (time >= 55.0) & (time < 95.0)
        command[rising] = peak * (time[rising] - 15.0) / 40.0
        command[falling] = peak * (95.0 - time[falling]) / 40.0
        trace = simulate_current_clamp(time, command)
        metadata = {"peak_uA_cm2": peak}
    elif family == "current_multisine":
        time = time_grid(160.0, dt_ms)
        frequencies = rng.uniform(1.0, 45.0, size=5)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=5)
        amplitudes = rng.uniform(0.5, 2.5, size=5)
        command = 5.0 + sum(
            amplitude * np.sin(2.0 * np.pi * frequency * time / 1000.0 + phase)
            for amplitude, frequency, phase in zip(amplitudes, frequencies, phases)
        )
        trace = simulate_current_clamp(time, command)
        metadata = {"frequency_min_hz": float(frequencies.min()), "frequency_max_hz": float(frequencies.max())}
    elif family == "voltage_ramp":
        time = time_grid(90.0, dt_ms)
        holding = float(rng.uniform(-100.0, -75.0))
        peak = float(rng.uniform(0.0, 50.0))
        command = np.full_like(time, holding)
        rising = (time >= 10.0) & (time < 45.0)
        falling = (time >= 45.0) & (time < 80.0)
        command[rising] = holding + (peak - holding) * (time[rising] - 10.0) / 35.0
        command[falling] = peak + (holding - peak) * (time[falling] - 45.0) / 35.0
        trace = simulate_voltage_clamp(time, command, include_capacitive_current=False)
        metadata = {"holding_mV": holding, "peak_mV": peak}
    elif family == "voltage_tail":
        time = time_grid(100.0, dt_ms)
        holding = float(rng.uniform(-100.0, -80.0))
        activate = float(rng.uniform(-10.0, 40.0))
        tail_1 = float(rng.uniform(-120.0, -70.0))
        tail_2 = float(rng.uniform(-65.0, -30.0))
        command = np.full_like(time, holding)
        command[(time >= 10.0) & (time < 40.0)] = activate
        command[(time >= 40.0) & (time < 65.0)] = tail_1
        command[(time >= 65.0) & (time < 90.0)] = tail_2
        trace = simulate_voltage_clamp(time, command, include_capacitive_current=False)
        metadata = {"holding_mV": holding, "activate_mV": activate, "tail_1_mV": tail_1, "tail_2_mV": tail_2}
    else:
        raise ValueError(f"unknown OOD family: {family}")
    metadata.update({"family": family, "control_mode": trace.control_mode, "ood": True})
    return trace, metadata


def _generate_one(arguments):
    root, index, dt_ms, seed_sequence = arguments
    rng = np.random.default_rng(seed_sequence)
    family = OOD_FAMILIES[index % len(OOD_FAMILIES)]
    trace, metadata = sample_ood_protocol(family, rng, dt_ms)
    record = _write_trajectory(Path(root), "test", f"trajectory_{index:06d}", trace, metadata)
    record.update({"dt_ms": dt_ms, "seed": int(seed_sequence.entropy)})
    return record


def generate_ood_dataset(output_dir, count: int = 200, dt_ms: float = 0.02, seed: int = 314, workers: int = 1):
    if count < len(OOD_FAMILIES):
        raise ValueError("count is too small")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seeds = np.random.SeedSequence(seed).spawn(count)
    arguments = [(str(root), index, dt_ms, seeds[index]) for index in range(count)]
    if workers == 1:
        records = [_generate_one(item) for item in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_generate_one, arguments))
    records.sort(key=lambda item: item["trajectory_id"])
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records
