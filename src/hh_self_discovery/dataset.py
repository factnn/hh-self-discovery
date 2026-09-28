"""Protocol sampling and leak-resistant trajectory serialization."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .events import observable_activity
from .protocols import chirp_current, current_step, prepulse_voltage, prbs_current, time_grid, voltage_step
from .simulator import HHParameters, Trajectory, simulate_current_clamp, simulate_voltage_clamp


PROTOCOL_FAMILIES = ("current_step", "current_chirp", "current_prbs", "voltage_step", "voltage_prepulse")


def sample_protocol(
    family: str,
    rng: np.random.Generator,
    dt_ms: float,
    params: HHParameters = HHParameters(),
) -> tuple[Trajectory, dict[str, float | str]]:
    """Sample and simulate one fixed-parameter intervention protocol."""
    if family == "current_step":
        time = time_grid(100.0, dt_ms)
        amplitude = float(rng.uniform(3.0, 16.0))
        start = float(rng.uniform(8.0, 20.0))
        stop = float(rng.uniform(55.0, 85.0))
        command = current_step(time, amplitude, start, stop)
        trace = simulate_current_clamp(time, command, params=params)
        metadata = {"amplitude_uA_cm2": amplitude, "start_ms": start, "stop_ms": stop}
    elif family == "current_chirp":
        time = time_grid(200.0, dt_ms)
        amplitude = float(rng.uniform(3.0, 12.0))
        offset = float(rng.uniform(2.0, 8.0))
        start_hz = float(rng.uniform(0.5, 5.0))
        stop_hz = float(rng.uniform(30.0, 100.0))
        command = chirp_current(time, amplitude, start_hz, stop_hz, offset)
        trace = simulate_current_clamp(time, command, params=params)
        metadata = {"amplitude_uA_cm2": amplitude, "offset_uA_cm2": offset, "start_hz": start_hz, "stop_hz": stop_hz}
    elif family == "current_prbs":
        time = time_grid(150.0, dt_ms)
        block = float(rng.uniform(1.0, 8.0))
        command = prbs_current(time, rng, block, -4.0, 14.0)
        trace = simulate_current_clamp(time, command, params=params)
        metadata = {"block_ms": block, "low_uA_cm2": -4.0, "high_uA_cm2": 14.0}
    elif family == "voltage_step":
        time = time_grid(60.0, dt_ms)
        holding = float(rng.uniform(-100.0, -60.0))
        step = float(rng.uniform(-80.0, 40.0))
        tail = float(rng.uniform(-100.0, -50.0))
        command = voltage_step(time, holding, step, 5.0, 45.0, tail)
        trace = simulate_voltage_clamp(time, command, params=params, include_capacitive_current=False)
        metadata = {"holding_mV": holding, "step_mV": step, "tail_mV": tail}
    elif family == "voltage_prepulse":
        time = time_grid(70.0, dt_ms)
        holding = float(rng.uniform(-100.0, -70.0))
        prepulse = float(rng.uniform(-120.0, 20.0))
        test = float(rng.uniform(-20.0, 30.0))
        command = prepulse_voltage(time, holding, prepulse, test, 5.0, 30.0, 55.0, holding)
        trace = simulate_voltage_clamp(time, command, params=params, include_capacitive_current=False)
        metadata = {"holding_mV": holding, "prepulse_mV": prepulse, "test_mV": test}
    else:
        raise ValueError(f"unknown protocol family: {family}")
    metadata["family"] = family
    metadata["control_mode"] = trace.control_mode
    return trace, metadata


def _write_trajectory(root: Path, split: str, trajectory_id: str, trace: Trajectory, metadata: dict) -> dict:
    observed_dir = root / split / "observed"
    evaluation_dir = root / split / "evaluation_only"
    observed_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    score, labels = observable_activity(trace.time_ms, trace.current_uA_cm2, trace.voltage_mV, trace.control_mode)
    observed_path = observed_dir / f"{trajectory_id}.npz"
    evaluation_path = evaluation_dir / f"{trajectory_id}.npz"
    np.savez_compressed(
        observed_path,
        time_ms=trace.time_ms,
        current_uA_cm2=trace.current_uA_cm2,
        voltage_mV=trace.voltage_mV,
        activity_score=score,
        event_label=labels,
    )
    np.savez_compressed(
        evaluation_path,
        n=trace.n,
        m=trace.m,
        h=trace.h,
        i_na_uA_cm2=trace.i_na_uA_cm2,
        i_k_uA_cm2=trace.i_k_uA_cm2,
        i_leak_uA_cm2=trace.i_leak_uA_cm2,
    )
    return {
        "trajectory_id": trajectory_id,
        "split": split,
        "observed_path": str(observed_path.relative_to(root)),
        "evaluation_path": str(evaluation_path.relative_to(root)),
        "num_samples": len(trace.time_ms),
        **metadata,
    }


def _generate_one(arguments: tuple[str, str, int, float, np.random.SeedSequence]) -> dict:
    root, split, index, dt_ms, seed_sequence = arguments
    rng = np.random.default_rng(seed_sequence)
    family = PROTOCOL_FAMILIES[index % len(PROTOCOL_FAMILIES)]
    trace, metadata = sample_protocol(family, rng, dt_ms)
    return _write_trajectory(Path(root), split, f"trajectory_{index:06d}", trace, metadata)


def generate_dataset(
    output_dir: str | Path,
    count: int,
    dt_ms: float = 0.02,
    seed: int = 42,
    workers: int = 1,
) -> list[dict]:
    """Generate a deterministic, protocol-stratified dataset and JSONL manifest."""
    if count < len(PROTOCOL_FAMILIES):
        raise ValueError(f"count must be at least {len(PROTOCOL_FAMILIES)}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if workers < 1:
        raise ValueError("workers must be positive")
    split_by_index = {}
    split_rng = np.random.default_rng(seed)
    family_indices = [list(range(offset, count, len(PROTOCOL_FAMILIES))) for offset in range(len(PROTOCOL_FAMILIES))]
    if min(len(indices) for indices in family_indices) >= 10:
        for indices in family_indices:
            order = split_rng.permutation(indices)
            train_end = int(round(0.8 * len(indices)))
            validation_end = train_end + int(round(0.1 * len(indices)))
            for rank, index in enumerate(order):
                split_by_index[int(index)] = (
                    "train" if rank < train_end else "validation" if rank < validation_end else "test"
                )
    else:
        order = split_rng.permutation(count)
        train_end = int(round(0.8 * count))
        validation_end = train_end + int(round(0.1 * count))
        for rank, index in enumerate(order):
            split_by_index[int(index)] = (
                "train" if rank < train_end else "validation" if rank < validation_end else "test"
            )

    seed_sequences = np.random.SeedSequence(seed).spawn(count)
    arguments = [(str(root), split_by_index[index], index, dt_ms, seed_sequences[index]) for index in range(count)]
    if workers == 1:
        records = [_generate_one(argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_generate_one, arguments))
    records.sort(key=lambda record: record["trajectory_id"])
    for record in records:
        record["dt_ms"] = dt_ms
        record["seed"] = seed
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return records
