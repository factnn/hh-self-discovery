"""Generate active voltage-clamp trajectories selected by observability search."""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .dataset import _write_trajectory
from .protocols import time_grid
from .simulator import simulate_voltage_clamp


ACTIVE_PROTOCOLS = {
    "active_34": (-108.0563, 8.8296, 42.1734, -76.3975, (2.60, 11.28, 5.06, 1.06)),
    "active_18": (-95.0618, 11.6128, 40.2245, -116.0812, (2.44, 3.44, 10.32, 3.80)),
    "active_09": (-93.0639, -32.7051, 20.5538, -116.6736, (0.82, 7.16, 5.64, 6.38)),
}


def sample_active_protocol(family, rng, dt_ms):
    holding, prepulse, test, tail, durations = ACTIVE_PROTOCOLS[family]
    levels = np.asarray([holding, prepulse, test, tail]) + rng.normal(0.0, 4.0, 4)
    phase = np.asarray(durations) * np.exp(rng.normal(0.0, 0.10, 4))
    phase *= 20.0 / phase.sum()
    time = time_grid(50.0, dt_ms)
    command = np.full_like(time, levels[0])
    boundaries = 15.0 + np.cumsum(phase)
    start = 15.0
    for stop, level in zip(boundaries, levels):
        command[(time >= start) & (time < stop)] = level
        start = stop
    command[time >= boundaries[-1]] = levels[0]
    trace = simulate_voltage_clamp(time, command, include_capacitive_current=False)
    metadata = {
        "family": family,
        "control_mode": "voltage_clamp",
        "active_search": True,
        "levels_mV": levels.tolist(),
        "phase_durations_ms": phase.tolist(),
    }
    return trace, metadata


def _generate_one(args):
    root, family, index, split, dt_ms, seed_sequence = args
    rng = np.random.default_rng(seed_sequence)
    trace, metadata = sample_active_protocol(family, rng, dt_ms)
    trajectory_id = f"{family}_{index:05d}"
    record = _write_trajectory(Path(root), split, trajectory_id, trace, metadata)
    record.update({"dt_ms": dt_ms, "seed": int(seed_sequence.entropy)})
    return record


def append_active_protocols(root, count_per_family=200, dt_ms=0.02, seed=8100, workers=1):
    root = Path(root)
    manifest = root / "manifest.jsonl"
    existing = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    tasks = []
    sequences = np.random.SeedSequence(seed).spawn(count_per_family * len(ACTIVE_PROTOCOLS))
    offset = 0
    for family in ACTIVE_PROTOCOLS:
        order = np.random.default_rng(seed + offset).permutation(count_per_family)
        for index in range(count_per_family):
            rank = int(np.flatnonzero(order == index)[0])
            split = "train" if rank < round(0.8 * count_per_family) else "validation" if rank < round(0.9 * count_per_family) else "test"
            tasks.append((str(root), family, index, split, dt_ms, sequences[offset + index]))
        offset += count_per_family
    if workers == 1:
        added = [_generate_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            added = list(executor.map(_generate_one, tasks))
    records = sorted(existing + added, key=lambda item: item["trajectory_id"])
    manifest.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records))
    return added
