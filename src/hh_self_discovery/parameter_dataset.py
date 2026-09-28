"""Fixed-protocol OOD data with one-at-a-time HH parameter mismatch."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .dataset import PROTOCOL_FAMILIES, _write_trajectory, sample_protocol
from .simulator import HHParameters


def _sample_parameters(rng: np.random.Generator, variation: float) -> HHParameters:
    base = HHParameters()
    scale = lambda value: float(value * rng.uniform(1.0 - variation, 1.0 + variation))
    voltage_shift = 50.0 * variation
    return HHParameters(
        capacitance_uF_cm2=scale(base.capacitance_uF_cm2),
        g_na_mS_cm2=scale(base.g_na_mS_cm2),
        g_k_mS_cm2=scale(base.g_k_mS_cm2),
        g_leak_mS_cm2=scale(base.g_leak_mS_cm2),
        e_na_mV=float(base.e_na_mV + rng.uniform(-voltage_shift, voltage_shift)),
        e_k_mV=float(base.e_k_mV + rng.uniform(-voltage_shift, voltage_shift)),
        e_leak_mV=float(base.e_leak_mV + rng.uniform(-voltage_shift, voltage_shift)),
    )


def _generate_one(arguments) -> dict:
    root, index, dt_ms, variation, seed_sequence = arguments
    rng = np.random.default_rng(seed_sequence)
    family = PROTOCOL_FAMILIES[index % len(PROTOCOL_FAMILIES)]
    params = _sample_parameters(rng, variation)
    trace, metadata = sample_protocol(family, rng, dt_ms, params=params)
    observed_arrays = (trace.time_ms, trace.current_uA_cm2, trace.voltage_mV)
    hidden_arrays = (trace.n, trace.m, trace.h)
    if not all(np.all(np.isfinite(values)) for values in observed_arrays + hidden_arrays):
        raise FloatingPointError(
            f"non-finite parameter-OOD trajectory {index} at dt_ms={dt_ms}"
        )
    metadata.update({
        "parameter_ood": True,
        "parameter_variation_fraction": variation,
        "hh_parameters": asdict(params),
    })
    record = _write_trajectory(Path(root), "test", f"trajectory_{index:06d}", trace, metadata)
    record.update({"dt_ms": dt_ms, "seed": int(seed_sequence.entropy)})
    return record


def generate_parameter_dataset(
    output_dir: str | Path,
    count: int = 200,
    dt_ms: float = 0.02,
    variation: float = 0.1,
    seed: int = 2718,
    workers: int = 1,
) -> list[dict]:
    if count < len(PROTOCOL_FAMILIES):
        raise ValueError("count is too small")
    if not 0.0 < variation < 0.5:
        raise ValueError("variation must be between zero and 0.5")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seeds = np.random.SeedSequence(seed).spawn(count)
    arguments = [(str(root), i, dt_ms, variation, seeds[i]) for i in range(count)]
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
