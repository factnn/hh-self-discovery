"""Simple baselines that establish whether learned models beat flat predictions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .events import EventClass
from .metrics import regression_metrics, waveform_metrics
from .sampling import BalancedWindowSampler, ObservedDataset


def evaluate_persistence(
    data_root: str,
    split: str = "validation",
    num_windows: int = 5000,
    history_steps: int = 500,
    prediction_steps: int = 250,
    seed: int = 42,
    control_mode: str | None = None,
    strict_events: bool = True,
) -> dict:
    """Predict every future response from the final observed response value."""
    dataset = ObservedDataset(data_root, split)
    dt_ms = float(dataset.records[0].get("dt_ms", 0.02))
    sampler = BalancedWindowSampler(dataset, history_steps, prediction_steps)
    if control_mode is None:
        references = sampler.sample_epoch(num_windows, seed, balance_control_modes=True)
    else:
        references = sampler.sample_mode_epoch(control_mode, num_windows, seed, strict=strict_events)
    grouped: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    mode_grouped: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    for reference in references:
        sample = sampler.extract(reference)
        target = np.asarray(sample["target_response"], dtype=np.float64)
        prediction = np.full_like(target, float(sample["history_response"][-1]))
        activity = np.asarray(sample["activity_score"], dtype=np.float64)
        mode = str(sample["control_mode"])
        event_name = EventClass(int(sample["event_class"])).name
        grouped[(mode, event_name)].append((target, prediction, activity))
        mode_grouped[mode].append((target, prediction, activity))

    def summarize(items, mode: str) -> dict:
        target = np.stack([item[0] for item in items])
        prediction = np.stack([item[1] for item in items])
        activity = np.stack([item[2] for item in items])
        return {
            "windows": len(items),
            **regression_metrics(target, prediction, activity),
            **waveform_metrics(target, prediction, mode, dt_ms),
        }

    return {
        "baseline": "persistence",
        "split": split,
        "num_windows": num_windows,
        "history_steps": history_steps,
        "prediction_steps": prediction_steps,
        "seed": seed,
        "control_mode": control_mode,
        "by_control_mode": {
            mode: summarize(items, mode) for mode, items in sorted(mode_grouped.items())
        },
        "by_control_mode_and_event": {
            f"{mode}/{event}": summarize(items, mode)
            for (mode, event), items in sorted(grouped.items())
        },
    }
