"""Summaries for generated datasets before model training."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .events import EventClass
from .sampling import ObservedDataset


def audit_dataset(root: str, split: str) -> dict:
    dataset = ObservedDataset(root, split)
    families = Counter()
    labels = Counter()
    family_labels: dict[str, Counter] = defaultdict(Counter)
    ranges = {"voltage_min": np.inf, "voltage_max": -np.inf, "current_min": np.inf, "current_max": -np.inf}
    spiking_current_clamp = 0
    total_samples = 0
    for record in dataset.records:
        arrays = dataset.load(record)
        family = record["family"]
        count = Counter(int(item) for item in arrays["event_label"])
        families[family] += 1
        labels.update(count)
        family_labels[family].update(count)
        total_samples += len(arrays["time_ms"])
        ranges["voltage_min"] = min(ranges["voltage_min"], float(arrays["voltage_mV"].min()))
        ranges["voltage_max"] = max(ranges["voltage_max"], float(arrays["voltage_mV"].max()))
        ranges["current_min"] = min(ranges["current_min"], float(arrays["current_uA_cm2"].min()))
        ranges["current_max"] = max(ranges["current_max"], float(arrays["current_uA_cm2"].max()))
        if record["control_mode"] == "current_clamp" and arrays["voltage_mV"].max() > 0.0:
            spiking_current_clamp += 1
    return {
        "split": split,
        "trajectories": len(dataset.records),
        "samples": total_samples,
        "families": dict(sorted(families.items())),
        "event_counts": {EventClass(key).name: value for key, value in sorted(labels.items())},
        "event_fractions": {EventClass(key).name: value / total_samples for key, value in sorted(labels.items())},
        "family_event_counts": {
            family: {EventClass(key).name: value for key, value in sorted(counts.items())}
            for family, counts in sorted(family_labels.items())
        },
        "spiking_current_clamp": spiking_current_clamp,
        **ranges,
    }

