from collections import Counter
from pathlib import Path

import numpy as np

from hh_self_discovery.dataset import generate_dataset
from hh_self_discovery.events import EventClass
from hh_self_discovery.sampling import (
    BalancedWindowSampler,
    ObservedDataset,
    WindowReference,
)


def test_balanced_sampler_matches_requested_proportions(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    generate_dataset(root, count=15, dt_ms=0.05, seed=13)
    dataset = ObservedDataset(root, "train")
    sampler = BalancedWindowSampler(dataset, history_steps=20, prediction_steps=10)
    references = sampler.sample_epoch(1000, seed=3)
    counts = Counter(reference.event_class for reference in references)
    assert counts == {
        EventClass.STEADY: 200,
        EventClass.COMMAND: 250,
        EventClass.FAST_RESPONSE: 250,
        EventClass.ACTIVE: 200,
        EventClass.RECOVERY: 100,
    }
    sample = sampler.extract(references[0])
    assert len(sample["history_time_ms"]) == 20
    assert len(sample["future_command"]) == 10
    assert len(sample["target_response"]) == 10
    assert not {"n", "m", "h"}.intersection(sample)


def test_sampling_is_reproducible(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    generate_dataset(root, count=10, dt_ms=0.05, seed=9)
    sampler = BalancedWindowSampler(ObservedDataset(root, "train"), 10, 5)
    first = sampler.sample_epoch(100, seed=99)
    second = sampler.sample_epoch(100, seed=99)
    assert first == second


def test_sampler_can_balance_control_modes_and_events(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    generate_dataset(root, count=50, dt_ms=0.1, seed=21, workers=2)
    sampler = BalancedWindowSampler(ObservedDataset(root, "train"), 10, 5)
    references = sampler.sample_epoch(1000, seed=4, balance_control_modes=True)
    mode_counts = Counter(reference.control_mode for reference in references)
    assert mode_counts == {"current_clamp": 500, "voltage_clamp": 500}
    for mode in mode_counts:
        event_counts = Counter(reference.event_class for reference in references if reference.control_mode == mode)
        assert event_counts == {
            EventClass.STEADY: 100,
            EventClass.COMMAND: 125,
            EventClass.FAST_RESPONSE: 125,
            EventClass.ACTIVE: 100,
            EventClass.RECOVERY: 50,
        }


def test_non_strict_mode_sampling_renormalizes_missing_class(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    generate_dataset(root, count=10, dt_ms=0.05, seed=31)
    sampler = BalancedWindowSampler(ObservedDataset(root, "train"), 10, 5)
    sampler._mode_candidates[("current_clamp", EventClass.COMMAND)] = []
    references = sampler.sample_mode_epoch("current_clamp", 100, seed=7, strict=False)
    assert len(references) == 100
    assert all(reference.event_class is not EventClass.COMMAND for reference in references)


def test_sampler_can_initialize_family_with_missing_events(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    generate_dataset(root, count=10, dt_ms=0.05, seed=31)
    dataset = ObservedDataset(root, "train")
    dataset.records = [
        record for record in dataset.records if record["family"] == "current_chirp"
    ]
    sampler = BalancedWindowSampler(
        dataset, 10, 5, require_all_events=False
    )
    references = sampler.sample_mode_epoch(
        "current_clamp", 100, seed=8, strict=False
    )
    assert len(references) == 100


def test_event_sampling_weights_trajectories_not_window_counts() -> None:
    short = WindowReference("short", "x", 1, EventClass.ACTIVE, "current_clamp")
    long = [
        WindowReference("long", "x", index, EventClass.ACTIVE, "current_clamp")
        for index in range(100)
    ]
    selected = BalancedWindowSampler._select_trajectory_balanced(
        [short, *long], 10000, np.random.default_rng(1)
    )
    short_fraction = sum(item.trajectory_id == "short" for item in selected) / len(selected)
    assert 0.47 < short_fraction < 0.53
