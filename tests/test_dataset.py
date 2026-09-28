from pathlib import Path

import numpy as np

from hh_self_discovery.dataset import PROTOCOL_FAMILIES, generate_dataset
from hh_self_discovery.events import EventClass, observable_activity


def test_observable_activity_uses_valid_classes() -> None:
    time = np.linspace(0.0, 10.0, 101)
    current = np.where((time >= 2.0) & (time < 7.0), 10.0, 0.0)
    voltage = -65.0 + 20.0 * np.exp(-((time - 4.0) / 0.5) ** 2)
    score, labels = observable_activity(time, current, voltage, "current_clamp")
    assert score.shape == time.shape
    assert labels.shape == time.shape
    assert np.all((score >= 0.0) & (score <= 1.0))
    assert set(np.unique(labels)).issubset(set(int(item) for item in EventClass))
    assert EventClass.COMMAND in labels


def test_dataset_separates_observed_and_hidden_truth(tmp_path: Path) -> None:
    records = generate_dataset(tmp_path / "dataset", count=10, dt_ms=0.05, seed=7)
    assert len(records) == 10
    assert {record["family"] for record in records} == set(PROTOCOL_FAMILIES)
    assert {record["split"] for record in records} == {"train", "validation", "test"}
    for record in records:
        observed = np.load(tmp_path / "dataset" / record["observed_path"])
        evaluation = np.load(tmp_path / "dataset" / record["evaluation_path"])
        assert set(observed.files) == {"time_ms", "current_uA_cm2", "voltage_mV", "activity_score", "event_label"}
        assert set(evaluation.files) == {"n", "m", "h", "i_na_uA_cm2", "i_k_uA_cm2", "i_leak_uA_cm2"}
        assert len(observed["time_ms"]) == record["num_samples"]


def test_generation_is_independent_of_worker_count(tmp_path: Path) -> None:
    serial = generate_dataset(tmp_path / "serial", count=10, dt_ms=0.05, seed=17, workers=1)
    parallel = generate_dataset(tmp_path / "parallel", count=10, dt_ms=0.05, seed=17, workers=2)
    for serial_record, parallel_record in zip(serial, parallel):
        comparable_keys = set(serial_record) - {"observed_path", "evaluation_path"}
        assert {key: serial_record[key] for key in comparable_keys} == {
            key: parallel_record[key] for key in comparable_keys
        }
        with np.load(tmp_path / "serial" / serial_record["observed_path"]) as serial_data:
            with np.load(tmp_path / "parallel" / parallel_record["observed_path"]) as parallel_data:
                for field in serial_data.files:
                    np.testing.assert_array_equal(serial_data[field], parallel_data[field])


def test_large_dataset_split_is_stratified_by_protocol(tmp_path: Path) -> None:
    records = generate_dataset(tmp_path / "stratified", count=50, dt_ms=0.1, seed=5, workers=2)
    counts = {}
    for family in PROTOCOL_FAMILIES:
        counts[family] = {
            split: sum(record["family"] == family and record["split"] == split for record in records)
            for split in ("train", "validation", "test")
        }
    assert all(value == {"train": 8, "validation": 1, "test": 1} for value in counts.values())
