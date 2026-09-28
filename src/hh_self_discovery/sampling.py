"""Leak-resistant loading and event-balanced temporal window sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .events import EventClass


@dataclass(frozen=True)
class WindowReference:
    trajectory_id: str
    observed_path: str
    center: int
    event_class: EventClass
    control_mode: str


class ObservedDataset:
    """Access only manifest records and files under observed directories."""

    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        manifest_path = self.root / "manifest.jsonl"
        records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
        self.records = [record for record in records if record["split"] == split]
        if not self.records:
            raise ValueError(f"no records found for split {split!r}")
        for record in self.records:
            path = Path(record["observed_path"])
            if "evaluation_only" in path.parts or "observed" not in path.parts:
                raise ValueError(f"unsafe observed path in manifest: {path}")
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    def load(self, record: dict) -> dict[str, np.ndarray]:
        trajectory_id = record["trajectory_id"]
        if trajectory_id not in self._cache:
            with np.load(self.root / record["observed_path"]) as archive:
                fields = {name: archive[name] for name in archive.files}
            forbidden = {"n", "m", "h", "i_na_uA_cm2", "i_k_uA_cm2", "i_leak_uA_cm2"}
            if forbidden.intersection(fields):
                raise ValueError(f"hidden evaluation fields leaked into {record['observed_path']}")
            self._cache[trajectory_id] = fields
        return self._cache[trajectory_id]

    @staticmethod
    def command_and_response(record: dict, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if record["control_mode"] == "current_clamp":
            return arrays["current_uA_cm2"], arrays["voltage_mV"]
        if record["control_mode"] == "voltage_clamp":
            return arrays["voltage_mV"], arrays["current_uA_cm2"]
        raise ValueError(f"unknown control mode: {record['control_mode']}")


class BalancedWindowSampler:
    """Sample window endpoints using configured event-class proportions."""

    def __init__(
        self,
        dataset: ObservedDataset,
        history_steps: int,
        prediction_steps: int,
        fractions: dict[EventClass, float] | None = None,
        require_all_events: bool = True,
    ):
        if history_steps < 1 or prediction_steps < 1:
            raise ValueError("history_steps and prediction_steps must be positive")
        self.dataset = dataset
        self.history_steps = history_steps
        self.prediction_steps = prediction_steps
        self.fractions = fractions or {
            EventClass.STEADY: 0.20,
            EventClass.COMMAND: 0.25,
            EventClass.FAST_RESPONSE: 0.25,
            EventClass.ACTIVE: 0.20,
            EventClass.RECOVERY: 0.10,
        }
        total = sum(self.fractions.values())
        if set(self.fractions) != set(EventClass) or not np.isclose(total, 1.0):
            raise ValueError("fractions must define every event class and sum to one")
        self._candidates: dict[EventClass, list[WindowReference]] = {event: [] for event in EventClass}
        self._mode_candidates: dict[tuple[str, EventClass], list[WindowReference]] = {}
        self._record_by_id = {record["trajectory_id"]: record for record in dataset.records}
        for record in dataset.records:
            arrays = dataset.load(record)
            labels = arrays["event_label"]
            valid_stop = len(labels) - prediction_steps
            for center in range(history_steps, valid_stop):
                event = EventClass(int(labels[center]))
                reference = WindowReference(
                    record["trajectory_id"],
                    record["observed_path"],
                    center,
                    event,
                    record["control_mode"],
                )
                self._candidates[event].append(reference)
                self._mode_candidates.setdefault((record["control_mode"], event), []).append(reference)
        empty = [event.name for event, values in self._candidates.items() if not values]
        if empty and require_all_events:
            raise ValueError(f"no valid windows for event classes: {empty}")

    @staticmethod
    def _allocate(total: int, fractions: dict) -> dict:
        raw = {key: total * fraction for key, fraction in fractions.items()}
        counts = {key: int(np.floor(value)) for key, value in raw.items()}
        remainder = total - sum(counts.values())
        for key in sorted(fractions, key=lambda item: raw[item] - counts[item], reverse=True)[:remainder]:
            counts[key] += 1
        return counts

    @staticmethod
    def _select_trajectory_balanced(
        candidates: list[WindowReference],
        count: int,
        rng: np.random.Generator,
    ) -> list[WindowReference]:
        """Select trajectories uniformly, then a window uniformly within each."""
        by_trajectory: dict[str, list[WindowReference]] = {}
        for reference in candidates:
            by_trajectory.setdefault(reference.trajectory_id, []).append(reference)
        trajectory_ids = sorted(by_trajectory)
        selected_trajectories = rng.choice(len(trajectory_ids), size=count, replace=True)
        selected = []
        for trajectory_index in selected_trajectories:
            windows = by_trajectory[trajectory_ids[int(trajectory_index)]]
            selected.append(windows[int(rng.integers(len(windows)))])
        return selected

    def sample_epoch(
        self,
        num_windows: int,
        seed: int,
        balance_control_modes: bool = False,
    ) -> list[WindowReference]:
        """Draw a shuffled epoch; minority classes are sampled with replacement."""
        if num_windows < len(EventClass):
            raise ValueError(f"num_windows must be at least {len(EventClass)}")
        rng = np.random.default_rng(seed)
        references = []
        if balance_control_modes:
            modes = sorted({record["control_mode"] for record in self.dataset.records})
            mode_counts = self._allocate(num_windows, {mode: 1.0 / len(modes) for mode in modes})
            for mode, mode_count in mode_counts.items():
                event_counts = self._allocate(mode_count, self.fractions)
                for event, count in event_counts.items():
                    candidates = self._mode_candidates.get((mode, event), [])
                    if not candidates:
                        raise ValueError(f"no valid windows for {mode}/{event.name}")
                    references.extend(self._select_trajectory_balanced(candidates, count, rng))
        else:
            counts = self._allocate(num_windows, self.fractions)
            for event, count in counts.items():
                candidates = self._candidates[event]
                references.extend(self._select_trajectory_balanced(candidates, count, rng))
        rng.shuffle(references)
        return references

    def sample_mode_epoch(
        self, control_mode: str, num_windows: int, seed: int, strict: bool = True
    ) -> list[WindowReference]:
        """Draw event-balanced windows for one mode.

        Training uses strict sampling. Evaluation may renormalize over event
        classes present in an OOD protocol family (for example, smooth ramps
        legitimately contain no command jumps).
        """
        rng = np.random.default_rng(seed)
        available = {
            event: fraction
            for event, fraction in self.fractions.items()
            if self._mode_candidates.get((control_mode, event), [])
        }
        missing = set(self.fractions) - set(available)
        if strict and missing:
            event = sorted(missing, key=int)[0]
            raise ValueError(f"no valid windows for {control_mode}/{event.name}")
        if not available:
            raise ValueError(f"no valid windows for {control_mode}")
        total_fraction = sum(available.values())
        normalized = {event: fraction / total_fraction for event, fraction in available.items()}
        counts = self._allocate(num_windows, normalized)
        references = []
        for event, count in counts.items():
            candidates = self._mode_candidates.get((control_mode, event), [])
            references.extend(self._select_trajectory_balanced(candidates, count, rng))
        rng.shuffle(references)
        return references


    def extract(self, reference: WindowReference) -> dict[str, np.ndarray | str | int]:
        record = self._record_by_id[reference.trajectory_id]
        arrays = self.dataset.load(record)
        command, response = self.dataset.command_and_response(record, arrays)
        start = reference.center - self.history_steps
        stop = reference.center + self.prediction_steps
        return {
            "trajectory_id": reference.trajectory_id,
            "control_mode": reference.control_mode,
            "event_class": int(reference.event_class),
            "history_time_ms": arrays["time_ms"][start : reference.center],
            "history_current_uA_cm2": arrays["current_uA_cm2"][start : reference.center],
            "history_voltage_mV": arrays["voltage_mV"][start : reference.center],
            "history_command": command[start : reference.center],
            "history_response": response[start : reference.center],
            "future_command": command[reference.center:stop],
            "target_response": response[reference.center:stop],
            "activity_score": arrays["activity_score"][reference.center:stop],
        }
