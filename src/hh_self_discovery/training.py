"""Data normalization, training, and evaluation for black-box baselines."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .events import EventClass
from .metrics import regression_metrics, waveform_metrics
from .sampling import BalancedWindowSampler, ObservedDataset, WindowReference
from .sequence_models import AutoregressiveSequenceModel


@dataclass(frozen=True)
class Standardizer:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


@dataclass(frozen=True)
class ModeNormalizers:
    command: Standardizer
    response: Standardizer


def fit_mode_normalizers(dataset: ObservedDataset, control_mode: str) -> ModeNormalizers:
    """Fit command/response statistics using complete training trajectories only."""
    moments = {"command": [0, 0.0, 0.0], "response": [0, 0.0, 0.0]}
    for record in dataset.records:
        if record["control_mode"] != control_mode:
            continue
        command, response = dataset.command_and_response(record, dataset.load(record))
        for name, values in (("command", command), ("response", response)):
            values64 = np.asarray(values, dtype=np.float64)
            moments[name][0] += values64.size
            moments[name][1] += float(values64.sum())
            moments[name][2] += float(np.square(values64).sum())

    def finish(values: list[float]) -> Standardizer:
        count, total, square_total = values
        if count == 0:
            raise ValueError(f"no trajectories found for {control_mode}")
        mean = total / count
        variance = max(square_total / count - mean * mean, 1e-12)
        return Standardizer(mean, float(np.sqrt(variance)))

    return ModeNormalizers(finish(moments["command"]), finish(moments["response"]))


class WindowTorchDataset(Dataset):
    def __init__(
        self,
        sampler: BalancedWindowSampler,
        references: list[WindowReference],
        normalizers: ModeNormalizers,
    ):
        self.sampler = sampler
        self.references = references
        self.normalizers = normalizers

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sample = self.sampler.extract(self.references[index])
        history_command = self.normalizers.command.transform(np.asarray(sample["history_command"]))
        history_response = self.normalizers.response.transform(np.asarray(sample["history_response"]))
        history = np.stack([history_command, history_response], axis=-1).astype(np.float32)
        future_command = self.normalizers.command.transform(np.asarray(sample["future_command"])).astype(np.float32)
        target = self.normalizers.response.transform(np.asarray(sample["target_response"])).astype(np.float32)
        return {
            "history": torch.from_numpy(history),
            "future_command": torch.from_numpy(future_command),
            "target": torch.from_numpy(target),
            "activity": torch.from_numpy(np.asarray(sample["activity_score"], dtype=np.float32)),
            "event_class": int(sample["event_class"]),
        }


def _loss(prediction: torch.Tensor, target: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
    weights = 1.0 + 4.0 * activity.clamp(0.0, 1.0)
    value_loss = (weights * (prediction - target).square()).sum() / weights.sum()
    if prediction.shape[1] > 1:
        derivative_loss = torch.mean(
            ((prediction[:, 1:] - prediction[:, :-1]) - (target[:, 1:] - target[:, :-1])).square()
        )
    else:
        derivative_loss = prediction.new_zeros(())
    return value_loss + 0.2 * derivative_loss


def evaluate_sequence_model(
    model: AutoregressiveSequenceModel,
    loader: DataLoader,
    normalizers: ModeNormalizers,
    device: torch.device,
    teacher_forcing: bool = False,
    control_mode: str | None = None,
    dt_ms: float = 0.02,
) -> dict:
    model.eval()
    grouped: dict[EventClass, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        event: [] for event in EventClass
    }
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device)
            command = batch["future_command"].to(device)
            target_tensor = batch["target"].to(device)
            prediction = model(
                history,
                command,
                target_tensor if teacher_forcing else None,
                1.0 if teacher_forcing else 0.0,
            ).cpu().numpy()
            target = target_tensor.cpu().numpy()
            prediction = normalizers.response.inverse(prediction)
            target = normalizers.response.inverse(target)
            activity = batch["activity"].numpy()
            for index, event_value in enumerate(batch["event_class"].numpy()):
                grouped[EventClass(int(event_value))].append((target[index], prediction[index], activity[index]))

    def summarize(items: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
        target = np.stack([item[0] for item in items])
        prediction = np.stack([item[1] for item in items])
        activity = np.stack([item[2] for item in items])
        result = {"windows": len(items), **regression_metrics(target, prediction, activity)}
        if control_mode is not None:
            result.update(waveform_metrics(target, prediction, control_mode, dt_ms))
        return result

    all_items = [item for items in grouped.values() for item in items]
    return {
        "overall": summarize(all_items),
        "by_event": {event.name: summarize(items) for event, items in grouped.items() if items},
    }


def train_sequence_baseline(
    data_root: str,
    output_dir: str,
    control_mode: str,
    cell_type: str,
    epochs: int = 10,
    train_windows: int = 10000,
    validation_windows: int = 2500,
    history_steps: int = 500,
    prediction_steps: int = 250,
    batch_size: int = 128,
    hidden_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device_name: str = "cuda",
) -> dict:
    if control_mode not in {"current_clamp", "voltage_clamp"}:
        raise ValueError("invalid control_mode")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    train_data = ObservedDataset(data_root, "train")
    validation_data = ObservedDataset(data_root, "validation")
    normalizers = fit_mode_normalizers(train_data, control_mode)
    train_sampler = BalancedWindowSampler(train_data, history_steps, prediction_steps)
    validation_sampler = BalancedWindowSampler(validation_data, history_steps, prediction_steps)
    validation_refs = validation_sampler.sample_mode_epoch(control_mode, validation_windows, seed + 10000)
    validation_loader = DataLoader(
        WindowTorchDataset(validation_sampler, validation_refs, normalizers), batch_size=batch_size, shuffle=False
    )
    model = AutoregressiveSequenceModel(cell_type, hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    history = []
    best_score = float("inf")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        references = train_sampler.sample_mode_epoch(control_mode, train_windows, seed + epoch)
        loader = DataLoader(
            WindowTorchDataset(train_sampler, references, normalizers),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        model.train()
        losses = []
        teacher_forcing = max(0.0, 0.5 * (1.0 - epoch / max(epochs - 1, 1)))
        for batch in loader:
            history_tensor = batch["history"].to(device)
            command = batch["future_command"].to(device)
            target = batch["target"].to(device)
            activity = batch["activity"].to(device)
            prediction = model(history_tensor, command, target, teacher_forcing)
            loss = _loss(prediction, target, activity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        free_roll_metrics = evaluate_sequence_model(
            model, validation_loader, normalizers, device, control_mode=control_mode
        )
        teacher_forced_metrics = evaluate_sequence_model(
            model, validation_loader, normalizers, device,
            teacher_forcing=True, control_mode=control_mode
        )
        metrics = {"free_roll": free_roll_metrics, "teacher_forced": teacher_forced_metrics}
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "teacher_forcing_ratio": teacher_forcing,
            "validation": metrics,
        }
        history.append(epoch_record)
        score = metrics["free_roll"]["overall"]["activity_weighted_rmse"]
        print(json.dumps(epoch_record, sort_keys=True))
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cell_type": cell_type,
                    "hidden_size": hidden_size,
                    "control_mode": control_mode,
                    "history_steps": history_steps,
                    "prediction_steps": prediction_steps,
                    "normalizers": asdict(normalizers),
                    "validation": metrics,
                    "seed": seed,
                },
                output / "best.pt",
            )
    summary = {
        "control_mode": control_mode,
        "cell_type": cell_type,
        "device": str(device),
        "epochs": epochs,
        "train_windows": train_windows,
        "validation_windows": validation_windows,
        "history": history,
        "best_activity_weighted_rmse": best_score,
    }
    (output / "training.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
