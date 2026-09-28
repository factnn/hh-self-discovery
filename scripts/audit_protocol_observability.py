"""Rank protocols by locked-model latent output sensitivity using observed I/V."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_self_discovery.blind_evaluation import load_latent_checkpoint
from hh_self_discovery.observability import gramian_metrics, output_jacobian
from hh_self_discovery.sampling import BalancedWindowSampler, ObservedDataset


def tensors(sample: dict, checkpoint: dict, mode: str, device: torch.device):
    stats = checkpoint["normalizers"][mode]
    command = (sample["history_command"] - stats["command"]["mean"]) / stats["command"]["std"]
    response = (sample["history_response"] - stats["response"]["mean"]) / stats["response"]["std"]
    history = np.stack([command, response], axis=-1)
    future = (sample["future_command"] - stats["command"]["mean"]) / stats["command"]["std"]
    return (
        torch.as_tensor(history[None], dtype=torch.float32, device=device),
        torch.as_tensor(future[None], dtype=torch.float32, device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs=2, required=True)
    parser.add_argument("--data", default="data/generated/full_stratified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--windows-per-family", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loaded = [load_latent_checkpoint(path, device) for path in args.checkpoints]
    prediction_steps = loaded[0][1]["prediction_steps"]
    if any(item[1]["prediction_steps"] != prediction_steps for item in loaded):
        raise ValueError("checkpoints must use the same prediction_steps")
    base = ObservedDataset(args.data, args.split)
    families = sorted({record["family"] for record in base.records})
    results = {}
    for family_index, family in enumerate(families):
        dataset = ObservedDataset(args.data, args.split)
        dataset.records = [record for record in dataset.records if record["family"] == family]
        mode = dataset.records[0]["control_mode"]
        sampler = BalancedWindowSampler(
            dataset,
            loaded[0][1]["history_steps"],
            prediction_steps,
            require_all_events=False,
        )
        refs = sampler.sample_mode_epoch(
            mode, args.windows_per_family, args.seed + family_index, strict=False
        )
        targets = []
        model_predictions = []
        model_metrics = []
        for model, checkpoint in loaded:
            predictions = []
            jacobians = []
            for ref in refs:
                sample = sampler.extract(ref)
                history, future = tensors(sample, checkpoint, mode, device)
                with torch.no_grad():
                    prediction = model(history, future, mode)
                    prediction = model.denormalize(prediction, mode, "response")
                predictions.append(prediction.cpu().numpy()[0])
                jacobians.append(output_jacobian(model, history, future, mode).cpu().numpy())
                if len(targets) < len(refs):
                    targets.append(np.asarray(sample["target_response"]))
            prediction_array = np.stack(predictions)
            target_array = np.stack(targets)
            scale = max(float(target_array.std()), 1e-8)
            metrics = gramian_metrics(np.concatenate(jacobians))
            metrics["prediction_nrmse"] = float(
                np.sqrt(np.mean((prediction_array - target_array) ** 2)) / scale
            )
            model_predictions.append(prediction_array)
            model_metrics.append(metrics)
        target_scale = max(float(np.stack(targets).std()), 1e-8)
        disagreement = float(
            np.sqrt(np.mean((model_predictions[0] - model_predictions[1]) ** 2))
            / target_scale
        )
        results[family] = {
            "mode": mode,
            "windows": len(refs),
            "model_metrics": model_metrics,
            "pair_prediction_disagreement_nrmse": disagreement,
        }
    output = {
        "checkpoints": args.checkpoints,
        "training_fields_used": ["current", "voltage", "event_labels", "protocol_family"],
        "evaluation_only_gates_read": False,
        "families": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
