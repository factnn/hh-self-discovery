"""Shardable active search for informative voltage-clamp protocols."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hh_self_discovery.blind_evaluation import load_latent_checkpoint
from hh_self_discovery.observability import gramian_metrics, output_jacobian
from hh_self_discovery.sampling import BalancedWindowSampler, ObservedDataset


def normalized_history(sample, checkpoint, device):
    stats = checkpoint["normalizers"]["voltage_clamp"]
    command = (sample["history_command"] - stats["command"]["mean"]) / stats["command"]["std"]
    response = (sample["history_response"] - stats["response"]["mean"]) / stats["response"]["std"]
    return torch.as_tensor(
        np.stack([command, response], axis=-1)[None], dtype=torch.float32, device=device
    )


def candidate_waveform(rng, steps, dt_ms):
    holding = float(rng.uniform(-110.0, -70.0))
    prepulse = float(rng.uniform(-130.0, 20.0))
    test = float(rng.uniform(-40.0, 50.0))
    tail = float(rng.uniform(-130.0, -50.0))
    fractions = rng.dirichlet([1.5, 2.0, 2.0, 1.5])
    boundaries = np.cumsum(np.maximum(1, np.floor(fractions * steps).astype(int)))
    boundaries[-1] = steps
    values = np.empty(steps, dtype=np.float32)
    start = 0
    for stop, level in zip(boundaries, (holding, prepulse, test, tail)):
        values[start:stop] = level
        start = stop
    return values, {
        "holding_mV": holding,
        "prepulse_mV": prepulse,
        "test_mV": test,
        "tail_mV": tail,
        "durations_ms": (np.diff(np.r_[0, boundaries]) * dt_ms).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs=2, required=True)
    parser.add_argument("--data", default="data/generated/full_stratified")
    parser.add_argument("--candidates", type=int, default=60)
    parser.add_argument("--histories", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--history-seed", type=int)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loaded = [load_latent_checkpoint(path, device) for path in args.checkpoints]
    checkpoint = loaded[0][1]
    steps, dt_ms = checkpoint["prediction_steps"], checkpoint["dt_ms"]
    dataset = ObservedDataset(args.data, "validation")
    dataset.records = [
        record for record in dataset.records if record["control_mode"] == "voltage_clamp"
    ]
    sampler = BalancedWindowSampler(
        dataset, checkpoint["history_steps"], steps, require_all_events=False
    )
    history_seed = args.seed if args.history_seed is None else args.history_seed
    refs = sampler.sample_mode_epoch(
        "voltage_clamp", args.histories, history_seed, strict=False
    )
    samples = [sampler.extract(ref) for ref in refs]
    rng = np.random.default_rng(args.seed)
    candidates = [candidate_waveform(rng, steps, dt_ms) for _ in range(args.candidates)]
    results = []
    for candidate_index in range(args.shard_index, args.candidates, args.shard_count):
        waveform, parameters = candidates[candidate_index]
        model_grams, model_outputs = [], []
        for model, model_checkpoint in loaded:
            stats = model_checkpoint["normalizers"]["voltage_clamp"]["command"]
            command = torch.as_tensor(
                ((waveform - stats["mean"]) / stats["std"])[None],
                dtype=torch.float32,
                device=device,
            )
            jacobians, predictions = [], []
            for sample in samples:
                history = normalized_history(sample, model_checkpoint, device)
                jacobians.append(
                    output_jacobian(model, history, command, "voltage_clamp").cpu().numpy()
                )
                with torch.no_grad():
                    prediction = model(history, command, "voltage_clamp")
                    prediction = model.denormalize(
                        prediction, "voltage_clamp", "response"
                    )
                predictions.append(prediction.cpu().numpy()[0])
            model_grams.append(gramian_metrics(np.concatenate(jacobians)))
            model_outputs.append(np.stack(predictions))
        robust_minimum = min(item["minimum_eigenvalue"] for item in model_grams)
        disagreement = float(
            np.sqrt(np.mean((model_outputs[0] - model_outputs[1]) ** 2))
            / max(float(np.std(np.concatenate(model_outputs))), 1e-8)
        )
        results.append({
            "candidate": candidate_index,
            "parameters": parameters,
            "model_gramians": model_grams,
            "robust_minimum_eigenvalue": robust_minimum,
            "pair_disagreement_nrmse": disagreement,
            "score": float(np.log10(robust_minimum + 1e-12) + 0.25 * disagreement),
        })
    output = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "candidates_total": args.candidates,
        "histories": args.histories,
        "candidate_seed": args.seed,
        "history_seed": history_seed,
        "evaluation_only_gates_read": False,
        "results": sorted(results, key=lambda item: item["score"], reverse=True),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output["results"][:3], indent=2))


if __name__ == "__main__":
    main()
