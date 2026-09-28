#!/usr/bin/env bash
set -euo pipefail

case "${CUDA_VISIBLE_DEVICES:-}" in
  0|1|2|3) ;;
  *)
    echo "Set CUDA_VISIBLE_DEVICES to one physical GPU in {0,1,2,3}." >&2
    exit 2
    ;;
esac

export PYTHONPATH=src
data_root=data/generated/full_stratified
ood_root=data/generated/ood_protocols
parameter_root=data/generated/parameter_ood_10pct

python -m pytest -q
test -f "${data_root}/manifest.jsonl" || python scripts/generate_dataset.py \
  --output "${data_root}" --count 2000 --dt-ms 0.02 --seed 4242 --workers 32
test -f "${ood_root}/manifest.jsonl" || python scripts/generate_ood_dataset.py \
  --output "${ood_root}" --count 200 --dt-ms 0.02 --seed 314 --workers 32
test -f "${parameter_root}/manifest.jsonl" || python scripts/generate_parameter_dataset.py \
  --output "${parameter_root}" --count 200 --dt-ms 0.02 --variation 0.1 --workers 32

for seed in 42 43 44; do
  output="runs/latent/full_balanced_k3_seed${seed}"
  test -f "${output}/best.pt" || python scripts/train_latent_model.py \
    --data "${data_root}" --output "${output}" --latent-size 3 --epochs 10 \
    --train-windows-per-mode 10000 --validation-windows-per-mode 1000 \
    --history-steps 500 --prediction-steps 250 --batch-size 128 \
    --lr 3e-4 --seed "${seed}" --device cuda:0
  python scripts/blind_evaluate.py --checkpoint "${output}/best.pt" \
    --data "${data_root}" --windows-per-mode 500 --seed 6060 --device cuda:0 \
    --output "runs/evaluation/blind_full_balanced_k3_seed${seed}.json"
done

selected=runs/latent/full_balanced_k3_seed43/best.pt

python scripts/evaluate_latent_model.py \
  --checkpoint "${selected}" --data "${data_root}" \
  --split test --windows-per-mode 500 --seed 6060 --device cuda:0 \
  --output runs/evaluation/ablation_full_balanced_k3_seed43.json
python scripts/evaluate_latent_model.py \
  --checkpoint "${selected}" --data "${ood_root}" \
  --split test --windows-per-mode 300 --seed 8080 --skip-ablations --device cuda:0 \
  --output runs/evaluation/ood_full_balanced_k3_seed43.json
python scripts/evaluate_latent_model.py \
  --checkpoint "${selected}" --data "${parameter_root}" \
  --split test --windows-per-mode 300 --seed 9090 --skip-ablations --device cuda:0 \
  --output runs/evaluation/parameter_ood_full_balanced_k3_seed43.json
python scripts/evaluate_latent_model.py \
  --checkpoint "${selected}" --data "${data_root}" \
  --split test --windows-per-mode 500 --seed 6060 --response-noise-fraction 0.10 \
  --skip-ablations --device cuda:0 \
  --output runs/evaluation/noise_010_full_balanced_k3_seed43.json
python scripts/evaluate_latent_model.py \
  --checkpoint "${selected}" --data "${data_root}" \
  --split test --windows-per-mode 500 --seed 6060 --history-observation-stride 10 \
  --skip-ablations --device cuda:0 \
  --output runs/evaluation/sparse_s10_full_balanced_k3_seed43.json
python scripts/evaluate_latent_model.py \
  --checkpoint "${selected}" --data "${data_root}" \
  --split test --windows-per-mode 300 --seed 6060 --prediction-steps 1000 \
  --skip-ablations --device cuda:0 \
  --output runs/evaluation/long_h1000_full_balanced_k3_seed43.json
python scripts/evaluate_mechanism.py \
  --checkpoint "${selected}" \
  --blind-result runs/evaluation/blind_full_balanced_k3_seed43.json \
  --output-json runs/evaluation/mechanism_full_balanced_k3_seed43.json \
  --output-figure runs/evaluation/mechanism_full_balanced_k3_seed43.png \
  --data "${data_root}" \
  --output-phase-figure runs/evaluation/phase_full_balanced_k3_seed43.png
