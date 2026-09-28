#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
cd "$ROOT" || exit 1
for k in 3 4 5 6; do
  for seed in 51 52 53 54 55; do
    key="latentdim${k}_kw24_h750_s${seed}"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PY" scripts/blind_evaluate.py \
      --checkpoint "runs/phase2/$key/best.pt" \
      --data data/generated/full_stratified_active_v1 \
      --evaluation-data data/generated/ood_protocols \
      --mapping-split validation --evaluation-split test \
      --windows-per-mode 300 --seed 3030 --device cuda:0 \
      --output "runs/phase2/${key}_ood_protocols_blind_strict_resample3030.json" \
      > "runs/phase2/e94_k${k}s${seed}_ood.log" 2>&1 || exit $?
  done
done
echo 0 > runs/phase2/e94_ood_resample_g0.exit
msg="All jobs finished with exit code 0."
tmux send-keys -t hh:0.0 -l "$msg"
sleep 2
tmux send-keys -t hh:0.0 C-m
