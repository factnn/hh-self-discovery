#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1

for seed in 52 53 54 55; do
  key="latentdim3_kw24_h750_s${seed}"
  for mapseed in 8989 9090; do
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "$PY" scripts/train_dynamics_aware_map.py \
      --checkpoint "runs/phase2/$key/best.pt" --data "$DATA" --device cuda:0 \
      --seed "$mapseed" --width 256 --depth 4 --steps 20000 --field-weights 1 \
      --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 --include-voltage \
      --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh \
      --state-gate-weights 1,100,1 --select-best --calibration-split trajectory \
      --output "runs/phase2/${key}_vc_trajcal_seed${mapseed}.json" \
      > "runs/phase2/e87_k3s${seed}_g0_chart${mapseed}.log" 2>&1 || exit $?
  done
done

echo 0 > runs/phase2/e87_remaining_g0.exit
msg="All jobs finished with exit code 0."
tmux send-keys -t hh:0.0 -l "$msg"
sleep 2
tmux send-keys -t hh:0.0 C-m
