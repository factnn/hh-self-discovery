#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1
for k in 1 2 3; do
  key="latentdim${k}_kw24_h750_s55"
  for mapseed in 8989 9090; do
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src "$PY" scripts/train_dynamics_aware_map.py \
      --checkpoint "runs/phase2/$key/best.pt" --data "$DATA" --device cuda:0 \
      --seed "$mapseed" --width 256 --depth 4 --steps 20000 --field-weights 1 \
      --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 \
      --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh \
      --state-gate-weights 1,100,1 --select-best --calibration-split trajectory \
      --output "runs/phase2/${key}_vc_nov_trajcal_seed${mapseed}.json" \
      > "runs/phase2/e92_k${k}s55_nov_${mapseed}.log" 2>&1 || exit $?
  done
done
echo 0 > runs/phase2/e92_no_voltage_g1.exit
msg="All jobs finished with exit code 0."
tmux send-keys -t hh:0.0 -l "$msg"
sleep 2
tmux send-keys -t hh:0.0 C-m
