#!/usr/bin/env bash
# Same-capacity control for the relaxation prior: replace dz/dt = (z_inf(V) - z)/tau(V)
# with a free vector field dz/dt = f(z, V) of comparable parameter count, and repeat
# the K = 2/3/4 dimension scan on two seeds. The question is whether the K = 3
# prediction transition survives without the relaxation parameterisation.
#
#   bash scripts/launch_free_dynamics_prior_control.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA="$ROOT/data/generated/full_stratified_active_v1"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/src"

run_one() {  # gpu, latent size, seed
  local gpu=$1 k=$2 seed=$3
  local key="freeform_latentdim${k}_kw24_h750_s${seed}"
  local log="$ROOT/runs/phase2/${key}"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/train_latent_model.py \
    --data "$DATA" --output "$ROOT/runs/phase2/$key" --latent-size "$k" \
    --encoder-size 128 --kinetics-width 24 --dynamics-mode free \
    --epochs 20 --train-windows-per-mode 8000 --validation-windows-per-mode 800 \
    --history-steps 750 --prediction-steps 1000 --batch-size 32 --lr 0.0003 \
    --complexity-scale 4 --weight-decay 0.000001 --curriculum-fraction 0.4 \
    --multi-horizon-loss --current-clamp-loss-weight 3 \
    --early-stopping-patience 5 --early-stopping-min-delta 0.001 \
    --seed "$seed" --device cuda:0 > "${log}.train.log" 2>&1 \
  && CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/evaluate_latent_model.py \
    --checkpoint "$ROOT/runs/phase2/$key/best.pt" --data "$DATA" --split test \
    --windows-per-mode 500 --seed 1234 --prediction-steps 1000 --skip-ablations \
    --device cuda:0 --output "$ROOT/runs/phase2/${key}_test20ms.json" > "${log}.eval.log" 2>&1
  echo "$? $key"
}

gpu=0
for seed in 51 52; do
  for k in 2 3 4; do
    run_one "$gpu" "$k" "$seed" &
    gpu=$((gpu + 1))
  done
done
wait
echo "free-dynamics control done"
