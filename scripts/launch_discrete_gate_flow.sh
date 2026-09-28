#!/usr/bin/env bash
# One-step gate-flow audit: structured K=3 (five seeds) plus S4D and GRU (three seeds each).
#
#   bash scripts/launch_discrete_gate_flow.sh
#
# Each job writes runs/phase2/<tag>_discrete_flow.json.  Jobs are placed one per
# GPU and the script waits for all of them before returning.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA="$ROOT/data/generated/full_stratified_active_v1"
RUNS="$ROOT/runs/phase2"
BASE="$ROOT/runs/baselines"
COMMON="--windows-per-mode 200 --seed 19800 --ridge-alpha 100 --chart-degree 3"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:$ROOT/external/s4:$ROOT/external/latent_ode"
cd "$ROOT" || exit 1

run_one() {  # gpu, checkpoint, tag
  local gpu=$1 ckpt=$2 tag=$3
  local out="$RUNS/${tag}_discrete_flow.json"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/audit_discrete_gate_flow.py \
    --checkpoint "$ckpt" --data "$DATA" --device cuda:0 $COMMON \
    --output "$out" > "${out%.json}.log" 2>&1
  echo "$? $tag"
}

gpu=0
for seed in 51 52 53 54 55; do
  run_one "$gpu" "$RUNS/latentdim3_kw24_h750_s${seed}/best.pt" "latentdim3_kw24_h750_s${seed}" &
  gpu=$(( (gpu + 1) % 8 ))
done
wait
echo "== structured done =="

gpu=0
for seed in 56 57 58; do
  run_one "$gpu" "$BASE/b3_s4d_bottleneck_k5_s${seed}/best.pt" "s4d_k5_s${seed}" &
  gpu=$((gpu + 1))
done
for seed in 56 57 58; do
  run_one "$gpu" "$BASE/b3_gru_bottleneck_k5_s${seed}/best.pt" "gru_k5_s${seed}" &
  gpu=$((gpu + 1))
done
wait
echo "== baselines done =="
