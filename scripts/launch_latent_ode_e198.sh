#!/usr/bin/env bash
# Reproduce the Latent ODE (and matched structured) transported-field audits.
#
#   bash scripts/launch_latent_ode_e198.sh
#
# Runs E198 settings (200 windows/mode, stride 5, ridge alpha 100, jvp eps 0.01,
# audit seed 19800) with charts fitted on the ID validation split and evaluation
# on either the ID test split or the OOD protocol test split.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
export PYTHONPATH=src:scripts:external/latent_ode
ID=data/generated/full_stratified_active_v1
OOD=data/generated/ood_protocols
RUNS=runs/phase2
COMMON="--windows-per-mode 200 --stride 5 --seed 19800 --ridge-alpha 100 --jvp-epsilon-ms 0.01"
cd "$ROOT" || exit 1

run() {  # gpu, checkpoint, evaluation_root_or_empty, output
  local gpu=$1 ckpt=$2 evalroot=$3 out=$4
  local extra=""
  test -n "$evalroot" && extra="--evaluation-data $evalroot"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/audit_latent_ode_chain.py \
    --checkpoint "$ckpt" --data "$ID" $extra --device cuda:0 $COMMON \
    --output "$out" > "${out%.json}.log" 2>&1
  echo "$? $out" >> "$RUNS/latent_ode_e198.exit"
}

rm -f "$RUNS/latent_ode_e198.exit"
for seed in 56 57 58; do
  run $((seed - 56)) "runs/baselines/b3_controlled_latent_ode_k2_s${seed}/best.pt" "$OOD" \
      "$RUNS/latent_ode_k2_s${seed}_claim_chain_e198.json" &
done
for seed in 56 57 58; do
  run $((seed - 53)) "runs/baselines/b3_controlled_latent_ode_k2_s${seed}/best.pt" "" \
      "$RUNS/latent_ode_k2_s${seed}_claim_chain_e198_idtest.json" &
done
for seed in 51 52 53 54 55; do
  run $((seed - 51)) "runs/phase2/latentdim3_kw24_h750_s${seed}/best.pt" "$OOD" \
      "$RUNS/latentdim3_kw24_h750_s${seed}_claim_chain_e198_ood.json" &
done
run 6 "runs/phase2/latentdim3_kw24_h750_s51/best.pt" "" \
    "$RUNS/latentdim3_kw24_h750_s51_claim_chain_e198_newscript_idcheck.json" &
wait
echo "launcher finished"
cat "$RUNS/latent_ode_e198.exit"
