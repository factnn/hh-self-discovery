#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1

chart_job() {
  b=$1; seed=$2; gpu=$3
  key="branch${b}_transientw1_latentdim5_kw24_h750_s${seed}"
  log="runs/phase2/e180_b${b}s${seed}_g${gpu}"
  printf '%s' "CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py --checkpoint runs/phase2/$key/best.pt --data $DATA --device cuda:0 --seed 9090 --width 256 --depth 4 --steps 20000 --field-weights 1 --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 --include-voltage --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh --state-gate-weights 1,100,1 --select-best --calibration-split trajectory --output runs/phase2/${key}_vc_trajcal_seed9090.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
}

launch_chain() {
  gpu=$1; shift; chain=""
  for spec in "$@"; do
    b=${spec%%:*}; seed=${spec##*:}; job=$(chart_job "$b" "$seed" "$gpu")
    if test -z "$chain"; then chain="$job"; else chain="$chain; $job"; fi
  done
  tmux new-session -d -s "e180_g${gpu}" "bash -lc 'cd $ROOT; $chain'"
}

launch_chain 2 2:56 5:57
launch_chain 5 2:57 5:58
launch_chain 6 2:58
launch_chain 7 5:56

tmux new-session -d -s e180_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e180_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 6; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 Enter; sleep 10; exit; fi; sleep 60; done'"
