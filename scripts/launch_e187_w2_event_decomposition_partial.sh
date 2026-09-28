#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1

chart_job() {
  b=$1; seed=$2; map_seed=$3; gpu=$4
  key="branch${b}_transientw2_latentdim5_kw24_h750_s${seed}"
  log="runs/phase2/e187_b${b}s${seed}m${map_seed}_g${gpu}"
  printf '%s' "CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py --checkpoint runs/phase2/$key/best.pt --data $DATA --device cuda:0 --seed $map_seed --width 256 --depth 4 --steps 20000 --field-weights 1 --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 --include-voltage --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh --state-gate-weights 1,100,1 --select-best --calibration-split trajectory --output runs/phase2/${key}_eventdecomp_e187_seed${map_seed}.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
}

launch_chain() {
  gpu=$1; shift; chain=""
  for spec in "$@"; do
    b=${spec%%:*}; rest=${spec#*:}; seed=${rest%%:*}; map_seed=${rest##*:}
    job=$(chart_job "$b" "$seed" "$map_seed" "$gpu")
    if test -z "$chain"; then chain="$job"; else chain="$chain; $job"; fi
  done
  tmux new-session -d -s "e187_g${gpu}" "bash -lc 'cd $ROOT; $chain'"
}

launch_chain 2 2:57:8989 5:58:9090
launch_chain 5 2:57:9090 5:57:9090
launch_chain 6 5:57:8989
launch_chain 7 5:58:8989

tmux new-session -d -s e187_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e187_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 6; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 Enter; sleep 10; exit; fi; sleep 60; done'"
