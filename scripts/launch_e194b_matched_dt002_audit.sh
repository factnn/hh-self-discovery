#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
RUNS=runs/phase2
cd "$ROOT" || exit 1

launch_one() {
  gpu=$1
  k=$2
  seed=$3
  key="latentdim${k}_kw24_h750_s${seed}"
  log="$RUNS/e194b_k${k}s${seed}_g${gpu}"
  cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/blind_evaluate.py --checkpoint $RUNS/$key/best.pt --data $DATA --mapping-split validation --evaluation-split test --windows-per-mode 500 --seed 19400 --device cuda:0 --ridge-alpha-grid 0.001 0.01 0.1 1 10 100 1000 10000 --ridge-cv-folds 5 --output $RUNS/${key}_id_blind_e194b_groupcv1se.json > ${log}.blind.log 2>&1"
  cmd="$cmd && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:scripts $PY scripts/audit_constrained_relaxation.py --checkpoint $RUNS/$key/best.pt --blind-result $RUNS/${key}_id_blind_e194b_groupcv1se.json --data $DATA --windows-per-mode 200 --seed 19400 --steps 2000 --device cuda:0 --output $RUNS/${key}_sampling_rate_control_e194b.json > ${log}.physics.log 2>&1"
  cmd="$cmd; echo \$? > ${log}.exit"
  tmux new-session -d -s "e194b_k${k}s${seed}_g${gpu}" "bash -lc '$cmd'"
}

launch_one 0 5 56
launch_one 1 5 57
launch_one 2 5 58
launch_one 3 6 56
launch_one 4 6 57
launch_one 5 6 58

tmux new-session -d -s e194b_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in $RUNS/e194b_k*s*_g*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 6; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 120; done'"
