#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
OOD=data/generated/ood_protocols
cd "$ROOT" || exit 1
for b in 2 5; do
  for seed in 56 57 58; do
    if test "$b" = 2; then gpu=$((seed-56)); else gpu=$((seed-53)); fi
    key="branch${b}_transientw05_latentdim5_kw24_h750_s${seed}"
    log="runs/phase2/e172_b${b}s${seed}_g${gpu}"
    blind="runs/phase2/${key}_ood_blind_e172_groupcv1se.json"
    cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/blind_evaluate.py --checkpoint runs/phase2/$key/best.pt --data $DATA --evaluation-data $OOD --mapping-split validation --evaluation-split test --windows-per-mode 1000 --seed 17200 --device cuda:0 --ridge-alpha-grid 0.001 0.01 0.1 1 10 100 1000 10000 --ridge-cv-folds 5 --output $blind > ${log}.blind.log 2>&1 && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:scripts $PY scripts/audit_constrained_relaxation.py --checkpoint runs/phase2/$key/best.pt --blind-result $blind --data $DATA --windows-per-mode 200 --seed 17250 --steps 2000 --device cuda:0 --output runs/phase2/${key}_tau_e172.json > ${log}.tau.log 2>&1; echo \$? > ${log}.exit"
    tmux new-session -d -s "e172_b${b}s${seed}_g${gpu}" "bash -lc '$cmd'"
  done
done
tmux new-session -d -s e172_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e172_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 6; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 60; done'"
