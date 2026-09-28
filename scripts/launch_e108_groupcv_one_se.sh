#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
cd "$ROOT" || exit 1
i=0
for k in 5 6; do
  for seed in 56 57 58; do
    key="latentdim${k}_kw24_h750_s${seed}"
    log="runs/phase2/e108_k${k}s${seed}_g${i}"
    cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$i PYTHONPATH=src $PY scripts/blind_evaluate.py --checkpoint runs/phase2/$key/best.pt --data data/generated/full_stratified_active_v1 --evaluation-data data/generated/ood_protocols --mapping-split validation --evaluation-split test --windows-per-mode 1000 --seed 10600 --device cuda:0 --ridge-alpha-grid 0.001 0.01 0.1 1 10 100 1000 10000 --ridge-cv-folds 5 --output runs/phase2/${key}_ood_protocols_blind_strict_e108_groupcv1se.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
    tmux new-session -d -s "e108_k${k}s${seed}_g${i}" "bash -lc '$cmd'"
    i=$((i + 1))
  done
done
tmux new-session -d -s e108_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e108_k*s*_g*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 6; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
