#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
cd "$ROOT" || exit 1
i=0
for k in 5 6; do
  for seed in 51 52 53 54 55; do
    gpu=$((i % 8))
    key="latentdim${k}_kw24_h750_s${seed}"
    log="runs/phase2/e106_k${k}s${seed}_g${gpu}"
    cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/blind_evaluate.py --checkpoint runs/phase2/$key/best.pt --data data/generated/full_stratified_active_v1 --evaluation-data data/generated/ood_protocols --mapping-split validation --evaluation-split test --windows-per-mode 1000 --seed 10400 --device cuda:0 --ridge-alpha 100 --output runs/phase2/${key}_ood_protocols_blind_strict_e106_alpha100.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
    tmux new-session -d -s "e106_k${k}s${seed}_g${gpu}" "bash -lc '$cmd'"
    i=$((i + 1))
  done
done
tmux new-session -d -s e106_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e106_k*s*_g*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 10; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
