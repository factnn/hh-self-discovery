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
    log="runs/phase2/e120_k${k}s${seed}_g${gpu}"
    cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/evaluate_mechanism.py --checkpoint runs/phase2/$key/best.pt --blind-result runs/phase2/${key}_ood_protocols_blind_strict_e110_groupcv1se.json --output-json runs/phase2/${key}_mechanism_e120.json --output-figure runs/phase2/${key}_mechanism_e120.png > ${log}.log 2>&1; echo \$? > ${log}.exit"
    tmux new-session -d -s "e120_k${k}s${seed}_g${gpu}" "bash -lc '$cmd'"
    i=$((i + 1))
  done
done
tmux new-session -d -s e120_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e120_k*s*_g*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 10; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
