#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
cd "$ROOT" || exit 1
i=0
for k in 3 4 5 6; do
  key="latentdim${k}_kw24_h750_s51"
  log="runs/phase2/e116_k${k}s51_g${i}"
  cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$i PYTHONPATH=src $PY scripts/audit_protocol_context.py --checkpoint runs/phase2/$key/best.pt --data data/generated/full_stratified_active_v1 --split validation --windows-per-mode 800 --seed 11600 --max-points-per-trajectory 200 --device cuda:0 --output runs/phase2/${key}_protocol_context_e116_encoder_history.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
  tmux new-session -d -s "e116_k${k}s51_g${i}" "bash -lc '$cmd'"
  i=$((i + 1))
done
tmux new-session -d -s e116_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e116_k*s*_g*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 4; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
