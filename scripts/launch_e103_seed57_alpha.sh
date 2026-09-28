#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
cd "$ROOT" || exit 1
i=0
for alpha in 0.001 0.01 0.1 1 10 100; do
  tag=${alpha//./p}
  cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$i PYTHONPATH=src $PY scripts/blind_evaluate.py --checkpoint runs/phase2/latentdim5_kw24_h750_s57/best.pt --data data/generated/full_stratified_active_v1 --evaluation-data data/generated/ood_protocols --mapping-split validation --evaluation-split test --windows-per-mode 1000 --seed 10200 --device cuda:0 --ridge-alpha $alpha --output runs/phase2/latentdim5_kw24_h750_s57_e103_alpha${tag}.json > runs/phase2/e103_alpha${tag}_g${i}.log 2>&1; echo \$? > runs/phase2/e103_alpha${tag}_g${i}.exit"
  tmux new-session -d -s "e103_a${tag}_g${i}" "bash -lc '$cmd'"
  i=$((i + 1))
done
tmux new-session -d -s e103_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e103_alpha*_g*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 6; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
