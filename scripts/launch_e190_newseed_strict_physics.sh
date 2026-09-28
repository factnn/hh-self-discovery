#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
OOD=data/generated/ood_protocols
cd "$ROOT" || exit 1

specs=(
  "2:1:59" "2:2:59" "5:1:59" "5:2:59"
  "2:1:60" "2:2:60" "5:1:60" "5:2:60"
)

for gpu in $(seq 0 7); do
  spec=${specs[$gpu]}
  branch=${spec%%:*}
  rest=${spec#*:}
  weight=${rest%%:*}
  seed=${rest##*:}
  key="branch${branch}_transientw${weight}_latentdim5_kw24_h750_s${seed}"
  blind="runs/phase2/${key}_ood_blind_e190_groupcv1se.json"
  tau="runs/phase2/${key}_tau_e190.json"
  log="runs/phase2/e190_b${branch}w${weight}s${seed}_g${gpu}"
  audit_seed=$((19000 + gpu))
  cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/blind_evaluate.py --checkpoint runs/phase2/$key/best.pt --data $DATA --evaluation-data $OOD --mapping-split validation --evaluation-split test --windows-per-mode 1000 --seed $audit_seed --device cuda:0 --ridge-alpha-grid 0.001 0.01 0.1 1 10 100 1000 10000 --ridge-cv-folds 5 --output $blind > ${log}.blind.log 2>&1 && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:scripts $PY scripts/audit_constrained_relaxation.py --checkpoint runs/phase2/$key/best.pt --blind-result $blind --data $DATA --windows-per-mode 200 --seed $((audit_seed + 100)) --steps 2000 --device cuda:0 --output $tau > ${log}.tau.log 2>&1; echo \$? > ${log}.exit"
  tmux new-session -d -s "e190_g${gpu}" "bash -lc '$cmd'"
done

tmux new-session -d -s e190_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e190_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 8; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 Enter; sleep 10; exit; fi; sleep 60; done'"
