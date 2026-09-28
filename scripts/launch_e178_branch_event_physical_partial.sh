#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
OOD=data/generated/ood_protocols
cd "$ROOT" || exit 1

physical_job() {
  b=$1
  weight=$2
  seed=$3
  gpu=$4
  wtag=${weight/./p}
  key="branch${b}_transientw${wtag}_latentdim5_kw24_h750_s${seed}"
  log="runs/phase2/e178_b${b}w${wtag}s${seed}_g${gpu}"
  blind="runs/phase2/${key}_ood_blind_e178_groupcv1se.json"
  printf '%s' "CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/blind_evaluate.py --checkpoint runs/phase2/$key/best.pt --data $DATA --evaluation-data $OOD --mapping-split validation --evaluation-split test --windows-per-mode 1000 --seed 17800 --device cuda:0 --ridge-alpha-grid 0.001 0.01 0.1 1 10 100 1000 10000 --ridge-cv-folds 5 --output $blind > ${log}.blind.log 2>&1 && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:scripts $PY scripts/audit_constrained_relaxation.py --checkpoint runs/phase2/$key/best.pt --blind-result $blind --data $DATA --windows-per-mode 200 --seed 17850 --steps 2000 --device cuda:0 --output runs/phase2/${key}_tau_e178.json > ${log}.tau.log 2>&1; echo \$? > ${log}.exit"
}

launch_chain() {
  gpu=$1
  shift
  chain=""
  for spec in "$@"; do
    b=${spec%%:*}
    rest=${spec#*:}
    weight=${rest%%:*}
    seed=${rest##*:}
    job=$(physical_job "$b" "$weight" "$seed" "$gpu")
    if test -z "$chain"; then chain="$job"; else chain="$chain; $job"; fi
  done
  tmux new-session -d -s "e178_g${gpu}" "bash -lc 'cd $ROOT; $chain'"
}

launch_chain 2 2:1:56 5:1:56
launch_chain 5 2:1:57 5:1:57
launch_chain 6 2:1:58 5:1:58
launch_chain 7 2:2:56 5:2:56

tmux new-session -d -s e178_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e178_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 8; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 Enter; sleep 10; exit; fi; sleep 60; done'"
