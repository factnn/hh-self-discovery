#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
OOD=data/generated/ood_protocols
cd "$ROOT" || exit 1

launch() {
    gpu=$1; seed=$2
    key="transient2_latentdim3_kw24_h750_s${seed}"
    log="runs/phase2/e161_k3s${seed}_g${gpu}"
    blind="runs/phase2/${key}_ood_protocols_blind_strict_e161_groupcv1se.json"
    cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/blind_evaluate.py --checkpoint runs/phase2/$key/best.pt --data $DATA --evaluation-data $OOD --mapping-split validation --evaluation-split test --windows-per-mode 1000 --seed 13800 --device cuda:0 --ridge-alpha-grid 0.001 0.01 0.1 1 10 100 1000 10000 --ridge-cv-folds 5 --output $blind > ${log}.blind.log 2>&1 && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:scripts $PY scripts/audit_constrained_relaxation.py --checkpoint runs/phase2/$key/best.pt --blind-result $blind --data $DATA --windows-per-mode 200 --seed 14100 --steps 2000 --device cuda:0 --output runs/phase2/${key}_tau_e161.json > ${log}.tau.log 2>&1; echo \$? > ${log}.exit"
    tmux new-session -d -s "e161_k3s${seed}_g${gpu}" "bash -lc '$cmd'"
}

launch 0 57
launch 1 58

tmux new-session -d -s e161_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e161_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 2; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
