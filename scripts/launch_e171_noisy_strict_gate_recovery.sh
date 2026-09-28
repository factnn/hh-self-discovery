#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
OOD=data/generated/ood_protocols
cd "$ROOT" || exit 1
launch_chain() {
    gpu=$1; k=$2; seed=$3; chain=""; key="transient2_latentdim${k}_kw24_h750_s${seed}"
    for condition in moderate extreme; do
        if test "$condition" = moderate; then noise=0.10; stride=10; else noise=0.40; stride=50; fi
        log="runs/phase2/e171_k${k}s${seed}_${condition}_g${gpu}"
        job="CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/blind_evaluate.py --checkpoint runs/phase2/$key/best.pt --data $DATA --evaluation-data $OOD --mapping-split validation --evaluation-split test --windows-per-mode 1000 --seed 17100 --device cuda:0 --ridge-alpha-grid 0.001 0.01 0.1 1 10 100 1000 10000 --ridge-cv-folds 5 --evaluation-response-noise-fraction $noise --evaluation-history-observation-stride $stride --output runs/phase2/${key}_ood_blind_e171_${condition}.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
        if test -z "$chain"; then chain="$job"; else chain="$chain; $job"; fi
    done
    tmux new-session -d -s "e171_k${k}s${seed}_g${gpu}" "bash -lc 'cd $ROOT; $chain'"
}
launch_chain 0 3 56
launch_chain 1 3 57
launch_chain 2 4 56
launch_chain 3 4 57
tmux new-session -d -s e171_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e171_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 8; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
