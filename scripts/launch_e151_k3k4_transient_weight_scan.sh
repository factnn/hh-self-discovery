#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1
launch() {
    gpu=$1; k=$2; weight=$3; tag=$4
    key="transient${tag}_latentdim${k}_kw24_h750_s56"
    log="runs/phase2/e151_k${k}_${tag}_g${gpu}"
    cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_latent_model.py --data $DATA --output runs/phase2/$key --latent-size $k --encoder-size 128 --kinetics-width 24 --epochs 20 --train-windows-per-mode 8000 --validation-windows-per-mode 800 --history-steps 750 --prediction-steps 1000 --batch-size 32 --lr 0.0003 --complexity-scale 4 --weight-decay 0.000001 --curriculum-fraction 0.4 --multi-horizon-loss --current-clamp-loss-weight 3 --voltage-transient-loss-weight $weight --voltage-transient-steps 25 --early-stopping-patience 5 --early-stopping-min-delta 0.001 --seed 56 --device cuda:0 > ${log}.train.log 2>&1 && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/evaluate_latent_model.py --checkpoint runs/phase2/$key/best.pt --data $DATA --split test --windows-per-mode 500 --seed 1234 --prediction-steps 1000 --skip-ablations --device cuda:0 --output runs/phase2/${key}_test20ms.json > ${log}.eval.log 2>&1; echo \$? > ${log}.exit"
    tmux new-session -d -s "e151_k${k}_${tag}_g${gpu}" "bash -lc '$cmd'"
}
launch 0 3 0.5 w05
launch 1 3 2 2
launch 2 4 0.5 w05
launch 3 4 2 2
tmux new-session -d -s e151_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e151_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 4; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
