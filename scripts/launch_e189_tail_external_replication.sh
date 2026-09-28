#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
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
  log="runs/phase2/e189_b${branch}w${weight}s${seed}_g${gpu}"
  cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_latent_model.py --data $DATA --output runs/phase2/$key --latent-size 5 --encoder-size 128 --branch-count $branch --kinetics-width 24 --epochs 20 --train-windows-per-mode 8000 --validation-windows-per-mode 800 --history-steps 750 --prediction-steps 1000 --batch-size 32 --lr 0.0003 --complexity-scale 4 --weight-decay 0.000001 --curriculum-fraction 0.4 --multi-horizon-loss --current-clamp-loss-weight 3 --voltage-transient-loss-weight $weight --voltage-transient-steps 25 --early-stopping-patience 5 --early-stopping-min-delta 0.001 --seed $seed --device cuda:0 > ${log}.train.log 2>&1 && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/evaluate_latent_model.py --checkpoint runs/phase2/$key/best.pt --data $DATA --split test --windows-per-mode 500 --seed 1234 --prediction-steps 1000 --skip-ablations --device cuda:0 --output runs/phase2/${key}_test20ms.json > ${log}.eval.log 2>&1; echo \$? > ${log}.exit"
  tmux new-session -d -s "e189_g${gpu}" "bash -lc '$cmd'"
done

tmux new-session -d -s e189_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e189_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 8; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 Enter; sleep 10; exit; fi; sleep 60; done'"
