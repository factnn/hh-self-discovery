#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_dt001
RUNS=runs/phase2
cd "$ROOT" || exit 1

# Regenerate the same deterministic protocol draws and trajectory split at
# twice the sampling rate.  The target directory is unique to this control.
if [ ! -f "$DATA/.complete" ]; then
  if [ -e "$DATA" ]; then
    echo "Incomplete target dataset already exists: $DATA" >&2
    exit 2
  fi
  PYTHONPATH=src "$PY" scripts/generate_dataset.py \
    --output "$DATA" --count 2000 --dt-ms 0.01 --seed 4242 --workers 64 \
    > "$RUNS/e194_dataset_dt001.log" 2>&1 || exit 3
  PYTHONPATH=src "$PY" scripts/augment_active_dataset.py \
    --data "$DATA" --count-per-family 200 --dt-ms 0.01 --seed 8100 --workers 64 \
    >> "$RUNS/e194_dataset_dt001.log" 2>&1 || exit 4
  PYTHONPATH=src "$PY" -c '
import json
from pathlib import Path
p = Path("data/generated/full_stratified_active_dt001/manifest.jsonl")
r = [json.loads(x) for x in p.read_text().splitlines()]
assert len(r) == 2600
assert {x["dt_ms"] for x in r} == {0.01}
assert len({x["trajectory_id"] for x in r}) == 2600
Path("data/generated/full_stratified_active_dt001/.complete").write_text("2600 trajectories; dt_ms=0.01\n")
' || exit 5
fi

launch_one() {
  gpu=$1
  k=$2
  seed=$3
  key="dt001_latentdim${k}_kw24_h1500_s${seed}"
  log="$RUNS/e194_k${k}s${seed}_g${gpu}"
  cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_latent_model.py --data $DATA --output $RUNS/$key --latent-size $k --encoder-size 128 --kinetics-width 24 --epochs 20 --train-windows-per-mode 8000 --validation-windows-per-mode 800 --history-steps 1500 --prediction-steps 2000 --batch-size 32 --lr 0.0003 --complexity-scale 4 --weight-decay 0.000001 --curriculum-fraction 0.4 --multi-horizon-loss --current-clamp-loss-weight 3 --early-stopping-patience 5 --early-stopping-min-delta 0.001 --seed $seed --device cuda:0 > ${log}.train.log 2>&1"
  cmd="$cmd && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/evaluate_latent_model.py --checkpoint $RUNS/$key/best.pt --data $DATA --split test --windows-per-mode 500 --seed 1234 --prediction-steps 2000 --skip-ablations --device cuda:0 --output $RUNS/${key}_test20ms.json > ${log}.eval.log 2>&1"
  cmd="$cmd && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/blind_evaluate.py --checkpoint $RUNS/$key/best.pt --data $DATA --mapping-split validation --evaluation-split test --windows-per-mode 500 --seed 19400 --device cuda:0 --ridge-alpha-grid 0.001 0.01 0.1 1 10 100 1000 10000 --ridge-cv-folds 5 --output $RUNS/${key}_id_blind_e194_groupcv1se.json > ${log}.blind.log 2>&1"
  cmd="$cmd && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:scripts $PY scripts/audit_constrained_relaxation.py --checkpoint $RUNS/$key/best.pt --blind-result $RUNS/${key}_id_blind_e194_groupcv1se.json --data $DATA --windows-per-mode 200 --seed 19400 --steps 2000 --device cuda:0 --output $RUNS/${key}_sampling_rate_physics_e194.json > ${log}.physics.log 2>&1"
  cmd="$cmd; echo \$? > ${log}.exit"
  tmux new-session -d -s "e194_k${k}s${seed}_g${gpu}" "bash -lc '$cmd'"
}

launch_one 0 5 56
launch_one 1 5 57
launch_one 2 5 58
launch_one 3 6 56
launch_one 4 6 57
launch_one 5 6 58

tmux new-session -d -s e194_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in $RUNS/e194_k*s*_g*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 6; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
