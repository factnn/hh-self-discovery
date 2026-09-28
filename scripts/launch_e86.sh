#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1

launch_one() {
  gpu=$1
  k=$2
  seed=$3
  key="latentdim${k}_kw24_h750_s${seed}"
  log="runs/phase2/e86_k${k}s${seed}_g${gpu}"
  cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_latent_model.py --data $DATA --output runs/phase2/$key --latent-size $k --encoder-size 128 --kinetics-width 24 --epochs 20 --train-windows-per-mode 8000 --validation-windows-per-mode 800 --history-steps 750 --prediction-steps 1000 --batch-size 32 --lr 0.0003 --complexity-scale 4 --weight-decay 0.000001 --curriculum-fraction 0.4 --multi-horizon-loss --current-clamp-loss-weight 3 --early-stopping-patience 5 --early-stopping-min-delta 0.001 --seed $seed --device cuda:0 > ${log}.train.log 2>&1"
  cmd="$cmd && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/evaluate_latent_model.py --checkpoint runs/phase2/$key/best.pt --data $DATA --split test --windows-per-mode 500 --seed 1234 --prediction-steps 1000 --skip-ablations --device cuda:0 --output runs/phase2/${key}_test20ms.json > ${log}.eval.log 2>&1"
  cmd="$cmd && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/audit_latent_intrinsic_dimension.py --checkpoint runs/phase2/$key/best.pt --data $DATA --split test --windows-per-mode 500 --seed 1234 --device cuda:0 --output runs/phase2/${key}_intrinsic_dimension_e86.json > ${log}.intrinsic.log 2>&1"
  for mapseed in 8989 9090; do
    cmd="$cmd && CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py --checkpoint runs/phase2/$key/best.pt --data $DATA --device cuda:0 --seed $mapseed --width 256 --depth 4 --steps 20000 --field-weights 1 --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 --include-voltage --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh --state-gate-weights 1,100,1 --select-best --calibration-split trajectory --output runs/phase2/${key}_vc_trajcal_seed${mapseed}.json > ${log}.chart${mapseed}.log 2>&1"
  done
  cmd="$cmd; echo \$? > ${log}.exit"
  tmux new-session -d -s "e86_k${k}s${seed}_g${gpu}" "bash -lc '$cmd'"
}

launch_one 2 5 56
launch_one 3 5 57
launch_one 4 5 58
launch_one 5 6 56
launch_one 6 6 57
launch_one 7 6 58

tmux new-session -d -s e86_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e86_k*s*_g*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 6; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
