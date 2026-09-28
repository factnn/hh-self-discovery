#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1

launch_chart() {
  b=$1
  seed=$2
  map_seed=$3
  gpu=$4
  tag=$5
  key="branch${b}_transientw05_latentdim5_kw24_h750_s${seed}"
  log="runs/phase2/e176_${tag}_g${gpu}"
  cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py --checkpoint runs/phase2/$key/best.pt --data $DATA --device cuda:0 --seed $map_seed --width 256 --depth 4 --steps 20000 --field-weights 1 --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 --include-voltage --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh --state-gate-weights 1,100,1 --select-best --calibration-split trajectory --output runs/phase2/${key}_vc_trajcal_seed${map_seed}.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
  tmux new-session -d -s "e176_${tag}_g${gpu}" "bash -lc '$cmd'"
}

gpu=0
for b in 2 5; do
  for seed in 56 57 58; do
    launch_chart "$b" "$seed" 9090 "$gpu" "b${b}s${seed}m9090"
    gpu=$((gpu+1))
  done
done

# A third chart initialization on the paired seed56 checkpoints distinguishes
# mapping instability from a genuinely topology-dependent bad-checkpoint tail.
launch_chart 2 56 9292 6 b2s56m9292
launch_chart 5 56 9292 7 b5s56m9292

tmux new-session -d -s e176_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e176_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 8; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 60; done'"
