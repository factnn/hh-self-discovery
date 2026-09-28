#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1

declare -a chains
for gpu in $(seq 0 7); do chains[$gpu]=""; done
index=0
for weight in 1 2; do
  for seed in 56 57 58 59 60; do
    for map_seed in 8989 9090; do
      gpu=$((index % 8))
      key="branch2_transientw${weight}_latentdim5_kw24_h750_s${seed}"
      log="runs/phase2/e192_b2w${weight}s${seed}m${map_seed}_g${gpu}"
      job="CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py --checkpoint runs/phase2/$key/best.pt --data $DATA --device cuda:0 --seed $map_seed --width 256 --depth 4 --steps 20000 --field-weights 1 --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 --include-voltage --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh --state-gate-weights 1,100,1 --select-best --calibration-split trajectory --calibration-field-objective true_field_far --output runs/phase2/${key}_farselect_e192_seed${map_seed}.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
      if test -z "${chains[$gpu]}"; then chains[$gpu]="$job"; else chains[$gpu]="${chains[$gpu]}; $job"; fi
      index=$((index + 1))
    done
  done
done

for gpu in $(seq 0 7); do
  tmux new-session -d -s "e192_g${gpu}" "bash -lc 'cd $ROOT; ${chains[$gpu]}'"
done

tmux new-session -d -s e192_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e192_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 20; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 Enter; sleep 10; exit; fi; sleep 60; done'"
