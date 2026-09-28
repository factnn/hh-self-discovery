#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
OOD=data/generated/ood_protocols
KEY=branch2_transientw2_latentdim5_kw24_h750_s58
cd "$ROOT" || exit 1

blind="runs/phase2/${KEY}_ood_blind_e188_groupcv1se.json"
cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY scripts/blind_evaluate.py --checkpoint runs/phase2/$KEY/best.pt --data $DATA --evaluation-data $OOD --mapping-split validation --evaluation-split test --windows-per-mode 1000 --seed 18800 --device cuda:0 --ridge-alpha-grid 0.001 0.01 0.1 1 10 100 1000 10000 --ridge-cv-folds 5 --output $blind > runs/phase2/e188_b2s58_g0.blind.log 2>&1 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:scripts $PY scripts/audit_constrained_relaxation.py --checkpoint runs/phase2/$KEY/best.pt --blind-result $blind --data $DATA --windows-per-mode 200 --seed 18850 --steps 2000 --device cuda:0 --output runs/phase2/${KEY}_tau_e188.json > runs/phase2/e188_b2s58_g0.tau.log 2>&1; echo \$? > runs/phase2/e188_b2s58_g0.exit"
tmux new-session -d -s e188_b2s58_physical_g0 "bash -lc '$cmd'"

for spec in 8989:1 9090:3; do
  map_seed=${spec%%:*}; gpu=${spec##*:}; log="runs/phase2/e188_b2s58m${map_seed}_g${gpu}"
  cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py --checkpoint runs/phase2/$KEY/best.pt --data $DATA --device cuda:0 --seed $map_seed --width 256 --depth 4 --steps 20000 --field-weights 1 --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 --include-voltage --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh --state-gate-weights 1,100,1 --select-best --calibration-split trajectory --output runs/phase2/${KEY}_eventdecomp_e188_seed${map_seed}.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
  tmux new-session -d -s "e188_b2s58m${map_seed}_g${gpu}" "bash -lc '$cmd'"
done

tmux new-session -d -s e188_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e188_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 3; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 Enter; sleep 10; exit; fi; sleep 60; done'"
