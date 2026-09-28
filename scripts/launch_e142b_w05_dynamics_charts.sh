#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1

launch_chart() {
    gpu=$1
    k=$2
    mapseed=$3
    key="transientw05_latentdim${k}_kw24_h750_s56"
    log="runs/phase2/e142b_chart_k${k}m${mapseed}_g${gpu}"
    cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py --checkpoint runs/phase2/$key/best.pt --data $DATA --device cuda:0 --seed $mapseed --width 256 --depth 4 --steps 20000 --field-weights 1 --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 --include-voltage --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh --state-gate-weights 1,100,1 --select-best --calibration-split trajectory --output runs/phase2/${key}_vc_trajcal_seed${mapseed}.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
    tmux new-session -d -s "e142b_chart_k${k}m${mapseed}_g${gpu}" "bash -lc '$cmd'"
}

launch_chart 4 5 8989
launch_chart 5 5 9090
launch_chart 6 6 8989
launch_chart 7 6 9090

tmux new-session -d -s e142b_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e142b_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 4; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
