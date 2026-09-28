#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1

launch_chain() {
    gpu=$1; seed=$2; chain=""
    key="transient2_latentdim3_kw24_h750_s${seed}"
    for mapseed in 8989 9090; do
        log="runs/phase2/e163_k3s${seed}m${mapseed}_g${gpu}"
        job="CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py --checkpoint runs/phase2/$key/best.pt --data $DATA --device cuda:0 --seed $mapseed --width 256 --depth 4 --steps 20000 --field-weights 1 --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 --include-voltage --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh --state-gate-weights 1,100,1 --select-best --calibration-split trajectory --output runs/phase2/${key}_vc_trajcal_seed${mapseed}.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
        if test -z "$chain"; then chain="$job"; else chain="$chain; $job"; fi
    done
    tmux new-session -d -s "e163_k3s${seed}_g${gpu}" "bash -lc 'cd $ROOT; $chain'"
}

launch_chain 0 57
launch_chain 1 58

tmux new-session -d -s e163_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e163_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 4; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
