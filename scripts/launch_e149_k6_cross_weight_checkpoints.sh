#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1
launch() {
    gpu=$1; prefix=$2; tag=$3; seed=$4
    key="${prefix}_latentdim6_kw24_h750_s${seed}"
    mapseed=8989
    log="runs/phase2/e149_${tag}_s${seed}_g${gpu}"
    cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py --checkpoint runs/phase2/$key/best.pt --data $DATA --device cuda:0 --seed $mapseed --width 256 --depth 4 --steps 20000 --field-weights 1 --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 --include-voltage --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh --state-gate-weights 1,100,1 --select-best --calibration-split trajectory --output runs/phase2/${key}_vc_trajcal_seed${mapseed}.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
    tmux new-session -d -s "e149_${tag}_s${seed}_g${gpu}" "bash -lc '$cmd'"
}
launch 0 transientw1 w1 57
launch 1 transientw1 w1 58
launch 2 transient2 w2 57
launch 3 transient2 w2 58
tmux new-session -d -s e149_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e149_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 4; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
