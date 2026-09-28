#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1
launch_pair() {
    gpu=$1; prefix=$2; k=$3; tag=$4
    key="${prefix}_latentdim${k}_kw24_h750_s56"
    common="--checkpoint runs/phase2/$key/best.pt --data $DATA --device cuda:0 --width 256 --depth 4 --steps 20000 --field-weights 1 --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 --include-voltage --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh --state-gate-weights 1,100,1 --select-best --calibration-split trajectory"
    cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py $common --seed 9191 --output runs/phase2/${key}_vc_trajcal_seed9191.json > runs/phase2/e142d_${tag}_k${k}m9191_g${gpu}.log 2>&1; a=\$?; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py $common --seed 9292 --output runs/phase2/${key}_vc_trajcal_seed9292.json > runs/phase2/e142d_${tag}_k${k}m9292_g${gpu}.log 2>&1; b=\$?; test \$a -eq 0 -a \$b -eq 0; echo \$? > runs/phase2/e142d_${tag}_k${k}_g${gpu}.exit"
    tmux new-session -d -s "e142d_${tag}_k${k}_g${gpu}" "bash -lc '$cmd'"
}
launch_pair 4 transientw1 5 w1
launch_pair 5 transientw1 6 w1
launch_pair 6 transient2 5 w2
launch_pair 7 transient2 6 w2
tmux new-session -d -s e142d_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e142d_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 4; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
