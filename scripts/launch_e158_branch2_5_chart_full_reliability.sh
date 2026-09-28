#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1
launch_pair() {
    gpu=$1; b=$2; s1=$3; m1=$4; s2=$5; m2=$6; tag=$7
    key1="branch${b}_latentdim5_kw24_h750_s${s1}"
    key2="branch${b}_latentdim5_kw24_h750_s${s2}"
    common="--data $DATA --device cuda:0 --width 256 --depth 4 --steps 20000 --field-weights 1 --learning-rate 0.0002 --batch-size 4096 --grad-clip 1 --include-voltage --mode voltage_clamp --jvp-epsilon 0.01 --field-target mapped_hh --state-gate-weights 1,100,1 --select-best --calibration-split trajectory"
    cmd="cd $ROOT; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py --checkpoint runs/phase2/$key1/best.pt $common --seed $m1 --output runs/phase2/${key1}_vc_trajcal_seed${m1}.json > runs/phase2/e158_${tag}_a_g${gpu}.log 2>&1; a=\$?; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/train_dynamics_aware_map.py --checkpoint runs/phase2/$key2/best.pt $common --seed $m2 --output runs/phase2/${key2}_vc_trajcal_seed${m2}.json > runs/phase2/e158_${tag}_b_g${gpu}.log 2>&1; c=\$?; test \$a -eq 0 -a \$c -eq 0; echo \$? > runs/phase2/e158_${tag}_g${gpu}.exit"
    tmux new-session -d -s "e158_${tag}_g${gpu}" "bash -lc '$cmd'"
}
launch_pair 4 2 56 8989 56 9090 b2s56
launch_pair 5 5 56 8989 56 9090 b5s56
launch_pair 6 2 57 9090 58 9090 b2_retest
launch_pair 7 5 57 9090 58 9090 b5_retest
tmux new-session -d -s e158_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e158_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 4; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
