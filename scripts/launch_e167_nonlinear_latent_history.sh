#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1
launch_chain() {
    gpu=$1; shift; chain=""
    for spec in "$@"; do
        k=${spec%%:*}; seed=${spec##*:}; key="transient2_latentdim${k}_kw24_h750_s${seed}"; log="runs/phase2/e167_k${k}s${seed}_g${gpu}"
        job="CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/audit_latent_history_reconstruction.py --checkpoint runs/phase2/$key/best.pt --data $DATA --split validation --windows-per-mode 1000 --seed 16700 --max-points-per-trajectory 200 --device cuda:0 --estimator extra_trees --trees 200 --output runs/phase2/${key}_latent_history_e167_nonlinear.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
        if test -z "$chain"; then chain="$job"; else chain="$chain; $job"; fi
    done
    tmux new-session -d -s "e167_g${gpu}" "bash -lc 'cd $ROOT; $chain'"
}
launch_chain 0 3:56 4:58
launch_chain 1 3:57 4:56
launch_chain 2 3:58
launch_chain 3 4:57
tmux new-session -d -s e167_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e167_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 6; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
