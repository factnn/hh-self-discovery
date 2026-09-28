#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
cd "$ROOT" || exit 1
launch_chain() {
    gpu=$1; shift; chain=""
    for spec in "$@"; do
        k=${spec%%:*}; seed=${spec##*:}; key="transient2_latentdim${k}_kw24_h750_s${seed}"
        for condition in clean extreme; do
            if test "$condition" = clean; then noise=0; stride=1; else noise=0.40; stride=50; fi
            log="runs/phase2/e170_k${k}s${seed}_${condition}_g${gpu}"
            job="CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PY scripts/evaluate_latent_model.py --checkpoint runs/phase2/$key/best.pt --data $DATA --split test --windows-per-mode 500 --seed 16900 --prediction-steps 1000 --skip-ablations --response-noise-fraction $noise --history-observation-stride $stride --device cuda:0 --output runs/phase2/${key}_test20ms_e170_${condition}.json > ${log}.log 2>&1; echo \$? > ${log}.exit"
            if test -z "$chain"; then chain="$job"; else chain="$chain; $job"; fi
        done
    done
    tmux new-session -d -s "e170_g${gpu}" "bash -lc 'cd $ROOT; $chain'"
}
launch_chain 0 3:56 4:58
launch_chain 1 3:57 4:56
launch_chain 2 3:58
launch_chain 3 4:57
tmux new-session -d -s e170_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in runs/phase2/e170_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 12; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 300; done'"
