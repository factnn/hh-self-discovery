#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
RUNS=runs/phase2
cd "$ROOT" || exit 1

keys=(
  latentdim3_kw24_h750_s51 latentdim3_kw24_h750_s52 latentdim3_kw24_h750_s53
  latentdim3_kw24_h750_s54 latentdim3_kw24_h750_s55
  latentdim5_kw24_h750_s51 latentdim5_kw24_h750_s52 latentdim5_kw24_h750_s53
  latentdim5_kw24_h750_s54 latentdim5_kw24_h750_s55
  latentdim6_kw24_h750_s51 latentdim6_kw24_h750_s52 latentdim6_kw24_h750_s53
  latentdim6_kw24_h750_s54 latentdim6_kw24_h750_s55
)

for gpu in 0 1 2 3 4 5 6 7; do
  jobs=()
  for idx in "${!keys[@]}"; do
    if test $((idx % 8)) -eq "$gpu"; then jobs+=("${keys[$idx]}"); fi
  done
  command="cd $ROOT;"
  for key in "${jobs[@]}"; do
    tag=${key#latentdim}
    command+=" CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:scripts $PY scripts/audit_claim_chain.py --checkpoint $RUNS/$key/best.pt --data $DATA --device cuda:0 --windows-per-mode 200 --stride 5 --seed 19600 --ridge-alpha 100 --jvp-epsilon-ms 0.01 --output $RUNS/${key}_claim_chain_e196r.json > $RUNS/e196r_${tag}_g${gpu}.log 2>&1; echo \$? > $RUNS/e196r_${tag}_g${gpu}.exit;"
  done
  tmux new-session -d -s "e196r_g${gpu}" "bash -lc '$command'"
done

tmux new-session -d -s e196r_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in $RUNS/e196r_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 15; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 Enter; exit; fi; sleep 60; done'"

echo "Launched E196R on GPUs 0-7."
