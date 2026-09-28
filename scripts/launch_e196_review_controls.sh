#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA=data/generated/full_stratified_active_v1
RUNS=runs/phase2
cd "$ROOT" || exit 1

run_claim() {
  gpu=$1
  key=$2
  tag=$3
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:scripts "$PY" scripts/audit_claim_chain.py \
    --checkpoint "$RUNS/$key/best.pt" --data "$DATA" --device cuda:0 \
    --windows-per-mode 200 --stride 5 --seed 19600 --ridge-alpha 100 \
    --jvp-epsilon-ms 0.01 --output "$RUNS/${key}_claim_chain_e196.json" \
    > "$RUNS/e196_${tag}_g${gpu}.log" 2>&1
  echo $? > "$RUNS/e196_${tag}_g${gpu}.exit"
}

run_control() {
  gpu=$1
  seed=$2
  tag=$3
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src:scripts "$PY" scripts/audit_known_coordinate_control.py \
    --checkpoint "$RUNS/latentdim5_kw24_h750_s51/best.pt" --data "$DATA" \
    --device cuda:0 --windows-per-mode 300 --stride 5 --seed "$seed" \
    --ridge-alpha 100 --jvp-epsilon-ms 0.01 \
    --output "$RUNS/known_coordinate_control_e197_seed${seed}.json" \
    > "$RUNS/e197_${tag}_g${gpu}.log" 2>&1
  echo $? > "$RUNS/e197_${tag}_g${gpu}.exit"
}

tmux new-session -d -s e196_g0 "bash -lc 'cd $ROOT; source scripts/launch_e196_review_controls.sh --worker-placeholder'" 2>/dev/null || true

if test "${1:-}" = "--worker-placeholder"; then
  exit 0
fi

tmux kill-session -t e196_g0 2>/dev/null || true

tmux new-session -d -s e196_g0 "bash -lc 'cd $ROOT; run_claim(){ gpu=\$1; key=\$2; tag=\$3; CUDA_VISIBLE_DEVICES=\$gpu PYTHONPATH=src:scripts $PY scripts/audit_claim_chain.py --checkpoint $RUNS/\$key/best.pt --data $DATA --device cuda:0 --windows-per-mode 200 --stride 5 --seed 19600 --ridge-alpha 100 --jvp-epsilon-ms 0.01 --output $RUNS/\${key}_claim_chain_e196.json > $RUNS/e196_\${tag}_g\${gpu}.log 2>&1; echo \$? > $RUNS/e196_\${tag}_g\${gpu}.exit; }; run_claim 0 latentdim3_kw24_h750_s51 k3s51; run_claim 0 latentdim5_kw24_h750_s54 k5s54'"
tmux new-session -d -s e196_g1 "bash -lc 'cd $ROOT; run_claim(){ gpu=\$1; key=\$2; tag=\$3; CUDA_VISIBLE_DEVICES=\$gpu PYTHONPATH=src:scripts $PY scripts/audit_claim_chain.py --checkpoint $RUNS/\$key/best.pt --data $DATA --device cuda:0 --windows-per-mode 200 --stride 5 --seed 19600 --ridge-alpha 100 --jvp-epsilon-ms 0.01 --output $RUNS/\${key}_claim_chain_e196.json > $RUNS/e196_\${tag}_g\${gpu}.log 2>&1; echo \$? > $RUNS/e196_\${tag}_g\${gpu}.exit; }; run_claim 1 latentdim3_kw24_h750_s52 k3s52; run_claim 1 latentdim5_kw24_h750_s55 k5s55'"
tmux new-session -d -s e196_g2 "bash -lc 'cd $ROOT; run_claim(){ gpu=\$1; key=\$2; tag=\$3; CUDA_VISIBLE_DEVICES=\$gpu PYTHONPATH=src:scripts $PY scripts/audit_claim_chain.py --checkpoint $RUNS/\$key/best.pt --data $DATA --device cuda:0 --windows-per-mode 200 --stride 5 --seed 19600 --ridge-alpha 100 --jvp-epsilon-ms 0.01 --output $RUNS/\${key}_claim_chain_e196.json > $RUNS/e196_\${tag}_g\${gpu}.log 2>&1; echo \$? > $RUNS/e196_\${tag}_g\${gpu}.exit; }; run_claim 2 latentdim3_kw24_h750_s53 k3s53; run_claim 2 latentdim6_kw24_h750_s51 k6s51'"
tmux new-session -d -s e196_g3 "bash -lc 'cd $ROOT; run_claim(){ gpu=\$1; key=\$2; tag=\$3; CUDA_VISIBLE_DEVICES=\$gpu PYTHONPATH=src:scripts $PY scripts/audit_claim_chain.py --checkpoint $RUNS/\$key/best.pt --data $DATA --device cuda:0 --windows-per-mode 200 --stride 5 --seed 19600 --ridge-alpha 100 --jvp-epsilon-ms 0.01 --output $RUNS/\${key}_claim_chain_e196.json > $RUNS/e196_\${tag}_g\${gpu}.log 2>&1; echo \$? > $RUNS/e196_\${tag}_g\${gpu}.exit; }; run_claim 3 latentdim3_kw24_h750_s54 k3s54; run_claim 3 latentdim6_kw24_h750_s52 k6s52'"
tmux new-session -d -s e196_g4 "bash -lc 'cd $ROOT; run_claim(){ gpu=\$1; key=\$2; tag=\$3; CUDA_VISIBLE_DEVICES=\$gpu PYTHONPATH=src:scripts $PY scripts/audit_claim_chain.py --checkpoint $RUNS/\$key/best.pt --data $DATA --device cuda:0 --windows-per-mode 200 --stride 5 --seed 19600 --ridge-alpha 100 --jvp-epsilon-ms 0.01 --output $RUNS/\${key}_claim_chain_e196.json > $RUNS/e196_\${tag}_g\${gpu}.log 2>&1; echo \$? > $RUNS/e196_\${tag}_g\${gpu}.exit; }; run_claim 4 latentdim3_kw24_h750_s55 k3s55; run_claim 4 latentdim6_kw24_h750_s53 k6s53'"
tmux new-session -d -s e196_g5 "bash -lc 'cd $ROOT; run_claim(){ gpu=\$1; key=\$2; tag=\$3; CUDA_VISIBLE_DEVICES=\$gpu PYTHONPATH=src:scripts $PY scripts/audit_claim_chain.py --checkpoint $RUNS/\$key/best.pt --data $DATA --device cuda:0 --windows-per-mode 200 --stride 5 --seed 19600 --ridge-alpha 100 --jvp-epsilon-ms 0.01 --output $RUNS/\${key}_claim_chain_e196.json > $RUNS/e196_\${tag}_g\${gpu}.log 2>&1; echo \$? > $RUNS/e196_\${tag}_g\${gpu}.exit; }; run_claim 5 latentdim5_kw24_h750_s51 k5s51; run_claim 5 latentdim6_kw24_h750_s54 k6s54'"
tmux new-session -d -s e196_g6 "bash -lc 'cd $ROOT; run_claim(){ gpu=\$1; key=\$2; tag=\$3; CUDA_VISIBLE_DEVICES=\$gpu PYTHONPATH=src:scripts $PY scripts/audit_claim_chain.py --checkpoint $RUNS/\$key/best.pt --data $DATA --device cuda:0 --windows-per-mode 200 --stride 5 --seed 19600 --ridge-alpha 100 --jvp-epsilon-ms 0.01 --output $RUNS/\${key}_claim_chain_e196.json > $RUNS/e196_\${tag}_g\${gpu}.log 2>&1; echo \$? > $RUNS/e196_\${tag}_g\${gpu}.exit; }; run_claim 6 latentdim5_kw24_h750_s52 k5s52; run_claim 6 latentdim6_kw24_h750_s55 k6s55'"
tmux new-session -d -s e196_g7 "bash -lc 'cd $ROOT; run_claim(){ gpu=\$1; key=\$2; tag=\$3; CUDA_VISIBLE_DEVICES=\$gpu PYTHONPATH=src:scripts $PY scripts/audit_claim_chain.py --checkpoint $RUNS/\$key/best.pt --data $DATA --device cuda:0 --windows-per-mode 200 --stride 5 --seed 19600 --ridge-alpha 100 --jvp-epsilon-ms 0.01 --output $RUNS/\${key}_claim_chain_e196.json > $RUNS/e196_\${tag}_g\${gpu}.log 2>&1; echo \$? > $RUNS/e196_\${tag}_g\${gpu}.exit; }; run_control(){ gpu=\$1; seed=\$2; tag=\$3; CUDA_VISIBLE_DEVICES=\$gpu PYTHONPATH=src:scripts $PY scripts/audit_known_coordinate_control.py --checkpoint $RUNS/latentdim5_kw24_h750_s51/best.pt --data $DATA --device cuda:0 --windows-per-mode 300 --stride 5 --seed \$seed --ridge-alpha 100 --jvp-epsilon-ms 0.01 --output $RUNS/known_coordinate_control_e197_seed\${seed}.json > $RUNS/e197_\${tag}_g\${gpu}.log 2>&1; echo \$? > $RUNS/e197_\${tag}_g\${gpu}.exit; }; run_claim 7 latentdim5_kw24_h750_s53 k5s53; run_control 7 19700 s19700; run_control 7 19701 s19701; run_control 7 19702 s19702'"

tmux new-session -d -s e196_monitor "bash -lc 'cd $ROOT; while :; do n=0; bad=0; for f in $RUNS/e196_*.exit $RUNS/e197_*.exit; do test -f \"\$f\" || continue; n=\$((n+1)); test \"\$(tr -d \"[:space:]\" < \"\$f\")\" = 0 || bad=1; done; if test \$n -eq 18; then if test \$bad -eq 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 Enter; exit; fi; sleep 60; done'"

echo "Launched E196/E197 in tmux sessions e196_g0..e196_g7 with e196_monitor."
