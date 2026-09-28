#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PY="${PYTHON:-python}"; cd "$ROOT" || exit 1
cmd="cd $ROOT; PYTHONPATH=src:scripts $PY scripts/audit_m_information_concentration.py --data data/generated/full_stratified_active_v1 --split test --output runs/phase2/e136_m_information_concentration.json > runs/phase2/e136_m_information_concentration.log 2>&1; echo \$? > runs/phase2/e136_m_information_concentration.exit"; tmux new-session -d -s e136_m_information "bash -lc '$cmd'"
tmux new-session -d -s e136_monitor "bash -lc 'cd $ROOT; while :; do if test -f runs/phase2/e136_m_information_concentration.exit; then code=\$(tr -d \"[:space:]\" < runs/phase2/e136_m_information_concentration.exit); if test \"\$code\" = 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 60; done'"
