#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PY="${PYTHON:-python}"; cd "$ROOT" || exit 1
cmd="cd $ROOT; PYTHONPATH=src:scripts $PY scripts/audit_m_instantaneous_observability.py --data data/generated/full_stratified_active_v1 --split test --output runs/phase2/e135_m_instantaneous_observability.json > runs/phase2/e135_m_observability.log 2>&1; echo \$? > runs/phase2/e135_m_observability.exit"; tmux new-session -d -s e135_m_observability "bash -lc '$cmd'"
tmux new-session -d -s e135_monitor "bash -lc 'cd $ROOT; while :; do if test -f runs/phase2/e135_m_observability.exit; then code=\$(tr -d \"[:space:]\" < runs/phase2/e135_m_observability.exit); if test \"\$code\" = 0; then msg=\"All jobs finished with exit code 0.\"; else msg=\"At least one job failed; check the exit codes and logs.\"; fi; tmux send-keys -t hh:0.0 -l \"\$msg\"; sleep 2; tmux send-keys -t hh:0.0 C-m; exit; fi; sleep 60; done'"
