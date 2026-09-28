#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmux_target="${BASELINE_TMUX_TARGET:-hh:0.0}"
poll_seconds="${BASELINE_POLL_SECONDS:-60}"

while true; do
  if [[ -s "${project_root}/runs/baselines/b3_fnn_k2_s57/training.json" &&
        -s "${project_root}/runs/baselines/b3_fnn_k2_s58/training.json" ]]; then
    message="Baseline training finished; run the unified tests and gate audits."
    tmux send-keys -t "${tmux_target}" -l "${message}"
    sleep 1
    tmux send-keys -t "${tmux_target}" Enter
    exit 0
  fi
  sleep "${poll_seconds}"
done
