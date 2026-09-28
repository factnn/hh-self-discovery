#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmux_target="${BASELINE_TMUX_TARGET:-hh:0.0}"
poll_seconds="${BASELINE_POLL_SECONDS:-60}"
runs=(
  b3_ddae_controlled_k3_s56
  b3_ddae_controlled_k3_s57
  b3_ddae_controlled_k3_s58
  b3_ddae_controlled_k5_s56
  b3_ddae_controlled_k5_s57
  b3_ddae_controlled_k5_s58
  b3_ddae_autonomous_k3_s56
  b3_ddae_autonomous_k3_s57
)

while true; do
  missing=0
  for run in "${runs[@]}"; do
    [[ -s "${project_root}/runs/baselines/${run}/training.json" ]] ||
      missing=$((missing + 1))
  done
  if [[ "${missing}" -eq 0 ]]; then
    message="Baseline training finished; run the unified tests and gate audits."
    tmux send-keys -t "${tmux_target}" -l "${message}"
    sleep 1
    tmux send-keys -t "${tmux_target}" Enter
    exit 0
  fi
  sleep "${poll_seconds}"
done
