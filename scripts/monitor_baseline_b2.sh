#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmux_target="${BASELINE_TMUX_TARGET:-hh:0.0}"
poll_seconds="${BASELINE_POLL_SECONDS:-60}"

expected=(
  b2_gru_bottleneck_k2_s51
  b2_gru_bottleneck_k3_s51
  b2_gru_bottleneck_k5_s51
  b2_s4d_bottleneck_k2_s51
  b2_s4d_bottleneck_k3_s51
  b2_s4d_bottleneck_k5_s51
  b2_controlled_latent_ode_k2_s51
  b2_controlled_latent_ode_k3_s51
  b2_controlled_latent_ode_k5_s51
  b2_fnn_k2_s51
  b2_fnn_k3_s51
  b2_fnn_k5_s51
)

while true; do
  missing=0
  for run in "${expected[@]}"; do
    if [[ ! -s "${project_root}/runs/baselines/${run}/training.json" ]]; then
      missing=$((missing + 1))
    fi
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
