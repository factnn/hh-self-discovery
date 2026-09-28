#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmux_target="${BASELINE_TMUX_TARGET:-hh:0.0}"
poll_seconds="${BASELINE_POLL_SECONDS:-60}"
ode_notified=0
extension_notified=0

ode_runs=(
  b3_controlled_latent_ode_k2_s56
  b3_controlled_latent_ode_k2_s57
  b3_controlled_latent_ode_k2_s58
)
extension_runs=(
  b2_ddae_controlled_k2_s51
  b2_ddae_controlled_k3_s51
  b2_ddae_controlled_k5_s51
  b2_ddae_autonomous_k3_s51
  b3_fnn_k2_s56
)

complete() {
  local run
  for run in "${@}"; do
    if [[ ! -s "${project_root}/runs/baselines/${run}/training.json" ]]; then
      return 1
    fi
  done
  return 0
}

notify() {
  tmux send-keys -t "${tmux_target}" -l "${1}"
  sleep 1
  tmux send-keys -t "${tmux_target}" Enter
}

while true; do
  if [[ "${ode_notified}" -eq 0 ]] && complete "${ode_runs[@]}"; then
    notify "Baseline training finished; run the unified tests and gate audits."
    ode_notified=1
  fi
  if [[ "${extension_notified}" -eq 0 ]] && complete "${extension_runs[@]}"; then
    notify "Baseline training finished; run the unified tests and gate audits."
    extension_notified=1
  fi
  if [[ "${ode_notified}" -eq 1 && "${extension_notified}" -eq 1 ]]; then
    exit 0
  fi
  sleep "${poll_seconds}"
done
