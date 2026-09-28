#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
poll_seconds="${1:-60}"
marker="runs/phase2/.e20_e21_training_complete"
pattern='scripts/train_latent_model.py.*(latentdim6_kw24_h750_s5[2-5]|fastproxy_d[36]_kw(16|24)_h750_s51)'

while pgrep -f "$pattern" >/dev/null; do
  sleep "$poll_seconds"
done

summaries=0
for seed in 52 53 54 55; do
  [[ -s "runs/phase2/latentdim6_kw24_h750_s${seed}/training.json" ]] && summaries=$((summaries + 1))
done
for dim in 3 6; do
  for width in 16 24; do
    [[ -s "runs/phase2/fastproxy_d${dim}_kw${width}_h750_s51/training.json" ]] && summaries=$((summaries + 1))
  done
done

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$marker"
printf 'E20_E21_TRAINING_FINISHED summaries=%d/8 marker=%s\n' "$summaries" "$marker"
