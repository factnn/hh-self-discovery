#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

poll_seconds="${1:-60}"
marker="runs/phase2/.e19_training_complete"
pattern='scripts/train_latent_model.py.*latentdim(3_kw24|6_kw16)_h750_s5[1-4]'

while pgrep -f "$pattern" >/dev/null; do
  sleep "$poll_seconds"
done

expected=0
for seed in 51 52 53 54; do
  for spec in 3_kw24 6_kw16; do
    if [[ -s "runs/phase2/latentdim${spec}_h750_s${seed}/training.json" ]]; then
      expected=$((expected + 1))
    fi
  done
done

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$marker"
printf 'E19_TRAINING_FINISHED summaries=%d/8 marker=%s\n' "$expected" "$marker"
