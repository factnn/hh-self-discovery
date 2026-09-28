#!/usr/bin/env bash
# Fetch the upstream baseline implementations into external/.
#
# The baseline comparisons use the official repositories below. They are not
# vendored, so clone them once before running any baseline pipeline:
#
#   bash scripts/fetch_baselines.sh
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p external

fetch() {  # directory, repository
  local dir="external/$1" url="$2"
  if [ -d "$dir/.git" ]; then
    echo "already present: $dir"
  else
    echo "cloning $url -> $dir"
    git clone --depth 1 "$url" "$dir"
  fi
}

fetch latent_ode https://github.com/YuliaRubanova/latent_ode.git
fetch s4 https://github.com/state-spaces/s4.git
fetch fnn https://github.com/williamgilpin/fnn.git
fetch deep-delay-autoencoder https://github.com/josephbakarji/deep-delay-autoencoder.git

echo
echo "Apply the recorded compatibility patch for the DDAE baseline with:"
echo "  git -C external/deep-delay-autoencoder apply ../../external_patches/deep-delay-autoencoder.patch"
