# HH Self-Discovery

This repository asks whether predictive models trained only on observable current and voltage histories can recover the hidden states and dynamics of the Hodgkin-Huxley (HH) model.

The workflow enforces a **truth firewall**: gate trajectories (`n`, `m`, and `h`) and the analytic HH generator are excluded from training, validation, checkpoint selection, and latent-dimension selection. They are opened only after a model is locked, for four increasingly strict audits:

1. long-horizon response prediction;
2. low-complexity gate-state recovery;
3. transfer of the same chart to unseen stimulation protocols;
4. agreement of the transported latent field with the controlled HH generator.

The central finding is that accurate prediction and near-perfect gate decoding do not by themselves certify mechanism recovery: the fast sodium-activation state can be decoded while its physical clock remains systematically distorted.

## Repository layout

- `src/hh_self_discovery/`: simulator, protocols, datasets, models, training, and audits.
- `scripts/`: data generation, training, evaluation, robustness controls, and audits.
- `tests/`: unit and integration tests for simulation, data isolation, models, and evaluation.
- `configs/`: experiment configurations.
- `external/`: upstream baseline repositories, fetched by `scripts/fetch_baselines.sh`.
- `external_patches/`: compatibility patches applied to upstream baselines.

Large generated datasets and checkpoints are intentionally excluded from Git.

## Installation

```bash
git clone https://github.com/factnn/hh-self-discovery.git
cd hh-self-discovery
python -m venv .venv
source .venv/bin/activate
pip install -e .
bash scripts/fetch_baselines.sh
```

The last step clones the upstream baseline implementations into `external/` and
prints the compatibility patch command for the DDAE baseline.

## Validation and reproduction

```bash
python -m pytest
PYTHONPATH=src python scripts/smoke_simulation.py
CUDA_VISIBLE_DEVICES=0 bash scripts/reproduce_final.sh
```

`scripts/reproduce_final.sh` generates missing data, trains the locked model family, and runs the observable, OOD, and post-lock audits. GPU selection is controlled through `CUDA_VISIBLE_DEVICES`; no physical device index is assumed by the code.

Aggregate machine-readable results for the reported audits are under `results/`; generated datasets, checkpoints, and run directories are recreated by the scripts above.

## Authors

Peiyu Zang, Jiayi Hao, and Yongqiang Cai (corresponding author),
School of Mathematical Sciences, Beijing Normal University.

## Citation

```bibtex
@misc{zang2026retracing,
  title  = {Retracing Hodgkin and Huxley: State Recovery Does Not Certify Mechanism},
  author = {Zang, Peiyu and Hao, Jiayi and Cai, Yongqiang},
  year   = {2026},
  note   = {Preprint}
}
```

## License

MIT. See `LICENSE`.
