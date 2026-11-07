# No Three In Line

Inpainting PatternBoost implementation for the no-three-in-line problem.

## Requirements

Python 3.10+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running experiments

Ready-to-run configs are provided in `exp-config/` (baseline, ablations,
and large `n` runs). Example:

```bash
# PatternBoost main loop on the n=10 baseline config
python train_patternboost.py --config exp-config/baseline/n10_main.yaml

# Random greedy baseline
python random_baseline.py --mode trials --n 15 --trials 100
```

All hyperparameters live in YAML configs. `config.yaml` at the repo root is
the default fallback when `--config` is omitted.

## Codebase structure

```
.
├── env.py, geometry.py, ...          # no-three-in-line environment
├── models/                           # column-pair Transformer
├── patternboost/                     # PatternBoost outer loop
├── supervised/                       # legacy; reused as a library by patternboost/
├── alphazero/                        # legacy; not used in the paper
├── examples/, tests/                 # demos and unit tests
├── config.yaml                       # all hyperparameters
├── train_patternboost.py             # PatternBoost entrypoint
└── random_baseline.py                # random greedy baseline
```

`supervised/` and `alphazero/` are legacy and not evaluated in the paper.
`supervised/` is kept because PatternBoost reuses its training, dataset, and
preprocessing modules as a library; `alphazero/` is kept only for reference.
