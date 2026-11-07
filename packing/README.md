# Circle Packing

Inpainting PatternBoost implementation for circle packing in the unit square.
A slot-based Transformer with DDPM or flow-matching (FM) heads inpaints free
circle centers; an LP solver scores complete layouts; an optional SRP +
L-BFGS-B + LP polish refines the top candidates.

## Requirements

Python 3.10+ and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running experiments

Ready-to-run configs are provided in `exp-config/`, organized by experiment
group (`main`, `ablation-oneshot`, `ablation-remask`) and supervision mode
(`ddpm`, `fm`). Example:

```bash
# Single PatternBoost experiment with reproducible artifacts
python scripts/run_patternboost_experiment.py \
    --config exp-config/main/ddpm/K15_ddpm_main.yaml --device cuda

# Batch-run every YAML in a folder
python run_experiment_folder.py exp-config/main/ddpm --device cuda
```

Output directory follows `patternboost.runtime.checkpoint_dir/run_name/`.
Each run writes `config_snapshot.yaml`, `resolved_config.yaml`,
`run_manifest.json`, `metrics.jsonl`, `generations.jsonl`, `results.json`,
and `top_pool.pt`.

## Codebase structure

```
.
├── circle_pack/                     # environment, geometry, LP solver, SRP polish
├── models/                          # Transformer + DDPM / FM heads
├── patternboost/                    # PatternBoost outer loop, TopPool, edit, polish
├── pipelines/                       # supervised data and outer-loop helpers
│                                    # (used as a library by patternboost/)
├── scripts/
│   ├── run_patternboost_experiment.py   # main entrypoint
│   ├── train_patternboost.py            # direct PatternBoost training
│   ├── train_supervised.py              # legacy: supervised training
│   ├── run_outer_loop.py                # legacy: non-PatternBoost inference
│   ├── generate_data_demo.py            # legacy: teacher-data generation
│   └── demo_*.py                        # legacy: smoke demos
├── exp-config/                      # ready experiment configs
├── tests/                           # unit tests
├── config.yaml                      # default config
├── requirements.txt                 # third-party dependencies
└── run_experiment_folder.py         # batch runner over a config folder
```

`scripts/train_supervised.py`, `scripts/run_outer_loop.py`, and the
`scripts/demo_*.py` and `scripts/generate_data_demo.py` scripts are legacy
and not used in the paper. The `pipelines/` package is retained because the
PatternBoost loop reuses its supervised-data, outer-loop, and priority
modules as a library.
