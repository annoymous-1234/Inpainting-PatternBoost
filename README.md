# Inpainting PatternBoost

Anonymous code repository for the inpainting PatternBoost paper. Each problem
lives in its own self-contained sub-repository:

- [`no-three-in-line/`](no-three-in-line/) — no-three-in-line problem
- [`packing/`](packing/) — circle packing in the unit square

To run experiments, enter the corresponding sub-repository and follow its
local `README.md`:

```bash
cd no-three-in-line && cat README.md
cd packing && cat README.md
```

Experiment reports for the paper are collected under
[`exp-result/`](exp-result/), organized as
`exp-result/<problem>/<group>/<method>/<run>/`.
