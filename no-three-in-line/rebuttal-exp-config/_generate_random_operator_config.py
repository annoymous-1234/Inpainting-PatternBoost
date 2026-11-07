#!/usr/bin/env python3
"""Generate the random-completion-operator ablation configs.

Each config is the published *iterative* main config it is compared against --
same n, same seed, same generation and candidate budget -- with only these
fields changed, so a diff against the template is the full statement of the
ablation:

  generation.completion_operator   learned -> random
  runtime.device                   cuda -> cpu     (no model is ever built)
  runtime.checkpoint_dir           unique per run

Under `completion_operator: random` the decode loop is skipped entirely and the
masked slots are filled by the environment's own `greedy_saturate`, which
already draws a legal row uniformly at random for the slot it selects. The
`decode` section is therefore inert and is left untouched, and `saturation` is
deliberately left at the template's `slot_order: mrv, y_order: random`: it is a
mechanism both arms share, so changing it would mix a second effect into the
Ours-minus-random difference.

Arms and their comparison targets:

  n=25 seed 44  vs exp-config/max/n25_main_stage1_rt44   Best 48
                   (d256/L6/H8, 64 gen x 4096 cand)
  n=30 seed 23  vs rebuttal-exp-config/n30_main_rt23     Best 57
                   (d256/L6/H8, 48 gen x 4096 cand)

Seed 0 at n=25 is not generated here: its config is n25_random_op.yaml, archived
as n25_random_op_mrv.

n=30 pairs against the *fixed-backbone* run
(checkpoints/n30_main_rt23, seed 23, 48 generations of 4096, d_model=256), NOT
exp-config/max/n30_main.yaml, which used a larger backbone (d_model=320) and a
3x budget (96 generations of 8192) under seed 11. Backbone, budget and seed all
have to match the arm being compared against.

Budget is inherited from each template rather than normalised, because the
comparison only means anything at equal search budget.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# (n, seed, template relative to REPO, template's patternboost checkpoint_dir)
SPECS = [
    (25, 44, "exp-config/max/n25_main_stage1_rt44.yaml", "checkpoints/n25_main_stage1_rt44"),
    (30, 23, "rebuttal-exp-config/n30_main_rt23.yaml", "checkpoints/n30_main_rt23"),
]


def make(n: int, seed: int, template_path: str, template_ckpt: str) -> Path:
    template = REPO / template_path
    text = template.read_text(encoding="utf-8")
    run_name = f"n{n}_random_op_rt{seed}"

    edits = [
        # Anchor at the head of the generation block: the templates differ in
        # which generation keys they set, but all of them open with
        # num_generations.
        ("  generation:\n    num_generations: ",
         "  generation:\n    completion_operator: random\n    num_generations: "),
        # `device: cuda` also appears in the supervised and alphazero runtime
        # sections, so anchor on the patternboost block to switch only that one.
        (f"    device: cuda\n    num_workers: 0\n    checkpoint_dir: {template_ckpt}\n",
         f"    device: cpu\n    num_workers: 0\n    checkpoint_dir: checkpoints/{run_name}\n"),
    ]
    for old, new in edits:
        assert text.count(old) == 1, f"expected exactly one {old!r} in {template}"
        text = text.replace(old, new, 1)

    # The n=30 template ships with candidate logging off. Turn it on so every
    # arm carries the per-candidate seed_score/delta_score trail the inheritance
    # analysis needs. This is observation only: nothing in the
    # search reads it, so the arms stay comparable.
    assert text.count("    log_candidates_jsonl: ") == 1, f"expected one log_candidates_jsonl in {template}"
    text = text.replace("    log_candidates_jsonl: false\n", "    log_candidates_jsonl: true\n", 1)

    assert text.startswith(f"n: {n}\n"), f"unexpected n in {template}"

    out = HERE / f"{run_name}.yaml"
    out.write_text(text, encoding="utf-8")
    return out


if __name__ == "__main__":
    for spec in SPECS:
        path = make(*spec)
        print(f"wrote {path.relative_to(REPO)}")
