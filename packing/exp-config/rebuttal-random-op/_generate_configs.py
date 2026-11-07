#!/usr/bin/env python3
"""Generate the random-completion-operator ablation configs.

Each config is the K=26 *iterative* main config of its head and seed, with only
these fields changed, so a diff against the template is the full statement of
the ablation:

  env.activation_budget            4 -> 26   complete every masked slot in one
                                             round (the published one-shot
                                             recipe; random completion gains
                                             nothing from iterating, and one
                                             round makes displacement a closed
                                             form of sigma)
  generation.completion_operator   learned -> random_perturb
  generation.random_center_jitter_std        the noise scale
  runtime.device                   cuda -> cpu     (no model is ever built)
  runtime.save_checkpoints         -> false        (nothing to checkpoint)
  runtime.checkpoint_dir/run_name  unique per run  (the entrypoint hardcodes
                                             checkpoint_dir="checkpoints" and
                                             uses run_name as the subdirectory,
                                             so a shared run_name overwrites)

Everything else -- top pool, edit policies and ratios, generation count,
candidate budget, polish settings, and both seeds -- is inherited verbatim from
the learned run each config is compared against.

Choosing sigma
--------------
A masked center reaches its final position in two steps, and the random arm
inherits the first one unchanged:

    parent --(draft jitter, train.draft_center_jitter_std)--> draft
           --(completion operator)--> output

`inference_center_distance_mean` -- the metric the paper reports -- measures
only the second leg. The quantity that says how far a child actually moved from
its parent is the combination of both, and for isotropic 2D Gaussians a
displacement of standard deviation s has mean length s*sqrt(pi/2), so the two
legs add in quadrature:

    total(sigma_r) = sqrt(sigma_draft^2 + sigma_r^2) * sqrt(pi/2)

The draft jitter is head-specific: every DDPM config uses 0.02, every FM config
uses 0.05. Measured on the comparison arms at K=26 (whole-run mean of
inference_center_distance_mean, over the seeds that record it):

    head   learned 2nd leg   learned total     sigma_r grid -> total
    ddpm            0.0451          0.0516     0.04 -> 0.052   (matched)
                                               0.05 -> 0.068
                                               0.06 -> 0.079
                                               0.10 -> 0.128   (= generation 1)
    fm              0.0524          0.0817     0.05 -> 0.089
                                               0.06 -> 0.098

The grid therefore starts at parity with the learned operator and sweeps up to
roughly 2.5x its movement, so "the random operator just needed a bigger step"
cannot explain a gap.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_CONFIG = HERE.parent

# (head, sigma tag, sigma, seed) -- tag keeps run names sortable and unique.
SPECS = [
    ("ddpm", "04", 0.04, 1234),
    ("ddpm", "10", 0.10, 1234),
    ("ddpm", "05", 0.05, 1234),
    ("ddpm", "05", 0.05, 44),
    ("ddpm", "06", 0.06, 1234),
    ("ddpm", "06", 0.06, 44),
    ("fm", "05", 0.05, 1234),
    ("fm", "05", 0.05, 44),
    ("fm", "06", 0.06, 1234),
    ("fm", "06", 0.06, 44),
]


def make(head: str, tag: str, sigma: float, seed: int) -> Path:
    template_stem = f"K26_{head}_main_rt{seed}"
    template = EXP_CONFIG / "main" / head / f"{template_stem}.yaml"
    text = template.read_text(encoding="utf-8")
    run_name = f"K26_{head}_randperturb_s{tag}_rt{seed}"

    edits = [
        ("  activation_budget: 4\n", "  activation_budget: 26\n"),
        ("    greedy_complete_unfinished: true\n",
         "    greedy_complete_unfinished: true\n"
         "    completion_operator: random_perturb\n"
         f"    random_center_jitter_std: {sigma}\n"),
        ("    device: cuda\n", "    device: cpu\n"),
        (f"checkpoints/{template_stem}", f"checkpoints/{run_name}"),
        (f"run_name: {template_stem}", f"run_name: {run_name}"),
    ]
    for old, new in edits:
        assert text.count(old) == 1, f"expected exactly one {old!r} in {template}"
        text = text.replace(old, new, 1)

    # The DDPM templates enable checkpointing and the FM ones do not; either way
    # the random arm has no model to save.
    assert text.count("    save_checkpoints: ") == 1, f"expected one save_checkpoints in {template}"
    text = text.replace("    save_checkpoints: true\n", "    save_checkpoints: false\n", 1)

    out = HERE / head / f"{run_name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


if __name__ == "__main__":
    for spec in SPECS:
        path = make(*spec)
        print(f"wrote {path.relative_to(EXP_CONFIG.parent)}")
