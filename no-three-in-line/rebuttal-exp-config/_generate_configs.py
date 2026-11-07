#!/usr/bin/env python3
"""Generate the fixed-budget configs for no-three-in-line (paper Table 9).

FIXED d256/L6/H8 backbone + FIXED budget across n; only `n` and the arm toggle
vary, giving a controlled AC-vs-no-AC gap at a single model.

Settings:
  - n in {20, 30}.
  - num_generations reduced 64 -> 48.

Arms per n:
  - main   = ITERATIVE decode (paper's "Ours"), template max/n25_main_stage1.yaml
  - no_ac  = full-mask [1.0], one-shot,        template ablation-remask/n25_abl_no_ac.yaml
Each new config = the n=25 template with ONLY the top-level `n:`, num_generations,
and patternboost.runtime.checkpoint_dir changed (verified by diff). Everything
else (model, pool/C/samples/seeds, mask schedule, decode, seed=0) is identical.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NS = [20, 30]
GENERATIONS = 48  # was 64 in the templates

ARMS = {
    "main":  (REPO / "exp-config/max/n25_main_stage1.yaml",
              "checkpoints/n25_main_stage1",
              lambda n: (f"checkpoints/n{n}_main", f"n{n}_main.yaml")),
    "no_ac": (REPO / "exp-config/ablation-remask/n25_abl_no_ac.yaml",
              "checkpoints/n25_abl_no_ac",
              lambda n: (f"checkpoints/n{n}_abl_no_ac", f"n{n}_abl_no_ac.yaml")),
}

def make(template: Path, old_ckpt: str, new_ckpt: str, out_name: str, n: int) -> Path:
    text = template.read_text(encoding="utf-8")
    assert text.startswith("n: 25\n"), f"unexpected first line in {template}"
    assert old_ckpt in text, f"{old_ckpt} not in {template}"
    assert "num_generations: 64" in text, f"num_generations: 64 not in {template}"
    text = text.replace("n: 25\n", f"n: {n}\n", 1)
    text = text.replace("num_generations: 64", f"num_generations: {GENERATIONS}")
    text = text.replace(old_ckpt, new_ckpt)
    out = HERE / out_name
    out.write_text(text, encoding="utf-8")
    return out

if __name__ == "__main__":
    for n in NS:
        for arm, (tpl, old_ck, fn) in ARMS.items():
            new_ck, out_name = fn(n)
            p = make(tpl, old_ck, new_ck, out_name, n)
            print(f"wrote {p.relative_to(REPO)}")
