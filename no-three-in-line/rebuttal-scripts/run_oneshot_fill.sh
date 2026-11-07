#!/usr/bin/env bash
#
# Run seeds 23 and 44 of the "active core + one-shot decode" row at n=20 and n=30.
#
# These four runs are wired into run_rebuttal_scaling.sh behind its ONESHOT flag,
# so this script just calls it; every completed run is skipped automatically.
#
# Single GPU, sequential. Reference timing: a few hours at n=20, longer at n=30.
#
# Usage:
#   conda activate ec
#   rebuttal-scripts/run_oneshot_fill.sh --dry-run
#   rebuttal-scripts/run_oneshot_fill.sh                 # the four n=20/30 runs
#   rebuttal-scripts/run_oneshot_fill.sh --aggregate-only

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PYTHON="${PYTHON:-python}"
DRY=0; AGG_ONLY=0
for a in "$@"; do
  case "$a" in
    --dry-run)        DRY=1 ;;
    --aggregate-only) AGG_ONLY=1 ;;
    -h|--help)        sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

# ---- the four n=20 / n=30 seeds --------------------------------------------
if [[ $AGG_ONLY -eq 0 ]]; then
  echo "=== one-shot seeds at n=20 and n=30 ==="
  if [[ $DRY -eq 1 ]]; then
    ONESHOT=all "rebuttal-scripts/run_rebuttal_scaling.sh" --dry-run | grep -E "abl_oneshot|run\(s\)"
  else
    ONESHOT=all "rebuttal-scripts/run_rebuttal_scaling.sh"
  fi
fi

[[ $DRY -eq 1 ]] && { echo; echo "(dry run)"; exit 0; }

# ---- aggregate the whole one-shot row --------------------------------------
echo; echo "=== active core + one-shot decode, across n ==="
"$PYTHON" - <<'PY'
import json, os, statistics as st

def read(path):
    """Return (best, mean_pool) from a run, falling back to the last generation
    record when the run never reached run_complete."""
    if not os.path.exists(path):
        return None
    best = counts = None
    last_gen_counts = last_gen = None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("event") == "run_complete":
            best, counts = rec.get("best_score"), rec.get("final_score_counts")
        elif rec.get("generation"):
            c = rec.get("pool_score_counts_after")
            if c:
                last_gen_counts, last_gen = c, rec["generation"]
    partial = False
    if counts is None and last_gen_counts:
        counts, partial = last_gen_counts, True
        best = max(int(k) for k in counts)
    if not counts:
        return None
    tot = sum(counts.values())
    return best, sum(int(k) * v for k, v in counts.items()) / tot, (last_gen if partial else None)

IP = "checkpoints"
# seed 0 of n=25 is the published run, archived outside this repo
PAPER_N25 = (48, 46.30)
rows = {
    20: [f"{IP}/n20_abl_oneshot/exp/metrics.jsonl",
         f"{IP}/n20_abl_oneshot_rt23/exp/metrics.jsonl",
         f"{IP}/n20_abl_oneshot_rt44/exp/metrics.jsonl"],
    25: [None,  # paper seed 0
         f"{IP}/n25_abl_oneshot_rt23/exp/metrics.jsonl",
         f"{IP}/n25_abl_oneshot_rt44/exp/metrics.jsonl"],
    30: [f"{IP}/n30_abl_oneshot/exp/metrics.jsonl",
         f"{IP}/n30_abl_oneshot_rt23/exp/metrics.jsonl",
         f"{IP}/n30_abl_oneshot_rt44/exp/metrics.jsonl"],
}
for n, paths in rows.items():
    bests, means, notes = [], [], []
    for p in paths:
        if p is None:
            bests.append(PAPER_N25[0]); means.append(PAPER_N25[1]); notes.append("seed0=paper")
            continue
        r = read(p)
        if r is None:
            continue
        b, m, partial_gen = r
        bests.append(b); means.append(m)
        notes.append(f"{os.path.basename(os.path.dirname(os.path.dirname(p)))}"
                     + (f"(gen{partial_gen})" if partial_gen else ""))
    if not bests:
        print(f"  n={n}: (no data)"); continue
    f = lambda v: f"{st.mean(v):.2f} +/- {st.stdev(v):.2f}" if len(v) > 1 else f"{v[0]:.2f}"
    print(f"  n={n}: Best {f(bests)}   Mean pool {f(means)}   (k={len(bests)})")
    print(f"        {', '.join(notes)}")
PY
