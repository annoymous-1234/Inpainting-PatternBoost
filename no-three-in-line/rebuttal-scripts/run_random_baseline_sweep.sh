#!/usr/bin/env bash
#
# Random + Saturation baseline for no-three-in-line (paper Table 3 row 1,
# Table 1 "Random+Sat" column and Table 9 row 1). This is the FROM-SCRATCH floor:
# random greedy candidates + greedy saturation into a TopPool, with NO active
# core and NO learned model. Uses random_baseline.py --mode toppool.
#
# Budgets are matched per n to whatever the learned runs use at that n:
#   n=20 : pool4096 C4096 G48   (matches rebuttal-exp-config/n20_*.yaml)
#   n=25 : pool4096 C4096 G64   (matches paper Table 3 / the n25 ablation runs)
#   n=30 : pool4096 C4096 G48   (matches rebuttal-exp-config/n30_*.yaml)
#   n=15 : pool4096 C8192 G30   (Table 1 only; enable with WITH_N15=1)
#
# Seeds: three at every n.
#
# CPU-only compute -- importing the repo pulls torch, so the `ec` env is still
# needed, but no GPU is used and this can run ALONGSIDE the GPU sweeps.
#
# Output : checkpoints/random_baseline/rebuttal/n<N>_seed<S>/  (summary.json, metrics.jsonl)
# Logs   : logs/random-baseline/n<N>_seed<S>.log
# Runs whose summary.json already exists are skipped, so re-invoking is safe.
#
# Usage (from anywhere):
#   conda activate ec
#   rebuttal-scripts/run_random_baseline_sweep.sh --dry-run
#   rebuttal-scripts/run_random_baseline_sweep.sh              # n=20,25,30 per spec
#   rebuttal-scripts/run_random_baseline_sweep.sh --only n25   # substring filter
#   WITH_N15=1 rebuttal-scripts/run_random_baseline_sweep.sh   # also do Table 1's n=15
#   rebuttal-scripts/run_random_baseline_sweep.sh --aggregate-only

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PYTHON="${PYTHON:-python}"
OUT="checkpoints/random_baseline/rebuttal"
LOG="logs/random-baseline"

# n : comma-separated seeds : candidates_per_generation : num_generations
SPEC=(
  "20:0,23,44:4096:48"
  "25:0,23,44:4096:64"
  "30:0,23,44:4096:48"
)
[[ "${WITH_N15:-0}" == "1" ]] && SPEC=("15:0,23,44:8192:30" "${SPEC[@]}")

POOL="${POOL:-4096}"
FILTER=""; DRY_RUN=0; AGG_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)           FILTER="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --aggregate-only) AGG_ONLY=1; shift ;;
    -h|--help)        sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG"
queued=0
for entry in "${SPEC[@]}"; do
  IFS=: read -r n seeds C G <<< "$entry"
  IFS=, read -ra seed_list <<< "$seeds"
  for s in "${seed_list[@]}"; do
    name="n${n}_seed${s}"
    [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && continue
    dir="$OUT/$name"
    if [[ -f "$dir/summary.json" ]]; then
      echo "[skip]  $name (summary.json exists)"; continue
    fi
    echo "[queue] $name  (n=$n pool=$POOL C=$C G=$G seed=$s)"
    queued=$((queued+1))
    [[ $DRY_RUN -eq 1 || $AGG_ONLY -eq 1 ]] && continue
    {
      echo "=== $name | n=$n pool=$POOL C=$C G=$G seed=$s | $(date -Is) ==="
      PYTHONUNBUFFERED=1 "$PYTHON" random_baseline.py --mode toppool \
        --n "$n" --top-pool-size "$POOL" --candidates-per-generation "$C" \
        --num-generations "$G" --seed "$s" \
        --output-dir "$OUT" --run-name "$name"
      echo "=== exit=$? | $(date -Is) ==="
    } 2>&1 | tee -a "$LOG/$name.log"
  done
done
[[ $DRY_RUN -eq 1 ]] && { echo; echo "$queued run(s) would launch. (dry run)"; exit 0; }

# ---- aggregate: per-n best_score, with mean+/-std where >1 seed -------------
echo; echo "=== Random + Saturation baseline (from scratch) ==="
"$PYTHON" - "$OUT" <<'PY'
import json, glob, os, sys, statistics as st
from collections import defaultdict
out = sys.argv[1]
g = defaultdict(dict)
for f in glob.glob(os.path.join(out, "n*_seed*", "summary.json")):
    name = os.path.basename(os.path.dirname(f))
    try:
        n = int(name.split("_")[0][1:]); s = int(name.split("seed")[1])
        g[n][s] = json.load(open(f))["best_score"]
    except Exception:
        pass
if not g:
    print("  (no results yet)")
for n in sorted(g):
    xs = [g[n][s] for s in sorted(g[n])]
    line = f"  n={n:>2}: per-seed={dict(sorted(g[n].items()))}  max={max(xs)}"
    if len(xs) > 1:
        line += f"  mean={st.mean(xs):.2f}  std={st.stdev(xs):.2f}"
    print(line + f"  (k={len(xs)})")
print("\n  reference (submitted, archived in ../exp-result/.../random_baseline):"
      "\n    n15=28  n20=36  n25=44   <- seed 0; n20/n25 used the same budgets as above")
PY
