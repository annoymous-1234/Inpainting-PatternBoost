#!/usr/bin/env bash
#
# Run seeds 23 and 44 of the "Random + LP" row of Table 4.
#
# K=26 has its three seeds {1234, 23, 44} archived, as does seed 1234 at K=15, 30,
# 32 and 40. This script runs the other eight cells (four K values x seeds 23 and
# 44) with the same protocol -- 40 rounds, 2048 layouts per round, pool 2048 --
# which is what the archived directory names record
# (``..._K<K>_r40_s2048_pool2048_seed<S>``).
#
# The generator is scripts/random_lp_baseline.py: uniform random centers, exact
# radius LP per layout, keep the best `pool` scores. No model, no active core.
#
# CPU only, so this can run alongside the GPU sweeps. Roughly 2 minutes per run
# at K=15 and appreciably longer at K=40 (the LP has C(K,2) pair constraints);
# measure the first run before committing to the whole sweep.
#
# Output : checkpoints/random_lp/rebuttal/K<K>_seed<S>/  (metrics.jsonl, results.json)
# Logs   : logs/random-lp/K<K>_seed<S>.log
# Runs whose results.json already exists are skipped, so re-invoking is safe.
#
# Usage:
#   conda activate ec
#   rebuttal-scripts/run_random_lp_sweep.sh --dry-run
#   rebuttal-scripts/run_random_lp_sweep.sh                 # the 8 default cells
#   rebuttal-scripts/run_random_lp_sweep.sh --only K40
#   SPEC_ALL=1 rebuttal-scripts/run_random_lp_sweep.sh      # also re-run seed 1234 and K=26
#   rebuttal-scripts/run_random_lp_sweep.sh --aggregate-only

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PYTHON="${PYTHON:-python}"
OUT="checkpoints/random_lp/rebuttal"
LOG="logs/random-lp"
ROUNDS="${ROUNDS:-40}"; SAMPLES="${SAMPLES:-2048}"; POOL="${POOL:-2048}"

# K : comma-separated seeds
SPEC=( "15:23,44" "30:23,44" "32:23,44" "40:23,44" )
[[ "${SPEC_ALL:-0}" == "1" ]] && SPEC=( "15:1234,23,44" "26:1234,23,44" "30:1234,23,44" "32:1234,23,44" "40:1234,23,44" )

FILTER=""; DRY_RUN=0; AGG_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)           FILTER="$2"; shift 2 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --aggregate-only) AGG_ONLY=1; shift ;;
    -h|--help)        sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG"
queued=0
for entry in "${SPEC[@]}"; do
  IFS=: read -r K seeds <<< "$entry"
  IFS=, read -ra seed_list <<< "$seeds"
  for s in "${seed_list[@]}"; do
    name="K${K}_seed${s}"
    [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && continue
    if [[ -f "$OUT/$name/results.json" ]]; then
      echo "[skip]  $name (results.json exists)"; continue
    fi
    echo "[queue] $name  (K=$K rounds=$ROUNDS samples=$SAMPLES pool=$POOL seed=$s)"
    queued=$((queued+1))
    [[ $DRY_RUN -eq 1 || $AGG_ONLY -eq 1 ]] && continue
    {
      echo "=== $name | K=$K r=$ROUNDS s=$SAMPLES pool=$POOL seed=$s | $(date -Is) ==="
      PYTHONUNBUFFERED=1 "$PYTHON" scripts/random_lp_baseline.py \
        --K "$K" --seed "$s" --rounds "$ROUNDS" --samples "$SAMPLES" --pool "$POOL" \
        --out-dir "$OUT" --run-name "$name"
      echo "=== exit=$? | $(date -Is) ==="
    } 2>&1 | tee -a "$LOG/$name.log"
  done
done
[[ $DRY_RUN -eq 1 ]] && { echo; echo "$queued run(s) would launch. (dry run)"; exit 0; }

# ---- aggregate: new runs merged with the archived seed-1234 runs ------------
echo; echo "=== Random + LP baseline, best sum of radii ==="
"$PYTHON" - "$OUT" <<'PY'
import json, glob, os, re, sys, statistics as st
from collections import defaultdict

out = sys.argv[1]
g = defaultdict(dict)

# runs produced here
for f in glob.glob(os.path.join(out, "K*_seed*", "results.json")):
    name = os.path.basename(os.path.dirname(f))
    m = re.match(r"K(\d+)_seed(\d+)", name)
    if m:
        g[int(m.group(1))][int(m.group(2))] = json.load(open(f))["summary"]["best_score"]

# archived seed-1234 runs shipped with the paper
for f in glob.glob("../exp-result/packing/main/ddpm/random_baseline/*/results.json"):
    name = os.path.basename(os.path.dirname(f))
    mk, ms = re.search(r"K(\d+)", name), re.search(r"seed(\d+)", name)
    if not (mk and ms):
        continue
    K, s = int(mk.group(1)), int(ms.group(1))
    if s in g[K]:
        continue
    try:
        g[K][s] = json.load(open(f))["summary"]["best_score"]
    except Exception:
        pass

if not g:
    print("  (no results yet)")
for K in sorted(g):
    xs = [g[K][s] for s in sorted(g[K])]
    line = f"  K={K:>2}: " + "  ".join(f"seed{s}={g[K][s]:.4f}" for s in sorted(g[K]))
    if len(xs) > 1:
        line += f"   -> {st.mean(xs):.4f} +/- {st.stdev(xs):.4f}"
    print(line + f"   (k={len(xs)})")
print("\n  paper Table 4 (seed 1234): K15 1.485  K26 1.840  K30 1.984  K32 2.025  K40 2.322")
PY
