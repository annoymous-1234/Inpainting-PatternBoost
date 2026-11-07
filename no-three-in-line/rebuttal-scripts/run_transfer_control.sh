#!/usr/bin/env bash
#
# Paired transfer control for no-three-in-line (paper Table 10).
#
# The model is retrained every generation on the pool the parents come from, so
# a gain over random completion cannot separate "learned transferable structure"
# from "memorised this pool". This adds an arm whose model never saw that pool.
#
#   parents  : checkpoints/n25_main_batch4/exp, snapshots of generations 32/48/64
#              (seed 0; the only n=25 run with per-generation pool snapshots).
#              Three well-separated generations already span the range of parent
#              quality; the remaining snapshots mostly re-contribute constructions
#              these three already hold.
#              PARENT_GENS=all uses every snapshot plus the final pool.
#   home     : that run's own generation-64 checkpoint
#   transfer : the generation-64 checkpoint of n25_main_stage1_rt44
#              (seed 44 -> a different pool, same architecture and budget).
#              Add _rt23 as a second donor through DONORS.
#   random   : completion_operator=random, the same arm as the random-operator
#              ablation
#
# Reading:
#   transfer - random > 0   structure a fresh model applies to constructions it
#                           never trained on; memorisation cannot produce this
#   home - transfer   > 0   the premium from having trained on this pool
#   transfer ~ random       the gain over random was memorisation
#
# One GPU. Cost is one forward per decode step per model arm; with one donor the
# three arms come to roughly 2.4 model-arm equivalents (the random arm loads no
# model and runs about twice as fast). Start with PARENTS=512 to time it.
#
# Output : checkpoints/n25_main_batch4/exp/analysis_transfer/
#            transfer_control.json, transfer_control_pairs.csv
# Logs   : logs/transfer-control/n25_main_batch4.log
#
# Usage:
#   conda activate ec
#   rebuttal-scripts/run_transfer_control.sh --dry-run
#   rebuttal-scripts/run_transfer_control.sh
#   PARENTS=512 rebuttal-scripts/run_transfer_control.sh     # timing probe
#   PARENT_GENS=all rebuttal-scripts/run_transfer_control.sh # every snapshot
#   DONORS="n25_main_stage1_rt23 n25_main_stage1_rt44" \
#     rebuttal-scripts/run_transfer_control.sh               # second donor
#   REPEATS=4 rebuttal-scripts/run_transfer_control.sh       # 4 masks per parent
#   POOL_RUN=checkpoints/n30_main/exp DONORS=n30_main_rt23 GEN=48 \
#     PARENT_GENS=all rebuttal-scripts/run_transfer_control.sh   # n=30 (no snapshots)

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PYTHON="${PYTHON:-python}"
POOL_RUN="${POOL_RUN:-checkpoints/n25_main_batch4/exp}"
DONORS="${DONORS:-n25_main_stage1_rt44}"
GEN="${GEN:-64}"                 # which generation's checkpoint to use as the model
PARENT_GENS="${PARENT_GENS:-32 48 64}"   # pool snapshots to draw parents from; 'all' for every one
PARENTS="${PARENTS:-8192}"
REPEATS="${REPEATS:-1}"
BATCH="${BATCH:-64}"
SEED="${SEED:-0}"
PARENT_SOURCE="${PARENT_SOURCE:-union}"
LOG_DIR="logs/transfer-control"
TAG="${TAG:-$(basename "$(dirname "$POOL_RUN")")}"

DRY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

ckpt_of() { echo "checkpoints/$1/exp/patternboost_gen_${GEN}_epoch_1.pt"; }

# ---- preflight: every file the run needs, named before anything launches ----
missing=0
for f in "$POOL_RUN/config_snapshot.yaml" "$POOL_RUN/top_pool.pt"; do
  [[ -f "$f" ]] || { echo "[missing] $f"; missing=1; }
done
HOME_CKPT="$POOL_RUN/patternboost_gen_${GEN}_epoch_1.pt"
[[ -f "$HOME_CKPT" ]] || { echo "[missing] $HOME_CKPT (home arm)"; missing=1; }

TRANSFER_ARGS=()
for donor in $DONORS; do
  path="$(ckpt_of "$donor")"
  if [[ -f "$path" ]]; then
    TRANSFER_ARGS+=(--transfer "${donor}=${path}")
  else
    echo "[missing] $path (transfer donor $donor)"; missing=1
  fi
done

# ---- which pool snapshots supply the parents -------------------------------
GEN_ARGS=()
if [[ "$PARENT_GENS" == "all" || -z "$PARENT_GENS" ]]; then
  parent_pools="every snapshot + top_pool.pt"
else
  GEN_ARGS=(--parent-generations $PARENT_GENS)
  pools=()
  for g in $PARENT_GENS; do
    path="$POOL_RUN/$(printf 'top_pool_generation_%03d.pt' "$g")"
    pools+=("$(basename "$path")")
    [[ -f "$path" ]] || { echo "[missing] $path (parent pool, generation $g)"; missing=1; }
  done
  parent_pools="${pools[*]}"
fi

snapshots=$(ls -1 "$POOL_RUN"/top_pool_generation_*.pt 2>/dev/null | wc -l)
echo "pool run       $POOL_RUN  (${snapshots} per-generation snapshot(s) on disk)"
echo "parent pools   $parent_pools"
echo "home           $HOME_CKPT"
echo "transfer       ${TRANSFER_ARGS[*]:-none}"
echo "parents        $PARENTS x $REPEATS  (source=$PARENT_SOURCE)  batch=$BATCH  seed=$SEED"

if [[ $missing -ne 0 ]]; then
  echo
  echo "Refusing to launch: the files above are gitignored (**/*.pt), so they exist"
  echo "only where the run was executed. Run this on that machine, or adjust GEN/DONORS."
  exit 1
fi
[[ $snapshots -eq 0 ]] && echo "note: no pool snapshots on disk; parents come from top_pool.pt alone" \
  && echo "      and by_parent_generation collapses to a single bucket (set PARENT_GENS=all)"

if [[ $DRY -eq 1 ]]; then echo; echo "(dry run)"; exit 0; fi

mkdir -p "$LOG_DIR"
{
  echo "=== transfer control | $TAG | $(date -Is) ==="
  PYTHONUNBUFFERED=1 "$PYTHON" analysis/transfer_control.py \
    --pool-run "$POOL_RUN" \
    --home "$HOME_CKPT" \
    "${TRANSFER_ARGS[@]}" \
    --num-parents "$PARENTS" \
    --repeats "$REPEATS" \
    --parent-source "$PARENT_SOURCE" \
    ${GEN_ARGS[@]+"${GEN_ARGS[@]}"} \
    --batch-size "$BATCH" \
    --seed "$SEED"
  echo "=== exit=$? | $(date -Is) ==="
} 2>&1 | tee -a "$LOG_DIR/$TAG.log"
