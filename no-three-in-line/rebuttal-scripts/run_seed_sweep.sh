#!/usr/bin/env bash
#
# Seed sweep for no-three-in-line (single GPU, sequential).
#
# Runs the three arms of the n=25 ablation (main / one-shot / no-active-core
# remask) under PatternBoost runtime seeds 44 and 23, which together with seed 0
# give each arm three seeds {0, 44, 23}.
#
# Output locations
#   working dir : checkpoints/<run>/exp/            (gitignored, full artifacts:
#                 metrics.jsonl, config_snapshot.yaml, top_pool.pt, *.pt)
#   archive     : ../exp-result/no-three-in-line/<group>/<run>/
#                 (metrics.jsonl + config_snapshot.yaml only, matching the
#                  layout of the runs already in exp-result/; written by
#                  --archive, or later by `rebuttal-scripts/run_seed_sweep.sh --archive-only`)
#   logs        : logs/seed-sweep/<run>.log
#
# Usage:
#   conda activate ec
#   rebuttal-scripts/run_seed_sweep.sh                      # run everything, sequentially
#   rebuttal-scripts/run_seed_sweep.sh --archive            # ... and copy results to exp-result/
#   rebuttal-scripts/run_seed_sweep.sh --only n25_abl       # substring filter on run name
#   rebuttal-scripts/run_seed_sweep.sh --dry-run
#   rebuttal-scripts/run_seed_sweep.sh --force              # discard existing run dirs and redo
#   rebuttal-scripts/run_seed_sweep.sh --archive-only       # archive finished runs, launch nothing
#
# Set PYTHON=... to pick an interpreter; the default `python` expects the conda
# environment to be active already.
#
# Completed runs are skipped, so the script is safe to re-invoke after an
# interruption. Runs whose directory exists but never reached "run_complete"
# are reported and left alone unless --force is given.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PYTHON="${PYTHON:-python}"
LOG_DIR="logs/seed-sweep"
ARCHIVE_ROOT="../exp-result/no-three-in-line"

# run name : config path : archive group
RUNS=(
  "n25_main_stage1_rt44:exp-config/max/n25_main_stage1_rt44.yaml:max"
  "n25_main_stage1_rt23:exp-config/max/n25_main_stage1_rt23.yaml:max"
  "n25_abl_oneshot_rt44:exp-config/ablation-oneshot/n25_abl_oneshot_rt44.yaml:ablation-oneshot"
  "n25_abl_oneshot_rt23:exp-config/ablation-oneshot/n25_abl_oneshot_rt23.yaml:ablation-oneshot"
  "n25_abl_no_ac_rt44:exp-config/ablation-remask/n25_abl_no_ac_rt44.yaml:ablation-remask"
  "n25_abl_no_ac_rt23:exp-config/ablation-remask/n25_abl_no_ac_rt23.yaml:ablation-remask"
)

FILTER=""
DRY_RUN=0
FORCE=0
ARCHIVE=0
ARCHIVE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)         FILTER="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --force)        FORCE=1; shift ;;
    --archive)      ARCHIVE=1; shift ;;
    --archive-only) ARCHIVE=1; ARCHIVE_ONLY=1; shift ;;
    -h|--help)      sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

field() { echo "$1" | cut -d: -f"$2"; }

# The run directory is patternboost.runtime.checkpoint_dir / run_name, and the
# generated configs use checkpoint_dir=checkpoints/<name> with run_name=exp.
run_dir_of() { echo "checkpoints/$1/exp"; }

is_complete() {
  local metrics="$(run_dir_of "$1")/metrics.jsonl"
  [[ -f "$metrics" ]] && grep -q '"event": "run_complete"' "$metrics"
}

archive_run() {
  local name="$1" group="$2" src dst
  src="$(run_dir_of "$name")"
  dst="$ARCHIVE_ROOT/$group/$name"
  mkdir -p "$dst"
  cp "$src/metrics.jsonl" "$dst/metrics.jsonl"
  [[ -f "$src/config_snapshot.yaml" ]] && cp "$src/config_snapshot.yaml" "$dst/config_snapshot.yaml"
  echo "[archive] $name -> ${dst#../}"
}

# Decide what to do with each run; prints "skip" or "run".
# Any existing run directory (complete OR partial) is skipped unless --force, so
# re-invoking resumes ONLY the runs that have no directory yet.
classify() {
  local dir; dir="$(run_dir_of "$1")"
  if [[ -d "$dir" && $FORCE -eq 0 ]]; then echo "skip"; else echo "run"; fi
}

launch() {
  local name="$1" config="$2" dir log
  dir="$(run_dir_of "$name")"
  log="$LOG_DIR/$name.log"

  # train_patternboost.py refuses to reuse an existing run directory.
  [[ -d "$dir" ]] && rm -rf "$dir"

  echo "[start] $name -> $log"
  # PYTHONUNBUFFERED=1 forces line-by-line output through the tee pipe (stdout
  # is block-buffered once it is not a terminal); tee mirrors it to terminal
  # and log. is_complete below reads the metrics file, not the pipe exit code,
  # so pipefail masking python's exit status here is harmless.
  {
    echo "=== $name | config=$config | $(date -Is) ==="
    PYTHONUNBUFFERED=1 "$PYTHON" train_patternboost.py --config "$config"
    echo "=== exit=$? | $(date -Is) ==="
  } 2>&1 | tee -a "$log"

  if is_complete "$name"; then
    echo "[done]  $name"
  else
    echo "[FAIL]  $name -- see $log"
  fi
}

# ---- plan -------------------------------------------------------------------

PENDING=()
for entry in "${RUNS[@]}"; do
  name="$(field "$entry" 1)"
  config="$(field "$entry" 2)"
  [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && continue
  if [[ ! -f "$config" ]]; then
    echo "[MISS]  $name (config not found: $config)"
    continue
  fi
  case "$(classify "$name")" in
    skip)  if is_complete "$name"; then
             echo "[skip]  $name (dir exists, complete)"
           else
             echo "[skip]  $name (dir exists, PARTIAL -- kept as-is; use --force to redo)"
           fi ;;
    run)   [[ $ARCHIVE_ONLY -eq 0 ]] && PENDING+=("$entry") && echo "[queue] $name <- $config" ;;
  esac
done

if [[ ${#PENDING[@]} -gt 0 ]]; then
  echo
  echo "${#PENDING[@]} run(s) queued, sequential on one GPU."
  echo "Reference timing on an RTX 5090 was 8-10 h per run; measure the first"
  echo "run on this machine before committing to the whole sweep."
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "(dry run: nothing launched)"
  exit 0
fi

# ---- execute ----------------------------------------------------------------

if [[ ${#PENDING[@]} -gt 0 ]]; then
  mkdir -p "$LOG_DIR"
  for entry in "${PENDING[@]}"; do
    launch "$(field "$entry" 1)" "$(field "$entry" 2)"
  done
fi

# ---- archive + summary ------------------------------------------------------

echo
echo "=== summary ==="
for entry in "${RUNS[@]}"; do
  name="$(field "$entry" 1)"
  group="$(field "$entry" 3)"
  [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && continue
  if is_complete "$name"; then
    best=$(grep '"event": "run_complete"' "$(run_dir_of "$name")/metrics.jsonl" \
           | sed -n 's/.*"best_score": \([0-9]*\).*/\1/p' | tail -1)
    init=$(grep '"event": "run_complete"' "$(run_dir_of "$name")/metrics.jsonl" \
           | sed -n 's/.*"initial_best_score": \([0-9]*\).*/\1/p' | tail -1)
    printf '  %-24s complete  init=%s best=%s\n' "$name" "${init:-?}" "${best:-?}"
    [[ $ARCHIVE -eq 1 ]] && archive_run "$name" "$group"
  else
    printf '  %-24s INCOMPLETE\n' "$name"
  fi
done
