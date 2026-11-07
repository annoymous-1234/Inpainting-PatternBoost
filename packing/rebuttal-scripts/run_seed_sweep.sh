#!/usr/bin/env bash
#
# Seed sweep for circle packing (single GPU, sequential).
#
# Runs the K=15/30/32/40 main configurations (both the FM and DDPM heads) under
# seeds 44 and 23. Together with seed 1234, every FM-vs-DDPM comparison in the
# main table rests on three seeds {1234, 44, 23}.
#
# Output locations
#   working dir : checkpoints/<run>/exp/            (gitignored, full artifacts:
#                 metrics.jsonl, generations.jsonl, results.json,
#                 run_manifest.json, config_snapshot.yaml, resolved_config.yaml,
#                 top_pool.pt)
#   archive     : ../exp-result/packing/main/<head>/<run>/
#                 (the six paper artifacts, matching the layout of the runs
#                  already in exp-result/; written by --archive, or later by
#                  `rebuttal-scripts/run_seed_sweep.sh --archive-only`)
#   logs        : logs/seed-sweep/<run>.log
#
# Usage:
#   conda activate ec
#   rebuttal-scripts/run_seed_sweep.sh                      # run everything, sequentially
#   rebuttal-scripts/run_seed_sweep.sh --archive            # ... and copy results to exp-result/
#   rebuttal-scripts/run_seed_sweep.sh --only K32           # substring filter on run name
#   rebuttal-scripts/run_seed_sweep.sh --dry-run
#   rebuttal-scripts/run_seed_sweep.sh --force              # discard existing run dirs and redo
#   rebuttal-scripts/run_seed_sweep.sh --archive-only       # archive finished runs, launch nothing
#   rebuttal-scripts/run_seed_sweep.sh --device cpu
#
# Set PYTHON=... to pick an interpreter; the default `python` expects the conda
# environment to be active already.
#
# Completed runs are skipped, so the script is safe to re-invoke after an
# interruption. Runs whose directory exists but never reached status "ok" are
# reported and left alone unless --force is given.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PYTHON="${PYTHON:-python}"
LOG_DIR="logs/seed-sweep"
ENTRYPOINT="scripts/run_patternboost_experiment.py"
ARCHIVE_ROOT="../exp-result/packing/main"
ARCHIVE_FILES=(
  metrics.jsonl generations.jsonl results.json
  run_manifest.json config_snapshot.yaml resolved_config.yaml
)

# run name : config path : archive group (the supervision head)
RUNS=()
for K in 15 30 32 40; do
  for HEAD in fm ddpm; do
    for SEED in 44 23; do
      RUNS+=("K${K}_${HEAD}_main_rt${SEED}:exp-config/main/${HEAD}/K${K}_${HEAD}_main_rt${SEED}.yaml:${HEAD}")
    done
  done
done

FILTER=""
DEVICE="cuda"
DRY_RUN=0
FORCE=0
ARCHIVE=0
ARCHIVE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)         FILTER="$2"; shift 2 ;;
    --device)       DEVICE="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --force)        FORCE=1; shift ;;
    --archive)      ARCHIVE=1; shift ;;
    --archive-only) ARCHIVE=1; ARCHIVE_ONLY=1; shift ;;
    -h|--help)      sed -n '2,35p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

field() { echo "$1" | cut -d: -f"$2"; }

# run_patternboost_experiment.py writes to "checkpoints"/<run_name> (the entrypoint
# hardcodes checkpoint_dir="checkpoints"; see resolve_experiment_config). Each config's
# run_name is now UNIQUE (= its filename stem), so runs no longer collide in checkpoints/exp.
run_dir_of() { echo "checkpoints/$1"; }

# Completion is judged from the run LOG (status=ok), NOT the checkpoint dir: the old
# sweep overwrote every run into checkpoints/exp, so only the logs are a reliable
# per-run record. New runs also print status=ok, so this stays correct going forward.
is_complete() {
  local log="$LOG_DIR/$1.log"
  [[ -f "$log" ]] && grep -q 'status=ok' "$log"
}

archive_run() {
  local name="$1" group="$2" src dst f
  src="$(run_dir_of "$name")"
  dst="$ARCHIVE_ROOT/$group/$name"
  mkdir -p "$dst"
  for f in "${ARCHIVE_FILES[@]}"; do
    [[ -f "$src/$f" ]] && cp "$src/$f" "$dst/$f"
  done
  echo "[archive] $name -> ${dst#../}"
}

# Decide what to do with each run; prints "skip" or "run".
# Skip iff the run's log shows a completed run (status=ok), unless --force.
classify() {
  if [[ $FORCE -eq 0 ]] && is_complete "$1"; then echo "skip"; else echo "run"; fi
}

launch() {
  local name="$1" config="$2" log
  log="$LOG_DIR/$name.log"

  echo "[start] $name -> $log"
  # PYTHONUNBUFFERED=1 forces line-by-line output through the tee pipe (stdout
  # is block-buffered once it is not a terminal, and packing's per-generation
  # print has no flush=True); tee mirrors it to terminal and log. is_complete
  # below reads results.json, not the pipe exit code, so pipefail masking
  # python's exit status here is harmless.
  {
    echo "=== $name | config=$config | $(date -Is) ==="
    # --force lets the entrypoint clear a partial directory itself.
    PYTHONUNBUFFERED=1 "$PYTHON" "$ENTRYPOINT" --config "$config" --device "$DEVICE" --force
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
    skip)  echo "[skip]  $name (log shows complete)" ;;
    run)   [[ $ARCHIVE_ONLY -eq 0 ]] && PENDING+=("$entry") && echo "[queue] $name <- $config" ;;
  esac
done

if [[ ${#PENDING[@]} -gt 0 ]]; then
  echo
  echo "${#PENDING[@]} run(s) queued, sequential on one GPU."
  echo "Reference timing on an RTX 5090 was 35-60 min per run; measure the first"
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
    # best from the log's last pool_score_best (results.json may be absent for the
    # pre-fix runs that were overwritten in checkpoints/exp).
    best=$(grep -oE 'pool_score_best=[0-9.]+' "$LOG_DIR/$name.log" | tail -1 | cut -d= -f2)
    printf '  %-24s complete  best=%s\n' "$name" "${best:-?}"
    [[ $ARCHIVE -eq 1 ]] && archive_run "$name" "$group"
  else
    printf '  %-24s INCOMPLETE\n' "$name"
  fi
done
