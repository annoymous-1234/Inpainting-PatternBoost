#!/usr/bin/env bash
#
# Random-completion-operator ablation for circle packing.
#
# Keeps the whole PatternBoost outer loop -- top pool, active-core edit sets,
# commit schedule, LP scoring -- and swaps ONLY the completion operator: masked
# centers are moved by isotropic Gaussian noise instead of by the DDPM model.
# Nothing is trained and no network is built, so these runs are CPU-only and
# much faster than the learned runs they are compared against.
#
# The comparison arm is the published ITERATIVE K=26 DDPM main run
# (Best = 2.370 +/- .043 over seeds 1234/44/23), already in exp-result/.
#
# Sigma is chosen so the mean masked-center displacement, sigma*sqrt(pi/2),
# brackets the whole range the learned model ever occupies:
#   sigma=0.04 -> 0.050  (learned: 0.038 final generation, 0.045 whole-run mean)
#   sigma=0.10 -> 0.125  (learned: 0.11-0.13 at generation 1)
# Each run reports its own inference_center_distance_mean, so the match can be
# checked rather than assumed.
#
# Output locations
#   working dir : checkpoints/<run>/                (gitignored, full artifacts)
#   archive     : ../exp-result/packing/rebuttal-random-op/<run>/
#   logs        : logs/random-operator/<run>.log
#
# Usage:
#   rebuttal-scripts/run_random_operator.sh                 # run everything, sequentially
#   rebuttal-scripts/run_random_operator.sh --archive       # ... and copy results to exp-result/
#   rebuttal-scripts/run_random_operator.sh --only s04      # substring filter on run name
#   rebuttal-scripts/run_random_operator.sh --dry-run
#   rebuttal-scripts/run_random_operator.sh --force         # discard existing run dirs and redo
#   rebuttal-scripts/run_random_operator.sh --archive-only  # archive finished runs, launch nothing
#
# Set PYTHON=... to pick an interpreter; the default `python` expects the conda
# environment to be active already.
#
# Completed runs are skipped, so the script is safe to re-invoke after an
# interruption.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PYTHON="${PYTHON:-python}"
LOG_DIR="logs/random-operator"
ENTRYPOINT="scripts/run_patternboost_experiment.py"
ARCHIVE_ROOT="../exp-result/packing/rebuttal-random-op"
ARCHIVE_FILES=(
  metrics.jsonl generations.jsonl results.json
  run_manifest.json config_snapshot.yaml resolved_config.yaml
)

# run name : config path : archive group (the supervision head)
#
# DDPM sigma=0.04 is the displacement-matched arm and sigma=0.10 the
# generation-1 arm; both are single-seed scouts. The sigma=0.05/0.06 grid is run
# under two seeds on both heads, so the arm that ends up in the table carries an
# error bar. See exp-config/rebuttal-random-op/_generate_configs.py for how each
# sigma maps to a parent-to-child displacement.
RUNS=()
for SEED in 1234; do
  for TAG in s04 s10; do
    RUNS+=("K26_ddpm_randperturb_${TAG}_rt${SEED}:exp-config/rebuttal-random-op/ddpm/K26_ddpm_randperturb_${TAG}_rt${SEED}.yaml:ddpm")
  done
done
for HEAD in ddpm fm; do
  for TAG in s05 s06; do
    for SEED in 1234 44; do
      RUNS+=("K26_${HEAD}_randperturb_${TAG}_rt${SEED}:exp-config/rebuttal-random-op/${HEAD}/K26_${HEAD}_randperturb_${TAG}_rt${SEED}.yaml:${HEAD}")
    done
  done
done

FILTER=""
DEVICE="cpu"
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
    -h|--help)      sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
  echo "${#PENDING[@]} run(s) queued, sequential on CPU."
  echo "These runs neither train nor run inference, so they should finish well"
  echo "inside the learned runs' budget; time the first one before queueing more."
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
