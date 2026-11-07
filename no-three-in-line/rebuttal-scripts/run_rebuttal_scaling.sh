#!/usr/bin/env bash
#
# Fixed-budget runner for no-three-in-line (paper Table 9).
#
# Uses a FIXED d256/L6/H8 backbone and FIXED budget (P4096/G48/C4096/T8192)
# across n, toggling ONLY the method: AC + iterative decode (the paper's "Ours",
# Table 3 row 5) vs no-AC full-mask one-shot (Table 3 row 3). ONESHOT=all adds
# the AC + one-shot arm (Table 3 row 4).
#
# n=25 is covered by run_seed_sweep.sh; this runner adds n=20,30.
# Seeds {0, 23, 44}. The _rt23/_rt44 configs are byte-identical to their seed-0
# parent except for patternboost.runtime.{seed, checkpoint_dir}, so every run
# owns a DISTINCT checkpoints/<name> directory and nothing is ever overwritten.
#
# NOTE: n=30 uses the SAME d256 backbone, not the d320 of the scaling table.
#
# Output locations
#   working dir : checkpoints/<run>/exp/   (full artifacts incl. top_pool.pt + *.pt)
#   archive     : ../exp-result/no-three-in-line/rebuttal-scaling/<run>/
#                 (metrics.jsonl + config_snapshot.yaml; via --archive)
#   logs        : logs/rebuttal-scaling/<run>.log
#
# Usage (from anywhere):
#   conda activate ec
#   rebuttal-scripts/run_rebuttal_scaling.sh              # run all, sequential
#   rebuttal-scripts/run_rebuttal_scaling.sh --dry-run
#   rebuttal-scripts/run_rebuttal_scaling.sh --only n20   # substring filter
#   rebuttal-scripts/run_rebuttal_scaling.sh --archive
#   rebuttal-scripts/run_rebuttal_scaling.sh --force      # redo existing dirs
#
# Set PYTHON=... to choose an interpreter (default expects `ec` conda env active).

set -uo pipefail

# cd to the no-three-in-line repo root (parent of this script's folder), so that
# train_patternboost.py and all relative paths resolve exactly as in run_seed_sweep.sh.
CFG_DIR_NAME="rebuttal-exp-config"
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PYTHON="${PYTHON:-python}"
LOG_DIR="logs/rebuttal-scaling"
ARCHIVE_ROOT="../exp-result/no-three-in-line"

# run name : config path : archive group   (name MUST match config checkpoint_dir basename)
# Ordered to complete one full seed across both n before starting the next. Within
# a seed the cheaper n=20 pair runs first, and each arm is immediately followed by
# its partner.
RUNS=(
  # --- seed 0 ---
  "n20_abl_no_ac:${CFG_DIR_NAME}/n20_abl_no_ac.yaml:rebuttal-scaling"
  "n20_main:${CFG_DIR_NAME}/n20_main.yaml:rebuttal-scaling"
  "n30_abl_no_ac:${CFG_DIR_NAME}/n30_abl_no_ac.yaml:rebuttal-scaling"
  "n30_main:${CFG_DIR_NAME}/n30_main.yaml:rebuttal-scaling"
  # --- seed 23 ---
  "n20_abl_no_ac_rt23:${CFG_DIR_NAME}/n20_abl_no_ac_rt23.yaml:rebuttal-scaling"
  "n20_main_rt23:${CFG_DIR_NAME}/n20_main_rt23.yaml:rebuttal-scaling"
  "n30_abl_no_ac_rt23:${CFG_DIR_NAME}/n30_abl_no_ac_rt23.yaml:rebuttal-scaling"
  "n30_main_rt23:${CFG_DIR_NAME}/n30_main_rt23.yaml:rebuttal-scaling"
  # --- seed 44 ---
  "n20_abl_no_ac_rt44:${CFG_DIR_NAME}/n20_abl_no_ac_rt44.yaml:rebuttal-scaling"
  "n20_main_rt44:${CFG_DIR_NAME}/n20_main_rt44.yaml:rebuttal-scaling"
  "n30_abl_no_ac_rt44:${CFG_DIR_NAME}/n30_abl_no_ac_rt44.yaml:rebuttal-scaling"
  "n30_main_rt44:${CFG_DIR_NAME}/n30_main_rt44.yaml:rebuttal-scaling"
)

# ---- Optional arm: AC + one-shot (paper Table 3 Row 4) ----------------------
# Adding this arm turns Delta(n) from "full method vs unconditional generation"
# into a clean 3-row decomposition at each n, matching Table 3:
#   no_ac  (mask 1.0, one-shot)  --(+active core)-->  oneshot (AC, one-shot)
#          --(+iterative decode)-->  main (AC, iterative)
# Each config is byte-identical to its n*_main parent except the decode block
# (num_steps 5->1, quota_ratios [0.2 x5]->[1.0]) and seed/checkpoint_dir, so the
# oneshot-vs-main contrast isolates decoding and oneshot-vs-no_ac isolates AC.
# SEED 0 ALONE is enough to close the isolation gap, so the flag is graded:
#   ONESHOT=1    -> seed 0 only        (2 runs, recommended)
#   ONESHOT=all  -> seeds 0, 23, 44    (6 runs)
# (a substring --only filter cannot separate seed 0 from the rt variants, hence the flag)
if [[ "${ONESHOT:-0}" != "0" ]]; then
  RUNS+=(
    "n20_abl_oneshot:${CFG_DIR_NAME}/n20_abl_oneshot.yaml:rebuttal-scaling"
    "n30_abl_oneshot:${CFG_DIR_NAME}/n30_abl_oneshot.yaml:rebuttal-scaling"
  )
  if [[ "${ONESHOT}" == "all" ]]; then
    RUNS+=(
      "n20_abl_oneshot_rt23:${CFG_DIR_NAME}/n20_abl_oneshot_rt23.yaml:rebuttal-scaling"
      "n30_abl_oneshot_rt23:${CFG_DIR_NAME}/n30_abl_oneshot_rt23.yaml:rebuttal-scaling"
      "n20_abl_oneshot_rt44:${CFG_DIR_NAME}/n20_abl_oneshot_rt44.yaml:rebuttal-scaling"
      "n30_abl_oneshot_rt44:${CFG_DIR_NAME}/n30_abl_oneshot_rt44.yaml:rebuttal-scaling"
    )
  fi
fi

FILTER=""; DRY_RUN=0; FORCE=0; ARCHIVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)    FILTER="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force)   FORCE=1; shift ;;
    --archive) ARCHIVE=1; shift ;;
    -h|--help) sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

field() { echo "$1" | cut -d: -f"$2"; }
run_dir_of() { echo "checkpoints/$1/exp"; }
is_complete() {
  local metrics="$(run_dir_of "$1")/metrics.jsonl"
  [[ -f "$metrics" ]] && grep -q '"event": "run_complete"' "$metrics"
}
classify() {
  local name="$1" dir; dir="$(run_dir_of "$name")"
  if is_complete "$name"; then echo "skip"
  elif [[ -d "$dir" && $FORCE -eq 0 ]]; then echo "stale"
  else echo "run"; fi
}
archive_run() {
  local name="$1" group="$2" src dst
  src="$(run_dir_of "$name")"; dst="$ARCHIVE_ROOT/$group/$name"
  mkdir -p "$dst"
  cp "$src/metrics.jsonl" "$dst/metrics.jsonl"
  [[ -f "$src/config_snapshot.yaml" ]] && cp "$src/config_snapshot.yaml" "$dst/config_snapshot.yaml"
  echo "[archive] $name -> ${dst#../}"
}
launch() {
  local name="$1" config="$2" dir log
  dir="$(run_dir_of "$name")"; log="$LOG_DIR/$name.log"
  [[ -d "$dir" ]] && rm -rf "$dir"    # train_patternboost.py refuses an existing run dir
  echo "[start] $name -> $log"
  {
    echo "=== $name | config=$config | $(date -Is) ==="
    PYTHONUNBUFFERED=1 "$PYTHON" train_patternboost.py --config "$config"
    echo "=== exit=$? | $(date -Is) ==="
  } 2>&1 | tee -a "$log"
  if is_complete "$name"; then echo "[done]  $name"; else echo "[FAIL]  $name -- see $log"; fi
}

# ---- plan ----
PENDING=()
for entry in "${RUNS[@]}"; do
  name="$(field "$entry" 1)"; config="$(field "$entry" 2)"
  [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && continue
  if [[ ! -f "$config" ]]; then echo "[MISS]  $name (config not found: $config)"; continue; fi
  case "$(classify "$name")" in
    skip)  echo "[skip]  $name (already complete)" ;;
    stale) echo "[stale] $name (incomplete dir $(run_dir_of "$name"); rerun with --force)" ;;
    run)   PENDING+=("$entry"); echo "[queue] $name <- $config" ;;
  esac
done

if [[ ${#PENDING[@]} -gt 0 ]]; then
  echo; echo "${#PENDING[@]} run(s) queued, sequential on one GPU."
  echo "Ref timing (RTX 5090): n<20 ~2-5h, n=20 ~12-15h, n=30 ~18-24h per run."
  echo "Measure the first (n15) run before committing to n20/n30."
fi
[[ $DRY_RUN -eq 1 ]] && { echo "(dry run: nothing launched)"; exit 0; }

# ---- execute ----
if [[ ${#PENDING[@]} -gt 0 ]]; then
  mkdir -p "$LOG_DIR"
  for entry in "${PENDING[@]}"; do launch "$(field "$entry" 1)" "$(field "$entry" 2)"; done
fi

# ---- summary + archive ----
echo; echo "=== summary ==="
for entry in "${RUNS[@]}"; do
  name="$(field "$entry" 1)"; group="$(field "$entry" 3)"
  [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && continue
  if is_complete "$name"; then
    best=$(grep '"event": "run_complete"' "$(run_dir_of "$name")/metrics.jsonl" \
           | sed -n 's/.*"best_score": \([0-9]*\).*/\1/p' | tail -1)
    init=$(grep '"event": "run_complete"' "$(run_dir_of "$name")/metrics.jsonl" \
           | sed -n 's/.*"initial_best_score": \([0-9]*\).*/\1/p' | tail -1)
    printf '  %-22s complete  init=%s best=%s\n' "$name" "${init:-?}" "${best:-?}"
    [[ $ARCHIVE -eq 1 ]] && archive_run "$name" "$group"
  else
    printf '  %-22s INCOMPLETE\n' "$name"
  fi
done
