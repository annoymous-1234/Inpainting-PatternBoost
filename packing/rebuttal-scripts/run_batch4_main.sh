#!/usr/bin/env bash
#
# K=26 DDPM main (Ours) rerun with candidate provenance logging.
#
# The config is exp-config/main/ddpm/K26_ddpm_main_rt1234.yaml (the paper's
# Table 4 K=26 DDPM row, seed 1234); only the run location differs, and --dry-run
# prints the diff. The transfer analysis uses the pool snapshots and the
# per-candidate child-minus-parent delta logged in candidates.jsonl.
#
# WHAT IT GUARANTEES
#   1. log_candidates_jsonl / log_jsonl are on            (checked before launch)
#   2. the installed code actually injects seed_score into
#      candidate metadata at all three call sites          (checked before launch)
#   3. per-generation pool snapshots are on, so the
#      control arm can pair against the pool a candidate
#      was really drawn from, not just the final pool     (checked before launch)
#   4. the first generation's log is validated on disk and
#      the run is aborted immediately if a field is missing
#
# Output
#   run dir : checkpoints/K26_ddpm_main_batch4/
#             metrics.jsonl, candidates.jsonl, config_snapshot.yaml,
#             results.json, run_manifest.json, top_pool.pt,
#             top_pool_generation_XXX.pt
#   log     : logs/batch4/K26_ddpm_main_batch4.log
#
# Usage
#   conda activate ec
#   rebuttal-scripts/run_batch4_main.sh                # preflight, then run
#   rebuttal-scripts/run_batch4_main.sh --dry-run      # preflight only
#   rebuttal-scripts/run_batch4_main.sh --force        # discard an existing run dir
#   rebuttal-scripts/run_batch4_main.sh --device cpu
#
# Set PYTHON=... to pick an interpreter; the default `python` expects the conda
# environment to be active already.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PYTHON="${PYTHON:-python}"
RUN_NAME="K26_ddpm_main_batch4"
CONFIG="exp-config/rebuttal-batch4/K26_ddpm_main_batch4.yaml"
PARENT_CONFIG="exp-config/main/ddpm/K26_ddpm_main_rt1234.yaml"
ENTRYPOINT="scripts/run_patternboost_experiment.py"
# The entrypoint forces checkpoint_dir to "checkpoints" and uses run_name as the
# subdirectory, so the run lands here regardless of the config's checkpoint_dir.
RUN_DIR="checkpoints/${RUN_NAME}"
LOG_DIR="logs/batch4"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

DEVICE=""
DRY_RUN=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force)   FORCE=1; shift ;;
    --device)  DEVICE="$2"; shift 2 ;;
    -h|--help) sed -n '2,35p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }

echo "=== preflight ==="

[[ -f "$CONFIG" ]] || fail "config not found: $CONFIG"

# 1. seed_score is injected at three sites: the serial generation path, the
#    batched path, and the batch fallback. Missing any one silently produces
#    candidates the analysis has to discard.
grep -q '"seed_score": float(seed_entry.score)' patternboost/runner.py \
  || fail "patternboost/runner.py does not inject seed_score in the serial path"
grep -q 'metadata\["seed_score"\] = float(item.seed_entry.score)' patternboost/runner.py \
  || fail "patternboost/runner.py does not inject seed_score in the batch-fallback path"
grep -q '"seed_score": float(item.seed_entry.score)' patternboost/generator.py \
  || fail "patternboost/generator.py does not inject seed_score in the batched path"
grep -q '_save_top_pool_snapshot' patternboost/runner.py \
  || fail "patternboost/runner.py has no per-generation pool snapshot support"
echo "  code       : seed_score injected at all 3 sites, pool snapshots present"

# 2. the config must switch them on, and must not collide with an existing run.
"$PYTHON" - "$CONFIG" "$PARENT_CONFIG" <<'PYEOF' || exit 1
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
parent = yaml.safe_load(open(sys.argv[2]))
rt = cfg["patternboost"]["runtime"]
problems = []
if not rt.get("log_jsonl"):
    problems.append("patternboost.runtime.log_jsonl must be true")
if not rt.get("log_candidates_jsonl"):
    problems.append("patternboost.runtime.log_candidates_jsonl must be true")
if int(rt.get("top_pool_snapshot_every_generations", 0)) <= 0:
    problems.append("patternboost.runtime.top_pool_snapshot_every_generations must be > 0")
if rt.get("run_name") == parent["patternboost"]["runtime"].get("run_name"):
    problems.append("run_name collides with the original run; the entrypoint would overwrite it")

# Everything except the run location must match the paper's config, otherwise
# this is a different experiment.
allowed = {"checkpoint_dir", "run_name"}
def flatten(node, prefix=""):
    out = {}
    for key, value in (node or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{path}."))
        else:
            out[path] = value
    return out
left, right = flatten(parent), flatten(cfg)
drift = sorted(
    key for key in set(left) | set(right)
    if left.get(key) != right.get(key) and key.split(".")[-1] not in allowed
)
if drift:
    problems.append(f"config drifted from {sys.argv[2]} on: {drift}")

srp = cfg["patternboost"]["polish"]["srp_enabled"]
print(f"  config     : K={cfg['problem']['K']} seed={rt['seed']} "
      f"gens={cfg['patternboost']['generation']['num_generations']} "
      f"cands/gen={cfg['patternboost']['generation']['candidates_per_generation']} "
      f"srp={srp} snapshot_every={rt['top_pool_snapshot_every_generations']}")
print(f"  run dir    : checkpoints/{rt['run_name']}")
if srp:
    print("  NOTE       : SRP is on, so the analysis must be run with --use-final-score")
if problems:
    for problem in problems:
        print(f"PREFLIGHT FAILED: {problem}", file=sys.stderr)
    sys.exit(1)
print("  config     : logging + snapshots enabled, matches the paper config")
PYEOF

if [[ -d "$RUN_DIR" ]]; then
  if [[ "$FORCE" -eq 1 ]]; then
    echo "  run dir    : removing existing $RUN_DIR (--force)"
    [[ "$DRY_RUN" -eq 1 ]] || rm -rf "$RUN_DIR"
  else
    fail "$RUN_DIR already exists; pass --force to discard it"
  fi
fi

CMD=("$PYTHON" "$ENTRYPOINT" --config "$CONFIG")
[[ -n "$DEVICE" ]] && CMD+=(--device "$DEVICE")

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "=== diff vs the paper config ($PARENT_CONFIG) ==="
  diff "$PARENT_CONFIG" "$CONFIG"
  echo
  echo "dry run: would execute"
  echo "  ${CMD[*]}"
  exit 0
fi

mkdir -p "$LOG_DIR"
echo
echo "=== launching $RUN_NAME (log: $LOG_FILE) ==="
"${CMD[@]}" > "$LOG_FILE" 2>&1 &
RUN_PID=$!
echo "  pid        : $RUN_PID"

# 3. Validate generation 1 on disk, then get out of the way.
echo "  validating : waiting for the first candidates.jsonl record ..."
CANDIDATES="${RUN_DIR}/candidates.jsonl"
for _ in $(seq 1 120); do
  kill -0 "$RUN_PID" 2>/dev/null || break
  [[ -s "$CANDIDATES" ]] && break
  sleep 15
done

if [[ ! -s "$CANDIDATES" ]]; then
  if kill -0 "$RUN_PID" 2>/dev/null; then
    kill "$RUN_PID" 2>/dev/null
    fail "no candidates.jsonl after 30 minutes; killed pid $RUN_PID (see $LOG_FILE)"
  fi
  fail "run exited before writing candidates.jsonl (see $LOG_FILE)"
fi

"$PYTHON" - "$CANDIDATES" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as handle:
    record = json.loads(handle.readline())
metadata = record.get("metadata") or {}
missing = [field for field in ("raw_score", "legal", "full_active") if field not in record]
missing += [f"metadata.{field}" for field in ("seed_hash", "seed_score", "edit_ratio", "edit_policy")
            if metadata.get(field) is None]
print(f"  validating : record {sorted(record)}")
print(f"  validating : metadata {sorted(metadata)}")
if missing:
    print(f"PREFLIGHT FAILED: candidates.jsonl is missing {missing}", file=sys.stderr)
    sys.exit(1)
print(f"  validating : ok (seed_score={metadata['seed_score']:.6f} "
      f"raw_score={record['raw_score']} edit_ratio={metadata['edit_ratio']})")
PYEOF

if [[ $? -ne 0 ]]; then
  kill "$RUN_PID" 2>/dev/null
  fail "candidate log schema is wrong; killed pid $RUN_PID so the run does not waste GPU hours"
fi

echo
echo "=== running ==="
echo "  the first generation logged everything the analysis needs; leaving it to finish."
echo "  tail -f $LOG_FILE"
wait "$RUN_PID"
STATUS=$?
echo "  exit code  : $STATUS"

if [[ "$STATUS" -eq 0 ]]; then
  echo
  echo "next:"
  echo "  $PYTHON analysis/batch41_random_control.py $RUN_DIR \\"
  echo "      --num-samples 5000 --sigma 0.04 --sigma-sweep 0.01 0.02 0.08 0.16"
fi
exit "$STATUS"
