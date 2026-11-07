#!/usr/bin/env bash
#
# n=25 main (Ours) rerun with candidate provenance logging and pool snapshots.
#
# The config is exp-config/max/n25_main_stage1.yaml (the paper's Table 3 "Ours"
# row, seed 0) plus per-generation pool snapshots; --dry-run prints the diff.
# The transfer analysis uses those snapshots and the per-candidate
# child-minus-parent delta logged in candidates.jsonl.
#
# WHAT IT GUARANTEES
#   1. log_candidates_jsonl / log_jsonl are on            (checked before launch)
#   2. the installed code actually emits seed_score +
#      delta_score into candidates.jsonl                  (checked before launch)
#   3. per-generation pool snapshots are on, so the
#      control arm can pair against the pool a candidate
#      was really drawn from, not just the final pool     (checked before launch)
#   4. the first generation's log is validated on disk and
#      the run is aborted immediately if a field is missing
#
# Output
#   run dir : checkpoints/n25_main_batch4/exp/
#             metrics.jsonl, candidates.jsonl, config_snapshot.yaml,
#             top_pool.pt, top_pool_generation_XXX.pt
#   log     : logs/batch4/n25_main_batch4.log
#
# Usage
#   conda activate ec
#   rebuttal-scripts/run_batch4_main.sh              # preflight, then run
#   rebuttal-scripts/run_batch4_main.sh --dry-run    # preflight only
#   rebuttal-scripts/run_batch4_main.sh --force      # discard an existing run dir
#
# Set PYTHON=... to pick an interpreter; the default `python` expects the conda
# environment to be active already.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PYTHON="${PYTHON:-python}"
RUN_NAME="n25_main_batch4"
CONFIG="rebuttal-exp-config/n25_main_batch4.yaml"
PARENT_CONFIG="exp-config/max/n25_main_stage1.yaml"
RUN_DIR="checkpoints/${RUN_NAME}/exp"
LOG_DIR="logs/batch4"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

# Fields the analysis reads out of candidates.jsonl.
REQUIRED_CANDIDATE_FIELDS=(seed_hash seed_score delta_score score mask_ratio remask_mode generation)

DRY_RUN=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force)   FORCE=1; shift ;;
    -h|--help) sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }

echo "=== preflight ==="

[[ -f "$CONFIG" ]] || fail "config not found: $CONFIG"

# 1. the code must emit the provenance fields.
grep -q '"seed_score": result.seed_score' patternboost/loop.py \
  || fail "patternboost/loop.py does not log seed_score -- this checkout predates the provenance logging"
grep -q '"delta_score": result.score - result.seed_score' patternboost/loop.py \
  || fail "patternboost/loop.py does not log delta_score -- this checkout predates the provenance logging"
grep -q 'seed_score=int(seed_entry.score)' patternboost/generator.py \
  || fail "patternboost/generator.py does not record the seed score on the candidate"
grep -q '_save_top_pool_snapshot' patternboost/loop.py \
  || fail "patternboost/loop.py has no per-generation pool snapshot support"
echo "  code       : provenance logging + pool snapshots present"

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
if rt.get("checkpoint_dir") == parent["patternboost"]["runtime"].get("checkpoint_dir"):
    problems.append("checkpoint_dir collides with the original run; pick a distinct one")

# Everything except the run location and the snapshot switch must match the
# paper's config, otherwise this is a different experiment.
allowed = {"checkpoint_dir", "top_pool_snapshot_every_generations"}
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

print(f"  config     : n={cfg['n']} seed={rt['seed']} "
      f"gens={cfg['patternboost']['generation']['num_generations']} "
      f"cands/gen={cfg['patternboost']['generation']['candidates_per_generation']} "
      f"snapshot_every={rt['top_pool_snapshot_every_generations']}")
print(f"  run dir    : {rt['checkpoint_dir']}/{rt['run_name']}")
if problems:
    for problem in problems:
        print(f"PREFLIGHT FAILED: {problem}", file=sys.stderr)
    sys.exit(1)
print("  config     : logging + snapshots enabled, matches the paper config")
PYEOF

if [[ -d "$RUN_DIR" ]]; then
  if [[ "$FORCE" -eq 1 ]]; then
    echo "  run dir    : removing existing $RUN_DIR (--force)"
    [[ "$DRY_RUN" -eq 1 ]] || rm -rf "checkpoints/${RUN_NAME}"
  else
    fail "$RUN_DIR already exists; pass --force to discard it"
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "=== diff vs the paper config ($PARENT_CONFIG) ==="
  diff "$PARENT_CONFIG" "$CONFIG"
  echo
  echo "dry run: would execute"
  echo "  $PYTHON train_patternboost.py --config $CONFIG"
  exit 0
fi

mkdir -p "$LOG_DIR"
echo
echo "=== launching $RUN_NAME (log: $LOG_FILE) ==="
"$PYTHON" train_patternboost.py --config "$CONFIG" > "$LOG_FILE" 2>&1 &
RUN_PID=$!
echo "  pid        : $RUN_PID"

# 3. Validate generation 1 on disk, then get out of the way. Catching a schema
#    problem here costs one generation instead of the whole run.
echo "  validating : waiting for the first candidates.jsonl record ..."
CANDIDATES="${RUN_DIR}/candidates.jsonl"
for _ in $(seq 1 240); do
  kill -0 "$RUN_PID" 2>/dev/null || break
  [[ -s "$CANDIDATES" ]] && break
  sleep 15
done

if [[ ! -s "$CANDIDATES" ]]; then
  if kill -0 "$RUN_PID" 2>/dev/null; then
    kill "$RUN_PID" 2>/dev/null
    fail "no candidates.jsonl after 60 minutes; killed pid $RUN_PID (see $LOG_FILE)"
  fi
  fail "run exited before writing candidates.jsonl (see $LOG_FILE)"
fi

"$PYTHON" - "$CANDIDATES" "${REQUIRED_CANDIDATE_FIELDS[@]}" <<'PYEOF'
import json, sys
path, required = sys.argv[1], sys.argv[2:]
with open(path) as handle:
    record = json.loads(handle.readline())
missing = [field for field in required if record.get(field) is None]
print(f"  validating : {len(record)} fields -> {sorted(record)}")
if missing:
    print(f"PREFLIGHT FAILED: candidates.jsonl is missing {missing}", file=sys.stderr)
    sys.exit(1)
print(f"  validating : ok (seed_score={record['seed_score']} "
      f"score={record['score']} delta_score={record['delta_score']})")
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
  echo "  $PYTHON analysis/batch41_random_control.py $RUN_DIR --num-samples 50000"
fi
exit "$STATUS"
