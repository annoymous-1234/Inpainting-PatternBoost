#!/usr/bin/env bash
#
# Transfer donor -- circle packing K=26 DDPM main (Ours), fresh seed.
#
# The transfer control needs a model that never trained on the pool the parents
# come from. The parents come from K26_ddpm_main_batch4 (seed 1234).
#
# This config is K26_ddpm_main_batch4's, with the seed changed to 7 and the run
# renamed. Nothing else differs (--dry-run prints the diff).
#
# DECODING
#   Iterative, matching the paper's "Ours" row: env.activation_budget=4 with
#   outer_loop.max_rounds=6, i.e. four of the 26 circles are activated,
#   committed, and re-conditioned on, six times over. The one-shot ablation is
#   the same config with activation_budget=30 and max_rounds=10. The DDPM
#   sampler itself is unchanged either way (infer_num_steps=32 of 64,
#   tau_start_infer=0.8, draft warm-start).
#
# COST
#   The identical run at seed 1234 took 55 minutes on one GPU
#   (checkpoints/K26_ddpm_main_batch4/results.json: elapsed_sec 3308).
#
# OUTPUT
#   run dir : checkpoints/K26_ddpm_transfer_rt7/
#             generation_{010,020,030,040}.pt   <- the donor model
#             latest.pt                          (= generation_040.pt)
#             top_pool.pt, top_pool_generation_{005..040}.pt
#             metrics.jsonl, candidates.jsonl, results.json, config_snapshot.yaml
#   log     : logs/transfer/K26_ddpm_transfer_rt7.log
#
# USAGE
#   conda activate ec
#   rebuttal-scripts/run_transfer_donor.sh --dry-run     # preflight + diff only
#   rebuttal-scripts/run_transfer_donor.sh
#   rebuttal-scripts/run_transfer_donor.sh --force       # discard an existing run dir
#   SEED=11 RUN_NAME=K26_ddpm_transfer_rt11 rebuttal-scripts/run_transfer_donor.sh   # another donor
#
# Set PYTHON=... to pick an interpreter; the default `python` expects the conda
# environment to be active already.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PYTHON="${PYTHON:-python}"
RUN_NAME="${RUN_NAME:-K26_ddpm_transfer_rt7}"
CONFIG="${CONFIG:-exp-config/rebuttal-transfer/K26_ddpm_transfer_rt7.yaml}"
PARENT_CONFIG="exp-config/rebuttal-batch4/K26_ddpm_main_batch4.yaml"
ENTRYPOINT="scripts/run_patternboost_experiment.py"
# The entrypoint forces checkpoint_dir to "checkpoints" and uses run_name as the
# subdirectory, so the run lands here regardless of the config's checkpoint_dir.
RUN_DIR="checkpoints/${RUN_NAME}"
LOG_DIR="logs/transfer"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

DEVICE=""; DRY_RUN=0; FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force)   FORCE=1; shift ;;
    --device)  DEVICE="$2"; shift 2 ;;
    -h|--help) sed -n '2,39p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }

echo "=== preflight ==="
[[ -f "$CONFIG" ]] || fail "config not found: $CONFIG"
[[ -f "$PARENT_CONFIG" ]] || fail "parent config not found: $PARENT_CONFIG"

# The donor is only a donor if it differs from the parents' run in the seed and
# nowhere else: a drifted architecture would fail the strict state-dict load in
# the analysis, and a drifted budget would make "home - transfer" unreadable.
"$PYTHON" - "$CONFIG" "$PARENT_CONFIG" "$RUN_NAME" <<'PYEOF' || exit 1
import sys, yaml

cfg = yaml.safe_load(open(sys.argv[1]))
parent = yaml.safe_load(open(sys.argv[2]))
expected_run_name = sys.argv[3]
rt = cfg["patternboost"]["runtime"]
problems = []

def flatten(node, prefix=""):
    out = {}
    for key, value in (node or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{path}."))
        else:
            out[path] = value
    return out

allowed = {"seed", "checkpoint_dir", "run_name"}
left, right = flatten(parent), flatten(cfg)
drift = sorted(
    key for key in set(left) | set(right)
    if left.get(key) != right.get(key) and key.split(".")[-1] not in allowed
)
if drift:
    problems.append(f"config drifted from {sys.argv[2]} on: {drift}")

# A donor that reuses the parents' seed is not a donor.
if rt.get("seed") == parent["patternboost"]["runtime"].get("seed"):
    problems.append(
        f"seed {rt.get('seed')} is the parents' own seed; the donor must train on a different pool"
    )
if rt.get("run_name") != expected_run_name:
    problems.append(f"run_name {rt.get('run_name')!r} != RUN_NAME {expected_run_name!r}")
if rt.get("run_name") == parent["patternboost"]["runtime"].get("run_name"):
    problems.append("run_name collides with the parents' run; the entrypoint would overwrite it")

# The analysis needs the model; the pool snapshots make this run reusable as a
# parent source later, and cost nothing.
if not rt.get("save_checkpoints"):
    problems.append("patternboost.runtime.save_checkpoints must be true (this run exists for its model)")
if int(cfg["patternboost"]["training"].get("checkpoint_every_generations", 0)) <= 0:
    problems.append("patternboost.training.checkpoint_every_generations must be > 0")
if int(rt.get("top_pool_snapshot_every_generations", 0)) <= 0:
    problems.append("patternboost.runtime.top_pool_snapshot_every_generations must be > 0")

env = cfg["env"]
outer = cfg["outer_loop"]
generation = cfg["patternboost"]["generation"]
print(f"  config     : K={cfg['problem']['K']} seed={rt['seed']} "
      f"gens={generation['num_generations']} cands/gen={generation['candidates_per_generation']} "
      f"srp={cfg['patternboost']['polish']['srp_enabled']}")
print(f"  decoding   : iterative (activation_budget={env['activation_budget']}, "
      f"max_rounds={outer['max_rounds']}); ddpm infer_num_steps={cfg['ddpm']['infer_num_steps']}")
print(f"  checkpoints: every {cfg['patternboost']['training']['checkpoint_every_generations']} gen(s); "
      f"pool snapshots every {rt['top_pool_snapshot_every_generations']}")
print(f"  run dir    : checkpoints/{rt['run_name']}")

if env["activation_budget"] >= cfg["problem"]["K"]:
    print("  WARNING    : activation_budget >= K, so this is the ONE-SHOT ablation, not Ours")

if problems:
    for problem in problems:
        print(f"PREFLIGHT FAILED: {problem}", file=sys.stderr)
    sys.exit(1)
print("  config     : differs from the parents' run in the seed only")
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
  echo "=== diff vs the parents' config ($PARENT_CONFIG) ==="
  diff "$PARENT_CONFIG" "$CONFIG"
  echo
  echo "dry run: would execute"
  echo "  ${CMD[*]}"
  exit 0
fi

mkdir -p "$LOG_DIR"
echo
echo "=== launching $RUN_NAME (log: $LOG_FILE) ==="
echo "  reference timing: the same run at seed 1234 took 55 minutes on one GPU"
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
STATUS="${PIPESTATUS[0]}"
echo "  exit code  : $STATUS"

if [[ "$STATUS" -eq 0 ]]; then
  echo
  echo "the donor model is $RUN_DIR/generation_040.pt (= latest.pt). next:"
  echo "  python analysis/transfer_control.py \\"
  echo "      --pool-run checkpoints/K26_ddpm_main_batch4 \\"
  echo "      --transfer ${RUN_NAME}=${RUN_DIR}/generation_040.pt"
fi
exit "$STATUS"
