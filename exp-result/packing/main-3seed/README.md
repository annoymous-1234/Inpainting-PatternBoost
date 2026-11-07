# Circle packing main results (3 seeds)

Records behind the main circle-packing table of the paper (Table 4). Each
method and `K` has three runs, with runtime seeds 1234, 23 and 44. The original
`../main/` folder is kept unchanged.

Seed 1234 runs are named `K{15,30,32,40}_<method>_main` and
`K26_<method>_main_rt1234`. Seeds 23 and 44 use the suffixes `_rt23` and `_rt44`.

## Log-only runs

Most runs contain the full record (`config_snapshot.yaml`, `generations.jsonl`,
`metrics.jsonl`, `resolved_config.yaml`, `results.json`, `run_manifest.json`).

These 13 runs contain only the training log `<run>.log` and the input config
`<run>.yaml`:

- `K15`, `K30`, `K32`, for both `ddpm` and `fm`, with `_rt23` and `_rt44`
- `K40_fm_main_rt44`

During the seed sweep these runs shared one output directory, so their other
files were overwritten. Their logs are complete: every run reaches generation 40
and ends with `status=ok`. The last `generation=` line gives the values in the
table: `pool_score_best` (Best), `pool_score_p90` (Pool p90), `pool_score_mean`
(Pool mean) and `final_score_mean` (Last mean).

## Random + LP

The Random + LP runs are in `ddpm/random_baseline/`. The `K<K>_seed23` and
`K<K>_seed44` runs have only `metrics.jsonl` and `results.json`.
