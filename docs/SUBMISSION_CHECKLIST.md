# SUBMISSION CHECKLIST

Branch: `submission/final-freeze`  
Team output path: `/work/OrbitSight/<DDMMYYYY>/`

## Freeze / science

- [x] final model freeze committed before Testing_sets access
- [x] sealed test run exactly once (`docs/runs/2026-09-08/final_sealed_test/SEALED_RUN_LOCK.json`)
- [x] no post-test tuning
- [x] TII / local evaluator parity on sealed holdout
- [x] no label leakage (`tests/test_label_leakage_final.py`)
- [x] temporal rescue **not** in final executable imports

## Packaging

- [x] Docker builds cleanly (`orbitsight-final:latest`)
- [x] Docker works `--network none`
- [x] dataset mounted read-only (`/OrbitSight_dataset:ro`)
- [x] correct `/work/<TEAM>/<DDMMYYYY>/` output path
- [x] deterministic Docker rerun (native == docker == docker; boxes identical, conf atol ≤ 1e-12)
- [x] automatic exit code 0 (`python -m orbitsight.submission`)
- [x] final model files bundled in image (`/models/final`)
- [x] README / `docs/SUBMISSION_RUNTIME.md` commands verified

## Latency

- [x] Docker overall p95 ≤ 40 ms (**36.31 ms**)
- [x] per-sensor p50/p95/p99 reported (EVK4 still >40 ms locally in container bench; overall PASS)

## Docs / handoff

- [x] `docs/FINAL_MODEL_FREEZE.md`
- [x] `docs/runs/2026-09-08_FINAL_SEALED_TEST.md`
- [x] `docs/FINAL_EVIDENCE_TABLE.md`
- [ ] proposal ≤5 pages (authoring outside this freeze)
- [x] Docker archive/image ready (`orbitsight-final:latest`)
- [x] submission filenames checked (`*_bb_windows_40ms.txt`)

## Do not

- [x] do not merge this PR automatically
- [x] do not run temporal folds 0/2/3
- [x] do not change architecture / threshold after sealed test
