# Challenge-metric detector baseline (2026-08-30)

Branch: `experiment/challenge-metric-baseline`  
Base: `experiment/tiny-foveated-refiner`  
Scope: TRAINING_sets sequence CV only (not test-set performance).  
No threshold/architecture tuning on validation. No motion rescue. No new NN/geometry.

Artifacts (CSV): `docs/runs/2026-08-30/challenge_metric_baseline/`

---

## PART 0 — H1 hybrid (final neural experiment)

Pipeline: Top-20 → ExtraTrees Top-3 → tiny neural **classification logits only** → C4_MEDIAN + S2 (ignore neural bbox).

| Config | pooled IoU50 | seq macro IoU50 | IoU75 | neural-selected compatible % |
|--------|-------------:|----------------:|------:|-----------------------------:|
| B_CURRENT | 43.121% | 37.645% | 10.685% | 89.132% |
| H1_NEURAL_SELECT_CLASSICAL_BOX | **43.663%** | **41.110%** | 10.718% | 90.714% |

Gate: `H1_beats_B_CURRENT=True` → PART 9 **H1_P1** run in all-window CV.  
CSVs: `h1_summary.csv`, `h1_by_sensor.csv`, `h1_by_sequence.csv`, `h1_cross_sensor.csv`.

---

## PART 1 — Profile B_CURRENT

500 windows × 5 sequences. Bottlenecks: candidate features, ExtraTrees ranker, S2 (~3–8 ms p50 each). EVK4 features/ranker p95 can exceed ~25 ms. Full table: `profile_components.csv`.

---

## PART 2 — Behaviour-preserving fast path

- Implementation: `src/orbitsight/inference/b_current.py`, `candidate_features_fast.py`
- Parity: `parity_failures=0` (`fast_parity.txt`); tests in `tests/test_fast_b_current.py`
- OLD vs FAST (2500 windows): p50 12.826 vs 12.834 ms; speedup_p50 ≈ 0.999 (negligible). CSVs: `fast_vs_old_latency.csv`

---

## PART 3 — All 40-ms windows (TRAINING_sets)

Convention matched to TII DataLoader (`window_convention.json`): `np.arange(t0, t1, 40000)` half-open.

| Metric | Value |
|--------|------:|
| total windows | 106192 |
| ≥1 GT | 15290 |
| empty | 90902 (85.6%) |
| 1 / 2 / 3+ GT among positive | 15290 / 0 / 0 |

By sensor: DAVIS 47011 (10184 GT), DVX 57101 (3905), EVK4 2080 (1201 windows / 1203 boxes).  
CSVs: `window_characterization_*.csv`

---

## PART 4 — All-window candidate cache

Local only (not committed): `artifacts/all_window_candidates.npz`

| rows | positives | negatives | pos_ratio | disk |
|-----:|----------:|----------:|----------:|-----:|
| 2095576 | 59073 | 2036503 | 0.0282 | 47.26 MB |

15 candidate features only; nearest-compatible GT; empty windows → all negatives.

---

## PARTS 5–8 — Confidence detector CV (P1/P3/P5)

Frozen ExtraTreesClassifier (64 / depth 14 / leaf 12 / balanced / seed 42).  
F1 threshold from **outer-train OOF only**. Geometry: FAST B_CURRENT (C4 + S2).  
NMS IoU=0.30 for P3/P5. Official TII evaluator ↔ local evaluator: **parity_ok=True**.

### Policy summary (pooled across 5 outer folds)

| Policy | Precision | Recall | F1 | mean AP@0.5 | TP | FP | FN | seq macro F1 | seq macro recall | pos-win loc recall | empty-win FPR |
|--------|----------:|-------:|---:|------------:|---:|---:|---:|-------------:|-----------------:|-------------------:|--------------:|
| P1 | 0.274 | 0.388 | 0.321 | 0.255 | 5936 | 15736 | 9356 | 0.327 | 0.368 | 0.344 | 0.124 |
| P3 | 0.147 | 0.390 | 0.213 | 0.255 | 5968 | 34715 | 9324 | 0.279 | 0.398 | 0.368 | 0.124 |
| P5 | 0.104 | 0.391 | 0.164 | 0.255 | 5975 | 51393 | 9317 | 0.266 | 0.406 | 0.373 | 0.124 |

Per-fold / per-sequence / per-sensor: `challenge_by_fold.csv`, `challenge_by_sequence.csv`, `challenge_by_sensor.csv`.

---

## PART 9 — H1_P1 (conditional)

Ran because PART 0 gate passed.

| Policy | Precision | Recall | F1 | mean AP@0.5 | TP | FP | FN | seq macro F1 |
|--------|----------:|-------:|---:|------------:|---:|---:|---:|-------------:|
| H1_P1 | 0.281 | 0.366 | 0.318 | 0.259 | 5592 | 14304 | 9700 | 0.346 |

---

## PART 10 — Latency (FAST complete paths)

Criteria: LATENCY_A overall p95 ≤ 40 ms; LATENCY_B each sensor p95 ≤ 40 ms.

| Policy | LATENCY_A (overall p95) | DAVIS | DVX | EVK4 | LATENCY_B |
|--------|------------------------:|------:|----:|-----:|-----------|
| P1 | 30.81 **PASS** | 18.29 PASS | 15.51 PASS | 43.36 **FAIL** | **FAIL** |
| P3 | 66.46 FAIL | 21.17 PASS | 14.79 PASS | 88.19 FAIL | FAIL |
| P5 | 73.27 FAIL | 19.81 PASS | 20.04 PASS | 97.97 FAIL | FAIL |
| H1_P1 | 61.20 FAIL | 68.03 FAIL | 25.91 PASS | 54.47 FAIL | FAIL |

Full: `challenge_latency.csv`, `latency_criteria.csv`.

---

## Criteria checklist

| Criterion | Result |
|-----------|--------|
| H1 beats B_CURRENT (pooled + seq macro IoU50) | **PASS** → H1_P1 evaluated |
| Fast-path prediction parity | **PASS** (0 failures) |
| Window enum vs TII convention | **PASS** (documented) |
| TII vs local evaluator parity | **PASS** |
| LATENCY_A P1 | **PASS** |
| LATENCY_A P3/P5/H1_P1 | **FAIL** |
| LATENCY_B (all sensors) all policies | **FAIL** (EVK4 p95 > 40 ms; H1 also DAVIS) |
| pytest -q | **PASS** (29) |

No algorithm changes after latency criteria check.
