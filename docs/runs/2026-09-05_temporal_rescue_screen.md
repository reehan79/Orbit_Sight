# Temporal rescue SCREEN (2026-09-05)

Branch: `sprint/temporal-rescue-screen`  
Base: `sprint/final-optimization`  
Mode: **SCREEN only** — outer folds **1, 4** (not full five-fold CV)  
Champion path: **D2** unchanged (accept → emit exactly D2; rescue only on reject)

Scientific interpretation is external. Method was not altered after seeing results.

---

## Core principle (enforced)

| Rule | Result |
|------|--------|
| If D2 accepts → TEMPORAL_RESCUE emits identical start/end/cx/cy/w/h/confidence | `accepted_path_mismatches = 0` |
| Rescue only on D2 reject; at most one box | Yes |
| No new CNN; no C4/S2/Top-20 change | Yes |
| Never Testing_sets | Yes |

---

## Method (fixed)

**Part A — causal temporal features (15):** history=7 windows (280 ms), decay=0.90, velocity dx,dy ∈ [-3,+3] cells/window. Incremental `CausalTemporalState` (no 7-window disk rescan).

**Part B — target:** on D2-rejected windows only; y=1 iff emitting current TRUE-P1 C4+S2 box is TII TP @ IoU≥0.5.

**Part C — model:** `StandardScaler` + `LogisticRegression(C=1, class_weight=balanced, max_iter=1000, random_state=42)`. Sequence-level INNER-OOF on OUTER TRAIN rejected rows → threshold maximizes TII detection F1 of (D2 accepts + OOF rescues) on OUTER TRAIN only → refit on all OUTER TRAIN rejected → apply frozen threshold to OUTER VAL.

---

## Pooled SCREEN (folds 1+4)

| method | precision | recall | F1 | AP@0.5 | TP | FP | FN | seq macro F1 | seq macro recall | empty FPR | mean matched IoU |
|--------|----------:|-------:|---:|-------:|---:|---:|---:|-------------:|-----------------:|----------:|-----------------:|
| CHAMPION_D2 | 0.5372 | 0.3052 | 0.3892 | 0.2075 | 3607 | 3108 | 8212 | 0.3353 | 0.2846 | 0.0210 | 0.6682 |
| TEMPORAL_RESCUE | 0.5372 | 0.3052 | 0.3892 | 0.2075 | 3607 | 3108 | 8212 | 0.3353 | 0.2846 | 0.0210 | 0.6682 |

TEMPORAL_RESCUE matched CHAMPION_D2 exactly: train-OOF selected rescue thresholds ≈ 1.0 → **0 rescue emissions** on validation.

Official TII ↔ local evaluator parity: required and passed for both methods on both folds.

---

## By fold

| fold | method | thr | precision | recall | F1 | AP@0.5 | TP | FP | FN | empty FPR |
|-----:|--------|----:|----------:|-------:|---:|-------:|---:|---:|---:|----------:|
| 1 | CHAMPION_D2 | 0.6330 | 0.3093 | 0.2046 | 0.2463 | 0.1067 | 493 | 1101 | 1917 | 0.0067 |
| 1 | TEMPORAL_RESCUE | ≈1.0 | 0.3093 | 0.2046 | 0.2463 | 0.1067 | 493 | 1101 | 1917 | 0.0067 |
| 4 | CHAMPION_D2 | 0.5187 | 0.6081 | 0.3310 | 0.4286 | 0.3083 | 3114 | 2007 | 6295 | 0.0352 |
| 4 | TEMPORAL_RESCUE | ≈1.0 | 0.6081 | 0.3310 | 0.4286 | 0.3083 | 3114 | 2007 | 6295 | 0.0352 |

D2 fold-1 threshold matches prior champion CV (`0.6330060940344227`).

Per-sequence / sensor tables: `docs/runs/2026-09-05/temporal_rescue_screen/compare_by_sequence.csv`.

---

## Part E — accepted-path invariance

`accepted_path_mismatches = 0` (folds 1+4).

---

## Part F — rescue accounting (pooled)

| metric | value |
|--------|------:|
| D2-rejected windows | 33316 |
| rescue attempts | 33316 |
| rescue emissions | **0** |
| rescue TP / FP | 0 / 0 |
| rescue precision | n/a (0 emissions) |
| new TP gained | 0 |
| new FP introduced | 0 |
| rescue TP with Top-20 present | 0 |

Train-side OOF F1 selection preferred never emitting rescues over any lower threshold.

---

## Part G — gate-score AP diagnostic (side only)

Unthresholded Top-1 rows persisted locally (`unthresholded_fold{1,4}.csv`, ~6 MB; not committed). Summary:

| fold | AP@0.5 BASE conf | AP@0.5 D2 GATE |
|-----:|-----------------:|---------------:|
| 1 | 0.1460 | 0.1124 |
| 4 | 0.3442 | 0.3608 |

Did not change the temporal method.

---

## Part H — latency screen (≤500 windows × 4 sequences × 2 folds pooled)

| method | sensor | n | p50 ms | p95 ms | p99 ms |
|--------|--------|--:|-------:|-------:|-------:|
| CHAMPION_D2 | ALL | 4000 | 15.43 | 29.27 | 41.73 |
| TEMPORAL_RESCUE | ALL | 4000 | 16.32 | 30.13 | 41.56 |
| CHAMPION_D2 | DAVIS | 1000 | 14.50 | 18.47 | 30.87 |
| TEMPORAL_RESCUE | DAVIS | 1000 | 15.18 | 19.78 | 30.36 |
| CHAMPION_D2 | DVX | 2000 | 14.94 | 19.21 | 22.91 |
| TEMPORAL_RESCUE | DVX | 2000 | 15.88 | 20.75 | 24.80 |
| CHAMPION_D2 | EVK4 | 1000 | 21.47 | 38.98 | 55.18 |
| TEMPORAL_RESCUE | EVK4 | 1000 | 22.32 | 38.55 | 60.19 |

Temporal state updated causally/incrementally.

---

## SCREEN promotion criteria (folds 1+4 pooled)

| ID | Criterion | Result | Pass? |
|----|-----------|--------|:-----:|
| SCREEN_A | F1_TEMPORAL ≥ F1_D2 + 0.03 | 0.3892 vs 0.3892 | **NO** |
| SCREEN_B | Recall_TEMPORAL ≥ Recall_D2 + 0.04 | 0.3052 vs 0.3052 | **NO** |
| SCREEN_C | Precision_TEMPORAL ≥ 0.50 | 0.5372 | YES |
| SCREEN_D | empty-window FPR ≤ 0.02 | 0.0210 | **NO** |
| SCREEN_E | accepted_path_mismatches == 0 | 0 | YES |
| SCREEN_F | temporal complete-path p95 ≤ 40 ms | 30.13 ms | YES |

### FULL_CV_RECOMMENDED=False

STOP. Do **not** run folds 0, 2, 3.

---

## Artifacts

- Report: `docs/runs/2026-09-05_temporal_rescue_screen.md`
- Compact CSVs / criteria: `docs/runs/2026-09-05/temporal_rescue_screen/`
- Code: `src/orbitsight/inference/temporal_rescue.py`, `scripts/cv_temporal_rescue_screen.py`
- Large local-only: `unthresholded_fold*.csv`, `latency_raw_fold*.json`
