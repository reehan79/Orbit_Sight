# Temporal rescue SCREEN (2026-09-05) — methodology v2

Branch: `sprint/temporal-rescue-screen`  
Base: `sprint/final-optimization`  
Mode: **SCREEN only** — outer folds **1, 4** (not full five-fold CV)  
Champion path: **D2** unchanged (accept → emit exactly D2; rescue only on reject)  
Methodology tag: **`v2_true_inner_oof`**

Scientific interpretation is external. Method was not altered after seeing results.

**Invalidation:** The first SCREEN numbers on this branch (commit `6f5d958`, leaky outer-train upstream features on inner held-out sequences) are **void**. All tables below are from the corrected v2 rerun.

---

## Core principle (enforced)

| Rule | Result |
|------|--------|
| If D2 accepts → TEMPORAL_RESCUE emits identical start/end/cx/cy/w/h/confidence | `accepted_path_mismatches = 0` |
| Rescue only on D2 reject; at most one box | Yes |
| No new CNN; no C4/S2/Top-20 change | Yes |
| Never Testing_sets | Yes |
| True inner-OOF upstream features (no outer-train leakage on held-out inner seq) | Yes |
| History advances through empty / proposal-free windows | Yes |
| Explicit NO_RESCUE threshold (above max train score) | Yes (fold 1 selected it) |
| Sequence macro over all val sequences across folds 1+4 | Yes |

---

## Method (fixed)

**Part A — causal temporal features (15):** history=7 windows (280 ms), decay=0.90, velocity dx,dy ∈ [-3,+3] cells/window. Incremental `CausalTemporalState` (no 7-window disk rescan). Empty windows push empty `HistorySlot`s so the 7-window buffer stays wall-clock aligned.

**Part B — target:** on D2-rejected windows only; y=1 iff emitting current TRUE-P1 C4+S2 box is TII TP @ IoU≥0.5.

**Part C — model:** `StandardScaler` + `LogisticRegression(C=1, class_weight=balanced, max_iter=1000, random_state=42)`.

**True inner-OOF:** for each inner split, fit conf / S2 / D2 gate on inner-train only, score the held-out inner-val sequence with those models, build temporal features from those scores. Outer-train models are **not** used for rescue-training rows.

**Threshold:** sequence-level INNER-OOF on OUTER TRAIN rejected rows → choose among candidate thresholds **including explicit NO_RESCUE** (above max OOF score) the value that maximizes TII detection F1 of (D2 accepts + OOF rescues) on OUTER TRAIN only → refit on all OUTER TRAIN rejected → apply frozen threshold to OUTER VAL.

---

## Pooled SCREEN (folds 1+4) — corrected v2

| method | precision | recall | F1 | AP@0.5 | TP | FP | FN | seq macro F1 | seq macro recall | empty FPR | mean matched IoU |
|--------|----------:|-------:|---:|-------:|---:|---:|---:|-------------:|-----------------:|----------:|-----------------:|
| CHAMPION_D2 | 0.5372 | 0.3052 | 0.3892 | 0.2075 | 3607 | 3108 | 8212 | 0.3186 | 0.2758 | 0.0210 | 0.6682 |
| TEMPORAL_RESCUE | 0.4054 | 0.3565 | 0.3794 | 0.1769 | 4214 | 6180 | 7605 | 0.3164 | 0.2933 | 0.0473 | 0.6554 |

TEMPORAL_RESCUE raises recall (+0.0514) but lowers F1 (−0.0098) and precision vs D2, driven by fold-4 rescue emissions (fold 1 chose NO_RESCUE).

Official TII ↔ local evaluator parity: required and passed for both methods on both folds.

---

## By fold

| fold | method | thr | precision | recall | F1 | AP@0.5 | TP | FP | FN | empty FPR |
|-----:|--------|----:|----------:|-------:|---:|-------:|---:|---:|---:|----------:|
| 1 | CHAMPION_D2 | 0.6330 | 0.3093 | 0.2046 | 0.2463 | 0.1067 | 493 | 1101 | 1917 | 0.0067 |
| 1 | TEMPORAL_RESCUE | NO_RESCUE (~2.0) | 0.3093 | 0.2046 | 0.2463 | 0.1067 | 493 | 1101 | 1917 | 0.0067 |
| 4 | CHAMPION_D2 | 0.5187 | 0.6081 | 0.3310 | 0.4286 | 0.3083 | 3114 | 2007 | 6295 | 0.0352 |
| 4 | TEMPORAL_RESCUE | 0.9975 | 0.4228 | 0.3955 | 0.4087 | 0.2471 | 3721 | 5079 | 5688 | 0.0878 |

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
| rescue emissions | **3679** (all on fold 4; fold 1 = 0) |
| rescue TP / FP | 607 / 3072 |
| rescue precision | 0.1650 |
| new TP gained | 607 |
| new FP introduced | 3072 |
| rescue TP with Top-20 present | 607 |

---

## Part G — gate-score AP diagnostic (side only)

Unthresholded Top-1 rows persisted locally (`unthresholded_fold{1,4}.csv`; not committed). Summary:

| fold | AP@0.5 BASE conf | AP@0.5 D2 GATE |
|-----:|-----------------:|---------------:|
| 1 | 0.1460 | 0.1124 |
| 4 | 0.3442 | 0.3608 |

Did not change the temporal method.

---

## Part H — latency screen (≤500 windows × 4 sequences × 2 folds pooled)

| method | sensor | n | p50 ms | p95 ms | p99 ms |
|--------|--------|--:|-------:|-------:|-------:|
| CHAMPION_D2 | ALL | 4000 | 15.90 | 28.98 | 39.13 |
| TEMPORAL_RESCUE | ALL | 4000 | 16.81 | 29.82 | 39.89 |
| CHAMPION_D2 | DAVIS | 1000 | 15.37 | 20.13 | 30.07 |
| TEMPORAL_RESCUE | DAVIS | 1000 | 16.26 | 20.70 | 32.83 |
| CHAMPION_D2 | DVX | 2000 | 15.62 | 19.70 | 24.53 |
| TEMPORAL_RESCUE | DVX | 2000 | 16.51 | 20.53 | 25.93 |
| CHAMPION_D2 | EVK4 | 1000 | 22.29 | 37.44 | 44.84 |
| TEMPORAL_RESCUE | EVK4 | 1000 | 23.16 | 37.62 | 45.09 |

Temporal state updated causally/incrementally.

---

## SCREEN promotion criteria (folds 1+4 pooled) — corrected v2

| ID | Criterion | Result | Pass? |
|----|-----------|--------|:-----:|
| SCREEN_A | F1_TEMPORAL ≥ F1_D2 + 0.03 | 0.3794 vs 0.3892 | **NO** |
| SCREEN_B | Recall_TEMPORAL ≥ Recall_D2 + 0.04 | 0.3565 vs 0.3052 | YES |
| SCREEN_C | Precision_TEMPORAL ≥ 0.50 | 0.4054 | **NO** |
| SCREEN_D | empty-window FPR ≤ 0.02 | 0.0473 | **NO** |
| SCREEN_E | accepted_path_mismatches == 0 | 0 | YES |
| SCREEN_F | temporal complete-path p95 ≤ 40 ms | 29.82 ms | YES |

### FULL_CV_RECOMMENDED=False

STOP. Do **not** run folds 0, 2, 3.

---

## Artifacts

- Report: `docs/runs/2026-09-05_temporal_rescue_screen.md`
- Compact CSVs / criteria: `docs/runs/2026-09-05/temporal_rescue_screen/` (committed: pooled/compare/latency/criteria/rescue; **not** `inner_oof_*.json`, `unthresholded_*.csv`, `latency_raw_*.json`)
- Code: `src/orbitsight/inference/temporal_rescue.py`, `scripts/cv_temporal_rescue_screen.py`, `src/orbitsight/sprint/s2_static_cache.py`
- Local execution caches (not committed): `artifacts/temporal_rescue_outer/`, `artifacts/temporal_rescue_static/`
