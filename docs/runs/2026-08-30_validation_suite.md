# Frozen OrbitSight validation suite — 2026-08-30

Branch: `run/2026-08-30-validation-suite`  
Base SHA at branch creation: `426c3e35be716f3ebe33436fced48fed4fae1de4`  
Local console logs: `_runs/2026-08-30_validation_suite_20260830_012000/` (not committed)  
Compact CSV summaries: `docs/runs/2026-08-30/`

Dataset root used: `D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset\Training_sets`  
Testing_sets were not accessed.

---

## Environment

| Item | Value |
|------|-------|
| Git SHA (start) | `426c3e35be716f3ebe33436fced48fed4fae1de4` |
| Python | 3.11.9 (MSC v.1938 64 bit, AMD64) |
| CPU | 12th Gen Intel(R) Core(TM) i7-12700 (12 cores / 20 threads) |
| RAM | 63.77 GB |
| OS | Windows-10-10.0.26200-SP0 |
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| scikit-learn | 1.8.0 |

Also recorded in `docs/runs/2026-08-30/environment.txt`.

## Git SHA

Start-of-run commit on `feat/candidate-models-m0-m2` after fast-forward:

`426c3e35be716f3ebe33436fced48fed4fae1de4`

## Tests

Command: `pip install -e ".[dev,ml]"` then `pytest -q`.

**Initial run:** 1 failed, 8 passed.

Failure: `tests/test_candidate_ranker.py::test_minimal_rankers_fit_and_score` asserted default ranker set `{M0_raw_rank, M1_logistic, M2_hist_gb}` but `DEFAULT_RANKERS` now also includes `M2a_tree` and `M2b_extra_trees`.

**Mechanical fix:** updated the expected set in the test only (see Mechanical changes). No algorithm or hyperparameter changes.

**Rerun:** `.........` — **9 passed in 2.90s**.

## Frozen CV results

Command:

```text
python -u .\scripts\cv_candidate_models.py --table .\artifacts\candidate_table.csv --folds .\sequence_folds.json --out .\artifacts\candidate_fast_cv.csv --models M0_raw_rank,M1_logistic,M2a_tree,M2b_extra_trees
```

Table: 305798 rows, 15 features, 59647 positives. Models not tuned after results.

### Per-fold (micro Top-1 / Top-3 / Top-5, macro Top-1, MRR)

| Fold | Model | T1 | T3 | T5 | macroT1 | MRR |
|------|-------|----|----|----|---------|-----|
| 0 | M0_raw_rank | 60.90% | 89.47% | 94.74% | 53.48% | 0.7567 |
| 0 | M1_logistic | 77.44% | 85.71% | 91.73% | 58.40% | 0.8345 |
| 0 | M2a_tree | 79.70% | 93.61% | 95.86% | 68.48% | 0.8700 |
| 0 | M2b_extra_trees | 80.83% | 93.98% | 97.74% | 71.00% | 0.8714 |
| 1 | M0_raw_rank | 79.24% | 86.38% | 89.00% | 64.82% | 0.8344 |
| 1 | M1_logistic | 84.18% | 87.92% | 89.20% | 73.47% | 0.8638 |
| 1 | M2a_tree | 70.27% | 82.68% | 86.67% | 73.01% | 0.7722 |
| 1 | M2b_extra_trees | 88.33% | 90.86% | 91.11% | 80.39% | 0.8962 |
| 2 | M0_raw_rank | 91.26% | 99.05% | 99.56% | 69.37% | 0.9513 |
| 2 | M1_logistic | 98.76% | 99.33% | 99.65% | 78.81% | 0.9911 |
| 2 | M2a_tree | 89.33% | 94.74% | 95.12% | 88.11% | 0.9225 |
| 2 | M2b_extra_trees | 99.30% | 99.78% | 99.81% | 95.61% | 0.9954 |
| 3 | M0_raw_rank | 34.00% | 72.00% | 80.00% | 39.74% | 0.5416 |
| 3 | M1_logistic | 42.00% | 64.00% | 76.00% | 42.82% | 0.5646 |
| 3 | M2a_tree | 54.00% | 88.00% | 94.00% | 54.62% | 0.7212 |
| 3 | M2b_extra_trees | 68.00% | 92.00% | 96.00% | 64.84% | 0.7990 |
| 4 | M0_raw_rank | 62.19% | 78.65% | 83.74% | 41.86% | 0.7198 |
| 4 | M1_logistic | 76.77% | 87.09% | 91.55% | 60.55% | 0.8298 |
| 4 | M2a_tree | 72.93% | 85.55% | 88.48% | 59.55% | 0.7989 |
| 4 | M2b_extra_trees | 86.22% | 89.00% | 90.93% | 74.45% | 0.8824 |

### Cross-fold mean ± std

| Model | top1_micro | top3_micro | top5_micro | top1_macro | mrr |
|-------|------------|------------|------------|------------|-----|
| M0_raw_rank | 0.655162 ± 0.193727 | 0.851101 ± 0.092585 | 0.894055 ± 0.071022 | 0.538542 ± 0.118645 | 0.760769 ± 0.135246 |
| M1_logistic | 0.758306 ± 0.186742 | 0.848102 ± 0.114863 | 0.896268 ± 0.076745 | 0.628078 ± 0.125984 | 0.816777 ± 0.139059 |
| M2a_tree | 0.732441 ± 0.116583 | 0.889159 ± 0.046264 | 0.920270 ± 0.037280 | 0.687529 ± 0.116407 | 0.816940 ± 0.071338 |
| M2b_extra_trees | 0.845352 ± 0.102224 | 0.931254 ± 0.036987 | 0.951203 ± 0.035562 | 0.772562 ± 0.104687 | 0.888868 ± 0.062964 |

CSV: `docs/runs/2026-08-30/candidate_fast_cv.csv`

## Candidate geometry

Command: `python .\scripts\diagnose_candidate_geometry.py --table .\artifacts\candidate_table.csv`

Diagnostic only (GT-oracle; not deployable). `positive_candidate_rows=59647`, `unique_gt_boxes=14550`.

| Sensor | median GT size | cand_center+oracle_size meanIoU / IoU50 | oracle_center+median_size meanIoU / IoU50 | cand_center+median_size meanIoU / IoU50 |
|--------|----------------|-------------------------------------------|--------------------------------------------|------------------------------------------|
| DAVIS | 9.00×9.00 | 0.2631 / 14.93% | 0.4854 / 44.21% | 0.1839 / 3.53% |
| DVX | 10.00×13.00 | 0.2802 / 15.72% | 0.6370 / 78.39% | 0.2300 / 7.86% |
| EVK4 | 63.00×60.00 | 0.3861 / 26.47% | 0.7130 / 83.42% | 0.3640 / 19.76% |

CSV: `docs/runs/2026-08-30/candidate_geometry_summary.csv`

## Full 17-sequence latency

Command pattern (sequential, memmap, windows=500 or all available if fewer, top-k=20):

```text
python -u .\scripts\benchmark_batched_stream.py --npy <path> --sequence <seq> --top-k 20 --windows 500 --mode memmap
```

| sequence | sensor | calls | mean events/window | amortized ms/window | proposal p50 | p95 | p99 |
|----------|--------|------:|-------------------:|--------------------:|-------------:|----:|----:|
| 2025_12_23_21_12_28_EVK4_mag5.2 | EVK4 | 499 | 8060.7 | 7.1836 | 0.1339 | 0.5150 | 0.6157 |
| DAVIS_COSMOS1933_18958_2024-12-04-18-37-01 | DAVIS | 500 | 1426.7 | 7.4451 | 0.0814 | 0.1059 | 0.1294 |
| DAVIS_EGS_16908_2024-11-01-19-10-44 | DAVIS | 500 | 950.2 | 7.1558 | 0.0551 | 0.0733 | 0.1087 |
| DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06 | DAVIS | 500 | 99.7 | 0.2316 | 0.0390 | 0.0547 | 0.1060 |
| DAVIS_RESURSDK1_29228_2024-12-04-18-37-01 | DAVIS | 500 | 1385.9 | 6.7849 | 0.1493 | 0.9585 | 3.2216 |
| DAVIS_SL12RB2_15772_2024-12-04-18-21-37 | DAVIS | 500 | 1457.8 | 1.7194 | 0.0587 | 0.0656 | 0.0729 |
| DAVIS_SL16RB_20625_2024-12-04-19-34-18 | DAVIS | 500 | 1482.7 | 7.1615 | 0.1312 | 0.1892 | 0.3357 |
| DAVIS_SL16RB_26070_2024-12-04-19-14-39 | DAVIS | 500 | 1447.2 | 1.5255 | 0.0605 | 0.0722 | 0.0924 |
| DAVIS_SL8RB_2025-01-13-19-15-36 | DAVIS | 500 | 1942.8 | 9.3272 | 0.1316 | 0.2264 | 0.3586 |
| DVX_Filtered_ACS3_59588_2025-01-20-19-35-44 | DVX | 500 | 503.5 | 3.7726 | 0.0477 | 0.0604 | 0.0747 |
| DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17 | DVX | 500 | 512.5 | 0.8893 | 0.0462 | 0.0622 | 0.1038 |
| DVX_Filtered_NOAA15_25338_2025-01-20-19-25-07 | DVX | 500 | 496.9 | 3.9067 | 0.0474 | 0.0788 | 0.1498 |
| DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50 | DVX | 500 | 513.0 | 3.9351 | 0.0473 | 0.0592 | 0.0790 |
| DVX_Filtered_NOAA6_11416_2025-01-20-19-11-35 | DVX | 500 | 508.2 | 1.1590 | 0.0448 | 0.0600 | 0.0865 |
| DVX_Filtered_Stars_2025-01-20-19-15-10 | DVX | 500 | 573.9 | 4.2369 | 0.0464 | 0.0702 | 0.1108 |
| DVX_Filtered_Stars2_2025-01-20-19-57-17 | DVX | 210 | 516.4 | 0.2212 | 0.0460 | 0.0550 | 0.0739 |
| DVX_NOAA6_11416_2025-01-20-19-06-31 | DVX | 457 | 5423.3 | 17.2703 | 0.2208 | 0.4640 | 0.9926 |

CSV: `docs/runs/2026-08-30/suite_d_latency_17.csv`

## RAM vs memmap

Same three sequences, 500 windows, Top-20, both modes.

| sequence | mode | np_load_ms | amortized ms/window | p50 | p95 | p99 |
|----------|------|----------:|--------------------:|----:|----:|----:|
| DAVIS_EGS_16908_2024-11-01-19-10-44 | memmap | 0.607 | 0.3940 | 0.0559 | 0.0726 | 0.1166 |
| DAVIS_EGS_16908_2024-11-01-19-10-44 | ram | 189.233 | 0.1943 | 0.0536 | 0.0735 | 0.1284 |
| DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50 | memmap | 0.669 | 0.2395 | 0.0529 | 0.0731 | 0.1061 |
| DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50 | ram | 96.743 | 0.1202 | 0.0470 | 0.0563 | 0.0691 |
| 2025_12_23_21_12_28_EVK4_mag5.2 | memmap | 0.651 | 0.4982 | 0.1352 | 0.4880 | 0.5834 |
| 2025_12_23_21_12_28_EVK4_mag5.2 | ram | 192.789 | 0.3156 | 0.1359 | 0.5007 | 0.6350 |

CSV: `docs/runs/2026-08-30/suite_e_ram_vs_memmap.csv`

Note: Suite E amortized times are lower than Suite D for the same sequences (warm disk / OS cache); both are reported as measured.

## B1 reproducibility

Command:

```text
python .\scripts\evaluate_raw_proposals.py --split-dir "$TRAIN" --top-k 20 --out .\artifacts\b1_raw_proposal_results_rerun.csv
```

| Source | Top-20 micro | Top-20 macro |
|--------|--------------|--------------|
| Expected (approx historical) | 95.148% | 95.291% |
| Recomputed from `artifacts/b1_raw_proposal_results.csv` | 95.1478% | 95.2914% |
| Rerun this suite | 95.0889% | 94.3175% |
| Δ rerun − historical CSV | −0.0589 pp | −0.9739 pp |

Match within numerical tolerance:

- Top-20 **micro**: near-match (absolute Δ ≈ 5.9e-4); within ~0.06 percentage points of historical.
- Top-20 **macro**: **does not match** within a tight numerical tolerance (absolute Δ ≈ 9.7e-3).

Parameters were not altered. Per-sequence CSV: `docs/runs/2026-08-30/b1_raw_proposal_results_rerun.csv`. Aggregate compare: `docs/runs/2026-08-30/b1_aggregate_compare.csv`.

## Hard-sequence table

B1 Top-1/3/5/20 from this suite’s rerun (and historical CSV for reference). No tuning.

| sequence | T1 rerun | T3 | T5 | T20 | T1 hist | T3 hist | T5 hist | T20 hist |
|----------|----------|----|----|-----|---------|---------|---------|----------|
| DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06 | 0.6373 | 0.7349 | 0.7841 | 0.8299 | 0.6166 | 0.7332 | 0.7807 | 0.8273 |
| DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17 | 0.1318 | 0.3787 | 0.4854 | 0.7531 | 0.1297 | 0.3849 | 0.4874 | 0.7510 |
| DVX_Filtered_Stars_2025-01-20-19-15-10 | 0.2303 | 0.5337 | 0.6479 | 0.8806 | 0.2297 | 0.5361 | 0.6458 | 0.8848 |
| 2025_12_23_21_12_28_EVK4_mag5.2 | 0.9667 | 0.9909 | 0.9942 | 0.9983 | 0.9667 | 0.9909 | 0.9942 | 0.9983 |

Candidate-model validation available without new per-sequence retrain:

- Primary 5-fold CV does **not** emit per-sequence metrics. Hard sequences appear only inside multi-sequence validation folds:
  - Fold 1 includes `DAVIS_Filtered_NOAA6_...` and `EVK4` (among others) — fold-level CSV: `docs/runs/2026-08-30/primary_cv_fold1_includes_noaa6_evk4.csv`
  - Fold 4 includes `DVX_Filtered_BlockDM_...` and `DVX_Filtered_Stars_...` (among others) — `docs/runs/2026-08-30/primary_cv_fold4_includes_blockdm_stars.csv`
- Cross-sensor stress fold 0 validates **EVK4 alone** (Suite I). Isolated metrics:

| Model | T1 | T3 | T5 | macroT1 | MRR |
|-------|----|----|----|---------|-----|
| M0_raw_rank | 96.84% | 99.25% | 99.58% | 96.84% | 0.9809 |
| M1_logistic | 92.67% | 94.67% | 95.92% | 92.67% | 0.9424 |
| M2a_tree | 86.18% | 96.09% | 96.67% | 86.18% | 0.9122 |
| M2b_extra_trees | 99.00% | 99.42% | 99.67% | 99.00% | 0.9927 |

CSV: `docs/runs/2026-08-30/hard_sequences_b1.csv`, `hard_sequence_evk4_cross_sensor_fold0.csv`

## Evaluator parity

TII evaluator: `OrbitSight_DataLoader/evaluate.py`  
Local evaluator: `scripts/evaluate_predictions.py` → `orbitsight.evaluation.metrics`

Training-only tiny deterministic case: first 3 GT windows from `DAVIS_SL12RB2_15772_2024-12-04-18-21-37`. Predictions: exact match (conf 0.95), spatially shifted FP (conf 0.80), exact match (conf 0.70). Identical numeric boxes fed to both (TII filename `*_bb_windows_40ms.txt`; local filename `*_pred.txt`).

| Metric | Local | TII console |
|--------|-------|-------------|
| TP / FP / FN | 2 / 1 / 1 | 2 / 1 / 1 |
| Precision | 0.6667 | 0.6667 |
| Recall | 0.6667 | 0.6667 |
| F1 | 0.6667 | 0.6667 |
| AP@0.5 / mAP | 0.5556 | 0.5556 |

**Agreement on this case.** Evaluators were not modified.

Semantic differences still present in code (not exercised by exact-match TP / clear FP):

- **IoU geometry:** TII uses inclusive pixel conversion `cx-(w-1)/2` with `+1` width/height in IoU; local uses continuous half-width `cx-w/2` without `+1`.
- **Prediction path naming:** TII requires prediction filename identical to GT (`*_bb_windows_40ms.txt`); local accepts `{sequence}_pred.txt` or `{sequence}.txt`.
- **AP endpoint padding:** minor formula differences in recall/precision pad sequences.

TII run printed metrics then raised `UnicodeEncodeError` on Excel save print (`→` under cp1252). Metrics above were captured from console before that failure.

## Cross-sensor stress test

**Label: cross-sensor stress testing, not primary CV.** No tuning.

Temporary folds in `docs/runs/2026-08-30/cross_sensor_folds.json`:

0. train DAVIS+DVX, validate EVK4  
1. train DAVIS+EVK4, validate DVX  
2. train DVX+EVK4, validate DAVIS  

Models: M0_raw_rank, M1_logistic, M2a_tree, M2b_extra_trees.

### Per-fold

| Fold (val sensor) | Model | T1 | T3 | T5 | macroT1 | MRR |
|-------------------|-------|----|----|----|---------|-----|
| 0 EVK4 | M0_raw_rank | 96.84% | 99.25% | 99.58% | 96.84% | 0.9809 |
| 0 EVK4 | M1_logistic | 92.67% | 94.67% | 95.92% | 92.67% | 0.9424 |
| 0 EVK4 | M2a_tree | 86.18% | 96.09% | 96.67% | 86.18% | 0.9122 |
| 0 EVK4 | M2b_extra_trees | 99.00% | 99.42% | 99.67% | 99.00% | 0.9927 |
| 1 DVX | M0_raw_rank | 22.18% | 52.27% | 63.05% | 35.70% | 0.4081 |
| 1 DVX | M1_logistic | 50.22% | 70.29% | 81.18% | 53.06% | 0.6271 |
| 1 DVX | M2a_tree | 29.96% | 60.00% | 68.55% | 36.98% | 0.4778 |
| 1 DVX | M2b_extra_trees | 49.60% | 68.73% | 77.67% | 46.57% | 0.6156 |
| 2 DAVIS | M0_raw_rank | 86.31% | 94.74% | 96.22% | 67.96% | 0.9075 |
| 2 DAVIS | M1_logistic | 88.38% | 93.35% | 95.03% | 67.47% | 0.9129 |
| 2 DAVIS | M2a_tree | 55.25% | 60.46% | 69.82% | 54.99% | 0.6257 |
| 2 DAVIS | M2b_extra_trees | 84.33% | 91.37% | 93.48% | 64.89% | 0.8843 |

### Cross-fold mean ± std (stress only)

| Model | top1_micro | top3_micro | top5_micro | top1_macro | mrr |
|-------|------------|------------|------------|------------|-----|
| M0_raw_rank | 0.684415 ± 0.329951 | 0.820846 ± 0.211651 | 0.862835 ± 0.164877 | 0.668317 ± 0.249733 | 0.765516 ± 0.254478 |
| M1_logistic | 0.770914 ± 0.190831 | 0.861060 ± 0.111934 | 0.907098 ± 0.067498 | 0.710682 ± 0.163708 | 0.827461 ± 0.142164 |
| M2a_tree | 0.571310 ± 0.229887 | 0.721814 ± 0.169046 | 0.783460 ± 0.129669 | 0.593812 ± 0.203252 | 0.671892 ± 0.180337 |
| M2b_extra_trees | 0.776441 ± 0.207130 | 0.865061 ± 0.129903 | 0.902722 ± 0.092624 | 0.701517 ± 0.217277 | 0.830882 ± 0.158517 |

CSV: `docs/runs/2026-08-30/cross_sensor_cv.csv`

## Mechanical changes made

1. `tests/test_candidate_ranker.py` — expected `fit_rankers` default set updated to include `M2a_tree` and `M2b_extra_trees`.
2. `scripts/cv_candidate_models.py` — MRR now assigns RR=0 to validation groups with no positive candidate (mean over all groups). Added `tests/test_cv_candidate_models.py`.
3. `src/orbitsight/proposals/raw_candidates.py` — deterministic Top-K ordering (`count` desc, cell id asc) after tie diagnostic showed argpartition nondeterminism (Top-20 micro delta −0.235 pp). Added tie unit tests in `tests/test_proposals.py`.
4. `scripts/diagnose_topk_ties.py` — Top-K tie diagnostic harness (Training_sets only).

No model hyperparameters were changed.

## Failures/errors encountered

1. Suite A initial pytest failure (stale default-rankers assertion) — fixed mechanically; rerun passed.
2. Suite H TII evaluator: metrics printed successfully; process exited with `UnicodeEncodeError` when printing Excel save path containing `→` under Windows cp1252. Excel write may still have occurred; console metrics were captured.
3. Suite F B1 macro Top-20 outside tight tolerance vs historical (see B1 section). Micro near-match.
4. Transient PowerShell heredoc/`$parity.gt` path bug while preparing Suite H — corrected and rerun successfully (not a product-code failure).

---

Suite completion status: **A–I all executed**. Suite A required one mechanical test fix. Suite H metrics completed with TII Excel Unicode print error after metrics. Suite F completed with partial numerical mismatch on macro Top-20.
