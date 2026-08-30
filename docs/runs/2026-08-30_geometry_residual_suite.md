# Geometry residual suite — 2026-08-30

Frozen M2b ExtraTrees candidate ranker. Centre methods C1/C4/C5/C6/C7 × size baselines S2/S3/S4.
Diagnostic only — no method recommendation, no hyperparameter tuning.

## Metric definitions

| Name | Definition |
|------|------------|
| **pooled micro** | sum(successful GT detections) / sum(all GT detections) across all folds |
| **sequence macro** | IoU≥0.5 rate computed independently per complete sequence, then unweighted mean over sequences |
| **fold mean** | unweighted mean of per-fold IoU≥0.5 percentages (`fold_mean_iou50_pct`); not a micro metric |
| **n_gt** | total GT detection count (not mean fold count) |

CSVs: `docs/runs/2026-08-30/geometry_residual/`

## Part 1 — Oracle bottleneck (A4 path: C1 + S2)

Proposal/ranking misses are preserved (IoU=0 for all oracle modes when the selected candidate is not GT-compatible). Oracle boxes isolate localization/size only after a target-compatible candidate is selected.

### Pooled

| mode | pooled micro IoU≥0.5 % | mean IoU | n_gt |
|------|-------------------------:|---------:|-----:|
| ACTUAL | 40.485 | 0.4326 | 15292 |
| ORACLE_SIZE | 56.637 | 0.5217 | 15292 |
| ORACLE_CENTRE | 61.457 | 0.5341 | 15292 |
| ORACLE_BOTH | 89.132 | 0.8913 | 15292 |

By-sensor and by-sequence oracle tables: `oracle_decomposition.csv` (scope=sensor / scope=sequence).

## Part 2 — Fixed centre refinement (size = S2)

| centre | pooled micro % | sequence macro % | fold mean % | IoU≥0.75 % | mean IoU | median IoU | centre err mean | centre err median | centre err p90 | p50 ms | p95 ms | p99 ms |
|--------|---------------:|-----------------:|------------:|-----------:|---------:|-----------:|----------------:|------------------:|---------------:|-------:|-------:|-------:|
| C1_CENTROID | 40.485 | 36.268 | 37.661 | 10.731 | 0.4329 | 0.4394 | 27.304 | 2.511 | 54.584 | 10.064 | 16.344 | 26.741 |
| C4_MEDIAN | 43.068 | 37.046 | 39.012 | 10.646 | 0.4441 | 0.4568 | 27.113 | 2.236 | 55.083 | 9.817 | 17.122 | 28.028 |
| C5_SOFT_BACKGROUND_CENTROID | 38.877 | 35.883 | 36.472 | 9.142 | 0.4224 | 0.4282 | 27.452 | 2.803 | 54.562 | 11.068 | 22.150 | 27.511 |
| C6_RIDGE_RESIDUAL | 39.753 | 35.696 | 36.356 | 8.534 | 0.4268 | 0.4395 | 27.615 | 2.644 | 54.344 | 11.272 | 17.663 | 28.208 |
| C7_EXTRATREES_RESIDUAL | 41.133 | 38.279 | 38.380 | 10.581 | 0.4355 | 0.4438 | 27.296 | 2.480 | 54.371 | 14.789 | 22.194 | 27.022 |

By-sensor / by-sequence: `by_sensor.csv`, `by_sequence.csv` (filter `size=S2`).

## Part 3 — Scale-robust size diagnostic (S2 / S3 / S4 × each centre)

| config | pooled micro % | sequence macro % | fold mean % | IoU≥0.75 % | mean IoU | median IoU |
|--------|---------------:|-----------------:|------------:|-----------:|---------:|-----------:|
| C1_CENTROID__S2 | 40.485 | 36.268 | 37.661 | 10.731 | 0.4329 | 0.4394 |
| C1_CENTROID__S3 | 9.835 | 9.291 | 10.231 | 0.654 | 0.2720 | 0.2647 |
| C1_CENTROID__S4 | 32.422 | 25.242 | 26.932 | 7.586 | 0.4002 | 0.4013 |
| C4_MEDIAN__S2 | 43.068 | 37.046 | 39.012 | 10.646 | 0.4441 | 0.4568 |
| C4_MEDIAN__S3 | 11.640 | 9.123 | 10.497 | 0.883 | 0.2785 | 0.2687 |
| C4_MEDIAN__S4 | 36.052 | 28.076 | 30.603 | 7.723 | 0.4144 | 0.4196 |
| C5_SOFT_BACKGROUND_CENTROID__S2 | 38.877 | 35.883 | 36.472 | 9.142 | 0.4224 | 0.4282 |
| C5_SOFT_BACKGROUND_CENTROID__S3 | 8.612 | 8.514 | 9.130 | 0.490 | 0.2660 | 0.2597 |
| C5_SOFT_BACKGROUND_CENTROID__S4 | 30.787 | 24.449 | 25.456 | 6.500 | 0.3905 | 0.3914 |
| C6_RIDGE_RESIDUAL__S2 | 39.753 | 35.696 | 36.356 | 8.534 | 0.4268 | 0.4395 |
| C6_RIDGE_RESIDUAL__S3 | 9.783 | 9.147 | 9.874 | 0.536 | 0.2701 | 0.2647 |
| C6_RIDGE_RESIDUAL__S4 | 32.710 | 26.857 | 27.457 | 6.709 | 0.3991 | 0.4047 |
| C7_EXTRATREES_RESIDUAL__S2 | 41.133 | 38.279 | 38.380 | 10.581 | 0.4355 | 0.4438 |
| C7_EXTRATREES_RESIDUAL__S3 | 10.352 | 9.244 | 10.129 | 0.634 | 0.2741 | 0.2674 |
| C7_EXTRATREES_RESIDUAL__S4 | 33.331 | 26.540 | 28.001 | 7.834 | 0.4048 | 0.4069 |

Full matrix with latency: `matrix_summary.csv`.

## Part 4 — Cross-sensor stress

Top three matrix configs by primary-CV pooled micro, evaluated on the existing three cross-sensor splits (no retuning).

| config | pooled micro % | sequence macro % | fold mean % | n_gt |
|--------|---------------:|-----------------:|------------:|-----:|
| C4_MEDIAN__S2 | 33.312 | 25.914 | 33.451 | 15292 |
| C7_EXTRATREES_RESIDUAL__S2 | 32.278 | 24.800 | 31.668 | 15292 |
| C1_CENTROID__S2 | 32.056 | 24.844 | 32.056 | 15292 |

Per-split latency rows: `cross_sensor_by_fold.csv`.

## Protocol notes

- Ranker: M2b ExtraTrees (`n_estimators=32`, `max_depth=12`, `min_samples_leaf=24`, `random_state=42`, `n_jobs=1`)
- C6: StandardScaler + Ridge(alpha=2.0); C7: ExtraTreesRegressor with the same frozen tree hyperparameters
- S3: local ROI extent (x/y p90−p10, min 1 px); S4: ExtraTrees log-extent residual on TRAIN only
- Training_sets only; Testing_sets not accessed
