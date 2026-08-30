# Tiny foveated neural refiner — 2026-08-30

Frozen tiny local neural refiner. Training_sets only. No architecture search, no hyperparameter changes after results.

## Setup

- Parameter count: **44121**
- Training device: cpu (CUDA not available)
- Optimizer: AdamW lr=1e-3, weight decay=1e-4, epochs=12, batch=256, seed=42
- Loss: BCEWithLogits (pos_weight = n_neg/n_pos on TRAIN) + 2.0 × SmoothL1 (positives only)
- Patch cache: 45870 rows, 26496 positives, 14.42 MB (`artifacts/local_patch_cache.npz`, not committed)
- B_CURRENT: C4_MEDIAN + S2 with C1-centred size features
- N1: ExtraTrees Top-1 + neural bbox (classification ignored)
- N2: ExtraTrees Top-3 in one ONNX batch; select max classification logit; use that bbox

## Main comparison (5 frozen sequence folds)

| config | n_gt | Top-20 contains % | ET Top-1 compat % | ET Top-3 contains % | neural-selected % | pooled micro IoU50 % | sequence macro % | fold mean % | IoU75 % | mean IoU | median IoU | centre p50 | centre p90 |
|--------|-----:|------------------:|------------------:|--------------------:|------------------:|---------------------:|-----------------:|------------:|--------:|---------:|-----------:|-----------:|-----------:|
| B_CURRENT | 15292 | 94.854 | 89.132 | 91.518 | 89.132 | 43.121 | 37.645 | 39.487 | 10.685 | 0.4444 | 0.4580 | 2.236 | 55.083 |
| N1_TOP1_BOX | 15292 | 94.854 | 89.132 | 91.518 | 89.132 | 36.248 | 28.942 | 32.535 | 6.984 | 0.4081 | 0.4230 | 2.886 | 63.115 |
| N2_TOP3_JOINT | 15292 | 94.854 | 89.132 | 91.518 | 90.714 | 40.557 | 36.323 | 39.026 | 7.723 | 0.4348 | 0.4484 | 2.688 | 23.966 |

Misses remain IoU=0 (all GT in the denominator).

## CPU deployment latency (ONNX Runtime CPUExecutionProvider)

Complete path: event slice + proposals + numerical features + ExtraTrees + patch rasterization + ONNX + selection + decode. Training time excluded. Mean of per-fold percentiles.

| config | p50 ms | p95 ms | p99 ms |
|--------|-------:|-------:|-------:|
| B_CURRENT | 42.221 | 102.886 | 163.109 |
| N1_TOP1_BOX | 27.490 | 75.796 | 119.178 |
| N2_TOP3_JOINT | 30.275 | 81.805 | 130.104 |

By-sensor latency: `latency_by_sensor.csv`. Observation cadence target: 40 ms (reported, architecture unchanged).

## By sensor / sequence / fold

CSVs: `by_sensor.csv`, `by_sequence.csv`, `by_fold.csv`.

## Cross-sensor stress (N1 and N2)

Same frozen recipe. Splits: DAVIS+DVX→EVK4; DAVIS+EVK4→DVX; DVX+EVK4→DAVIS.

| config | pooled micro IoU50 % | sequence macro % | IoU75 % | CPU p95 ms |
|--------|---------------------:|-----------------:|--------:|-----------:|
| N1_TOP1_BOX | 29.682 | 19.164 | 3.799 | 61.385 |
| N2_TOP3_JOINT | 38.432 | 25.689 | 5.970 | 72.004 |

## Decision gate (report only — no action taken)

| quantity | value |
|----------|------:|
| delta_N1_vs_B_CURRENT pooled IoU50 | −6.873 pp |
| delta_N2_vs_B_CURRENT pooled IoU50 | −2.563 pp |
| delta_N1_vs_B_CURRENT sequence macro | −8.703 pp |
| delta_N2_vs_B_CURRENT sequence macro | −1.322 pp |
| CRITERION_A: N2 pooled IoU50 ≥ B_CURRENT + 8 pp | False |
| CRITERION_B: N2 sequence macro IoU50 ≥ B_CURRENT + 5 pp | False |
| CRITERION_C: N2 CPU p95 ≤ 30 ms | False |

CSVs: `docs/runs/2026-08-30/tiny_foveated_refiner/`
