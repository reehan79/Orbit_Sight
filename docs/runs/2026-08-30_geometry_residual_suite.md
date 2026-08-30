# Geometry residual suite — 2026-08-30

Frozen M2b ranker. Centre methods C1/C4/C5/C6/C7 × size baselines S2/S3/S4.
Metrics: **pooled micro** = sum(success)/sum(GT); **sequence macro** = unweighted mean of per-sequence IoU>=0.5; **fold mean** = unweighted mean of fold percentages.

## Part 2 — Centre methods (S2 size)

| centre | pooled micro IoU>=0.5 % | sequence macro % | fold mean % | n_gt | mean IoU | centre err mean | p95 ms |
|--------|-------------------------:|-----------------:|------------:|-----:|---------:|----------------:|-------:|
| C1_CENTROID | 40.485 | 36.268 | 37.661 | 15292 | 0.4329 | 27.304 | 26.192 |
| C4_MEDIAN | 43.121 | 37.645 | 39.487 | 15292 | 0.4444 | 27.113 | 78.049 |
| C5_SOFT_BACKGROUND_CENTROID | 38.844 | 35.893 | 36.455 | 15292 | 0.4224 | 27.452 | 89.013 |
| C6_RIDGE_RESIDUAL | 39.642 | 35.632 | 36.196 | 15292 | 0.4267 | 27.615 | 22.054 |
| C7_EXTRATREES_RESIDUAL | 41.106 | 38.252 | 38.297 | 15292 | 0.4358 | 27.296 | 54.563 |

## Part 3 — Centre × size matrix

S3 is centre-dependent (local extent at the output box centre). S2/S4 size features always use C1 centre at train and inference (single S2/S4 regressor per fold, not retrained per centre).

| config | pooled micro IoU>=0.5 % | sequence macro % | fold mean % | IoU>=0.75 % | median IoU |
|--------|-------------------------:|-----------------:|------------:|------------:|-----------:|
| C1_CENTROID__S2 | 40.485 | 36.268 | 37.661 | 10.731 | 0.4394 |
| C1_CENTROID__S3 | 9.835 | 9.291 | 10.231 | 0.654 | 0.2647 |
| C1_CENTROID__S4 | 32.422 | 25.242 | 26.932 | 7.586 | 0.4013 |
| C4_MEDIAN__S2 | 43.121 | 37.645 | 39.487 | 10.685 | 0.4580 |
| C4_MEDIAN__S3 | 11.640 | 9.123 | 10.497 | 0.883 | 0.2687 |
| C4_MEDIAN__S4 | 36.176 | 27.541 | 30.857 | 7.618 | 0.4215 |
| C5_SOFT_BACKGROUND_CENTROID__S2 | 38.844 | 35.893 | 36.455 | 9.142 | 0.4287 |
| C5_SOFT_BACKGROUND_CENTROID__S3 | 8.612 | 8.514 | 9.130 | 0.490 | 0.2597 |
| C5_SOFT_BACKGROUND_CENTROID__S4 | 30.702 | 24.717 | 25.671 | 6.507 | 0.3919 |
| C6_RIDGE_RESIDUAL__S2 | 39.642 | 35.632 | 36.196 | 8.547 | 0.4396 |
| C6_RIDGE_RESIDUAL__S3 | 9.783 | 9.147 | 9.874 | 0.536 | 0.2647 |
| C6_RIDGE_RESIDUAL__S4 | 32.344 | 26.119 | 27.009 | 6.683 | 0.4046 |
| C7_EXTRATREES_RESIDUAL__S2 | 41.106 | 38.252 | 38.297 | 10.574 | 0.4445 |
| C7_EXTRATREES_RESIDUAL__S3 | 10.352 | 9.244 | 10.129 | 0.634 | 0.2674 |
| C7_EXTRATREES_RESIDUAL__S4 | 33.292 | 27.048 | 28.516 | 7.873 | 0.4067 |

## Part 1 — Oracle bottleneck (A4 path: C1 + S2)

- ACTUAL: pooled micro IoU>=0.5=40.485% mean IoU=0.4326 (n_gt=15292)
- ORACLE_BOTH: pooled micro IoU>=0.5=89.132% mean IoU=0.8913 (n_gt=15292)
- ORACLE_CENTRE: pooled micro IoU>=0.5=61.457% mean IoU=0.5341 (n_gt=15292)
- ORACLE_SIZE: pooled micro IoU>=0.5=56.637% mean IoU=0.5217 (n_gt=15292)

## Part 4 — Cross-sensor stress (predefined configs)

Fixed configs (no primary-CV selection): C1_CENTROID__S2, C4_MEDIAN__S2, C7_EXTRATREES_RESIDUAL__S2.

- C1_CENTROID__S2: pooled micro IoU>=0.5=32.056% sequence macro=24.844% n_gt=15292
- C4_MEDIAN__S2: pooled micro IoU>=0.5=33.338% sequence macro=25.894% n_gt=15292
- C7_EXTRATREES_RESIDUAL__S2: pooled micro IoU>=0.5=32.357% sequence macro=25.274% n_gt=15292

## Methodology notes

- S2/S4 size features always extracted around C1 centre (train and inference).
- S3 local extent is centre-dependent (uses the output box centre ROI).
- Residual training assigns one GT per candidate: nearest compatible by Euclidean centre distance.

CSVs: `docs\runs\2026-08-30\geometry_residual/`

