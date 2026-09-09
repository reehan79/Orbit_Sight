# FINAL SEALED HOLDOUT — NO POST-TEST TUNING

Date: 2026-09-08
Model dir: `models/final/` (threshold=0.848222565946)
Model train git SHA (manifest): `382c70ee80975b566bbfab8cc4916567e64e32c1`
Test sequences: 4 (Testing_sets only)

**This is the one and only sealed holdout run. Do not change the model.**

Confidence values in these predictions come from the frozen D2 gate runtime (gate probability), not raw ExtraTrees confidence.

## Overall (local evaluator; TII parity required)

| metric | local | TII |
|--------|------:|----:|
| precision | 0.5943 | 0.5943 |
| recall | 0.5420 | 0.5420 |
| F1 | 0.5669 | 0.5669 |
| AP@0.5 | 0.3864 | 0.3864 |
| TP / FP / FN | 3835 / 2618 / 3241 | — |
| mean matched IoU | 0.6657 | — |

## Per sequence

| sequence | sensor | P | R | F1 | AP@0.5 | TP | FP | FN | mean IoU | p95 ms |
|----------|--------|--:|--:|---:|------:|---:|---:|---:|---------:|-------:|
| 2025_12_23_20_53_46_EVK4_mag7.3 | EVK4 | 0.8817 | 0.8998 | 0.8906 | 0.8689 | 1140 | 153 | 127 | 0.7066 | 39.79 |
| DAVIS_SAOCOM1B_46265_2024-12-04-18-21-37 | DAVIS | 0.6263 | 0.3402 | 0.4409 | 0.2519 | 181 | 108 | 351 | 0.6033 | 29.72 |
| DVX_Filtered_Stars3_2025-01-20-20-22-53 | DVX | 0.5139 | 0.4985 | 0.5061 | 0.3457 | 2492 | 2357 | 2507 | 0.6516 | 24.14 |
| DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43 | DVX | 1.0000 | 0.0791 | 0.1467 | 0.0791 | 22 | 0 | 256 | 0.6573 | 22.12 |

## Per sensor

| sensor | P | R | F1 | TP | FP | FN | p50 ms | p95 ms | p99 ms |
|--------|--:|--:|---:|---:|---:|---:|-------:|-------:|-------:|
| DAVIS | 0.6263 | 0.3402 | 0.4409 | 181 | 108 | 351 | 14.55 | 29.72 | 53.81 |
| DVX | 0.5161 | 0.4764 | 0.4955 | 2514 | 2357 | 2763 | 14.93 | 23.64 | 33.34 |
| EVK4 | 0.8817 | 0.8998 | 0.8906 | 1140 | 153 | 127 | 15.60 | 39.79 | 68.72 |

## Runtime overall

| p50 | p95 | p99 | n |
|----:|----:|----:|--:|
| 14.91 | 25.68 | 42.72 | 37558 |

Artifacts: `docs/runs/2026-09-08/final_sealed_test/`

