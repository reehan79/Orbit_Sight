# End-to-end positive-window CV — 2026-08-30

Deployable path: deterministic Top-20 → ranker → C1 centre → size baseline. No GT at inference.

Metric definitions:
- **pooled micro** = sum(successful GT) / sum(all GT) across folds
- **sequence macro** = unweighted mean of per-sequence IoU≥0.5 rates
- **fold mean** = unweighted mean of fold IoU≥0.5 percentages (`fold_mean_iou50_pct`); not a micro metric
- **n_gt** = total GT count

## Main CV (ranker × size)

### M1_S0
- pooled micro IoU>=0.5: 13.536% (n_gt=15292)
- sequence macro IoU>=0.5: 5.103% fold mean IoU>=0.5: 10.718%
- fold 0: proposal=98.87% ranker=78.95% pooled50=12.78% seq_macro50=4.92% mean_iou=0.2348 infer_p95=7.495ms
- fold 1: proposal=91.08% ranker=84.15% pooled50=8.96% seq_macro50=4.66% mean_iou=0.1563 infer_p95=22.840ms
- fold 2: proposal=99.87% ranker=98.76% pooled50=18.82% seq_macro50=6.31% mean_iou=0.3459 infer_p95=37.513ms
- fold 3: proposal=100.00% ranker=42.00% pooled50=0.00% seq_macro50=0.00% mean_iou=0.0698 infer_p95=2.923ms
- fold 4: proposal=94.00% ranker=76.66% pooled50=13.03% seq_macro50=9.83% mean_iou=0.2749 infer_p95=7.152ms

### M1_S1
- pooled micro IoU>=0.5: 32.985% (n_gt=15292)
- sequence macro IoU>=0.5: 16.800% fold mean IoU>=0.5: 24.153%
- fold 0: proposal=98.87% ranker=78.95% pooled50=12.03% seq_macro50=7.20% mean_iou=0.2662 infer_p95=8.443ms
- fold 1: proposal=91.08% ranker=84.15% pooled50=9.50% seq_macro50=7.65% mean_iou=0.2183 infer_p95=22.137ms
- fold 2: proposal=99.87% ranker=98.76% pooled50=63.86% seq_macro50=39.87% mean_iou=0.5615 infer_p95=12.794ms
- fold 3: proposal=100.00% ranker=42.00% pooled50=6.00% seq_macro50=7.95% mean_iou=0.1347 infer_p95=5.340ms
- fold 4: proposal=94.00% ranker=76.66% pooled50=29.38% seq_macro50=27.58% mean_iou=0.3561 infer_p95=10.664ms

### M1_S2
- pooled micro IoU>=0.5: 36.817% (n_gt=15292)
- sequence macro IoU>=0.5: 23.503% fold mean IoU>=0.5: 27.921%
- fold 0: proposal=98.87% ranker=78.95% pooled50=11.65% seq_macro50=13.12% mean_iou=0.2747 infer_p95=11.854ms
- fold 1: proposal=91.08% ranker=84.15% pooled50=23.15% seq_macro50=16.82% mean_iou=0.3394 infer_p95=25.025ms
- fold 2: proposal=99.87% ranker=98.76% pooled50=62.12% seq_macro50=50.83% mean_iou=0.5463 infer_p95=47.433ms
- fold 3: proposal=100.00% ranker=42.00% pooled50=10.00% seq_macro50=11.61% mean_iou=0.1583 infer_p95=4.750ms
- fold 4: proposal=94.00% ranker=76.66% pooled50=32.68% seq_macro50=30.82% mean_iou=0.3628 infer_p95=11.319ms

### M2b_S0
- pooled micro IoU>=0.5: 14.204% (n_gt=15292)
- sequence macro IoU>=0.5: 5.495% fold mean IoU>=0.5: 11.419%
- fold 0: proposal=98.87% ranker=81.20% pooled50=12.78% seq_macro50=4.92% mean_iou=0.2402 infer_p95=10.305ms
- fold 1: proposal=91.08% ranker=88.46% pooled50=9.09% seq_macro50=4.73% mean_iou=0.1621 infer_p95=100.273ms
- fold 2: proposal=99.87% ranker=99.33% pooled50=19.32% seq_macro50=6.48% mean_iou=0.3481 infer_p95=10.156ms
- fold 3: proposal=100.00% ranker=68.00% pooled50=2.00% seq_macro50=1.28% mean_iou=0.1165 infer_p95=3.515ms
- fold 4: proposal=94.00% ranker=86.22% pooled50=13.90% seq_macro50=10.51% mean_iou=0.2991 infer_p95=10.036ms

### M2b_S1
- pooled micro IoU>=0.5: 37.503% (n_gt=15292)
- sequence macro IoU>=0.5: 25.920% fold mean IoU>=0.5: 29.623%
- fold 0: proposal=98.87% ranker=81.20% pooled50=15.04% seq_macro50=17.05% mean_iou=0.2858 infer_p95=16.785ms
- fold 1: proposal=91.08% ranker=88.46% pooled50=9.50% seq_macro50=10.45% mean_iou=0.2282 infer_p95=105.605ms
- fold 2: proposal=99.87% ranker=99.33% pooled50=62.62% seq_macro50=47.30% mean_iou=0.5605 infer_p95=12.337ms
- fold 3: proposal=100.00% ranker=68.00% pooled50=24.00% seq_macro50=25.93% mean_iou=0.2779 infer_p95=4.406ms
- fold 4: proposal=94.00% ranker=86.22% pooled50=36.95% seq_macro50=36.98% mean_iou=0.4136 infer_p95=13.971ms

### M2b_S2
- pooled micro IoU>=0.5: 40.485% (n_gt=15292)
- sequence macro IoU>=0.5: 36.268% fold mean IoU>=0.5: 37.661%
- fold 0: proposal=98.87% ranker=81.20% pooled50=17.67% seq_macro50=26.27% mean_iou=0.3015 infer_p95=15.195ms
- fold 1: proposal=91.08% ranker=88.46% pooled50=26.72% seq_macro50=26.68% mean_iou=0.3677 infer_p95=100.274ms
- fold 2: proposal=99.87% ranker=99.33% pooled50=62.75% seq_macro50=55.19% mean_iou=0.5498 infer_p95=18.969ms
- fold 3: proposal=100.00% ranker=68.00% pooled50=44.00% seq_macro50=41.90% mean_iou=0.3484 infer_p95=5.289ms
- fold 4: proposal=94.00% ranker=86.22% pooled50=37.17% seq_macro50=37.83% mean_iou=0.4145 infer_p95=14.921ms

## Ablation

| config | pooled micro IoU>=0.5 % | sequence macro IoU>=0.5 % | fold mean IoU>=0.5 % | n_gt | mean IoU | ranker hit % |
|--------|-------------------------:|---------------------------:|---------------------:|-----:|---------:|-------------:|
| A0 | 6.069 | 2.741 | 5.751 | 15292 | 0.2059 | 66.12 |
| A1 | 3.721 | 1.950 | 3.718 | 15292 | 0.1925 | 84.64 |
| A2 | 14.204 | 5.495 | 11.419 | 15292 | 0.2860 | 84.64 |
| A2_M1 | 13.536 | 5.103 | 10.718 | 15292 | 0.2695 | 76.10 |
| A3 | 37.503 | 25.920 | 29.623 | 15292 | 0.4120 | 84.64 |
| A4 | 40.485 | 36.268 | 37.661 | 15292 | 0.4329 | 84.64 |
| A4_M1 | 36.817 | 23.503 | 27.921 | 15292 | 0.3948 | 76.10 |

## Failure buckets (config=A4)

- box_size_error: 2640 (17.26% of all GT, 29.01% of failures)
- centre_error_too_large: 4799 (31.38% of all GT, 52.73% of failures)
- proposal_miss: 787 (5.15% of all GT, 8.65% of failures)
- ranking_error: 875 (5.72% of all GT, 9.61% of failures)
- success_iou50: 6191 (40.49% of all GT, 0.00% of failures)
