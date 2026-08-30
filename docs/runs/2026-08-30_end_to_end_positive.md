# End-to-end positive-window CV — 2026-08-30

Deployable path: deterministic Top-20 → ranker → C1 centre → size baseline. No GT at inference.

## Main CV (ranker × size)

### M1_S0
- cross-fold mean micro IoU>=0.5: 10.718%
- fold 0: proposal=98.87% ranker=78.95% micro50=12.78% macro50=4.92% mean_iou=0.2348 infer_p95=8.986ms
- fold 1: proposal=91.08% ranker=84.15% micro50=8.96% macro50=4.66% mean_iou=0.1563 infer_p95=22.707ms
- fold 2: proposal=99.87% ranker=98.76% micro50=18.82% macro50=6.31% mean_iou=0.3459 infer_p95=3.273ms
- fold 3: proposal=100.00% ranker=42.00% micro50=0.00% macro50=0.00% mean_iou=0.0698 infer_p95=3.674ms
- fold 4: proposal=94.00% ranker=76.66% micro50=13.03% macro50=9.83% mean_iou=0.2749 infer_p95=23.531ms

### M1_S1
- cross-fold mean micro IoU>=0.5: 24.153%
- fold 0: proposal=98.87% ranker=78.95% micro50=12.03% macro50=7.20% mean_iou=0.2662 infer_p95=9.254ms
- fold 1: proposal=91.08% ranker=84.15% micro50=9.50% macro50=7.65% mean_iou=0.2183 infer_p95=27.428ms
- fold 2: proposal=99.87% ranker=98.76% micro50=63.86% macro50=39.87% mean_iou=0.5615 infer_p95=8.072ms
- fold 3: proposal=100.00% ranker=42.00% micro50=6.00% macro50=7.95% mean_iou=0.1347 infer_p95=4.251ms
- fold 4: proposal=94.00% ranker=76.66% micro50=29.38% macro50=27.58% mean_iou=0.3561 infer_p95=10.083ms

### M1_S2
- cross-fold mean micro IoU>=0.5: 27.921%
- fold 0: proposal=98.87% ranker=78.95% micro50=11.65% macro50=13.12% mean_iou=0.2747 infer_p95=12.239ms
- fold 1: proposal=91.08% ranker=84.15% micro50=23.15% macro50=16.82% mean_iou=0.3394 infer_p95=25.012ms
- fold 2: proposal=99.87% ranker=98.76% micro50=62.12% macro50=50.83% mean_iou=0.5463 infer_p95=11.056ms
- fold 3: proposal=100.00% ranker=42.00% micro50=10.00% macro50=11.61% mean_iou=0.1583 infer_p95=5.742ms
- fold 4: proposal=94.00% ranker=76.66% micro50=32.68% macro50=30.82% mean_iou=0.3628 infer_p95=12.599ms

### M2b_S0
- cross-fold mean micro IoU>=0.5: 11.419%
- fold 0: proposal=98.87% ranker=81.20% micro50=12.78% macro50=4.92% mean_iou=0.2402 infer_p95=11.310ms
- fold 1: proposal=91.08% ranker=88.46% micro50=9.09% macro50=4.73% mean_iou=0.1621 infer_p95=26.193ms
- fold 2: proposal=99.87% ranker=99.33% micro50=19.32% macro50=6.48% mean_iou=0.3481 infer_p95=10.700ms
- fold 3: proposal=100.00% ranker=68.00% micro50=2.00% macro50=1.28% mean_iou=0.1165 infer_p95=5.891ms
- fold 4: proposal=94.00% ranker=86.22% micro50=13.90% macro50=10.51% mean_iou=0.2991 infer_p95=11.553ms

### M2b_S1
- cross-fold mean micro IoU>=0.5: 29.623%
- fold 0: proposal=98.87% ranker=81.20% micro50=15.04% macro50=17.05% mean_iou=0.2858 infer_p95=12.442ms
- fold 1: proposal=91.08% ranker=88.46% micro50=9.50% macro50=10.45% mean_iou=0.2282 infer_p95=25.103ms
- fold 2: proposal=99.87% ranker=99.33% micro50=62.62% macro50=47.30% mean_iou=0.5605 infer_p95=10.934ms
- fold 3: proposal=100.00% ranker=68.00% micro50=24.00% macro50=25.93% mean_iou=0.2779 infer_p95=5.929ms
- fold 4: proposal=94.00% ranker=86.22% micro50=36.95% macro50=36.98% mean_iou=0.4136 infer_p95=12.026ms

### M2b_S2
- cross-fold mean micro IoU>=0.5: 37.661%
- fold 0: proposal=98.87% ranker=81.20% micro50=17.67% macro50=26.27% mean_iou=0.3015 infer_p95=14.900ms
- fold 1: proposal=91.08% ranker=88.46% micro50=26.72% macro50=26.68% mean_iou=0.3677 infer_p95=32.675ms
- fold 2: proposal=99.87% ranker=99.33% micro50=62.75% macro50=55.19% mean_iou=0.5498 infer_p95=13.912ms
- fold 3: proposal=100.00% ranker=68.00% micro50=44.00% macro50=41.90% mean_iou=0.3484 infer_p95=7.689ms
- fold 4: proposal=94.00% ranker=86.22% micro50=37.17% macro50=37.83% mean_iou=0.4145 infer_p95=12.417ms

## Ablation

| config | micro IoU>=0.5 % | macro IoU>=0.5 % | mean IoU | ranker hit % |
|--------|-----------------:|-----------------:|---------:|-------------:|
| A0 | 5.751 | 2.690 | 0.1633 | 66.12 |
| A1 | 3.718 | 1.883 | 0.1519 | 84.64 |
| A2 | 11.419 | 5.584 | 0.2332 | 84.64 |
| A2_M1 | 10.718 | 5.144 | 0.2163 | 76.10 |
| A3 | 29.623 | 27.543 | 0.3532 | 84.64 |
| A4 | 37.661 | 37.574 | 0.3964 | 84.64 |
| A4_M1 | 27.921 | 24.640 | 0.3363 | 76.10 |

## Failure buckets (config=A4)

- box_size_error: 2640 (17.26%)
- centre_error_too_large: 4969 (32.49%)
- other: 6021 (39.37%)
- proposal_miss: 787 (5.15%)
- ranking_error: 875 (5.72%)
