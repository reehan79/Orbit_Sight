# Local foveation diagnostic — 2026-08-30

Training data only. GT used for evaluation only, not inside estimators.

Timing note: `combined_localization_block_*` columns measure the full C0–C3 `estimate_all()` block once per GT-compatible window, not per-method latency.

## By sensor

| sensor | method | n | centre err mean | median | p90 | IoU oracle mean | IoU>=0.5 % | combined block p50 ms | p95 ms |
|--------|--------|---:|----------------:|-------:|----:|----------------:|------------:|---------------:|-------:|
| DAVIS | C0_grid | 9936 | 3.786 | 3.606 | 6.083 | 0.4138 | 40.29 | 0.0908 | 0.2557 |
| DAVIS | C1_centroid | 9936 | 2.455 | 1.891 | 4.818 | 0.5579 | 57.34 | 0.0908 | 0.2557 |
| DAVIS | C2_recent_centroid | 9936 | 2.471 | 1.901 | 4.864 | 0.5556 | 57.32 | 0.0908 | 0.2557 |
| DAVIS | C3_linear_motion | 9936 | 2.571 | 1.956 | 4.927 | 0.5510 | 56.57 | 0.0908 | 0.2557 |
| DVX | C0_grid | 3368 | 5.401 | 5.000 | 8.062 | 0.3776 | 27.23 | 0.2097 | 0.3503 |
| DVX | C1_centroid | 3368 | 3.220 | 2.404 | 5.714 | 0.5782 | 65.50 | 0.2097 | 0.3503 |
| DVX | C2_recent_centroid | 3368 | 3.221 | 2.434 | 5.956 | 0.5764 | 64.70 | 0.2097 | 0.3503 |
| DVX | C3_linear_motion | 3368 | 3.686 | 2.486 | 6.000 | 0.5672 | 63.78 | 0.2097 | 0.3503 |
| EVK4 | C0_grid | 1201 | 10.948 | 9.487 | 18.439 | 0.6513 | 83.35 | 0.2836 | 1.5739 |
| EVK4 | C1_centroid | 1201 | 7.150 | 5.320 | 12.797 | 0.7614 | 93.92 | 0.2836 | 1.5739 |
| EVK4 | C2_recent_centroid | 1201 | 7.058 | 5.354 | 12.894 | 0.7610 | 94.09 | 0.2836 | 1.5739 |
| EVK4 | C3_linear_motion | 1201 | 10.418 | 5.328 | 13.484 | 0.7514 | 92.92 | 0.2836 | 1.5739 |

## By sequence and method

- `2025_12_23_21_12_28_EVK4_mag5.2` C0_grid: err_mean=10.948, iou50=83.35%, combined_block_p95=1.5739 ms
- `2025_12_23_21_12_28_EVK4_mag5.2` C1_centroid: err_mean=7.150, iou50=93.92%, combined_block_p95=1.5739 ms
- `2025_12_23_21_12_28_EVK4_mag5.2` C2_recent_centroid: err_mean=7.058, iou50=94.09%, combined_block_p95=1.5739 ms
- `2025_12_23_21_12_28_EVK4_mag5.2` C3_linear_motion: err_mean=10.418, iou50=92.92%, combined_block_p95=1.5739 ms
- `DAVIS_COSMOS1933_18958_2024-12-04-18-37-01` C0_grid: err_mean=11.651, iou50=25.58%, combined_block_p95=0.1527 ms
- `DAVIS_COSMOS1933_18958_2024-12-04-18-37-01` C1_centroid: err_mean=10.305, iou50=48.84%, combined_block_p95=0.1527 ms
- `DAVIS_COSMOS1933_18958_2024-12-04-18-37-01` C2_recent_centroid: err_mean=9.361, iou50=55.81%, combined_block_p95=0.1527 ms
- `DAVIS_COSMOS1933_18958_2024-12-04-18-37-01` C3_linear_motion: err_mean=13.353, iou50=23.26%, combined_block_p95=0.1527 ms
- `DAVIS_EGS_16908_2024-11-01-19-10-44` C0_grid: err_mean=3.213, iou50=62.76%, combined_block_p95=0.1720 ms
- `DAVIS_EGS_16908_2024-11-01-19-10-44` C1_centroid: err_mean=1.398, iou50=89.54%, combined_block_p95=0.1720 ms
- `DAVIS_EGS_16908_2024-11-01-19-10-44` C2_recent_centroid: err_mean=1.428, iou50=89.54%, combined_block_p95=0.1720 ms
- `DAVIS_EGS_16908_2024-11-01-19-10-44` C3_linear_motion: err_mean=1.499, iou50=88.20%, combined_block_p95=0.1720 ms
- `DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06` C0_grid: err_mean=4.393, iou50=16.51%, combined_block_p95=0.1280 ms
- `DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06` C1_centroid: err_mean=3.469, iou50=27.72%, combined_block_p95=0.1280 ms
- `DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06` C2_recent_centroid: err_mean=3.592, iou50=26.35%, combined_block_p95=0.1280 ms
- `DAVIS_Filtered_NOAA6_11416_2025-01-13-19-51-06` C3_linear_motion: err_mean=3.758, iou50=27.30%, combined_block_p95=0.1280 ms
- `DAVIS_RESURSDK1_29228_2024-12-04-18-37-01` C0_grid: err_mean=11.215, iou50=30.43%, combined_block_p95=0.2881 ms
- `DAVIS_RESURSDK1_29228_2024-12-04-18-37-01` C1_centroid: err_mean=11.494, iou50=30.43%, combined_block_p95=0.2881 ms
- `DAVIS_RESURSDK1_29228_2024-12-04-18-37-01` C2_recent_centroid: err_mean=10.713, iou50=34.78%, combined_block_p95=0.2881 ms
- `DAVIS_RESURSDK1_29228_2024-12-04-18-37-01` C3_linear_motion: err_mean=19.069, iou50=21.74%, combined_block_p95=0.2881 ms
- `DAVIS_SL12RB2_15772_2024-12-04-18-21-37` C0_grid: err_mean=12.392, iou50=12.50%, combined_block_p95=0.2107 ms
- `DAVIS_SL12RB2_15772_2024-12-04-18-21-37` C1_centroid: err_mean=13.223, iou50=25.00%, combined_block_p95=0.2107 ms
- `DAVIS_SL12RB2_15772_2024-12-04-18-21-37` C2_recent_centroid: err_mean=12.218, iou50=25.00%, combined_block_p95=0.2107 ms
- `DAVIS_SL12RB2_15772_2024-12-04-18-21-37` C3_linear_motion: err_mean=16.688, iou50=12.50%, combined_block_p95=0.2107 ms
- `DAVIS_SL16RB_20625_2024-12-04-19-34-18` C0_grid: err_mean=4.883, iou50=27.55%, combined_block_p95=0.1431 ms
- `DAVIS_SL16RB_20625_2024-12-04-19-34-18` C1_centroid: err_mean=4.882, iou50=24.49%, combined_block_p95=0.1431 ms
- `DAVIS_SL16RB_20625_2024-12-04-19-34-18` C2_recent_centroid: err_mean=4.637, iou50=28.06%, combined_block_p95=0.1431 ms
- `DAVIS_SL16RB_20625_2024-12-04-19-34-18` C3_linear_motion: err_mean=5.234, iou50=22.96%, combined_block_p95=0.1431 ms
- `DAVIS_SL16RB_26070_2024-12-04-19-14-39` C0_grid: err_mean=12.661, iou50=20.00%, combined_block_p95=0.1365 ms
- `DAVIS_SL16RB_26070_2024-12-04-19-14-39` C1_centroid: err_mean=11.933, iou50=40.00%, combined_block_p95=0.1365 ms
- `DAVIS_SL16RB_26070_2024-12-04-19-14-39` C2_recent_centroid: err_mean=11.275, iou50=40.00%, combined_block_p95=0.1365 ms
- `DAVIS_SL16RB_26070_2024-12-04-19-14-39` C3_linear_motion: err_mean=17.966, iou50=30.00%, combined_block_p95=0.1365 ms
- `DAVIS_SL8RB_2025-01-13-19-15-36` C0_grid: err_mean=3.848, iou50=32.36%, combined_block_p95=0.2910 ms
- `DAVIS_SL8RB_2025-01-13-19-15-36` C1_centroid: err_mean=2.662, iou50=45.65%, combined_block_p95=0.2910 ms
- `DAVIS_SL8RB_2025-01-13-19-15-36` C2_recent_centroid: err_mean=2.674, iou50=45.65%, combined_block_p95=0.2910 ms
- `DAVIS_SL8RB_2025-01-13-19-15-36` C3_linear_motion: err_mean=2.680, iou50=45.43%, combined_block_p95=0.2910 ms
- `DVX_Filtered_ACS3_59588_2025-01-20-19-35-44` C0_grid: err_mean=13.529, iou50=40.00%, combined_block_p95=0.3298 ms
- `DVX_Filtered_ACS3_59588_2025-01-20-19-35-44` C1_centroid: err_mean=14.588, iou50=20.00%, combined_block_p95=0.3298 ms
- `DVX_Filtered_ACS3_59588_2025-01-20-19-35-44` C2_recent_centroid: err_mean=12.225, iou50=30.00%, combined_block_p95=0.3298 ms
- `DVX_Filtered_ACS3_59588_2025-01-20-19-35-44` C3_linear_motion: err_mean=17.376, iou50=20.00%, combined_block_p95=0.3298 ms
- `DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17` C0_grid: err_mean=4.930, iou50=33.43%, combined_block_p95=0.3203 ms
- `DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17` C1_centroid: err_mean=2.777, iou50=74.22%, combined_block_p95=0.3203 ms
- `DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17` C2_recent_centroid: err_mean=2.886, iou50=73.94%, combined_block_p95=0.3203 ms
- `DVX_Filtered_BlockDM_SLRB_32405_2025-01-20-19-57-17` C3_linear_motion: err_mean=2.709, iou50=75.92%, combined_block_p95=0.3203 ms
- `DVX_Filtered_NOAA15_25338_2025-01-20-19-25-07` C0_grid: err_mean=14.284, iou50=42.31%, combined_block_p95=0.4339 ms
- `DVX_Filtered_NOAA15_25338_2025-01-20-19-25-07` C1_centroid: err_mean=16.697, iou50=26.92%, combined_block_p95=0.4339 ms
- `DVX_Filtered_NOAA15_25338_2025-01-20-19-25-07` C2_recent_centroid: err_mean=14.647, iou50=42.31%, combined_block_p95=0.4339 ms
- `DVX_Filtered_NOAA15_25338_2025-01-20-19-25-07` C3_linear_motion: err_mean=29.733, iou50=15.38%, combined_block_p95=0.4339 ms
- `DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50` C0_grid: err_mean=15.631, iou50=32.35%, combined_block_p95=0.4989 ms
- `DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50` C1_centroid: err_mean=17.883, iou50=23.53%, combined_block_p95=0.4989 ms
- `DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50` C2_recent_centroid: err_mean=15.641, iou50=23.53%, combined_block_p95=0.4989 ms
- `DVX_Filtered_NOAA16_26536_2025-01-20-19-46-50` C3_linear_motion: err_mean=38.924, iou50=8.82%, combined_block_p95=0.4989 ms
- `DVX_Filtered_NOAA6_11416_2025-01-20-19-11-35` C0_grid: err_mean=14.794, iou50=50.00%, combined_block_p95=0.4820 ms
- `DVX_Filtered_NOAA6_11416_2025-01-20-19-11-35` C1_centroid: err_mean=14.733, iou50=28.57%, combined_block_p95=0.4820 ms
- `DVX_Filtered_NOAA6_11416_2025-01-20-19-11-35` C2_recent_centroid: err_mean=13.650, iou50=50.00%, combined_block_p95=0.4820 ms
- `DVX_Filtered_NOAA6_11416_2025-01-20-19-11-35` C3_linear_motion: err_mean=29.265, iou50=7.14%, combined_block_p95=0.4820 ms
- `DVX_Filtered_Stars2_2025-01-20-19-57-17` C0_grid: err_mean=7.132, iou50=66.67%, combined_block_p95=0.3773 ms
- `DVX_Filtered_Stars2_2025-01-20-19-57-17` C1_centroid: err_mean=6.231, iou50=100.00%, combined_block_p95=0.3773 ms
- `DVX_Filtered_Stars2_2025-01-20-19-57-17` C2_recent_centroid: err_mean=4.362, iou50=88.89%, combined_block_p95=0.3773 ms
- `DVX_Filtered_Stars2_2025-01-20-19-57-17` C3_linear_motion: err_mean=6.778, iou50=88.89%, combined_block_p95=0.3773 ms
- `DVX_Filtered_Stars_2025-01-20-19-15-10` C0_grid: err_mean=5.165, iou50=25.99%, combined_block_p95=0.3439 ms
- `DVX_Filtered_Stars_2025-01-20-19-15-10` C1_centroid: err_mean=2.857, iou50=65.60%, combined_block_p95=0.3439 ms
- `DVX_Filtered_Stars_2025-01-20-19-15-10` C2_recent_centroid: err_mean=2.910, iou50=64.47%, combined_block_p95=0.3439 ms
- `DVX_Filtered_Stars_2025-01-20-19-15-10` C3_linear_motion: err_mean=2.946, iou50=63.82%, combined_block_p95=0.3439 ms
- `DVX_NOAA6_11416_2025-01-20-19-06-31` C0_grid: err_mean=13.230, iou50=33.33%, combined_block_p95=0.4547 ms
- `DVX_NOAA6_11416_2025-01-20-19-06-31` C1_centroid: err_mean=13.403, iou50=16.67%, combined_block_p95=0.4547 ms
- `DVX_NOAA6_11416_2025-01-20-19-06-31` C2_recent_centroid: err_mean=12.735, iou50=16.67%, combined_block_p95=0.4547 ms
- `DVX_NOAA6_11416_2025-01-20-19-06-31` C3_linear_motion: err_mean=20.963, iou50=16.67%, combined_block_p95=0.4547 ms

## Fold variation (validation sequences only)

- fold 0 C0_grid: err_mean=7.155, iou50=28.90%
- fold 0 C1_centroid: err_mean=7.510, iou50=24.71%
- fold 0 C2_recent_centroid: err_mean=6.880, iou50=28.14%
- fold 0 C3_linear_motion: err_mean=11.261, iou50=20.91%
- fold 1 C0_grid: err_mean=8.146, iou50=53.30%
- fold 1 C1_centroid: err_mean=5.644, iou50=64.33%
- fold 1 C2_recent_centroid: err_mean=5.626, iou50=63.96%
- fold 1 C3_linear_motion: err_mean=7.637, iou50=63.10%
- fold 2 C0_grid: err_mean=3.247, iou50=62.64%
- fold 2 C1_centroid: err_mean=1.442, iou50=89.41%
- fold 2 C2_recent_centroid: err_mean=1.463, iou50=89.38%
- fold 2 C3_linear_motion: err_mean=1.552, iou50=88.01%
- fold 3 C0_grid: err_mean=14.102, iou50=40.00%
- fold 3 C1_centroid: err_mean=15.194, iou50=30.00%
- fold 3 C2_recent_centroid: err_mean=13.693, iou50=44.00%
- fold 3 C3_linear_motion: err_mean=27.248, iou50=16.00%
- fold 4 C0_grid: err_mean=4.326, iou50=30.30%
- fold 4 C1_centroid: err_mean=2.731, iou50=53.37%
- fold 4 C2_recent_centroid: err_mean=2.761, iou50=52.99%
- fold 4 C3_linear_motion: err_mean=2.769, iou50=52.71%
