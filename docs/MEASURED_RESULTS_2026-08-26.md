# Measured results — 2026-08-26

These values are development measurements on the 17 supplied training sequences. They are not official test-set scores and proposal recall is not AP/F1/IoU.

## Dataset integrity

- 17 / 17 training sequences passed structural checks.
- Event arrays have shape `(N, 6)`.
- Timestamps were monotonic in every checked sequence.
- Labels were binary 0/1.

## B1 raw proposal coverage

- Targets: 15,292
- Top-1 micro / macro: 70.749% / 54.468%
- Top-3 micro / macro: 84.234% / 78.056%
- Top-5 micro / macro: 88.000% / 83.350%
- Top-10 micro / macro: 92.571% / 89.751%
- Top-20 micro / macro: 95.148% / 95.291%
- Mean GT-window pipeline time: 2.4346 ms

The macro Top-20 result is as important as the micro result because it shows the coverage is not solely created by one very long sequence. The principal weak sequences at Top-20 are DAVIS_Filtered_NOAA6 (82.73%), DVX_Filtered_BlockDM (75.10%), DVX_Filtered_ACS3 (75.00%), and DVX_Filtered_Stars (88.48%).

## Stream-path timing spot checks

Scope: timestamp search + memmap slice + raw proposal; initial `np.load` excluded.

| Sequence / sensor | mean ms | p50 ms | p95 ms | p99 ms | mean events/window |
|---|---:|---:|---:|---:|---:|
| DAVIS EGS | 1.1044 | 0.1927 | 6.5967 | 18.4526 | 950.2 |
| DVX NOAA16 | 0.5868 | 0.0691 | 1.8602 | 9.3142 | 513.0 |
| EVK4 mag5.2 | 7.9091 | 3.9166 | 29.6495 | 46.1600 | 8044.6 |

The long-tail timing, especially EVK4 p99, means the proposal kernel itself is not the full latency story. Sequential streaming / page-fault behavior must be profiled before claiming sub-millisecond end-to-end latency.

## Candidate-learning table

- Rows: 305,798
- Positive candidate rows: 59,647
- GT target boxes: 15,292
- Target boxes preserved by Top-20 proposals: 14,550 (95.148%)

The table currently contains GT-positive windows only. It is appropriate for ranking and box-refinement development, but not for final false-positive calibration or official AP/precision.

## Current decision

B1 survives as the high-recall front end. The next experiment is candidate ranking with whole-sequence cross-validation, starting from the simplest models and measuring whether learned evidence closes the Top-20 → Top-1 gap. Motion and richer local representations remain deferred until the residual errors are measured.
