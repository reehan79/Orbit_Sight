# Challenge-aligned confidence (2026-08-31)

Branch: `experiment/challenge-aligned-confidence`  
Base: `experiment/challenge-metric-baseline`  
Scope: TRAINING_sets sequence CV only (not test-set performance).  
No threshold/architecture tuning on outer validation. No motion rescue. No new CNN.

Artifacts: `docs/runs/2026-08-31/challenge_aligned_confidence/`

---

## PART 0 — PR #9 correctness fixes

### mean_matched_iou
Pooled matched IoU is now TP-weighted:

`sum(seq_mean_iou * seq_tp) / sum(seq_tp)`

`sequence_macro_matched_iou` kept separately (unweighted mean of sequence means).  
Unit test: seq A (1 TP @ 1.0) + seq B (9 TP @ 0.5) → pooled **0.55**, macro **0.75**.

### Latency percentiles
Criteria use **pooled samples** p50/p95/p99 (not mean of fold p95).  
`mean_fold_p95` retained as a separately named diagnostic in the baseline script.

---

## PART 1 — EVK4 GT audit

Sequence: `2025_12_23_21_12_28_EVK4_mag5.2`  
Event span: first=71700, last=83253998. Enumeration: `np.arange(t0, t1, 40000)`.

| gt_index | start_us | end_us | overlaps enum window? | reason |
|---------:|---------:|-------:|:---------------------:|--------|
| 1201 | 83271700 | 83311700 | No | outside_event_span |
| 1202 | 83311700 | 83351700 | No | outside_event_span |

Not an enumeration bug vs TII convention. Documented as out-of-event-range / unassignable GT records.  
CSV: `evk4_gt_audit.csv`

---

## PART 2 — TRUE P1 fast path

Semantics: Top-20 → features → proba all → Top-1 only → geometry **only if** conf ≥ threshold.

| Path | n | p50 ms | p95 ms | p99 ms |
|------|--:|-------:|-------:|-------:|
| OLD_P1_reference | 2500 | 10.05 | 30.47 | 55.07 |
| FAST_P1 | 2500 | 10.23 | 21.09 | 30.75 |

Parity: `parity_failures=0` (integer box rows bit-identical; confidence atol 1e-12).  
Tests: `tests/test_p1_fast_parity.py`

---

## PART 3 — Confidence ceiling (diagnostic)

| Mode | pooled note |
|------|-------------|
| P1_ALWAYS | emit Top-1 every window → very low precision (many empty windows) |
| ORACLE_EMIT | emit Top-1 iff TP@0.5 → precision=1.0; recall = localization ceiling |

Fold-level: `ceiling_summary.csv`.  
ORACLE_EMIT recall ≈ 0.24–0.63 by fold (pooled TP 6641 / GT 15292 ≈ 0.434 if summed).  
This bounds rejection only; GT not used in deployable inference.

---

## PARTS 4–7 — D0 / D1 / D2 / D3 comparison (P1 only)

| Method | Precision | Recall | F1 | AP@0.5 (mean folds) | TP | FP | FN | seq macro F1 | empty FPR |
|--------|----------:|-------:|---:|--------------------:|---:|---:|---:|-------------:|----------:|
| D0 (T0 candidate-F1) | 0.274 | 0.388 | 0.321 | 0.269 | 5936 | 15736 | 9356 | 0.327 | 0.132 |
| D1 (T1 detection-F1) | 0.345 | 0.351 | 0.348 | 0.269 | 5367 | 10189 | 9925 | 0.358 | 0.069 |
| D2 (G1 logistic gate) | **0.572** | 0.319 | **0.409** | 0.269 | 4875 | 3645 | 10417 | **0.390** | **0.010** |
| D3 (G2 ExtraTrees gate) | 0.421 | 0.264 | 0.325 | 0.269 | 4044 | 5560 | 11248 | 0.353 | 0.046 |

**AP ranking unchanged by threshold/gate strategy** (identical AP@0.5 across D0–D3; continuous Top-1 scores retained for AP).

TII official ↔ local evaluator: parity OK on all folds/methods.

Per-fold / per-sequence / per-sensor: `compare_by_fold.csv`, `compare_by_sequence.csv`, `compare_by_sensor.csv`.

---

## PART 8 — Threshold stability

| Method | mean | std | min | max |
|--------|-----:|----:|----:|----:|
| D0_T0 | 0.577 | 0.161 | 0.286 | 0.742 |
| D1_T1 | 0.743 | 0.095 | 0.568 | 0.824 |
| D2_G1 | 0.707 | 0.114 | 0.519 | 0.822 |
| D3_G2 | 0.473 | 0.164 | 0.207 | 0.654 |

Fold thresholds: `threshold_stability.csv`. No method changes from variability.

---

## PART 9 — Latency (TRUE P1 fast; pooled samples folds 1–4)

Fold 0 latency raw JSON was lost after reboot; pooled latency uses folds 1–4 (n=10000 windows / method).

| Method | overall p95 | DAVIS | DVX | EVK4 | LATENCY_A | LATENCY_B |
|--------|------------:|------:|----:|-----:|:---------:|:---------:|
| D1 | 25.48 | 15.39 | 17.39 | 33.72 | **PASS** | **PASS** |
| D2 | 32.23 | 22.33 | 22.17 | 43.07 | **PASS** | FAIL (EVK4) |
| D3 | 38.80 | 29.20 | 30.05 | 51.78 | **PASS** | FAIL (EVK4) |

---

## PART 10 — Fixed decision criteria (report only; no algorithm changes)

Vs D0 means:

| Criterion | D1 | D2 | D3 |
|-----------|:--:|:--:|:--:|
| CONF_A (F1 ≥ D0+0.05) | FAIL (+0.008) | **PASS** (+0.055) | FAIL (−0.005) |
| CONF_B (AP ≥ D0+0.03) | FAIL (0) | FAIL (0) | FAIL (0) |
| CONF_C (empty FPR ≤ 0.05) | FAIL (0.069) | **PASS** (0.010) | **PASS** (0.046) |
| CONF_D (precision ≥ 0.40) | FAIL (0.377) | **PASS** (0.509) | **PASS** (0.476) |
| LATENCY_A | PASS | PASS | PASS |
| LATENCY_B | PASS | FAIL | FAIL |

Full: `decision_criteria.csv`

---

## Criteria checklist

| Item | Result |
|------|--------|
| Pooled mean_matched_iou + unit test | **PASS** |
| Pooled latency percentiles | **PASS** |
| EVK4 2 GT boxes | out-of-range; not enum bug |
| TRUE P1 fast parity | **PASS** |
| TII ↔ local parity | **PASS** |
| AP unchanged by threshold strategy | **PASS** (identical) |
| pytest -q | see commit checklist |

No algorithm changes after seeing results.
