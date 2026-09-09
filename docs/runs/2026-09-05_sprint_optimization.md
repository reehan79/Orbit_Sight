# Sprint optimization (2026-09-05)

Branch: `sprint/final-optimization`  
Base: `experiment/challenge-aligned-confidence`  
Champion: D2 logistic gate + TRUE P1 + C4 + S2

---

## PART A — Security / reproducibility

Searched tracked repository text files for `sk-`, `api_key`, `apikey`, `token=`.

**Result:** no matches in tracked files. No suspicious credential content reported.

---

## PART B — D2_GATE_SCORE vs D2_BASE_SCORE — STOPPED

### Required artifacts (not found)

To evaluate D2_GATE_SCORE AP without multi-hour re-inference, the following would be needed **per outer fold**:

1. Unthresholded P1 Top-1 boxes for every validation window: `(ws, we, cx, cy, w, h)`
2. Paired continuous scores on those same boxes:
   - `base_confidence` (ExtraTrees candidate proba)
   - `gate_probability` (D2 logistic gate proba)
3. Existing D2 train-OOF threshold (`threshold_stability.csv` has this)

### What exists today

| Artifact | Present? | Notes |
|----------|:--------:|-------|
| `tii_fold{N}_D2.xlsx` | Yes | **Thresholded** D2 preds; confidence column is already gate score (from emit path) |
| `compare_by_fold.csv` AP for D2 | Yes | Computed with **base** confidence on **unthresholded** Top-1 set |
| Unthresholded window cache with base+gate scores | **No** | Never persisted |
| Fitted D2 gate model pickles / OOF gate score vectors | **No** | Never persisted |
| Gate feature matrices for val windows | **No** | Never persisted |

### Action

**STOP Part B** as instructed — do not launch multi-hour rebuild.

**Known from prior FULL CV (unchanged):**

- D2 thresholded F1/TP/FP/FN use gate score for emit/reject
- D2 reported AP@0.5 used base candidate confidence ranking (identical AP across D0–D3)
- Therefore D2_BASE_SCORE AP is already documented in `docs/runs/2026-08-31/challenge_aligned_confidence/`
- D2_GATE_SCORE AP cannot be computed from persisted artifacts alone

To complete Part B later: persist per-fold NPZ/JSON of unthresholded Top-1 rows with both scores during the next FULL MODE run (no model changes).

---

## PART C — D2 latency deduplication

### Duplication found

Deployable D2 path previously:

1. `run_p1_window_fast(..., always_emit=True)` → computes C4, C1, local18, S2 `log_w/log_h`
2. `build_gate_features(...)` → **recomputed** the same geometry via `classical_box_from_candidate`

Also reused without recompute already: selected feat15 row, candidate confidence / top2/top3 / mean/std / rank fraction (from stored `probs` / `features`).

### Fix

Cache a `GeometryBundle` on `P1WindowResult` and reuse it in `build_gate_features` (`reuse_geometry=True` default).  
Legacy path kept: `reuse_geometry=False` forces fresh recompute for OLD vs NEW benchmarks.

### Parity

Strict checks on representative training windows (500 DAVIS + 500 DVX + 500 EVK4):

`parity_failures=0` (`d2_dedup_parity.txt`)

| sensor | path | n | p50 ms | p95 ms | p99 ms |
|--------|------|--:|-------:|-------:|-------:|
| DAVIS | OLD_recompute | 500 | 83.50 | 280.33 | 613.49 |
| DAVIS | NEW_reuse | 500 | 100.28 | 141.26 | 176.61 |
| DVX | OLD_recompute | 500 | 101.55 | 142.04 | 183.76 |
| DVX | NEW_reuse | 500 | 59.99 | 85.04 | 106.98 |
| EVK4 | OLD_recompute | 500 | 10.84 | 22.80 | 29.17 |
| EVK4 | NEW_reuse | 500 | 7.28 | 16.58 | 18.11 |

CSV: `docs/runs/2026-09-05/sprint_optimization/d2_dedup_latency.csv`  
Note: absolute ms levels depend on concurrent load; relative OLD vs NEW on the same machine is the comparison. No 5-fold CV rerun.

---

## PART D — Sprint cache / resume

- Added `orbitsight.sprint.parse_fold_ids` / atomic JSON+text writers
- `cv_challenge_aligned_confidence.py`: `--fold-ids`, `--resume` / `--no-resume` (already had fold checkpoints)
- `cv_challenge_metric_baseline.py`: `--fold-ids`, `--resume` / `--no-resume`, skip completed folds
- Unit tests: `tests/test_sprint_checkpoint.py`
- Guide: `docs/SPRINT_EXECUTION.md`

Did **not** rerun any FULL CV merely to exercise resume.

---

## Criteria / checklist

| Item | Result |
|------|--------|
| Secrets scan | PASS (no matches) |
| D2_GATE_SCORE eval | **STOPPED** — missing unthresholded gate-score cache |
| Geometry reuse parity | see Part C outputs |
| Sprint resume helpers + docs | PASS |
| pytest -q | see commit |

No motion rescue. No Testing_sets. No new architecture. No full 5-fold detector experiment.
