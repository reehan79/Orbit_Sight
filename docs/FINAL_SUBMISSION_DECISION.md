# FINAL SUBMISSION DECISION — Phase 1

Participation name (ChallengeON): **SparseSight-SSA**

Branch: `submission/final-freeze`  
Decision date: 2026-09-09

```
ADMIN_CLARIFICATION_PENDING=True
```

This document records packaging / output-contract decisions only. It does **not**
change scientific model content, threshold, or sealed-holdout results.

---

## Frozen model

| Item | Value |
|------|-------|
| Architecture | D2 logistic challenge-aligned gate + TRUE P1 + C4_MEDIAN + S2 ExtraTrees |
| Threshold | `0.848222565945547` |
| Geometry | Parity-preserving reuse |
| Temporal rescue | Rejected for deployment (not in runtime) |
| Tiny neural refiner | Rejected for deployment |

## Sealed holdout (one run; no post-test tuning)

| Metric | Value |
|--------|------:|
| F1 | 0.5669 |
| Precision | 0.5943 |
| Recall | 0.5420 |
| AP@0.5 | 0.3864 |

Docker complete-path overall p95 (prior release bench): **36.31 ms**

---

## Output naming decision (operational)

**Production default filename:**

`<sequence>_bb_windows_40ms.txt`

**TSV columns (exact order):**

```
window_start_timestamp_us
window_end_timestamp_us
center_x
center_y
width
height
confidence
```

**Not emitted:** `sequence_id`, `class_id`  
**Not emitted:** dual naming variants (`*_pred.txt` alongside)  
**Not emitted by inference:** `Evaluation_Metrics.xlsx` / `Evaluation_Metric.xlsx`

### Why this filename

Administrator clarification on ChallengeON portal packaging is still pending.
Until then, the **organizer-supplied executable evaluator** is the operational
source of truth because:

1. `OrbitSight_DataLoader/evaluate.py` opens predictions using the GT filename
   `<sequence>_bb_windows_40ms.txt`.
2. Empirical organizer-evaluator round-trip **PASSED** with that name.
3. Empirical `<sequence>_pred.txt` **FAILED** with
   `Missing prediction` / `No sequences evaluated`.

README Docker prose that mentions `_pred.txt` remains an unresolved documentation
conflict (`ADMIN_CLARIFICATION_PENDING=True`). That ambiguity is **packaging /
portal-serialization only** — it does **not** scientifically affect the frozen
detector, threshold, or sealed-holdout metrics.

### Why no evaluation spreadsheet from inference

Production inference must stay **GT-independent** (events `x,y,polarity,t` only).
The supplied `evaluate.py` itself generates the evaluation workbook and requires GT.
Therefore the container does **not** write `Evaluation_Metrics.xlsx` at inference
time. If administrators later require an entrant-shipped sheet, only serialization /
packaging may change — never detection logic.

---

## Team / output root

Default team folder:

`/work/SparseSight-SSA/<DDMMYYYY>/`

Override via `ORBITSIGHT_TEAM_NAME` or `--team` if needed.

---

## Related docs

- `docs/FINAL_OUTPUT_CONTRACT_AUDIT.md`
- `docs/FINAL_MODEL_FREEZE.md`
- `docs/runs/2026-09-08_FINAL_SEALED_TEST.md`
- `docs/SUBMISSION_RUNTIME.md`
- `docs/OPEN_SOURCE_DISCLOSURE.md`
