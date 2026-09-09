# OFFICIAL OUTPUT CONTRACT V2 — SparseSight-SSA Phase 1

Participation: **SparseSight-SSA**  
Decision date: 2026-09-09  
Branch: `submission/final-freeze`

```
SCIENCE_CHANGED=False
MODEL_CHANGED=False
THRESHOLD_CHANGED=False
SEALED_TEST_RERUN=False
```

Frozen science unchanged:

| Item | Value |
|------|-------|
| Architecture | D2 + TRUE P1 + C4_MEDIAN + S2 ExtraTrees |
| Threshold | `0.848222565945547` |

---

## Organizer clarification (summary)

Organizer confirmed in writing that the submission must use the **public timeline**
output schema, **not** the supplied `evaluate.py` internal
`*_bb_windows_40ms.txt` / `center_x|width|height` schema.

Required detection row fields:

```
sequence_id
window_start_timestamp_us
window_end_timestamp_us
centre_x
centre_y
w
h
class_id
confidence
```

Also confirmed:

- `sequence_id` required
- `class_id` required
- participant **must** generate `Evaluation_Metrics.xlsx`

Filename literal still has residual ambiguity:

| Source | Pattern |
|--------|---------|
| Administrator answer #1 | `<sequence>_pred.txt` |
| Public timeline wording | `<sequencename>.txt` |

**Release-candidate default:** `admin_pred` → `<sequence>_pred.txt`  
Switchable via `ORBITSIGHT_PRED_FILENAME_MODE=public_plain` with **no** inference change.

---

## Serialization adapter (no detection change)

Internal frozen row remains:

`(ws, we, cx, cy, w, h, confidence)`

Public mapping:

| Public field | Source |
|--------------|--------|
| sequence_id | sequence stem |
| window_start_timestamp_us | ws |
| window_end_timestamp_us | we |
| centre_x | round(cx) |
| centre_y | round(cy) |
| w | round(w) |
| h | round(h) |
| class_id | `ORBITSIGHT_CLASS_ID` / default **1** |
| confidence | D2 gate probability (unchanged) |

### class_id default

Dataset documentation states event `label`: **0 = background, 1 = RSO Object**.  
Therefore default `DEFAULT_CLASS_ID=1`. Numeric confirmation of the submission
`class_id` column is still awaiting final admin reply; override with
`ORBITSIGHT_CLASS_ID` / `--class-id` without touching the detector.

### Text format decision

Public timeline reply did not restate delimiter/header. Chosen:

- **TSV** (tab-separated)
- **exact nine-field header row**

Rationale: consistent with OrbitSight_DataLoader README historical submission
tables (tab-separated + header). Documented in
`orbitsight.submission.public_contract.PUBLIC_FORMAT_DECISION`.

---

## Evaluation_Metrics.xlsx

Generated **after** inference from the same frozen detections + GT
`*_bb_windows_40ms.txt` under `/OrbitSight_dataset` when present.

- Filename exact: `Evaluation_Metrics.xlsx`
- Sheets: `Evaluation` (organizer-style PRF1/AP) + `InferenceEfficiency`
- If GT absent: predictions still written; metrics recorded unavailable
  (`Evaluation_Metrics_UNAVAILABLE.txt`) — exit code remains 0
- GT never feeds proposals / ranking / geometry / gate / threshold

---

## Split behavior

`ORBITSIGHT_SPLIT_MODE`:

| Mode | Behavior |
|------|----------|
| `auto` (**default**) | Testing_sets if non-empty; else Training_sets; else direct events |
| `testing` | Testing_sets only |
| `training` | Training_sets only |
| `both` | Training then Testing (parameterized; **not** default until clarified) |

See `SPLIT_BEHAVIOR_DOC` in `orbitsight.submission`.

---

## Packaging knobs (env)

```
ORBITSIGHT_TEAM_NAME=SparseSight-SSA
ORBITSIGHT_PRED_FILENAME_MODE=admin_pred
ORBITSIGHT_CLASS_ID=1
ORBITSIGHT_SPLIT_MODE=auto
```

Docker tag for this contract: `sparsesight-ssa:phase1-final-v2`
