# FINAL OUTPUT CONTRACT AUDIT

FINAL_OUTPUT_CONTRACT_STATUS=AMBIGUOUS

Branch: `submission/final-freeze`  
Frozen commit audited: `1921a06`  
Audit date: 2026-09-08  
Frozen threshold unchanged: `0.848222565945547`  
**No production output-format change was made** (ambiguity unresolved).

---

## Executive conflict (why AMBIGUOUS)

Two organizer-supplied sources disagree on the **prediction filename**:

| Source | Filename required |
|--------|-------------------|
| **`OrbitSight_DataLoader/evaluate.py` (executable)** | Prediction file must match GT name exactly: `<sequence>_bb_windows_40ms.txt` |
| **`OrbitSight_DataLoader/README.md` Submission / Docker sections** | `<sequence>_pred.txt` |

Empirical probe (synthetic fixture, this audit):

- `SYNTH_SEQ_bb_windows_40ms.txt` → evaluate.py scores successfully, writes Excel
- `SYNTH_SEQ_pred.txt` → `[WARN] Missing prediction for: SYNTH_SEQ_bb_windows_40ms.txt — skipping` then `[ERROR] No sequences evaluated.`

README also claims `_pred.txt` files are “ready to pass to the evaluation script,” which is **false** for the shipped `evaluate.py` without a rename step.

Priority rule used (job instructions):

1. supplied evaluator/loader code  
2. explicit README  
3. examples  
4. prose  

Executable compatibility therefore favors `_bb_windows_40ms.txt`, but portal packaging prose favors `_pred.txt`. Choosing either silently could cause rejection depending on whether ChallengeON renames files before scoring. **Confirmation from TII/ChallengeON is required.**

A second naming conflict exists for the scoring sheet:

| Source | Name |
|--------|------|
| README Docker “Required files” | `Evaluation_Metrics.xlsx` (plural) |
| `evaluate.py --excel-out` default | `Evaluation_Metric.xlsx` (singular) |

---

## A. Organizer evidence table

| Topic | Evidence | Finding |
|-------|----------|---------|
| Pred discovery | `evaluate.py` L194–204 | Lists GT `*_bb_windows_40ms.txt`; opens **same filename** under `--pred-dir`; missing → WARN skip |
| Pred columns | `evaluate.py` L98–115 `load_pred` | TSV DictReader; keys: `window_start_timestamp_us`, `window_end_timestamp_us`, `center_x`, `center_y`, `width`, `height`; `confidence` optional → default 1.0 |
| GT columns | `evaluate.py` L81–95; GT sample files | Same six fields, no confidence |
| Delimiter | `evaluate.py` L85, L102 | Tab (`delimiter="\t"`) |
| Header required | DictReader | Yes — column names, not positional-only |
| `class_id` | evaluate.py / README / DOCX | **Not present** |
| `sequence_id` in rows | evaluate.py | **Not present**; sequence implied by filename stem |
| Confidence | evaluate.py L104; README L235–248 | Optional for evaluate.py; used for AP ranking when present |
| Extra files in pred_dir | evaluate.py L194–204 | Only opens files whose names match GT list; extras ignored |
| Excel artifact | evaluate.py L30, L412–413, L444; README L375 | Evaluator **generates** Excel; README also lists it as Docker required output |
| Docker mounts | README L345–350 | `/OrbitSight_dataset` RO; `/work/teamName/DDMMYYYY` write |
| Docker pred name | README L357, L363 | `<sequence>_pred.txt` |
| DOCX | `Read First ... June 2026.docx` | Mentions toolkit outputs `<sequence>_pred.txt`; no `/work` / Docker detail |
| Visualizer | `visualize_dataset.py` L78–91 | Loads GT `*_bb_windows_40ms.txt` only |

Authoritative paths:

- `D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_DataLoader\evaluate.py`
- `D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_DataLoader\README.md`
- `D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\Read First (Dataset Description and Contents)_June 2026.docx`

---

## B. Exact final input path

Organizer Docker mount (README):

`/OrbitSight_dataset` (read-only)

Expected contents: Training_sets / Testing_sets event `*.npy` and GT `*.txt`, plus dataloader materials.

Frozen runtime resolves split via `resolve_split_dir()`: prefers Testing_sets when present under the mount root.

---

## C. Exact final output path

Organizer (README):

`/work/<teamName>/<DDMMYYYY>/`

Current runtime:

`/work/<ORBITSIGHT_TEAM_NAME>/<DDMMYYYY>/` (default team `OrbitSight`)

**PASS** for path shape.

---

## D. Exact filename format — CONFLICT

| Authority | Pattern |
|-----------|---------|
| evaluate.py | `<sequence>_bb_windows_40ms.txt` |
| README Docker / Submission format | `<sequence>_pred.txt` |

Current frozen runtime writes: `<sequence>_bb_windows_40ms.txt`  
→ **PASS vs evaluate.py**, **FAIL vs README Docker prose**.

---

## E. Exact header / column schema — AGREED

Tab-separated, header row, order:

1. `window_start_timestamp_us` (int)  
2. `window_end_timestamp_us` (int)  
3. `center_x` (int)  
4. `center_y` (int)  
5. `width` (int)  
6. `height` (int)  
7. `confidence` (float, optional in evaluate.py; current runtime always writes it)

Current `write_tii_prediction_file()` matches this schema. **PASS**.

---

## F. Confidence semantics

- Organizer evaluate.py: optional; if present used to sort for AP (`load_pred` sorts by confidence descending).
- Frozen D2 runtime: writes **D2 gate probability** (threshold `0.848222565945547` for emit/reject).  
**PASS** for evaluate.py optional confidence column.

---

## G. class_id / sequence_id

- **class_id required?** No evidence in evaluate.py, README tables, or DOCX. **Not required.**
- **sequence_id in each row?** No. Sequence identity is the filename stem. **Not required.**

---

## H. Evaluation_Metrics.xlsx conclusion

| Question | Answer |
|----------|--------|
| Who generates Excel in evaluate.py? | The evaluation script (`write_excel`, default `Evaluation_Metric.xlsx`) |
| Is it listed as Docker required output? | README yes: `Evaluation_Metrics.xlsx` |
| Exact spelling | **Conflict**: Metrics vs Metric |
| Entrant must ship it? | README says yes; no portal code available to confirm whether organizers regenerate it |

**AMBIGUOUS** — confirm with TII whether the container must emit the xlsx, and the exact filename.

Current runtime does **not** write Excel. Left unchanged pending confirmation.

---

## I. Current-runtime changes

**None** for production format (job rule: do not change format while AMBIGUOUS).

Audit-only additions: this report + contract/probe tests + Training-only organizer round-trip script.

---

## J. Comparison matrix (frozen runtime vs organizer)

| Requirement | Organizer evidence | Current runtime | Verdict |
|-------------|--------------------|-----------------|:-------:|
| Output root `/work/<team>/<DDMMYYYY>/` | README Docker | `output_dir()` | PASS |
| Pred filename = GT name `_bb_windows_40ms.txt` | evaluate.py L194–200 | `write_sequence_prediction` | PASS (code) |
| Pred filename `_pred.txt` | README Submission/Docker | writes `_bb_windows_40ms.txt` | FAIL (prose) |
| Header 7 fields TSV | evaluate.py `load_pred` + README | `write_tii_prediction_file` | PASS |
| Integer boxes | evaluate.py casts `int(...)` | `int(round(...))` | PASS |
| Confidence optional/present | evaluate.py | always written (gate p) | PASS |
| No class_id | no evidence | not written | PASS |
| No per-row sequence_id | no evidence | not written | PASS |
| Extra files ignored | evaluate.py only opens GT-matched names | only pred txts | PASS vs evaluate.py |
| `Evaluation_Metrics.xlsx` entrant | README required | not written | AMBIGUOUS |
| Automatic exit 0 | README | `__main__` returns 0 | PASS |
| Offline / no train at infer | README | frozen joblibs | PASS |

---

## K. Organizer evaluator round-trip

Script: `scripts/organizer_evaluate_roundtrip.py`  
Uses **one Training_sets sequence copy** (`DVX_NOAA6_11416_2025-01-20-19-06-31`; not Testing_sets).

**Result: PASS (format acceptance)**

- Frozen submission wrote `.../OrbitSight/08092026/<seq>_bb_windows_40ms.txt`
- Organizer `evaluate.py` consumed it with **no** “Missing prediction”
- Wrote `Evaluation_Metric.xlsx`
- (Numeric F1 on this hard sequence may be low; this probe is for **format acceptance**, not score quality.)

---

## L. Docker smoke

Rebuild only if packaging sources change — **not required for format** (no production change).  
Existing `--network none` smoke remains valid for current `_bb_windows_40ms.txt` emitter.

---

## What to confirm with TII / ChallengeON

1. Portal scoring: does the scorer call the shipped `evaluate.py` **as-is** on `/work/...` files?  
   - If **yes** → keep/require `<sequence>_bb_windows_40ms.txt`.  
   - If **no** and portal expects `_pred.txt` → rename adapter only after written confirmation.
2. Must the Docker image emit `Evaluation_Metrics.xlsx` / `Evaluation_Metric.xlsx`? Exact spelling?
3. Should Training predictions also be written, or Testing_sets only under the mount?

Until answered: **do not change** the frozen production filename.

---

## STOP

Science / threshold / sealed holdout untouched. PR not merged.
