# Sprint execution guide

## Modes

### SCREEN MODE
Small representative-sequence run (typically ≤500 windows / sensor).  
May **reject** an idea early.  
Must **not** promote a method to champion.

### FULL MODE
Frozen full sequence-level training CV (all outer folds).  
**Required** before promotion to champion.  
Use `--fold-ids` / `--resume` to avoid repeating completed folds after interruption.

## Current champion

**D2** logistic challenge-aligned gate + TRUE P1 + C4_MEDIAN + S2 ExtraTrees size.

- Candidate confidence: ExtraTreesClassifier (frozen hyperparams from challenge-metric baseline)
- Detection gate: StandardScaler + LogisticRegression (C=1.0, balanced)
- Geometry: C4 centre, S2 size (C1 used only for S2 local features / gate features)
- Threshold: selected on outer-TRAIN inner-OOF detection F1 only

## Resume / fold selection

Long CV scripts support where practical:

```text
--fold-ids 0,2,4      # or 1-3, or all
--resume              # default: reuse completed fold checkpoints
--no-resume           # ignore checkpoints
```

Each completed fold writes CSV / latency checkpoints under the run `out-dir`.  
Do not rerun FULL MODE merely to test resume — use unit tests (`tests/test_sprint_checkpoint.py`).

## Constraints (sprint)

- Never access Testing_sets
- No new neural architecture
- No motion rescue unless explicitly instructed
- No proposal / geometry definition changes
- No full 5-fold detector experiment unless explicitly instructed
