# OrbitSight submission runtime (SparseSight-SSA)

## Frozen champion

**D2** = Top-20 RawGrid proposals → ExtraTrees confidence → TRUE P1 Top-1 → C4_MEDIAN + S2 size → StandardScaler + logistic detection gate.

Temporal rescue is **rejected** and is not imported by this entrypoint.

Default ChallengeON participation folder name: **SparseSight-SSA**.

## Local run

```bash
python -m orbitsight.submission \
  --dataset /path/to/OrbitSight_dataset \
  --work /path/to/work \
  --model-dir models/final \
  --team SparseSight-SSA
```

Writes: `/work/SparseSight-SSA/<DDMMYYYY>/*_bb_windows_40ms.txt`

## Docker

```bash
docker build -t sparsesight-ssa:phase1-final .
docker run --rm --network none \
  -v /path/to/dataset:/OrbitSight_dataset:ro \
  -v /path/to/work:/work \
  sparsesight-ssa:phase1-final
```

Environment defaults inside the image:
- `ORBITSIGHT_DATASET=/OrbitSight_dataset`
- `ORBITSIGHT_WORK=/work`
- `ORBITSIGHT_MODEL_DIR=/models/final`
- `ORBITSIGHT_TEAM_NAME=SparseSight-SSA`

## Output contract (operational)

- Filename: `<sequence>_bb_windows_40ms.txt` (organizer `evaluate.py` compatibility)
- No `Evaluation_Metrics.xlsx` from inference (GT-independent; evaluator generates workbook)
- See `docs/FINAL_SUBMISSION_DECISION.md`
