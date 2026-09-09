# OrbitSight submission runtime (SparseSight-SSA) — Output Contract V2

## Frozen champion

**D2** = Top-20 RawGrid proposals → ExtraTrees confidence → TRUE P1 Top-1 → C4_MEDIAN + S2 size → StandardScaler + logistic detection gate.

Temporal rescue is **rejected** and is not imported by this entrypoint.

Default ChallengeON participation folder name: **SparseSight-SSA**.

## Official public packaging (V2)

See `docs/OFFICIAL_OUTPUT_CONTRACT_V2.md`.

- Filename mode default: `admin_pred` → `<sequence>_pred.txt`
- Nine public columns including `sequence_id`, `centre_x`/`centre_y`/`w`/`h`, `class_id`
- Writes `Evaluation_Metrics.xlsx` after inference (GT read only for metrics)

## Local run

```bash
python -m orbitsight.submission \
  --dataset /path/to/OrbitSight_dataset \
  --work /path/to/work \
  --model-dir models/final \
  --team SparseSight-SSA
```

Writes: `/work/SparseSight-SSA/<DDMMYYYY>/<sequence>_pred.txt` + `Evaluation_Metrics.xlsx`

## Docker

```bash
docker build -t sparsesight-ssa:phase1-final-v2 .
docker run --rm --network none \
  -v /path/to/dataset:/OrbitSight_dataset:ro \
  -v /path/to/work:/work \
  sparsesight-ssa:phase1-final-v2
```

Environment defaults inside the image:

- `ORBITSIGHT_DATASET=/OrbitSight_dataset`
- `ORBITSIGHT_WORK=/work`
- `ORBITSIGHT_MODEL_DIR=/models/final`
- `ORBITSIGHT_TEAM_NAME=SparseSight-SSA`
- `ORBITSIGHT_PRED_FILENAME_MODE=admin_pred`
- `ORBITSIGHT_CLASS_ID=1`
- `ORBITSIGHT_SPLIT_MODE=auto`
