# OrbitSight submission runtime

## Frozen champion

**D2** = Top-20 RawGrid proposals → ExtraTrees confidence → TRUE P1 Top-1 → C4_MEDIAN + S2 size → StandardScaler + logistic detection gate.

Temporal rescue is **rejected** and is not imported by this entrypoint.

## Local run

```bash
python -m orbitsight.submission \
  --dataset /path/to/OrbitSight_dataset \
  --work /path/to/work \
  --model-dir models/final \
  --team OrbitSight
```

Writes: `/work/<TEAM>/<DDMMYYYY>/*_bb_windows_40ms.txt`

## Docker

```bash
docker build -t orbitsight-final:latest .
docker run --rm --network none \
  -v /path/to/dataset:/OrbitSight_dataset:ro \
  -v /path/to/work:/work \
  orbitsight-final:latest
```

Environment defaults inside the image:
- `ORBITSIGHT_DATASET=/OrbitSight_dataset`
- `ORBITSIGHT_WORK=/work`
- `ORBITSIGHT_MODEL_DIR=/models/final`
- `ORBITSIGHT_TEAM_NAME=OrbitSight`
