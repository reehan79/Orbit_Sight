# OrbitSight

Competition engineering repository for the **TII OrbitSight Challenge**.

The current goal is not to guess one "novel" architecture. It is to build a reproducible system in which every component earns its place through measured accuracy, generalization and CPU latency.

## Current strategy

```text
raw NVS events
      |
high-recall sparse proposals
      |
per-candidate evidence
(spatial / polarity / temporal / motion / background / sensor regime)
      |
small reliability-aware scorer + local box refinement
      |
strong detections + independent motion-rescue candidates
      |
final confidence + boxes
```

The official dataset is **not** stored in Git.

## Quick start on Windows / PowerShell

```powershell
cd D:\Projects\Orbit_Sight
py -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

Assuming the dataset is at:

```text
D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset
```

check the training split:

```powershell
$DATA="D:\OrbitSight_SSA_Challenge\OrbitSight_SSA_Challenge\Phase_1\OrbitSight_Dataset"
python .\scripts\check_dataset.py --split-dir "$DATA\Training_sets"
```

create deterministic whole-sequence CV folds:

```powershell
python .\scripts\make_splits.py `
  --train-dir "$DATA\Training_sets" `
  --folds 5 `
  --out .\sequence_folds.json
```

benchmark the raw sparse proposer on one sequence:

```powershell
python .\scripts\benchmark_raw_candidates.py `
  --npy "$DATA\Training_sets\DAVIS_EGS_16908_2024-11-01-19-10-44_labeled_events.npy" `
  --sequence "DAVIS_EGS_16908_2024-11-01-19-10-44" `
  --windows 500 `
  --top-k 20
```

Evaluate a prediction directory:

```powershell
python .\scripts\evaluate_predictions.py `
  --gt-dir "$DATA\Training_sets" `
  --pred-dir .\outputs\example
```

**Important:** before using local AP/F1 numbers as authoritative, compare this implementation against the challenge-provided `OrbitSight_DataLoader/evaluate.py` using identical prediction files. The repository intentionally treats the organizer evaluator as the final metric oracle.

See [`docs/COMPETITION_PLAN.md`](docs/COMPETITION_PLAN.md) for the development plan.
