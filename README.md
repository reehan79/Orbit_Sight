# OrbitSight

Competition engineering repository for the **TII OrbitSight Challenge**.

## Frozen deployed architecture

```text
raw events
      |
deterministic Top-20 sparse proposals (RawGrid)
      |
15-D candidate evidence
      |
ExtraTrees confidence
      |
TRUE P1 Top-1 selection
      |
C4 median centre + S2 learned size
      |
D2 challenge-aligned logistic gate
      |
TII prediction boxes + confidence
```

This is the **frozen** final detector. Architecture, features, hyperparameters, and
threshold (`0.848222565945547`) are fixed before Testing_sets evaluation.

Experimentally evaluated and **rejected for deployment**:
- temporal rescue (SCREEN failed promotion; not in the final executable)
- tiny neural / foveated bbox refiner (below classical baseline)

No post-test tuning. No novelty claims beyond measured components that earned their place.

Production entrypoint:

```text
python -m orbitsight.submission --dataset /OrbitSight_dataset --work /work
```

See [`docs/FINAL_MODEL_FREEZE.md`](docs/FINAL_MODEL_FREEZE.md),
[`docs/SUBMISSION_RUNTIME.md`](docs/SUBMISSION_RUNTIME.md), and
[`docs/SUBMISSION_CHECKLIST.md`](docs/SUBMISSION_CHECKLIST.md).

The official dataset is **not** stored in Git.

## Quick start on Windows / PowerShell

```powershell
cd D:\Projects\Orbit_Sight
py -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,ml]"
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

**Important:** before using local AP/F1 numbers as authoritative, compare this implementation against the challenge-provided `OrbitSight_DataLoader/evaluate.py` using identical prediction files. The repository intentionally treats the organizer evaluator as the final metric oracle.

See [`docs/COMPETITION_PLAN.md`](docs/COMPETITION_PLAN.md) for historical development context.
