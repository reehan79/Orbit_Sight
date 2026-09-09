# Open-source disclosure — SparseSight-SSA Phase 1 final runtime

Disclosure for packages **actually used** by the frozen Docker submission image
(`sparsesight-ssa:phase1-final`) and `python -m orbitsight.submission`.

Licenses are taken from installed package metadata where available. If not
readily available in this environment, marked **VERIFY**.

| Package | Pinned version | License (metadata) | Purpose |
|---------|----------------|--------------------|---------|
| Python | 3.11.9 (image base) | PSF (VERIFY base-image notice) | Runtime |
| numpy | 1.26.4 | BSD License (OSI classifier) | Event arrays / numerics |
| scikit-learn | 1.8.0 | BSD-3-Clause (`License-Expression`) | ExtraTrees + LogisticRegression + StandardScaler |
| joblib | 1.5.3 | BSD-3-Clause (`License-Expression`) | Model serialization load |
| scipy | 1.13.1 | BSD License (OSI classifier) | Transitive scientific dependency of scikit-learn |
| openpyxl | 3.1.5 | MIT (`License` metadata) | Write `Evaluation_Metrics.xlsx` post-inference |

## Not bundled in final image

- PyTorch / ONNX / foveated refiner (optional research path; not imported by submission)
- Training caches (`artifacts/`, `*.npz`, raw `.npy` datasets)
- Temporal-rescue experiment code (not on inference import path)

## How licenses were checked

```text
python -c "import importlib.metadata as m; print(m.metadata('numpy')['License'])"
```

(and similarly for scikit-learn, joblib, scipy on the training/submission
environment that matches `requirements-submission.txt`).

Re-verify against the exact image layers before portal upload if organizers
require SPDX identifiers from wheel METADATA.
