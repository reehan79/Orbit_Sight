# FINAL MODEL FREEZE

**Date (UTC):** 2026-09-07  
**Branch:** `submission/final-freeze`  
**Training code git SHA:** `382c70ee80975b566bbfab8cc4916567e64e32c1` (`sprint/final-optimization`)

## Statement

Architecture, features, hyperparameters, and threshold frozen before Testing_sets evaluation.

## Architecture components

| Stage | Choice |
|-------|--------|
| Proposals | RawGridProposer Top-20 |
| Confidence | ExtraTreesClassifier |
| Selection | TRUE P1 (Top-1 only) |
| Centre | C4_MEDIAN (S2 local features at C1) |
| Size | S2 ExtraTreesRegressor |
| Gate | D2 StandardScaler + LogisticRegression |
| Geometry | Parity-preserving reuse (`reuse_geometry=True`) |

Temporal rescue: **rejected for deployment** (not in final executable).

## Feature dimensions

| Block | Dim |
|-------|----:|
| Candidate features | 15 |
| Gate scalars | 15 |
| Local geometry | 18 |
| **Gate vector total** | **48** |

## Hyperparameters (frozen)

**Confidence ExtraTrees:** n_estimators=64, max_depth=14, min_samples_leaf=12, max_features=None, class_weight=balanced, random_state=42, n_jobs=1

**S2 ExtraTrees:** n_estimators=32, max_depth=12, min_samples_leaf=24, max_features=None, random_state=42, n_jobs=1

**D2 gate:** StandardScaler + LogisticRegression(C=1.0, class_weight=balanced, max_iter=1000, random_state=42)

**Inner OOF:** KFold on sequences, shuffle=True, random_state=42, n_splits=min(5, n_seq)

## Final D2 threshold

`0.848222565945547`

Selected by maximizing TII detection F1 on sequence-level inner-OOF gate scores over all **17** Training_sets sequences. Testing_sets were **not** used.

Train-OOF diagnostic (not a test score): F1=0.4160, P=0.6416, R=0.3078, empty FPR=0.0083.

## Training sequence count

**17** (full Training_sets)

## Model artifact SHA256

| File | SHA256 |
|------|--------|
| confidence_extratrees.joblib | 91267d192baf5fc146acd7f0ceefe8c3563798fa941e97d623d1b380104fa2dc |
| size_s2_extratrees.joblib | 79e65e6848425adda6d9f00da795d4a52a6a6fe7c3dbe6f84084d5f4a5fb0238 |
| d2_gate_scaler.joblib | 43339633fdb9eb37faccf1c08211c1e4b359cb8090ace7b1377479fdebb8b9c1 |
| d2_gate_logistic.joblib | 0d39234a43691295e24747462cf2427d1df9a322d9bdab77278ecb01b16b58b9 |
| d2_threshold.json | e49dfdc182070d1ac60b554aa9669ed61468803f0b7e9cb1a85b8aec35a22a25 |
| feature_names.json | 0be282f562261670bb62e734a334aae828cf4363ad60906d0188ffd2736ab43c |

See also `models/final/manifest.json`.

## Seeds

`random_state = 42` for ExtraTrees, LogisticRegression, and KFold.
