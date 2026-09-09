# OrbitSight competition plan

## Objective

Win on the combination that the challenge actually measures: detection quality, robust generalization, CPU real-time performance, technical innovation, and reproducibility.

## Architecture hypothesis

Do not commit to one cue. Build a high-recall sparse proposal stage, then measure complementary evidence per candidate:

1. instantaneous spatial evidence;
2. polarity structure;
3. multi-timescale temporal persistence;
4. motion coherence / rescue;
5. background reliability;
6. sensor/scene regime statistics;
7. a small local learned representation for ranking and box refinement.

Fusion must be reliability-aware. Motion is an additive rescue path; it is not allowed to destroy strong instantaneous detections.

## Evidence already established during exploration

- destructive filtering can erase faint targets;
- target statistics differ strongly between DAVIS, DVX and EVK4;
- temporal/motion evidence helps specific weak regimes but can hurt if used as a universal re-ranker;
- cheap raw proposals can achieve very high candidate coverage;
- cheap local geometry is not sufficient for accurate IoU, so learned local refinement is justified.

These observations guide experiments, but they are not treated as final performance claims. Final numbers must come from the reproducible evaluation framework in this repository.

## Development order

1. Validate local metric implementation against TII's supplied `OrbitSight_DataLoader/evaluate.py`.
2. Freeze deterministic whole-sequence cross-validation splits.
3. Establish a conventional end-to-end baseline (B0).
4. Benchmark the label-free raw sparse proposer (B1).
5. Generate candidate-level training examples without random-window leakage.
6. Train a minimal numerical candidate scorer first; compare with a tiny local CNN.
7. Add local box refinement and optimize AP@0.5, not merely candidate recall.
8. Add one evidence channel at a time and keep only measured improvements.
9. Add motion as an independent false-negative rescue path.
10. Freeze the architecture, then use the supplied testing sequences as a late generalization check.
11. Optimize CPU inference and package the offline Docker submission.

## Competition scoreboard

Every serious experiment should report at least:

- AP@0.5
- precision
- recall
- F1
- mean matched IoU
- p50 / p95 latency per 40-ms-equivalent interval
- model size / parameter count where applicable

Do not retain a component because it sounds novel. Keep it only if it improves accuracy, robustness, latency, or the technical case at acceptable cost.

## Latency budget

Initial engineering target: **< 40 ms end-to-end per 40-ms-equivalent interval on CPU**, preferably 20–30 ms to leave Docker/I/O margin.
