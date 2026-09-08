# FINAL EVIDENCE TABLE

Measured values only (Training_sets CV / screens / sealed holdout / Docker). Scientific interpretation external.

| Stage | What was measured | Key measured result | Disposition |
|-------|-------------------|---------------------|-------------|
| B1 sparse proposal | Top-k GT coverage on 17 train seqs | Top-20 micro **95.15%** / macro **95.29%**; Top-1 micro 70.75% | Kept as front-end |
| M2b ranker | Positive-window CV (M2b ExtraTrees) | M2b_S2 pooled micro IoU≥0.5 **40.49%**; seq macro **36.27%** | Kept |
| C1 → C4 localization | Geometry residual / e2e | C4_MEDIAN selected over C1 for deploy centre | Kept (C4) |
| S2 size | ExtraTrees log-w/h at C1 centre | S2 preferred vs S0/S1 in e2e matrix | Kept |
| All-window D0 | Candidate-F1 threshold on P1 | F1 **0.321**, P 0.274, R 0.388 (5-fold pooled) | Baseline only |
| **D2 challenge-aligned gate** | Logistic gate + TII detection-F1 thr | F1 **0.409**, P **0.572**, R 0.319; empty FPR **0.010** | **CHAMPION** |
| Tiny CNN / foveated refiner | N1/N2 vs B_CURRENT | N1/N2 **below** B_CURRENT IoU50 (36.2% / 40.6% vs 43.1%) | **Rejected** |
| H1 neural select | H1_P1 all-window | F1 **0.318** < D2; latency fail | **Rejected** |
| Temporal rescue | Corrected SCREEN folds 1+4 | F1 **0.379** vs D2 **0.389**; FULL_CV_RECOMMENDED=**False** | **Rejected for deployment** |
| Geometry reuse (PR #11) | OLD recompute vs NEW reuse | `parity_failures=0`; deploy path kept | Kept (parity-preserving) |
| **Final sealed holdout** | Testing_sets ×1 (frozen D2) | F1 **0.5669**, P 0.5943, R 0.5420, AP@0.5 0.3864; TII↔local parity | Recorded; **no post-test tuning** |
| **Final Docker latency** | Image complete path, 500 win/seq | Overall p95 **36.31 ms** (≤40); DAVIS 8.37; DVX 12.04; EVK4 62.83 | Overall criterion PASS |

## Explicitly rejected ideas

| Idea | Why rejected (measured / freeze rule) |
|------|----------------------------------------|
| Hard hot-pixel deletion | Not promoted; risk to weak-signal regimes; not in champion path |
| Universal trajectory reranking | Plan note: can hurt; never championed |
| Poisson / surprise central ranking | Not selected over ExtraTrees confidence |
| Tiny neural bbox refiner | Below classical B_CURRENT IoU50 |
| Temporal rescue deployment | SCREEN failed A/C/D; `FULL_CV_RECOMMENDED=False` |

## Final freeze pointer

- Models: `models/final/`
- Freeze doc: `docs/FINAL_MODEL_FREEZE.md`
- Sealed report: `docs/runs/2026-09-08_FINAL_SEALED_TEST.md`
- Docker latency CSV: `docs/runs/2026-09-08/docker_latency/latency_by_sensor.csv`
- Threshold: **0.848222565945547**
