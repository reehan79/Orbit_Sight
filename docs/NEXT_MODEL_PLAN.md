# Next model plan

This document is intentionally small. The next stacked PR will implement whole-sequence cross-validated candidate ranking and positive-candidate box regression from `artifacts/candidate_table.csv`.

Planned baselines:

- M0: raw candidate rank only.
- M1: standardized logistic regression on the 15 numerical evidence features.
- M2: compact nonlinear histogram gradient boosting classifier.
- R1: ridge regression for candidate-relative box offsets/sizes on positive candidates.

The first decision metric is candidate Top-1 / Top-3 / Top-5 recall on held-out whole sequences. Box regression is evaluated separately on held-out positive candidate rows until a full empty-window inference path is implemented.
