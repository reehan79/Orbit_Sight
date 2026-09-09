from orbitsight.validation.splits import build_sequence_folds, infer_sensor


def test_infer_sensor():
    assert infer_sensor("DAVIS_X") == "DAVIS"
    assert infer_sensor("DVX_X") == "DVX"
    assert infer_sensor("2025_EVK4_mag") == "EVK4"


def test_folds_keep_sequences_whole_and_cover_once():
    sequences = [f"DAVIS_{i}" for i in range(6)] + [f"DVX_{i}" for i in range(6)] + ["x_EVK4_y"]
    folds = build_sequence_folds(sequences, n_splits=5, seed=7)
    seen = []
    for fold in folds:
        assert set(fold.train).isdisjoint(fold.validation)
        assert set(fold.train) | set(fold.validation) == set(sequences)
        seen.extend(fold.validation)
    assert sorted(seen) == sorted(sequences)
