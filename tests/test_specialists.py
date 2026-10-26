"""Check specialist calibration boundaries and binary-versus-fusion semantics."""
import joblib
import numpy as np
import pandas as pd

from src.data.full_isic import CLASS_NAMES
from src.models.lesion_specialists import SpecialistSystem
from src.utils.specialist_training import binary_metrics, binary_weights, build_folds, f1_threshold


class ConstantSpecialist:
    def __init__(self, score, threshold):
        self.score, self.threshold = score, threshold

    def predict_positive(self, features):
        return np.full(len(features), self.score)


def test_grouped_selection_keeps_calibration_and_scoring_separate():
    rows = []
    for name in CLASS_NAMES:
        for lesion in range(30):
            for view in range(1 + lesion % 3):
                rows.append({"image_id": f"{name}_{lesion}_{view}", "lesion_id": f"{name}_{lesion}", "label": name})
    frame = pd.DataFrame(rows)
    folds = build_folds(frame)
    holdouts = []
    for fold in folds:
        groups = {key: set(frame.iloc[indices].lesion_id) for key, indices in fold.items()}
        assert groups["fit"].isdisjoint(groups["calibration"])
        assert groups["holdout"].isdisjoint(groups["fit"] | groups["calibration"])
        assert groups["outer_train"] == groups["fit"] | groups["calibration"]
        holdouts.extend(fold["holdout"])
    assert sorted(holdouts) == list(range(len(frame)))


def test_binary_metric_definitions_and_threshold_selection():
    labels = np.array([1, 1, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.6, 0.7, 0.3, 0.2, 0.1])
    metrics = binary_metrics(labels, probabilities, 0.5)
    assert metrics["tp"] == 2 and metrics["fp"] == 1
    np.testing.assert_allclose(metrics["precision"], 2 / 3)
    np.testing.assert_allclose(metrics["recall"], 1)
    np.testing.assert_allclose(metrics["f1"], 0.8)
    np.testing.assert_allclose(metrics["f05"], 5 / 7)
    np.testing.assert_allclose(metrics["accuracy"], 5 / 6)
    assert 0 < f1_threshold(labels, probabilities) < 1
    for power in [0.5, 1]:
        np.testing.assert_allclose(binary_weights(labels, power)[labels].mean(), 1)


def test_normalization_alone_keeps_winner_but_threshold_adjustment_can_change_it(tmp_path):
    heads = [ConstantSpecialist(0.6, 0.9), ConstantSpecialist(0.3, 0.1)]
    system = SpecialistSystem(heads, strength=0)
    features = np.zeros((2, 2560), dtype=np.float32)
    assert np.all(system.predict_proba(features).argmax(1) == 0)
    np.testing.assert_allclose(system.predict_proba(features).sum(1), 1)
    system.strength = 1
    assert np.all(system.predict_proba(features).argmax(1) == 1)
    joblib.dump(system, tmp_path / "model.joblib")
    restored = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_allclose(restored.predict_proba(features), system.predict_proba(features))
