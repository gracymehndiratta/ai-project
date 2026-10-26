"""A fixed nearest-neighbor readout, explicitly separate from bare PanDerm."""
import numpy as np
from threadpoolctl import threadpool_limits


class PanDermNearestNeighbor:
    """No learned parameters; retrieve a labelled TRAIN embedding by cosine."""
    def __init__(self, features, labels, groups):
        self.features = self.normalize(features)
        self.labels = np.asarray(labels, dtype=int)
        self.groups = np.asarray(groups)
        self.classes_ = np.arange(7)

    @staticmethod
    def normalize(features):
        values = np.asarray(features, dtype=np.float32)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    def predict_proba(self, features, excluded_groups=None):
        values = self.normalize(features)
        predictions = []
        with threadpool_limits(limits=4):
            for start in range(0, len(values), 256):
                score = values[start:start + 256] @ self.features.T
                if excluded_groups is not None:
                    excluded = np.asarray(excluded_groups)[start:start + 256]
                    mask = excluded[:, None] == self.groups[None, :]
                    score[mask] = -np.inf
                    if not np.isfinite(score).any(axis=1).all():
                        raise ValueError('No reference lesion remains after group exclusion')
                predictions.extend(self.labels[score.argmax(axis=1)])
        # These are hard votes, not calibrated probabilities. The floor only
        # makes the generic log-loss reporter numerically defined.
        probability = np.full((len(values), 7), 1e-12, dtype=np.float64)
        probability[np.arange(len(values)), predictions] = 1.
        return probability / probability.sum(1, keepdims=True)
