import numpy as np

from src.models.panderm_baseline import PanDermNearestNeighbor


def test_retrieval_uses_cosine_and_excludes_all_views_of_the_query_lesion():
    features = np.array([[10, 0], [9, 1], [0, 1], [-1, 0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 2])
    groups = np.array(['same', 'same', 'other', 'third'])
    model = PanDermNearestNeighbor(features, labels, groups)
    assert model.predict_proba([[2, 0]]).argmax(1).tolist() == [0]
    # Neither exact self match nor the second image of this lesion may vote.
    assert model.predict_proba([[2, 0]], excluded_groups=['same']).argmax(1).tolist() == [1]
    np.testing.assert_allclose(model.predict_proba(features).sum(1), 1)
