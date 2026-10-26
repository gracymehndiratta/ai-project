import numpy as np

from src.utils.regularized_specialists import PatientStopping, resample_fit


def test_smote_is_bounded_and_retains_original_observations():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 8)).astype(np.float32)
    y = np.array([0] * 90 + [1] * 10)
    original_x, original_y = x.copy(), y.copy()
    sampled_x, sampled_y = resample_fit(x, y, True, 42)
    np.testing.assert_array_equal(sampled_x[:100], original_x)
    np.testing.assert_array_equal(sampled_y[:100], original_y)
    assert np.bincount(sampled_y).tolist() == [90, 30]
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
    untouched_x, untouched_y = resample_fit(x, y, False, 42)
    np.testing.assert_array_equal(untouched_x, original_x)
    np.testing.assert_array_equal(untouched_y, original_y)


def test_patience_keeps_early_best_but_allows_recovery_before_stopping():
    class Booster:
        def set_attr(self, **kwargs):
            self.attrs = kwargs
    model = Booster()
    callback = PatientStopping()
    for epoch in range(200):
        value = -.8 if epoch == 10 else -.9 if epoch == 190 else -.7
        stop = callback.after_iteration(model, epoch, {'validation_0': {'negative_ap': [value]}})
        assert not stop
    assert model.attrs['best_iteration'] == '190'
    for epoch in range(200, 291):
        stop = callback.after_iteration(model, epoch, {'validation_0': {'negative_ap': [-.7]}})
        assert stop == (epoch == 290)
