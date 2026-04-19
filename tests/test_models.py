import numpy as np
import pytest

from ms_var_prediction.models.gaussian_mixture_model import GaussianMixtureVaR
from ms_var_prediction.models.markov_swiching_model import MarkovSwitchingVaR


class TestMarkovSwitchingVaR:
    def test_fit_returns_self(self, synthetic_returns):
        model = MarkovSwitchingVaR(n_states=2)
        result = model.fit(synthetic_returns)
        assert result is model

    def test_fit_sets_params(self, fitted_msm):
        assert fitted_msm._params is not None
        assert fitted_msm._params.shape == (2 * (2 + 2),)  # n_states=2

    def test_fit_sets_state_probs(self, fitted_msm):
        assert fitted_msm._state_probs is not None
        assert fitted_msm._state_probs.shape == (2,)
        assert np.isclose(fitted_msm._state_probs.sum(), 1.0, atol=1e-3)

    def test_predict_returns_float(self, fitted_msm):
        var = fitted_msm.predict(0.05)
        assert isinstance(var, (float, np.floating))

    def test_predict_negative_var(self, fitted_msm):
        """VaR at left tail should be negative for typical return distributions."""
        var = fitted_msm.predict(0.05)
        assert var < 0.0

    def test_predict_monotone_in_alpha(self, fitted_msm):
        var_01 = fitted_msm.predict(0.01)
        var_05 = fitted_msm.predict(0.05)
        assert var_01 <= var_05

    def test_predict_without_fit_raises(self):
        model = MarkovSwitchingVaR()
        with pytest.raises(ValueError):
            model.predict(0.05)

    def test_invalid_n_states(self):
        with pytest.raises(ValueError):
            MarkovSwitchingVaR(n_states=1)

    def test_invalid_optimizer_method(self):
        with pytest.raises(ValueError):
            MarkovSwitchingVaR(optimizer_method="nonexistent")

    def test_get_fitted_params_keys(self, fitted_msm):
        params = fitted_msm.get_fitted_params()
        assert "mus" in params
        assert "sigmas" in params
        assert "transition_matrix" in params
        assert "state_probs" in params

    def test_load_fitted_params_roundtrip(self, synthetic_returns):
        original = MarkovSwitchingVaR(n_states=2)
        original.fit(synthetic_returns)
        params = original.get_fitted_params()

        restored = MarkovSwitchingVaR(n_states=2)
        restored.load_fitted_params(params)

        assert np.allclose(original.predict(0.05), restored.predict(0.05))

    def test_load_fitted_params_skips_fit(self, synthetic_returns):
        model = MarkovSwitchingVaR(n_states=2)
        model.fit(synthetic_returns)
        params = model.get_fitted_params()

        fresh = MarkovSwitchingVaR(n_states=2)
        fresh.load_fitted_params(params)
        # Should be able to predict without calling fit
        var = fresh.predict(0.05)
        assert np.isfinite(var)


class TestGaussianMixtureVaR:
    def test_fit_returns_self(self, synthetic_returns):
        model = GaussianMixtureVaR(n_components=2)
        result = model.fit(synthetic_returns)
        assert result is model

    def test_fit_sets_attributes(self, fitted_gmm):
        assert fitted_gmm.weights_ is not None
        assert fitted_gmm.means_ is not None
        assert fitted_gmm.covariances_ is not None
        assert len(fitted_gmm.weights_) == 2
        assert np.isclose(fitted_gmm.weights_.sum(), 1.0, atol=1e-6)

    def test_predict_returns_float(self, fitted_gmm):
        var = fitted_gmm.predict(0.05)
        assert isinstance(var, (float, np.floating))

    def test_predict_negative_var(self, fitted_gmm):
        assert fitted_gmm.predict(0.05) < 0.0

    def test_predict_monotone_in_alpha(self, fitted_gmm):
        assert fitted_gmm.predict(0.01) <= fitted_gmm.predict(0.05)

    def test_invalid_n_components(self):
        with pytest.raises(ValueError):
            GaussianMixtureVaR(n_components=0)

    def test_invalid_tol(self):
        with pytest.raises(ValueError):
            GaussianMixtureVaR(tol=-1.0)

    def test_get_fitted_params_keys(self, fitted_gmm):
        params = fitted_gmm.get_fitted_params()
        assert "weights" in params
        assert "means" in params
        assert "covariances" in params

    def test_load_fitted_params_roundtrip(self, synthetic_returns):
        original = GaussianMixtureVaR(n_components=2)
        original.fit(synthetic_returns)
        params = original.get_fitted_params()

        restored = GaussianMixtureVaR(n_components=2)
        restored.load_fitted_params(params)

        assert np.allclose(original.predict(0.05), restored.predict(0.05))
