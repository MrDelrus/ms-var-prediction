import numpy as np
import pytest

from ms_var_prediction.backtester.backtester import Backtester
from ms_var_prediction.models.gaussian_mixture_model import GaussianMixtureVaR
from ms_var_prediction.models.markov_switching_model import MarkovSwitchingVaR
from ms_var_prediction.pipeline.backtesting import rolling_returns_and_var


WINDOW = 50


class TestRollingReturnsAndVar:
    @pytest.mark.parametrize(
        "ModelClass,kwargs",
        [
            (GaussianMixtureVaR, {"n_components": 2}),
            (MarkovSwitchingVaR, {"n_states": 2}),
        ],
    )
    def test_output_shapes(self, synthetic_returns, ModelClass, kwargs):
        model = ModelClass(**kwargs)
        returns, var_series = rolling_returns_and_var(
            model, synthetic_returns, alpha=0.05, window_shape=WINDOW
        )
        expected_len = len(synthetic_returns) - WINDOW
        assert len(returns) == expected_len
        assert len(var_series) == expected_len

    def test_var_series_finite(self, synthetic_returns):
        model = GaussianMixtureVaR(n_components=2)
        _, var_series = rolling_returns_and_var(
            model, synthetic_returns, alpha=0.05, window_shape=WINDOW
        )
        assert np.all(np.isfinite(var_series))

    def test_var_series_negative(self, synthetic_returns):
        """Most VaR estimates at 5% should be negative for typical returns."""
        model = GaussianMixtureVaR(n_components=2)
        _, var_series = rolling_returns_and_var(
            model, synthetic_returns, alpha=0.05, window_shape=WINDOW
        )
        assert np.mean(var_series < 0) > 0.9

    def test_returns_trimmed_correctly(self, synthetic_returns):
        model = GaussianMixtureVaR(n_components=2)
        returns, _ = rolling_returns_and_var(
            model, synthetic_returns, alpha=0.05, window_shape=WINDOW
        )
        assert np.allclose(returns, synthetic_returns[WINDOW:])


class TestBacktester:
    def _make_backtester(self, n=500, alpha=0.05, breach_rate=0.05):
        rng = np.random.default_rng(1)
        returns = rng.normal(0, 0.01, size=n)
        vars_ = np.percentile(returns, alpha * 100) * np.ones(n)
        return Backtester(returns, vars_, alpha)

    def test_test_returns_dict(self):
        bt = self._make_backtester()
        result = bt.test()
        assert isinstance(result, dict)

    def test_test_keys(self):
        bt = self._make_backtester()
        keys = bt.test().keys()
        for expected in [
            "binomial",
            "independence_simple",
            "kupiec",
            "christoffersen_independence",
            "christoffersen_conditional",
        ]:
            assert expected in keys

    def test_test_values_are_bool(self):
        bt = self._make_backtester()
        for v in bt.test().values():
            assert isinstance(v, (bool, np.bool_))

    def test_perfect_var_passes_coverage(self):
        """Exact quantile VaR should pass binomial and kupiec tests."""
        rng = np.random.default_rng(99)
        n = 1000
        returns = rng.normal(0, 0.01, size=n)
        alpha = 0.05
        threshold = np.percentile(returns, alpha * 100)
        vars_ = np.full(n, threshold)
        bt = Backtester(returns, vars_, alpha)
        result = bt.test()
        assert result["binomial"]
        assert result["kupiec"]

    def test_all_breach_fails_tests(self):
        """If every return exceeds VaR, coverage tests must fail."""
        n = 500
        returns = np.full(n, -0.1)
        vars_ = np.zeros(n)  # all returns < 0 < vars
        bt = Backtester(returns, vars_, alpha=0.05)
        result = bt.test()
        assert not result["binomial"]
        assert not result["kupiec"]
