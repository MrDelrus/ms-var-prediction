"""Smoke tests for run_experiment — uses a tiny synthetic dataset, no network calls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from unittest.mock import MagicMock, patch

from ms_var_prediction.experiment import build_model, run_experiment


class TestBuildModel:
    def test_builds_msm(self) -> None:
        from ms_var_prediction.models.markov_switching_model import MarkovSwitchingVaR

        m = build_model("msm", {"n_states": 2})
        assert isinstance(m, MarkovSwitchingVaR)

    def test_builds_gmm(self) -> None:
        from ms_var_prediction.models.gaussian_mixture_model import GaussianMixtureVaR

        m = build_model("gmm", {"n_components": 2})
        assert isinstance(m, GaussianMixtureVaR)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model_type"):
            build_model("unknown", {})


class TestRunExperiment:
    """run_experiment with mocked data fetch — no network, minimal windows."""

    def _make_cfg(self, model_type: str = "msm") -> dict:
        return {
            "name": f"test_{model_type}",
            "model_type": model_type,
            "hyperparams": (
                {"n_states": 2} if model_type == "msm" else {"n_components": 2}
            ),
            "experiment_params": {
                "start_date": "2020-01-01",
                "end_date": "2021-01-01",
                "alpha": 0.05,
                "window_shape": 60,
            },
            "logging": {},
        }

    def _make_prices(self) -> pd.Series:
        rng = np.random.default_rng(0)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        idx = pd.date_range("2020-01-01", periods=200, freq="B")
        return pd.Series(prices, index=idx, name="Close")

    def test_msm_returns_dataframe(self) -> None:
        prices = self._make_prices()
        mock_data = pd.DataFrame({"Close": prices})

        tracker = MagicMock()

        with patch(
            "ms_var_prediction.experiment.cached_history", return_value=mock_data
        ):
            with patch("ms_var_prediction.experiment.yf.Ticker"):
                df = run_experiment(self._make_cfg("msm"), tracker, ["SPY"])

        assert isinstance(df, pd.DataFrame)
        assert "SPY" in df.index

    def test_gmm_returns_dataframe(self) -> None:
        prices = self._make_prices()
        mock_data = pd.DataFrame({"Close": prices})

        tracker = MagicMock()

        with patch(
            "ms_var_prediction.experiment.cached_history", return_value=mock_data
        ):
            with patch("ms_var_prediction.experiment.yf.Ticker"):
                df = run_experiment(self._make_cfg("gmm"), tracker, ["SPY"])

        assert isinstance(df, pd.DataFrame)
        assert "SPY" in df.index
