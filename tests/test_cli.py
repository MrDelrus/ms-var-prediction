import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ms_var_prediction.cli import _resolve_dirs, build_parser
from ms_var_prediction.models.gaussian_mixture_model import GaussianMixtureVaR
from ms_var_prediction.pipeline import features as fs


@pytest.fixture
def gmm_features(fitted_gmm: GaussianMixtureVaR) -> pd.DataFrame:
    """A small features table built from one fitted GMM (alpha-independent)."""
    params = json.dumps(fitted_gmm.get_fitted_params())
    rng = np.random.default_rng(1)
    rows = [
        {
            "ticker": "TST",
            "date": f"2025-01-{i + 1:02d}",
            "realized_return": float(rng.normal(0, 0.02)),
            "params": params,
        }
        for i in range(20)
    ]
    return pd.DataFrame(rows, columns=fs.FEATURE_COLUMNS)


def test_predict_var_reloads_params(gmm_features: pd.DataFrame) -> None:
    preds = fs.predict_var(gmm_features, "gmm", alpha=0.05)
    assert list(preds.columns) == ["ticker", "date", "realized_return", "var"]
    assert len(preds) == len(gmm_features)
    # VaR at 5% should be a moderately negative return.
    assert (preds["var"] < 0).all()


def test_alpha_orders_var(gmm_features: pd.DataFrame) -> None:
    var_05 = fs.predict_var(gmm_features, "gmm", 0.05)["var"].iloc[0]
    var_01 = fs.predict_var(gmm_features, "gmm", 0.01)["var"].iloc[0]
    assert var_01 < var_05  # tighter quantile is more negative


def test_backtest_predictions_shape(gmm_features: pd.DataFrame) -> None:
    preds = fs.predict_var(gmm_features, "gmm", 0.05)
    per_stock, score = fs.backtest_predictions(preds, 0.05)
    assert per_stock.shape == (1, 5)
    assert per_stock.index.name == "ticker"
    assert set(score.index) == set(per_stock.columns)


def test_features_roundtrip(gmm_features: pd.DataFrame, tmp_path: Path) -> None:
    path = fs.year_features_path(tmp_path, "gmm", "nc2_w252", 2025, "default")
    assert path.name == "gmm_nc2_w252_2025-01-01_2026-01-01_default.csv"
    fs.save_features(gmm_features, path)
    loaded = fs.load_features(path)
    pd.testing.assert_frame_equal(loaded, gmm_features)


def test_full_years_skips_warmup() -> None:
    assert fs.full_years("2010-01-01", "2026-01-01") == list(range(2011, 2026))
    assert len(fs.full_years("2010-01-01", "2026-01-01")) == 15
    assert fs.full_years("2025-01-01", "2026-01-01") == []


def test_load_features_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fs.load_features(tmp_path / "nope.csv")


def test_save_results_multiindex(gmm_features: pd.DataFrame, tmp_path: Path) -> None:
    preds = fs.predict_var(gmm_features, "gmm", 0.05)
    _, score = fs.backtest_predictions(preds, 0.05)
    path = fs.results_path(
        tmp_path, "gmm", "nc2_w252", "2025-01-01", "2026-01-01", 0.05, "default"
    )
    fs.save_results(score, "gmm", 0.05, path)

    loaded = pd.read_csv(path, index_col=[0, 1])
    assert loaded.index.names == ["model", "alpha"]
    assert loaded.index[0] == ("gmm", 0.05)
    assert "kupiec" in loaded.columns


def test_resolve_dirs_requires_existing_root(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        _resolve_dirs(tmp_path / "does_not_exist")
    feats, results = _resolve_dirs(tmp_path)
    assert feats.is_dir() and results.is_dir()


def test_parser_defaults() -> None:
    args = build_parser().parse_args(
        [
            "--model",
            "msm",
            "--begin",
            "2025-01-01",
            "--end",
            "2026-01-01",
            "--output",
            ".",
        ]
    )
    assert args.model == "MSM"  # upper-cased
    assert args.mode == "full"
    assert args.alpha == 0.05
    assert args.window == 252
    assert args.tickers == ["SPY"]
    assert args.monitor is False
