"""
Feature building and VaR testing for the ``backtester run`` CLI.

The backtest is split into an alpha-independent phase and an alpha-dependent one:

* **build features** — roll a window over each ticker's returns, fit the model at
  every position and persist the *fitted parameters* (flattened to one column per
  parameter). These features do not depend on ``alpha``.
* **predict + test** — reload the parameters, evaluate VaR at each requested
  ``alpha`` and run the statistical backtests, aggregating pass counts per alpha.

Both models reduce VaR to the same mixture quantile, so prediction is fully
model-agnostic once the parameters are flattened.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from ms_var_prediction.backtester import Backtester
from ms_var_prediction.logger import logger
from ms_var_prediction.models import build_model
from ms_var_prediction.pipeline.data import cached_history
from ms_var_prediction.utils import (
    gaussian_mixture_cdf,
    prices_to_returns,
    quantile_bisection,
)

SEED = 42
META_COLUMNS = ["date", "ticker", "realized_return"]

# The five backtests, in the order they appear in the results table.
TEST_NAMES = [
    "binomial",
    "independence_simple",
    "kupiec",
    "christoffersen_independence",
    "christoffersen_conditional",
]


# ---------------------------------------------------------------------------
# Parameter flattening (one column per parameter, deterministic order)
# ---------------------------------------------------------------------------


def param_columns(model_type: str, n: int) -> list[str]:
    """Ordered names of the flattened parameter columns for a model with ``n`` states."""
    if model_type == "msm":
        cols = [f"mu_{i}" for i in range(n)]
        cols += [f"sigma_{i}" for i in range(n)]
        cols += [f"trans_{i}_{j}" for i in range(n) for j in range(n)]
        cols += [f"state_prob_{i}" for i in range(n)]
        return cols
    if model_type == "gmm":
        cols = [f"weight_{i}" for i in range(n)]
        cols += [f"mean_{i}" for i in range(n)]
        cols += [f"cov_{i}" for i in range(n)]
        return cols
    raise ValueError(f"Unknown model_type: {model_type!r}")


def feature_columns(model_type: str, n: int) -> list[str]:
    return META_COLUMNS + param_columns(model_type, n)


def _flatten_params(model_type: str, params: dict, n: int) -> dict[str, float]:
    flat: dict[str, float] = {}
    if model_type == "msm":
        for i in range(n):
            flat[f"mu_{i}"] = float(params["mus"][i])
            flat[f"sigma_{i}"] = float(params["sigmas"][i])
            flat[f"state_prob_{i}"] = float(params["state_probs"][i])
        for i in range(n):
            for j in range(n):
                flat[f"trans_{i}_{j}"] = float(params["transition_matrix"][i][j])
    else:  # gmm
        for i in range(n):
            flat[f"weight_{i}"] = float(params["weights"][i])
            flat[f"mean_{i}"] = float(params["means"][i])
            flat[f"cov_{i}"] = float(params["covariances"][i])
    return flat


def _mixture_matrices(
    df: pd.DataFrame, model_type: str, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (means, stds, weights) matrices of shape (n_rows, n) for VaR prediction.

    Both models predict VaR as the alpha-quantile of a Gaussian mixture; only the
    source columns differ (MSM: mu/sigma/state_prob, GMM: mean/cov/weight).
    """
    if model_type == "msm":
        means = df[[f"mu_{i}" for i in range(n)]].to_numpy(dtype=float)
        stds = df[[f"sigma_{i}" for i in range(n)]].to_numpy(dtype=float)
        weights = df[[f"state_prob_{i}" for i in range(n)]].to_numpy(dtype=float)
    else:
        means = df[[f"mean_{i}" for i in range(n)]].to_numpy(dtype=float)
        stds = df[[f"cov_{i}" for i in range(n)]].to_numpy(dtype=float)
        weights = df[[f"weight_{i}" for i in range(n)]].to_numpy(dtype=float)
    return means, stds, weights


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------


def build_features(
    model_type: str,
    n: int,
    tickers: list[str],
    begin: str,
    end: str,
    window: int,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Roll a window over each ticker and collect flattened fitted parameters.

    One row per (ticker, evaluation day); columns are ``date, ticker,
    realized_return`` followed by the flattened model parameters. The first
    ``window`` observations of each ticker seed the initial window and are not
    emitted (warm-up).
    """
    np.random.seed(seed)
    hyper = {"n_states": n} if model_type == "msm" else {"n_components": n}
    cols = feature_columns(model_type, n)
    rows: list[dict[str, object]] = []

    for symbol in tickers:
        try:
            data = cached_history(yf.Ticker(symbol), begin, end)
            prices = data["Close"].dropna()
            returns = prices_to_returns(prices)
            dates = prices.index[1:]  # returns[i] is realised on dates[i]
            if len(returns) <= window:
                raise ValueError(
                    f"not enough data: {len(returns)} returns <= window {window}"
                )

            model = build_model(model_type, hyper)
            for t in range(window, len(returns)):
                model.fit(returns[t - window : t])
                row: dict[str, object] = {
                    "date": str(dates[t].date()),
                    "ticker": symbol,
                    "realized_return": float(returns[t]),
                }
                row.update(_flatten_params(model_type, model.get_fitted_params(), n))
                rows.append(row)
            logger.info("Fitted %d windows for %s", len(returns) - window, symbol)
        except Exception as exc:  # noqa: BLE001 - skip and report bad tickers
            logger.error("Failed to fit %s: %s", symbol, exc)

    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Prediction + backtesting
# ---------------------------------------------------------------------------


def predict_and_test(
    df: pd.DataFrame,
    model_type: str,
    n: int,
    alphas: list[float],
) -> pd.DataFrame:
    """
    Evaluate VaR at each alpha and count, per test, how many stocks pass.

    Returns a table with one row per alpha and columns
    ``alpha, total, <test>_passed`` for each of the five tests.
    """
    means, stds, weights = _mixture_matrices(df, model_type, n)
    tickers = df["ticker"].to_numpy()
    returns = df["realized_return"].to_numpy(dtype=float)
    n_rows = len(df)

    records: list[dict[str, float]] = []
    for alpha in alphas:
        var = np.empty(n_rows, dtype=float)
        for k in range(n_rows):
            var[k] = float(
                quantile_bisection(
                    gaussian_mixture_cdf,
                    alpha,
                    -1.0,
                    1.0,
                    means[k],
                    stds[k],
                    weights[k],
                )
            )
        per_ticker = pd.DataFrame({"ticker": tickers, "return": returns, "var": var})
        counts = {t: 0 for t in TEST_NAMES}
        total = 0
        for _, group in per_ticker.groupby("ticker"):
            result = Backtester(
                group["return"].to_numpy(), group["var"].to_numpy(), alpha
            ).test()
            total += 1
            for t in TEST_NAMES:
                if result[t]:
                    counts[t] += 1
        rec: dict[str, float] = {"alpha": alpha, "total": total}
        rec.update({f"{t}_passed": counts[t] for t in TEST_NAMES})
        records.append(rec)

    cols = ["alpha", "total"] + [f"{t}_passed" for t in TEST_NAMES]
    return pd.DataFrame(records, columns=cols)


# ---------------------------------------------------------------------------
# Persistence + verification
# ---------------------------------------------------------------------------


def save_features(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    return pd.read_csv(path)


def verify_features(
    df: pd.DataFrame, model_type: str, n: int, begin: str, end: str
) -> None:
    """Raise ValueError if a loaded features file doesn't match the run settings."""
    expected = feature_columns(model_type, n)
    if list(df.columns) != expected:
        raise ValueError(
            f"features file columns do not match model={model_type.upper()} "
            f"n_components={n}.\n  expected: {expected}\n  found:    {list(df.columns)}"
        )
    if df.empty:
        raise ValueError("features file is empty")
    dates = pd.to_datetime(df["date"])
    if dates.min() < pd.Timestamp(begin) or dates.max() >= pd.Timestamp(end):
        raise ValueError(
            f"features file dates [{dates.min().date()} .. {dates.max().date()}] "
            f"fall outside the requested range [{begin}, {end})"
        )
