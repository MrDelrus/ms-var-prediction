"""
Feature store for the two-phase ``backtester`` CLI.

The backtest is split into two alpha-independent / alpha-dependent phases:

* **fit** — roll a window over each ticker's returns, fit the model at every
  position and persist the *fitted parameters* (``get_fitted_params``) together
  with the realised next-day return. These "features" do not depend on ``alpha``.
* **predict** — reload the persisted parameters (``load_fitted_params``),
  evaluate VaR at a given ``alpha`` and run the statistical backtests.

Keeping the two apart means an expensive ``fit`` sweep can be reused for many
different ``alpha`` values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from ms_var_prediction.backtester import Backtester
from ms_var_prediction.models import build_model
from ms_var_prediction.logger import logger
from ms_var_prediction.pipeline.data import cached_history
from ms_var_prediction.utils import prices_to_returns

FEATURE_COLUMNS = ["ticker", "date", "realized_return", "params"]


@dataclass
class FitStats:
    """Bookkeeping for how many tickers were processed successfully."""

    n_total: int
    n_fitted: int

    @property
    def pct_fitted(self) -> float:
        if self.n_total == 0:
            return 0.0
        return 100.0 * self.n_fitted / self.n_total


def compute_features(
    model_type: str,
    tickers: list[str],
    begin: str,
    end: str,
    window: int,
    hyperparams: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, FitStats]:
    """
    Roll a window over each ticker and collect per-window fitted parameters.

    Returns a long-format DataFrame with columns ``FEATURE_COLUMNS`` and a
    :class:`FitStats` describing how many tickers were fitted without error.
    """
    hyperparams = hyperparams or {}
    rows: list[dict[str, Any]] = []
    n_fitted = 0

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

            model = build_model(model_type, hyperparams)
            for t in range(window, len(returns)):
                model.fit(returns[t - window : t])
                rows.append(
                    {
                        "ticker": symbol,
                        "date": str(dates[t].date()),
                        "realized_return": float(returns[t]),
                        "params": json.dumps(model.get_fitted_params()),
                    }
                )
            n_fitted += 1
            logger.info("Fitted %d windows for %s", len(returns) - window, symbol)

        except Exception as exc:  # noqa: BLE001 - skip and report bad tickers
            logger.error("Failed to fit %s: %s", symbol, exc)

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return df, FitStats(n_total=len(tickers), n_fitted=n_fitted)


def compute_year_features(
    model_type: str,
    tickers: list[str],
    year: int,
    data_begin: str,
    data_end: str,
    window: int,
    hyperparams: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, FitStats]:
    """
    Compute per-window fitted parameters for the evaluation dates in one ``year``.

    The full price history ``[data_begin, data_end)`` is fetched so the rolling
    window for early-year dates can reach back into the previous year, but only
    windows whose *realised date* falls in ``year`` are kept. The model is fit
    fresh per ticker, so each year's file is reproducible on its own (it does not
    depend on which other years were computed). ``seed`` is fixed before the
    sweep for deterministic results.
    """
    np.random.seed(seed)
    hyperparams = hyperparams or {}
    rows: list[dict[str, Any]] = []
    fitted: set[str] = set()

    for symbol in tickers:
        try:
            data = cached_history(yf.Ticker(symbol), data_begin, data_end)
            prices = data["Close"].dropna()
            returns = prices_to_returns(prices)
            dates = prices.index[1:]  # returns[i] is realised on dates[i]

            if len(returns) <= window:
                raise ValueError(
                    f"not enough data: {len(returns)} returns <= window {window}"
                )

            model = build_model(model_type, hyperparams)
            for t in range(window, len(returns)):
                if dates[t].year != year:
                    continue
                model.fit(returns[t - window : t])
                rows.append(
                    {
                        "ticker": symbol,
                        "date": str(dates[t].date()),
                        "realized_return": float(returns[t]),
                        "params": json.dumps(model.get_fitted_params()),
                    }
                )
                fitted.add(symbol)

        except Exception as exc:  # noqa: BLE001 - skip and report bad tickers
            logger.error("Failed to fit %s for %d: %s", symbol, year, exc)

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return df, FitStats(n_total=len(tickers), n_fitted=len(fitted))


def predict_var(
    features_df: pd.DataFrame,
    model_type: str,
    alpha: float,
) -> pd.DataFrame:
    """
    Reload persisted parameters and evaluate VaR at ``alpha`` for every row.

    Returns a DataFrame with columns ``ticker, date, realized_return, var``.
    """
    out: list[dict[str, Any]] = []
    for record in features_df.to_dict("records"):
        model = build_model(model_type, {})
        model.load_fitted_params(json.loads(record["params"]))
        out.append(
            {
                "ticker": record["ticker"],
                "date": record["date"],
                "realized_return": float(record["realized_return"]),
                "var": float(model.predict(alpha)),
            }
        )
    return pd.DataFrame(out, columns=["ticker", "date", "realized_return", "var"])


def backtest_predictions(
    predictions_df: pd.DataFrame,
    alpha: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run the statistical backtests per ticker.

    Returns ``(per_stock, score)`` where ``per_stock`` is a 0/1 pass table
    (rows = tickers, columns = tests) and ``score`` is the per-test pass
    fraction across all tickers.
    """
    per_stock: dict[str, dict[str, bool]] = {}
    for ticker, group in predictions_df.groupby("ticker"):
        bt = Backtester(
            group["realized_return"].to_numpy(),
            group["var"].to_numpy(),
            alpha,
        )
        per_stock[str(ticker)] = bt.test()

    per_stock_df = pd.DataFrame(per_stock).T.astype(int)
    per_stock_df.index.name = "ticker"
    score = per_stock_df.mean().rename("score")
    return per_stock_df, score


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def params_str(n_components: int, window: int) -> str:
    """Compact encoding of the run hyperparameters for filenames, e.g. ``nc2_w252``."""
    return f"nc{n_components}_w{window}"


def full_years(begin: str, end: str) -> list[int]:
    """
    Years to produce features for, given a ``[begin, end)`` range.

    The first calendar year is reserved as warm-up (needed to build the initial
    rolling window) and is not saved. With ``begin=2010-01-01, end=2026-01-01``
    this returns ``[2011, ..., 2025]`` — 15 full years.
    """
    return list(range(int(begin[:4]) + 1, int(end[:4])))


def year_features_path(
    features_dir: Path, model_type: str, p_str: str, year: int, tag: str
) -> Path:
    """One feature file per year: ``{model}_{params}_{year}-01-01_{year+1}-01-01_{tag}.csv``."""
    return (
        features_dir / f"{model_type}_{p_str}_{year}-01-01_{year + 1}-01-01_{tag}.csv"
    )


def results_path(
    results_dir: Path,
    model_type: str,
    p_str: str,
    begin: str,
    end: str,
    alpha: float,
    tag: str,
) -> Path:
    return results_dir / f"{model_type}_{p_str}_{begin}_{end}_{alpha}_{tag}.csv"


def save_features(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Features file not found: {path}. Run with --mode fit (or full) first."
        )
    return pd.read_csv(path)


def save_results(score: pd.Series, model_type: str, alpha: float, path: Path) -> Path:
    """Write the aggregate score as a single MultiIndex (model, alpha) row."""
    row = score.to_frame().T
    row.index = pd.MultiIndex.from_tuples(
        [(model_type, alpha)], names=["model", "alpha"]
    )
    row.to_csv(path)
    return path
