import yfinance as yf
import numpy as np
import pandas as pd
from typing import List
from sklearn.base import BaseEstimator
from tqdm.auto import tqdm

from ms_var_prediction.pipeline.data import cached_history
from ms_var_prediction.backtester import Backtester
from ms_var_prediction.utils import prices_to_returns
from ms_var_prediction.logger import logger


def rolling_returns_and_var(
    model: BaseEstimator, returns: np.ndarray, alpha: float, window_shape: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform rolling-window model fitting and VaR prediction.

    Parameters
    ----------
    returns : array-like
        Time series of returns.
    model : estimator
        Model with .fit() and .predict().
    alpha : float
        Significance level for VaR.
    window_shape : int
        Number of observations used for each model fit.

    Returns
    -------
    tuple of np.ndarray
        (returns, var_series)
        Both arrays have length n - window_shape.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    var_series = np.empty(n - window_shape, dtype=float)

    for time_idx, time in enumerate(range(window_shape, n)):
        window = returns[time - window_shape : time]
        model.fit(window)
        var_series[time_idx] = model.predict(alpha)

    returns = returns[window_shape:]

    return returns, var_series


def ticker_evaluate_var(
    model: BaseEstimator,
    ticker: yf.Ticker,
    start_date: str,
    end_date: str,
    alpha: float,
    window_shape: int,
) -> dict[str, int]:
    logger.debug("Started backtesting for %s", ticker.ticker)
    data = cached_history(ticker, start_date, end_date)
    prices = data["Close"].dropna()

    returns = prices_to_returns(prices)

    return rolling_returns_and_var(model, returns, alpha, window_shape)


def batch_backtest_model(
    model: BaseEstimator,
    tickers: List[yf.Ticker],
    start_date: str = "2015-01-01",
    end_date: str = "2023-01-01",
    alpha: float = 0.05,
    window_shape: int = 250,
) -> pd.DataFrame:
    """
    Run VaR backtest for multiple tickers and return results as a DataFrame.

    Parameters
    ----------
    tickers : list of yf.Ticker
        List of Ticker objects.
    start_date : str
        Start date for price history.
    end_date : str
        End date for price history.
    alpha : float
        VaR significance level.
    window_shape : int
        Rolling window size for model fitting.

    Returns
    -------
    pd.DataFrame
        Rows: ticker symbols
        Columns: backtest names
        Values: 0/1 for test passed/failed
    """
    results = []

    for ticker in tqdm(
        tickers, desc="Yahoo Finance Tickers Backtested", leave=True, position=0
    ):
        try:
            returns, vars = ticker_evaluate_var(
                model, ticker, start_date, end_date, alpha, window_shape
            )

            backtester = Backtester(returns, vars, alpha)
            test_results = backtester.test()

            test_results["ticker"] = ticker.ticker
            results.append(test_results)

        except Exception as e:
            tqdm.write(f"Error processing {ticker.ticker}: {e}")

    return pd.DataFrame(results).set_index("ticker").astype(int)
