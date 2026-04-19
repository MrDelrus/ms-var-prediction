from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from ms_var_prediction.backtester.backtester import Backtester
from ms_var_prediction.config import OUTPUT_DIR
from ms_var_prediction.models.gaussian_mixture_model import GaussianMixtureVaR
from ms_var_prediction.models.markov_switching_model import MarkovSwitchingVaR
from ms_var_prediction.pipeline.backtesting import rolling_returns_and_var
from ms_var_prediction.pipeline.data import cached_history
from ms_var_prediction.tracking.wandb_tracker import WandBTracker
from ms_var_prediction.utils import prices_to_returns


def build_model(model_type: str, hyperparams: dict[str, Any]) -> Any:
    if model_type == "msm":
        return MarkovSwitchingVaR(**hyperparams)
    if model_type == "gmm":
        return GaussianMixtureVaR(**hyperparams)
    raise ValueError(f"Unknown model_type: {model_type!r}")


def run_experiment(
    exp_cfg: dict[str, Any],
    tracker: WandBTracker,
    tickers: list[str],
) -> pd.DataFrame:
    exp_name: str = exp_cfg["name"]
    model_type: str = exp_cfg["model_type"]
    hyperparams: dict[str, Any] = exp_cfg.get("hyperparams", {})
    ep = exp_cfg["experiment_params"]
    log_cfg = exp_cfg.get("logging", {})

    start_date: str = ep["start_date"]
    end_date: str = ep["end_date"]
    alpha: float = float(ep["alpha"])
    window_shape: int = int(ep["window_shape"])

    results = []

    for symbol in tickers:
        ticker = yf.Ticker(symbol)
        try:
            data = cached_history(ticker, start_date, end_date)
            prices = data["Close"].dropna()
            rets = prices_to_returns(prices)
            dates = prices.index[1:]

            model = build_model(model_type, hyperparams)

            def _on_fit(m, t: int) -> None:
                if log_cfg.get("log_window_csv"):
                    w_start = str(dates[t - window_shape].date())
                    w_end = str(dates[t - 1].date())
                    tracker.log_window_row(m, exp_name, w_start, w_end)

            returns_eval, var_arr = rolling_returns_and_var(
                model, rets, alpha, window_shape, on_fit=_on_fit
            )
            var_series_list: list[float] = var_arr.tolist()
            eval_dates = dates[window_shape:]

            if log_cfg.get("log_var_series"):
                _save_var_series(
                    exp_name, symbol, eval_dates, returns_eval, var_series_list, alpha
                )

            bt = Backtester(returns_eval, pd.array(var_series_list, dtype=float), alpha)
            test_res = bt.test()
            test_res["ticker"] = symbol
            results.append(test_res)

        except Exception as exc:
            print(f"[{exp_name}] Error on {symbol}: {exc}")

    results_df = pd.DataFrame(results).set_index("ticker").astype(int)

    if log_cfg.get("log_window_csv"):
        tracker.flush_window_csv(exp_name)
    if log_cfg.get("log_model_csv"):
        tracker.log_model_results(exp_name, results_df)

    return results_df


def _save_var_series(
    exp_name: str,
    symbol: str,
    dates: pd.Index,
    returns: pd.Series,
    var_series: list[float],
    alpha: float,
) -> Path:
    """Save per-ticker VaR series to outputs/var_series/<exp_name>/<symbol>.csv"""
    out_dir = OUTPUT_DIR / "var_series" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}.csv"

    df = pd.DataFrame(
        {"date": dates, "return": returns, "var": var_series, "alpha": alpha}
    ).set_index("date")
    df.to_csv(path)
    return path
