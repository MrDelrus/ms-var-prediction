from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from ms_var_prediction.backtester.backtester import Backtester
from ms_var_prediction.models.gaussian_mixture_model import GaussianMixtureVaR
from ms_var_prediction.models.markov_swiching_model import MarkovSwitchingVaR
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
            var_series_list: list[float] = []

            for t in range(window_shape, len(rets)):
                window = rets[t - window_shape : t]
                model.fit(window)
                var_series_list.append(float(model.predict(alpha)))

                if log_cfg.get("log_window_csv"):
                    w_start = str(dates[t - window_shape].date())
                    w_end = str(dates[t - 1].date())
                    tracker.log_window_row(model, exp_name, w_start, w_end)

            returns_eval = rets[window_shape:]
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
