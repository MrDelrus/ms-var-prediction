from ms_var_prediction import (
    OUTPUT_DIR,
    GaussianMixtureVaR,
    MarkovSwitchingVaR,
    batch_backtest_model,
    fetch_valid_sp500_tickers,
    settings,
)
from typing import List
from yfinance import Ticker

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_gmm(tickers: List[Ticker]) -> None:
    gmm_model = GaussianMixtureVaR()
    gmm_res = batch_backtest_model(
        gmm_model,
        tickers,
        settings.START_DATE,
        settings.END_DATE,
        settings.ALPHA,
        settings.WINDOW_SHAPE,
    )
    gmm_res.to_csv(
        OUTPUT_DIR / f"backtesting_gmm_{settings.START_DATE}_{settings.END_DATE}"
    )


def run_msm(tickers: List[Ticker]) -> None:
    msm_model = MarkovSwitchingVaR()
    msm_res = batch_backtest_model(
        msm_model,
        tickers,
        settings.START_DATE,
        settings.END_DATE,
        settings.ALPHA,
        settings.WINDOW_SHAPE,
    )
    msm_res.to_csv(
        OUTPUT_DIR / f"backtesting_msm_{settings.START_DATE}_{settings.END_DATE}"
    )


if __name__ == "__main__":
    tickers = fetch_valid_sp500_tickers(settings.START_DATE, settings.END_DATE)
    run_gmm(tickers)
    run_msm(tickers)
