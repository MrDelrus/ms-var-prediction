import yfinance as yf
import pandas as pd
import os

from tqdm.auto import tqdm
from contextlib import redirect_stderr
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional
from ms_var_prediction.config import DATA_DIR
from ms_var_prediction.logger import logger

_SP500_FILE_NAME = "constituents.csv"
_DEFAULT_CACHE_DIR = DATA_DIR / "yfinance_cache"
# Tolerance (calendar days) for the first/last available trading day vs the
# requested range, to absorb weekends and holidays.
_FULL_DATA_TOLERANCE_DAYS = 10


def _has_full_data(prices: pd.Series, begin: str, end: str) -> bool:
    """True if ``prices`` spans essentially the whole [begin, end) range."""
    if prices.empty:
        return False
    first = prices.index[0].date()
    last = prices.index[-1].date()
    b = date.fromisoformat(begin)
    e = date.fromisoformat(end)
    tol = timedelta(days=_FULL_DATA_TOLERANCE_DAYS)
    return first <= b + tol and last >= e - tol


def select_universe(begin: str, end: str, limit: Optional[int] = None) -> List[str]:
    """
    S&P 500 tickers with full price history over [begin, end).

    Ranked by average dollar volume (Close x Volume) descending; if ``limit`` is
    given, only the top ``limit`` are returned (a limit above the available count
    simply returns all of them).
    """
    scored: list[tuple[str, float]] = []
    for symbol in get_sp500_tickers():
        try:
            data = cached_history(yf.Ticker(symbol), begin, end)
            prices = data["Close"].dropna()
            if not _has_full_data(prices, begin, end):
                continue
            dollar_volume = float((data["Close"] * data["Volume"]).mean())
            scored.append((symbol, dollar_volume))
        except Exception as exc:  # noqa: BLE001 - skip unusable tickers
            logger.error("Skipping %s during universe selection: %s", symbol, exc)

    scored.sort(key=lambda pair: pair[1], reverse=True)
    symbols = [s for s, _ in scored]
    if limit is not None:
        symbols = symbols[:limit]
    logger.info("Universe: %d full-data tickers selected", len(symbols))
    return symbols


def get_sp500_tickers(cache_dir: Path = _DEFAULT_CACHE_DIR) -> List[str]:
    """
    Load the S&P 500 tickers from cache.

    Parameters
    ----------
    cache_dir : Path
        Directory where the tickers will be fetched from.

    Returns
    -------
    list of str
        List of S&P 500 ticker symbols.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    sp500_file = cache_dir / _SP500_FILE_NAME

    if not sp500_file.exists():
        error_message = "File with S&P 500 tickers can't be found."
        logger.error(error_message)
        raise RuntimeError(error_message)

    logger.info("Fetch list of S&P 500 tickers from cache.")
    return pd.read_csv(cache_dir / _SP500_FILE_NAME)["Symbol"].astype(str).to_list()


def fetch_valid_sp500_tickers(
    start_date: str, end_date: str, cache_dir: Path = _DEFAULT_CACHE_DIR
) -> List[yf.Ticker]:
    """
    Fetch all S&P 500 tickers with data in a given date range.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD'.
    end_date : str
        End date in 'YYYY-MM-DD' (exclusive).
    cache_dir : Path
        Directory to save cache files.

    Returns
    -------
    list of yf.Ticker
        List of Ticker objects with valid price history in the date range.
    """
    tickers = get_sp500_tickers(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"valid_sp500_{start_date}_{end_date}.txt"

    if cache_file.exists():
        logger.info("Loading valid S&P 500 tickers from cache.")
        with open(cache_file, "r") as f:
            valid_tickers_str = [line.strip() for line in f.readlines()]
        return [yf.Ticker(t) for t in valid_tickers_str]

    valid_tickers = []
    valid_tickers_str = []

    logger.info(f"Filtering S&P 500 tickers for range {start_date} - {end_date}")

    for ticker_symbol in tqdm(
        tickers, desc="Yahoo Finance Tickers Validated", leave=True, position=0
    ):
        try:
            with open(os.devnull, "w") as fnull:
                with redirect_stderr(fnull):
                    ticker = yf.Ticker(ticker_symbol)
                    df = ticker.history(start=start_date, end=end_date)
            if not df.empty:
                valid_tickers.append(ticker)
                valid_tickers_str.append(ticker_symbol)
        except Exception as e:
            tqdm.write("Error fetching %s: %s", ticker_symbol, e)

    with open(cache_file, "w") as f:
        for t in valid_tickers_str:
            f.write(f"{t}\n")

    logger.info("Fetching complete, found %i valid tickers", len(valid_tickers))
    return valid_tickers


def cached_history(
    ticker: yf.Ticker,
    start: str,
    end: str,
    cache_dir: Path = _DEFAULT_CACHE_DIR / "tickers",
) -> pd.DataFrame:
    """
    Download price history with caching for local development.

    Parameters
    ----------
    ticker : yf.Ticker
        Ticker object.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    cache_dir : str
        Directory to store cached CSV files.

    Returns
    -------
    pd.DataFrame
        Price history.
    """
    # Ensure cache folder exists
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker.ticker}_{start}_{end}.csv")

    if os.path.exists(cache_file):
        logger.info("Loading %s from local cache", ticker.ticker)
        data = pd.read_csv(cache_file, index_col=0)
        # Coerce the index to naive datetimes explicitly: bare parse_dates=True
        # leaves the index as strings on newer pandas, and yfinance CSVs may be
        # tz-aware. Drop any non-date header rows that slip in.
        idx = pd.to_datetime(data.index, errors="coerce", utc=True)
        data.index = idx.tz_convert(None)
        data = data[data.index.notna()]
    else:
        logger.info("Fetching %s from Yahoo Finance", ticker.ticker)
        data = ticker.history(start=start, end=end)
        data.to_csv(cache_file)

    return data
