"""
``backtester`` — command-line VaR backtesting tool.

Two-phase backtest of a Markov-Switching or Gaussian-Mixture VaR model over a
rolling window, with optional Weights & Biases monitoring.

Examples
--------
# Full pipeline (fit + predict) on SPY, default window/alpha, into ./out
backtester --model GMM --begin 2025-01-01 --end 2026-01-01 --output ./out

# Reuse cached features for a different alpha (predict only)
backtester --model GMM --begin 2025-01-01 --end 2026-01-01 \\
    --output ./out --mode predict --alpha 0.01

# Multiple tickers with W&B monitoring
backtester --model MSM --begin 2025-01-01 --end 2026-01-01 \\
    --output ./out --tickers SPY QQQ IWM --monitor

Output layout (``--output`` must already exist; subfolders are created)::

    <output>/
      features/{model}_{params}_{year}-01-01_{nextyear}-01-01_{tag}.csv  # one per year
      results/{model}_{params}_{begin}_{end}_{alpha}_{tag}.csv           # score row

Features are split per year and alpha-independent: a year already on disk is
reused automatically (unless ``--no-cache``), so an interrupted multi-year run
resumes cheaply. The first calendar year is warm-up and is not saved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ms_var_prediction.pipeline import features as fs

# NYSE averages ~252 trading days per calendar year.
DEFAULT_WINDOW = 252
DEFAULT_ALPHA = 0.05
SEED = 42
# Fixed set of popular, liquid mega-caps for the detailed per-ticker W&B panels
# (4 graphs each). Tickers with shorter histories simply start later.
PANEL_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "JNJ",
    "XOM",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtester",
        description="Rolling-window VaR backtester for MSM / GMM models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- required: what / when / where -------------------------------------
    parser.add_argument(
        "--model",
        required=True,
        type=str.upper,
        choices=["MSM", "GMM"],
        help="Model to backtest.",
    )
    parser.add_argument(
        "--begin",
        required=True,
        help="Start date YYYY-MM-DD (inclusive).",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date YYYY-MM-DD (exclusive).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Existing directory; 'features/' and 'results/' are written inside.",
    )
    # --- optional: how -----------------------------------------------------
    parser.add_argument(
        "--mode",
        choices=["fit", "predict", "full"],
        default="full",
        help="fit: dump per-window features; predict: VaR + tests from saved "
        "features; full: both.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="VaR significance level.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Rolling window width in trading days (~1 year).",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=None,
        help="Number of mixture components (GMM) / hidden states (MSM). "
        "Defaults to the model's own default (2).",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY"],
        help="One or more ticker symbols to backtest (ignored when --sp500 is set).",
    )
    parser.add_argument(
        "--sp500",
        action="store_true",
        help="Backtest the full S&P 500 constituent list instead of --tickers.",
    )
    parser.add_argument(
        "--tag",
        default="default",
        help="Run label embedded in feature/result filenames; keep it stable to "
        "reuse cached features across runs.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore (and overwrite) existing cached per-year features.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Push params, fit rate, plots and test tables to Weights & Biases.",
    )
    return parser


def _resolve_dirs(output: Path) -> tuple[Path, Path]:
    """Validate the output root exists and create features/ + results/ inside."""
    if not output.is_dir():
        raise NotADirectoryError(
            f"--output directory does not exist: {output} "
            "(it must be created in advance)."
        )
    features_dir = output / "features"
    results_dir = output / "results"
    features_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    return features_dir, results_dir


def _hyperparams(model_type: str, n_components: int | None) -> dict[str, int]:
    """Map the shared --n_components flag onto each model's constructor arg."""
    if n_components is None:
        return {}
    key = "n_states" if model_type == "msm" else "n_components"
    return {key: n_components}


def _resolve_tickers(args: argparse.Namespace) -> list[str]:
    if args.sp500:
        from ms_var_prediction.pipeline.data import get_sp500_tickers

        return get_sp500_tickers()
    return list(args.tickers)


def _build_ticker_panels(
    model_type: str,
    panel_features: pd.DataFrame,
    alpha: float,
    begin: str,
    end: str,
) -> dict[str, Any]:
    """
    Build the 4 detailed W&B panels for each PANEL_TICKERS stock present in
    ``panel_features``: full returns+VaR, per-regime std, per-regime mean, and
    probability of the calm regime. Model-agnostic (MSM and GMM alike).
    """
    import matplotlib.pyplot as plt
    import wandb
    import yfinance as yf

    from ms_var_prediction.pipeline.data import cached_history
    from ms_var_prediction.plotting import (
        build_regime_timeseries,
        plot_regime_metric,
        plot_regime_prob,
        plot_returns_and_var,
    )
    from ms_var_prediction.utils import prices_to_returns

    payload: dict[str, Any] = {}
    for tk in PANEL_TICKERS:
        rows = panel_features[panel_features["ticker"] == tk]
        if rows.empty:
            continue

        preds = fs.predict_var(rows, model_type, alpha)
        try:
            prices = cached_history(yf.Ticker(tk), begin, end)["Close"].dropna()
            returns = pd.Series(prices_to_returns(prices), index=prices.index[1:])
            fig = plot_returns_and_var(tk, returns, preds, alpha)
            payload[f"ticker/{tk}/1_returns_var"] = wandb.Image(fig)
            plt.close(fig)
        except Exception:  # noqa: BLE001 - a missing price series shouldn't abort
            pass

        ts = build_regime_timeseries(rows)
        f_std = plot_regime_metric(
            tk, ts, "std", "Std dev (σ)", "per-regime volatility"
        )
        payload[f"ticker/{tk}/2_regime_std"] = wandb.Image(f_std)
        plt.close(f_std)
        f_mean = plot_regime_metric(
            tk, ts, "mean", "Mean (μ)", "per-regime mean return"
        )
        payload[f"ticker/{tk}/3_regime_mean"] = wandb.Image(f_mean)
        plt.close(f_mean)
        f_prob = plot_regime_prob(tk, ts)
        payload[f"ticker/{tk}/4_regime_prob"] = wandb.Image(f_prob)
        plt.close(f_prob)

    return payload


def run(args: argparse.Namespace) -> None:
    # Fix all randomness before the fit process for reproducibility.
    np.random.seed(SEED)

    model_type = args.model.lower()
    features_dir, results_dir = _resolve_dirs(args.output)
    hyperparams = _hyperparams(model_type, args.n_components)
    n_comp = args.n_components if args.n_components is not None else 2
    p_str = fs.params_str(n_comp, args.window)
    years = fs.full_years(args.begin, args.end)
    if not years:
        print(
            f"error: range {args.begin}..{args.end} leaves no full year after "
            "warm-up; widen the dates.",
            file=sys.stderr,
        )
        sys.exit(1)
    tickers = _resolve_tickers(args)

    # Initialise W&B up front so the run link is live from the start and shows
    # per-year progress (and live CPU/RAM) while fitting, not only at the end.
    run = _wandb_init(args, model_type, p_str, len(tickers)) if args.monitor else None

    def year_path(year: int) -> Path:
        return fs.year_features_path(features_dir, model_type, p_str, year, args.tag)

    frames: list[pd.DataFrame] = []
    # Accumulates only the panel tickers' rows so the per-year panels can be
    # rebuilt cheaply (and the VaR line grows year by year on W&B).
    panel_frames: list[pd.DataFrame] = []

    # ---- fit phase: one file per year, reused when already present --------
    if args.mode in ("fit", "full"):
        print(
            f"Fitting {model_type.upper()} | {p_str} | {len(tickers)} tickers | "
            f"years {years[0]}-{years[-1]}"
        )
        for i, year in enumerate(years, start=1):
            path = year_path(year)
            if path.exists() and not args.no_cache:
                print(f"[{year}] cache hit -> reuse {path.name}")
                df_year = fs.load_features(path)
                frames.append(df_year)
                pct = None
            else:
                if path.exists() and args.no_cache:
                    path.unlink()
                    print(f"[{year}] --no-cache: removed {path.name}")
                print(f"[{year}] fitting {len(tickers)} tickers...")
                df_year, stats = fs.compute_year_features(
                    model_type,
                    tickers,
                    year,
                    args.begin,
                    args.end,
                    args.window,
                    hyperparams=hyperparams,
                    seed=SEED,
                )
                fs.save_features(df_year, path)
                print(
                    f"[{year}] saved {len(df_year)} rows -> {path.name} "
                    f"({stats.pct_fitted:.0f}% of tickers fitted)"
                )
                frames.append(df_year)
                pct = stats.pct_fitted
            if run is not None:
                log: dict[str, Any] = {
                    "progress/year": year,
                    "progress/years_done": i,
                    "progress/years_total": len(years),
                    "progress/rows": len(df_year),
                }
                if pct is not None:
                    log["progress/pct_fitted_year"] = pct
                panel_frames.append(df_year[df_year["ticker"].isin(PANEL_TICKERS)])
                panels = _build_ticker_panels(
                    model_type,
                    pd.concat(panel_frames, ignore_index=True),
                    args.alpha,
                    args.begin,
                    args.end,
                )
                run.log({**log, **panels})
        if args.mode == "fit":
            if run is not None:
                run.finish()
            return
    else:  # predict: load whatever year files exist
        for year in years:
            path = year_path(year)
            if path.exists():
                frames.append(fs.load_features(path))
        if not frames:
            raise FileNotFoundError(
                f"No cached year features for {model_type} {p_str} tag={args.tag} "
                f"in {features_dir}. Run with --mode fit (or full) first."
            )

    features_df = pd.concat(frames, ignore_index=True)
    if features_df.empty:
        print("error: no features available to predict on; aborting.", file=sys.stderr)
        sys.exit(1)

    # ---- predict + backtest phase ----------------------------------------
    print(f"Predicting VaR at alpha={args.alpha} over {len(features_df)} windows...")
    predictions_df = fs.predict_var(features_df, model_type, args.alpha)
    per_stock_df, score = fs.backtest_predictions(predictions_df, args.alpha)

    res_path = fs.results_path(
        results_dir, model_type, p_str, args.begin, args.end, args.alpha, args.tag
    )
    fs.save_results(score, model_type, args.alpha, res_path)
    print(f"Saved results -> {res_path}")

    pct_fitted = 100.0 * len(per_stock_df) / max(len(tickers), 1)
    print(
        f"\nTickers with results: {len(per_stock_df)}/{len(tickers)} ({pct_fitted:.0f}%)"
    )
    print("\nScore (pass fraction per test):")
    print(score.to_string())

    if run is not None:
        _wandb_log_final(
            run, args, model_type, features_df, per_stock_df, score, pct_fitted
        )
        run.finish()


def _wandb_init(
    args: argparse.Namespace, model_type: str, p_str: str, n_tickers: int
) -> Any:
    import wandb

    name = f"{model_type}_{p_str}_{args.begin}_{args.end}_a{args.alpha}_{args.tag}"
    config = {
        "model": model_type,
        "begin": args.begin,
        "end": args.end,
        "alpha": args.alpha,
        "window": args.window,
        "n_components": args.n_components if args.n_components is not None else 2,
        "mode": args.mode,
        "tag": args.tag,
        "universe": "sp500" if args.sp500 else "tickers",
        "n_tickers": n_tickers,
    }
    settings = wandb.Settings(init_timeout=180)

    # A transient W&B outage must never abort a multi-hour backtest: retry the
    # online init a few times, then fall back to offline (logs locally; can be
    # `wandb sync`-ed later) so the run always completes.
    for attempt in range(1, 4):
        try:
            run = wandb.init(
                project="var-research",
                name=name,
                config=config,
                mode="online",
                reinit=True,
                settings=settings,
            )
            print(f"W&B run: {run.url}")
            return run
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on any W&B error
            print(
                f"W&B online init failed (attempt {attempt}/3): {exc}",
                file=sys.stderr,
            )

    run = wandb.init(
        project="var-research",
        name=name,
        config=config,
        mode="offline",
        reinit=True,
        settings=settings,
    )
    print(f"W&B online unavailable; logging OFFLINE to {run.dir} (sync later).")
    return run


def _wandb_log_final(
    run: Any,
    args: argparse.Namespace,
    model_type: str,
    features_df: pd.DataFrame,
    per_stock_df: pd.DataFrame,
    score: pd.Series,
    pct_fitted: float,
) -> None:
    import wandb

    final: dict[str, Any] = {"pct_fitted": pct_fitted}

    # In predict mode the per-year fit loop never ran, so build the per-ticker
    # panels here. In full/fit mode they were already logged live each year.
    if args.mode == "predict":
        panel_features = features_df[features_df["ticker"].isin(PANEL_TICKERS)]
        final.update(
            _build_ticker_panels(
                model_type, panel_features, args.alpha, args.begin, args.end
            )
        )

    # Score (per-test pass fraction) and the full per-stock pass table.
    score_tbl = score.to_frame().T
    score_tbl.insert(0, "model", model_type)
    final["score"] = wandb.Table(dataframe=score_tbl)
    final["per_stock"] = wandb.Table(dataframe=per_stock_df.reset_index())
    run.log(final)


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except (NotADirectoryError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
