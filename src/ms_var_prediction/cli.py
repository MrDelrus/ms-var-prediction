"""
``backtester`` — command-line VaR backtesting tool.

Subcommands
-----------
``backtester run`` fits a Markov-Switching or Gaussian-Mixture VaR model on a
rolling window over the S&P 500 (stocks with full data in the range), optionally
caches the fitted parameters to a features file, and — when alphas are given —
scores the predicted VaR with five statistical backtests, writing a results
table.

See the README for the full option reference.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ms_var_prediction.pipeline import features as fs

DEFAULT_WINDOW = 252  # NYSE averages ~252 trading days per calendar year
SEED = 42


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backtester",
        description="Rolling-window VaR backtester for MSM / GMM models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="Fit features and (optionally) run the VaR backtests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run.add_argument(
        "--model",
        "-m",
        required=True,
        type=str.upper,
        choices=["MSM", "GMM"],
        help="Model to backtest.",
    )
    run.add_argument(
        "--n-components",
        "-n",
        type=int,
        default=1,
        help="Number of mixture components / hidden states (>= 1; MSM needs >= 2).",
    )
    run.add_argument(
        "--window-size",
        "--window_size",
        "-w",
        type=int,
        default=DEFAULT_WINDOW,
        dest="window_size",
        help="Rolling window width in trading days (~1 year of warm-up).",
    )
    run.add_argument(
        "--begin", "-b", required=True, help="Start date YYYY-MM-DD (inclusive)."
    )
    run.add_argument(
        "--end", "-e", required=True, help="End date YYYY-MM-DD (exclusive)."
    )
    run.add_argument(
        "--features-path",
        "-f",
        type=Path,
        default=None,
        dest="features_path",
        help="Features CSV. If it exists it is verified and reused; otherwise the "
        "built features are saved there.",
    )
    run.add_argument(
        "--alphas",
        "-a",
        default=None,
        help="Comma-separated VaR levels, e.g. '0.01,0.05' (no spaces). If omitted, "
        "only features are built.",
    )
    run.add_argument(
        "--results-path",
        "-r",
        type=Path,
        default=None,
        dest="results_path",
        help="Results CSV path. Required with --alphas, forbidden without it.",
    )
    run.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Use only the top N stocks by average dollar volume.",
    )
    run.add_argument(
        "--monitor",
        action="store_true",
        help="Log run config and the results table to Weights & Biases.",
    )
    run.set_defaults(func=cmd_run)
    return parser


# ---------------------------------------------------------------------------
# Validation (clear errors, no stack traces)
# ---------------------------------------------------------------------------


def _parse_date(value: str, flag: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{flag}: '{value}' is not a valid date (expected YYYY-MM-DD)")


def _parse_alphas(raw: str) -> list[float]:
    if raw.strip() != raw or " " in raw:
        raise ValueError("--alphas must not contain spaces (format: 0.01,0.05,0.001)")
    parts = raw.split(",")
    alphas: list[float] = []
    for part in parts:
        if not re.fullmatch(r"\d*\.?\d+", part):
            raise ValueError(
                f"--alphas: '{part}' is not a valid number (format: 0.01,0.05,0.001)"
            )
        value = float(part)
        if not 0.0 < value < 1.0:
            raise ValueError(f"--alphas: {value} must be strictly between 0 and 1")
        alphas.append(value)
    if not alphas:
        raise ValueError("--alphas is empty")
    return alphas


def _require_parent_dir(path: Path, flag: str) -> None:
    parent = path.parent if str(path.parent) else Path(".")
    if not parent.is_dir():
        raise ValueError(
            f"{flag}: directory '{parent}' does not exist (create it first)"
        )


def validate(args: argparse.Namespace) -> None:
    """Validate/normalise args in place; raise ValueError with a clear message."""
    if args.n_components < 1:
        raise ValueError("--n-components must be an integer >= 1")
    if args.model == "MSM" and args.n_components < 2:
        raise ValueError("MSM requires --n-components >= 2")
    if args.window_size <= 0:
        raise ValueError("--window-size must be a positive integer (days)")

    begin = _parse_date(args.begin, "--begin")
    end = _parse_date(args.end, "--end")
    if begin >= end:
        raise ValueError(
            f"--begin ({args.begin}) must be strictly before --end ({args.end})"
        )

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer")

    if args.alphas is not None:
        args.alphas = _parse_alphas(args.alphas)
        if args.results_path is None:
            raise ValueError("--results-path is required when --alphas is provided")
        _require_parent_dir(args.results_path, "--results-path")
    else:
        if args.results_path is not None:
            raise ValueError("--results-path may only be used together with --alphas")
        if args.features_path is None:
            raise ValueError(
                "nothing to do: pass --alphas (to run tests) and/or --features-path "
                "(to save features)"
            )

    if args.features_path is not None:
        _require_parent_dir(args.features_path, "--features-path")


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> None:
    np.random.seed(SEED)
    model = args.model.lower()
    n = args.n_components

    # ---- features: reuse if the file exists, else build (and save if a path given)
    if args.features_path is not None and args.features_path.exists():
        print(f"Reusing features from {args.features_path}")
        df = fs.load_features(args.features_path)
        fs.verify_features(df, model, n, args.begin, args.end)
    else:
        tickers = _select_tickers(args)
        print(
            f"Building {args.model} features | n={n} window={args.window_size} | "
            f"{len(tickers)} stocks | {args.begin}..{args.end}"
        )
        df = fs.build_features(
            model, n, tickers, args.begin, args.end, args.window_size, seed=SEED
        )
        if df.empty:
            print(
                "error: no features produced for any stock; aborting.", file=sys.stderr
            )
            sys.exit(1)
        if args.features_path is not None:
            fs.save_features(df, args.features_path)
            print(f"Saved features -> {args.features_path} ({len(df)} rows)")

    if not args.alphas:
        print("Done (features only; no --alphas given).")
        return

    # ---- predict + backtest
    print(f"Testing VaR at alphas={args.alphas} over {len(df)} windows...")
    results = fs.predict_and_test(df, model, n, args.alphas)
    results.to_csv(args.results_path, index=False)
    print(f"Saved results -> {args.results_path}")
    print(results.to_string(index=False))

    if args.monitor:
        _log_to_wandb(args, model, n, results)


def _select_tickers(args: argparse.Namespace) -> list[str]:
    from ms_var_prediction.pipeline.data import select_universe

    tickers = select_universe(args.begin, args.end, args.limit)
    if not tickers:
        raise ValueError(
            f"no S&P 500 stocks have full data over [{args.begin}, {args.end})"
        )
    return tickers


def _log_to_wandb(args: argparse.Namespace, model: str, n: int, results: Any) -> None:
    import wandb

    settings = wandb.Settings(init_timeout=180)
    config = {
        "model": model,
        "n_components": n,
        "window_size": args.window_size,
        "begin": args.begin,
        "end": args.end,
        "alphas": args.alphas,
        "limit": args.limit,
    }
    name = f"{model}_nc{n}_{args.begin}_{args.end}"
    run = None
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
            break
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            print(
                f"W&B online init failed (attempt {attempt}/3): {exc}", file=sys.stderr
            )
    if run is None:
        run = wandb.init(
            project="var-research",
            name=name,
            config=config,
            mode="offline",
            reinit=True,
            settings=settings,
        )
        print(f"W&B online unavailable; logging OFFLINE to {run.dir}")
    run.log({"results": wandb.Table(dataframe=results)})
    run.finish()


def main() -> None:
    args = build_parser().parse_args()
    try:
        validate(args)
        args.func(args)
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
