"""
Run all experiments defined in a suite TOML under a single W&B run.

Usage
-----
# Explicit tickers:
poetry run python scripts/run_suite.py \\
    --suite configs/experiments/baseline.toml \\
    --wandb-project var-research \\
    --tickers SPY AAPL MSFT

# Full S&P 500 (filtered for date range from first experiment):
poetry run python scripts/run_suite.py \\
    --suite configs/experiments/sp500_16y_2state.toml \\
    --wandb-project var-research \\
    --sp500
"""

from __future__ import annotations

import argparse
import tomllib
from datetime import datetime

from ms_var_prediction.experiment import run_experiment
from ms_var_prediction.pipeline.data import fetch_valid_sp500_tickers
from ms_var_prediction.tracking.wandb_tracker import WandBTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--wandb-project", default="var-research")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument(
        "--sp500",
        action="store_true",
        help="Use all valid S&P 500 tickers for the experiment date range",
    )
    args = parser.parse_args()

    with open(args.suite, "rb") as f:
        suite_cfg = tomllib.load(f)

    suite_name: str = suite_cfg.get("suite", {}).get("name", "suite")
    run_name = f"{suite_name}_{datetime.now():%Y%m%d_%H%M%S}"
    first_exp = next(iter(suite_cfg["experiments"].values()))
    global_config = {"suite": suite_name, **first_exp.get("experiment_params", {})}

    if args.sp500:
        ep = first_exp["experiment_params"]
        print(
            f"Fetching valid S&P 500 tickers for {ep['start_date']} – {ep['end_date']}..."
        )
        ticker_objs = fetch_valid_sp500_tickers(ep["start_date"], ep["end_date"])
        tickers = [t.ticker for t in ticker_objs]
        print(f"Found {len(tickers)} valid tickers.")
    else:
        tickers = args.tickers or ["SPY", "QQQ", "IWM"]

    tracker = WandBTracker(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=run_name,
    )
    tracker.start(config=global_config)

    all_results: dict = {}
    for exp_name, exp_cfg in suite_cfg["experiments"].items():
        print(f"\n=== Running experiment: {exp_name} ===")
        results_df = run_experiment(exp_cfg, tracker, tickers)
        all_results[exp_name] = (results_df, exp_cfg.get("hyperparams", {}))
        print(results_df)

    tracker.log_summary(all_results)
    tracker.finish()

    print("\n=== Suite complete ===")
    for exp_name, (df, _) in all_results.items():
        print(f"\n{exp_name}:")
        print(df)


if __name__ == "__main__":
    main()
