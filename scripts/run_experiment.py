"""
Run a single named experiment from a suite TOML and log to W&B.

Usage
-----
poetry run python scripts/run_experiment.py \\
    --suite configs/experiments/baseline.toml \\
    --experiment msm_2state \\
    --wandb-project var-research \\
    --wandb-entity my-entity \\
    --tickers SPY AAPL MSFT
"""

from __future__ import annotations

import argparse
import tomllib
from datetime import datetime

from ms_var_prediction.experiment import run_experiment
from ms_var_prediction.tracking.wandb_tracker import WandBTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, help="Path to suite TOML")
    parser.add_argument("--experiment", required=True, help="Experiment name in TOML")
    parser.add_argument("--wandb-project", default="var-research")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run", default=None)
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM"])
    args = parser.parse_args()

    with open(args.suite, "rb") as f:
        suite_cfg = tomllib.load(f)

    exp_cfg = suite_cfg["experiments"][args.experiment]
    run_name = args.wandb_run or f"{args.experiment}_{datetime.now():%Y%m%d_%H%M%S}"

    tracker = WandBTracker(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=run_name,
    )
    tracker.start(
        config={**exp_cfg.get("hyperparams", {}), **exp_cfg["experiment_params"]}
    )

    results_df = run_experiment(exp_cfg, tracker, args.tickers)

    if exp_cfg.get("logging", {}).get("log_summary"):
        tracker.log_summary(
            {args.experiment: (results_df, exp_cfg.get("hyperparams", {}))}
        )

    tracker.finish()
    print(results_df)


if __name__ == "__main__":
    main()
