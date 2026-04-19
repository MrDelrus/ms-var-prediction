"""
Entry point: python -m ms_var_prediction [--suite PATH] [--tickers TICKER ...]

Defaults to configs/experiments/baseline.toml with tickers SPY QQQ IWM.
"""

from __future__ import annotations

import argparse
import tomllib
from datetime import datetime

from ms_var_prediction.config import PROJECT_ROOT
from ms_var_prediction.experiment import run_experiment
from ms_var_prediction.tracking.wandb_tracker import WandBTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        default=str(PROJECT_ROOT / "configs" / "experiments" / "baseline.toml"),
    )
    parser.add_argument("--wandb-project", default="var-research")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "IWM"])
    args = parser.parse_args()

    with open(args.suite, "rb") as f:
        suite_cfg = tomllib.load(f)

    suite_name: str = suite_cfg.get("suite", {}).get("name", "suite")
    run_name = f"{suite_name}_{datetime.now():%Y%m%d_%H%M%S}"
    first_exp = next(iter(suite_cfg["experiments"].values()))

    tracker = WandBTracker(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=run_name,
    )
    tracker.start(
        config={"suite": suite_name, **first_exp.get("experiment_params", {})}
    )

    all_results: dict = {}
    for exp_name, exp_cfg in suite_cfg["experiments"].items():
        print(f"\n=== Running experiment: {exp_name} ===")
        results_df = run_experiment(exp_cfg, tracker, args.tickers)
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
