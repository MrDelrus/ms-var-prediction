# ms-var-prediction

This repository implements the `MarkovSwitchingVaR` model and compares it to a
`GaussianMixtureVaR` baseline (based on `sklearn`) for **Value-at-Risk (VaR)**
prediction on equities. Models are fit on a rolling window and evaluated with
five statistical backtests (Kupiec, Christoffersen independence & conditional
coverage, a simple independence test, and a binomial coverage test).

## Installation

```bash
poetry install
```

## `backtester` CLI

A command-line tool for rolling-window VaR backtesting. After `poetry install`
it is available as the `backtester` entry point (or run it as a module:
`poetry run python -m ms_var_prediction.cli`).

### Quick start

```bash
mkdir -p out                      # --output must already exist
backtester --model GMM --begin 2025-01-01 --end 2026-01-01 --output out
```

### Options

| Flag         | Required | Default | Description |
|--------------|:--------:|---------|-------------|
| `--model`    | yes      | —       | `MSM` or `GMM` (case-insensitive). |
| `--begin`    | yes      | —       | Start date `YYYY-MM-DD`, **inclusive**. |
| `--end`      | yes      | —       | End date `YYYY-MM-DD`, **exclusive**. |
| `--output`   | yes      | —       | An **existing** directory. `features/` and `results/` subfolders are created inside it; the root is never created for you. |
| `--mode`     | no       | `full`  | `fit` (dump per-window features only), `predict` (VaR + tests from saved features), or `full` (both). |
| `--alpha`    | no       | `0.05`  | VaR significance level. |
| `--window`   | no       | `252`   | Rolling-window width in trading days (~1 calendar year on the NYSE). |
| `--n_components` | no   | model default (2) | Number of mixture components (GMM) / hidden states (MSM). |
| `--tickers`  | no       | `SPY`   | One or more ticker symbols, space-separated. |
| `--sp500`    | no       | off     | Flag; backtest the full S&P 500 constituent list instead of `--tickers`. |
| `--tag`      | no       | `default` | Run label embedded in filenames; keep stable to reuse cached features. |
| `--no-cache` | no       | off     | Flag; ignore and overwrite existing cached per-year features. |
| `--monitor`  | no       | off     | Flag; pushes the run to Weights & Biases (requires `WANDB_API_KEY`). |

Flags may be given in any order. Randomness is fixed to seed 42 before fitting.

### Output layout

`--output` must already exist. The tool writes:

```
out/
├── features/
│   └── {model}_{params}_{year}-01-01_{nextyear}-01-01_{tag}.csv   # one file PER YEAR
└── results/
    └── {model}_{params}_{begin}_{end}_{alpha}_{tag}.csv           # score row
```
`{params}` encodes the run hyperparameters, e.g. `nc2_w252` (2 states/components,
252-day window).

* **`features/`** — long-format table (`ticker, date, realized_return, params`),
  **split per calendar year**, where `params` is the JSON of the model's fitted
  parameters at each window position. Features are **alpha-independent**.
  - The **first calendar year is warm-up** (needed to build the initial window)
    and is not saved; for `2010-01-01 → 2026-01-01` that yields 15 yearly files
    (2011–2025).
  - A year already on disk is **reused automatically** (loaded, not recomputed)
    unless `--no-cache` is set, so an interrupted multi-year run resumes cheaply.
* **`results/`** — aggregate scores: a single row indexed by `(model, alpha)`
  with one column per test, the value being the fraction of tickers that passed.

### The fit / predict split

`fit` and `predict` let you separate the expensive part (rolling model fits)
from the cheap part (evaluating VaR at a given quantile):

```bash
# 1. Fit once — produces one feature file per year under features/
backtester --model GMM --begin 2024-01-01 --end 2026-01-01 --output out --mode fit

# 2. Backtest the same fits at several alphas, reusing the features
backtester --model GMM --begin 2024-01-01 --end 2026-01-01 --output out --mode predict --alpha 0.05
backtester --model GMM --begin 2024-01-01 --end 2026-01-01 --output out --mode predict --alpha 0.01
```

`full` (the default) runs both phases in one shot.

### Full S&P 500 backtest

```bash
mkdir -p outputs
backtester --model MSM --n_components 2 \
    --begin 2010-01-01 --end 2026-01-01 \
    --sp500 --output outputs --monitor
```
This fits 15 yearly files (2011–2025; 2010 is warm-up). Re-running resumes from
whatever years are already cached.

### Weights & Biases monitoring

With `--monitor`, each run logs to the `var-research` W&B project:

* all run parameters (model, dates, alpha, window, n_components, tag, universe) as config;
* `pct_fitted` — the percentage of tickers with results;
* **returns vs. predicted-VaR** plots for a sample of tickers (capped at 10);
* a **score** table (per-test pass fraction) and a **full per-stock** pass/fail table.

```bash
backtester --model MSM --begin 2025-01-01 --end 2026-01-01 \
    --output out --tickers SPY QQQ IWM --monitor
```

## Development

```bash
pre-commit run --all-files            # black + ruff
mypy src/                             # strict typing
poetry run python -m pytest tests/ -v
```
