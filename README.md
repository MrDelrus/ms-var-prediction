# ms-var-prediction

## Overview
Backtests two **Value-at-Risk** models on S&P 500 stocks and compares them:
a custom **Markov-switching** Gaussian HMM (`MarkovSwitchingVaR`) and a
**Gaussian-mixture** baseline (`GaussianMixtureVaR`). Each model is refit on a
rolling window, predicts the next day's VaR, and is scored with five statistical
backtests (binomial, simple independence, Kupiec, Christoffersen independence and
conditional coverage). It ships as a `backtester` command-line tool with optional
Weights & Biases monitoring.

## Structure
```
src/ms_var_prediction/
  cli.py            # `backtester` entry point: args -> fit/predict -> backtest -> W&B
  models/           # MarkovSwitchingVaR, GaussianMixtureVaR + build_model() factory
  backtester/       # VaR backtest runner and the 5 statistical tests
  pipeline/
    data.py         # yfinance price fetching + local caching, S&P 500 tickers
    features.py     # rolling fit -> per-year features, predict VaR, backtest, persistence
  plotting.py       # returns-vs-VaR and per-regime panels (model-agnostic)
  utils.py          # Numba-JIT Gaussian PDF/CDF and quantile bisection
  config.py         # PROJECT_ROOT / DATA_DIR / OUTPUT_DIR (CWD-anchored)
notebooks/          # model_comparison.ipynb — the analysis notebook
```
Fitting is split into two phases: **fit** writes per-year, alpha-independent
feature files (fitted model parameters per window) that are reused across runs;
**predict** reloads them, evaluates VaR at a given alpha, and runs the backtests.

## How to run
```bash
poetry install                       # installs deps + the `backtester` command

mkdir -p outputs                     # --output must already exist
# Full S&P 500 backtest, MSM with 2 states, monitored to W&B:
backtester --model MSM --n_components 2 \
    --begin 2010-01-01 --end 2026-01-01 --sp500 --output outputs --monitor

# Reuse the cached features to score another alpha (fast, no refitting):
backtester --model MSM --begin 2010-01-01 --end 2026-01-01 \
    --sp500 --output outputs --mode predict --alpha 0.01
```

Main flags: `--model {MSM,GMM}`, `--begin/--end` (YYYY-MM-DD), `--output` (an
existing dir), `--mode {fit,predict,full}` (default `full`), `--alpha` (default
0.05), `--window` (default 252), `--n_components` (default 2), `--sp500`,
`--tag`, `--no-cache`, `--monitor`. Results are written to
`<output>/features/` (per year) and `<output>/results/` (aggregate scores);
runtime data is anchored to the working directory (override with `MS_VAR_DATA_DIR`).
