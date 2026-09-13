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
  cli.py            # `backtester` entry point: args -> build features -> test -> W&B
  models/           # MarkovSwitchingVaR, GaussianMixtureVaR + build_model() factory
  backtester/       # VaR backtest runner and the 5 statistical tests
  pipeline/
    data.py         # yfinance fetching + caching; S&P 500 universe selection
    features.py     # rolling fit -> flattened features, predict VaR, run tests
  plotting.py       # returns-vs-VaR and per-regime panels
  utils.py          # Numba-JIT Gaussian PDF/CDF and quantile bisection
  config.py         # PROJECT_ROOT / DATA_DIR / OUTPUT_DIR (CWD-anchored)
notebooks/          # model_comparison.ipynb — the analysis notebook
```

## How to run

Install (adds the `backtester` command to the environment):
```bash
poetry install
```

The tool has one subcommand, `backtester run`, with two phases:

1. **Build features** — roll the window over every eligible stock, fit the model
   at each position, and store the fitted parameters (one row per stock-day,
   one column per parameter). Features are **alpha-independent**.
2. **Predict + test** — reload those parameters, evaluate VaR at each `--alphas`
   value, and count how many stocks pass each test. Runs only when `--alphas` is
   given.

**Universe:** every S&P 500 constituent with *full* price history over
`[--begin, --end)` (stocks that IPO'd late or stopped trading are dropped),
ranked by average dollar volume. `--limit N` keeps the top N.

### Options
| Flag | Short | Required | Default | Meaning |
|------|-------|----------|---------|---------|
| `--model` | `-m` | yes | — | `MSM` or `GMM` (case-insensitive). |
| `--n-components` | `-n` | no | `1` | Components / hidden states. Must be ≥ 1; **MSM requires ≥ 2**. |
| `--window-size` | `-w` | no | `252` | Rolling window in trading days (~1 year of warm-up). |
| `--begin` | `-b` | yes | — | Start date `YYYY-MM-DD`, inclusive. |
| `--end` | `-e` | yes | — | End date `YYYY-MM-DD`, exclusive. Must be after `--begin`. |
| `--features-path` | `-f` | no | — | Features CSV. If it **exists**, it is verified (model, states, date range) and reused; if not, the built features are saved there. |
| `--alphas` | `-a` | no | — | VaR levels as `0.01,0.05,0.001` (comma-separated, `.` decimal, **no spaces**), each in (0, 1). Omit to only build features. |
| `--results-path` | `-r` | with `--alphas` | — | Where to write the results CSV. Required iff `--alphas` is given. |
| `--limit` | `-l` | no | — | Use only the top N stocks by dollar volume (N above the count is allowed). |
| `--monitor` | | no | off | Log run config + the results table to Weights & Biases (needs `WANDB_API_KEY`). |

Any invalid argument stops the run with a one-line `error: …` explanation
(exit code 2). You must pass `--alphas` and/or `--features-path` — otherwise
there is nothing to do.

### Examples
```bash
# Build & cache features only (no testing):
backtester run -m MSM -n 2 -b 2010-01-01 -e 2026-01-01 -f features/msm_n2.csv

# Reuse those features and score several alphas into a results table:
backtester run -m MSM -n 2 -b 2010-01-01 -e 2026-01-01 \
    -f features/msm_n2.csv -a 0.05,0.01 -r results/msm_n2.csv

# One-shot build + test, top-50 stocks, with W&B monitoring:
backtester run -m GMM -n 3 -b 2010-01-01 -e 2026-01-01 \
    -a 0.05,0.01 -r results/gmm_n3.csv -l 50 --monitor
```

### Output formats
- **Features CSV** (`--features-path`): `date, ticker, realized_return,` then the
  flattened parameters. MSM → `mu_i, sigma_i, trans_i_j, state_prob_i`;
  GMM → `weight_i, mean_i, cov_i` (`cov_i` is the component std dev).
- **Results CSV** (`--results-path`): one row per alpha —
  `alpha, total, <test>_passed` for each of the five tests, where `total` is the
  number of stocks tested and `<test>_passed` is how many passed.

Runtime data (price cache, outputs) is anchored to the working directory;
override with the `MS_VAR_DATA_DIR` environment variable. Randomness is seeded
(42) before fitting for reproducibility.
