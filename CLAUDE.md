# Markov Switching Model — VaR Prediction

## Project Purpose
Compares two statistical models for Value-at-Risk (VaR) prediction on S&P 500 stocks:
- **MarkovSwitchingVaR** (`models/markov_switching_model.py`) — custom N-state Hidden Markov model with Gaussian emissions, fit via scipy MLE
- **GaussianMixtureVaR** (`models/gaussian_mixture_model.py`) — sklearn GMM baseline

Both predict VaR quantiles via a mixture-of-Gaussians CDF bisection, then are backtested with binomial, simple-independence, Kupiec, and Christoffersen (independence + conditional) statistical tests.

## Package Layout
```
src/ms_var_prediction/
  models/           # MSM and GMM model classes + build_model() factory
  backtester/       # VaR backtest runner + statistical tests
  pipeline/
    data.py         # yfinance fetching + local caching, S&P 500 tickers
    features.py     # rolling fit -> per-year features, predict, backtest, persistence
  cli.py            # `backtester` CLI entry point (argparse + orchestration + W&B)
  plotting.py       # returns-vs-VaR and per-regime (model-agnostic) panels
  config.py         # PROJECT_ROOT / DATA_DIR / OUTPUT_DIR (CWD-anchored)
  constants.py      # EPS, SIGMA_EPS
  utils.py          # Numba-JIT Gaussian PDF/CDF, quantile bisection
notebooks/          # model_comparison.ipynb — the analysis notebook
yfinance_cache/     # Local cache to avoid re-downloading price data
```

## Running
The entry point is the `backtester` CLI (installed via `poetry install`, or run as
`python -m ms_var_prediction.cli`). See `README.md` for the full flag reference.

```bash
# Full backtest of the S&P 500 with W&B monitoring
backtester --model MSM --n_components 2 \
    --begin 2010-01-01 --end 2026-01-01 --sp500 --output outputs --monitor

# Reuse cached per-year features for another alpha (predict only, fast)
backtester --model MSM --begin 2010-01-01 --end 2026-01-01 \
    --sp500 --output outputs --mode predict --alpha 0.01
```

- `--output` must exist; the tool writes `features/` (per-year, alpha-independent,
  auto-reused) and `results/` (aggregate score row) inside it.
- Runtime data (cache, outputs) is anchored to the working directory; override with
  the `MS_VAR_DATA_DIR` env var. Randomness is seeded (42) for reproducibility.

## Key Results (see notebooks/model_comparison.ipynb)
Across {MSM, GMM} × {2, 3 states} × {α=5%, 1%} on ~498 S&P 500 stocks:
- MSM beats GMM on **independence tests** — captures regime clustering better
- GMM beats MSM on **unconditional coverage** (binomial / Kupiec)
- **MSM with 2 states** is the sweet spot — the only config strong on both axes
  (best conditional coverage); 3 states over-tighten coverage. α=1% is hard for all.

## Performance Notes
- `utils.py` and `models/loglikelihood.py` use **Numba JIT** — the first call is slow (compilation)
- yfinance data is cached under `yfinance_cache/`

## Code Quality
```bash
pre-commit run --all-files   # black + ruff
mypy src/                    # strict typing enforced
poetry run python -m pytest tests/ -v
```

## Dependencies
Managed with Poetry (`pyproject.toml`). Core: numpy, pandas, scipy, scikit-learn, numba, yfinance, matplotlib, wandb.
