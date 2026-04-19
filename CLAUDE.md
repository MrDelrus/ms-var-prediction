# Markov Switching Model — VaR Prediction

## Project Purpose
Compares two statistical models for Value-at-Risk (VaR) prediction on S&P 500 stocks:
- **MarkovSwitchingVaR** (`models/markov_swiching_model.py`) — custom N-state Hidden Markov model with Gaussian emissions, fit via scipy MLE
- **GaussianMixtureVaR** (`models/gaussian_mixture_model.py`) — sklearn GMM baseline

Both predict VaR quantiles via a mixture-of-Gaussians CDF bisection, then are backtested with Kupiec, Christoffersen, and binomial statistical tests.

## Package Layout
```
src/ms_var_prediction/
  models/           # MSM and GMM model classes
  backtester/       # VaR backtest runner + statistical tests
  pipeline/         # Rolling-window evaluation, yfinance data fetching
  config.py         # Pydantic settings from .env
  constants.py      # EPS, SIGMA_EPS
  utils.py          # Numba-JIT Gaussian PDF/CDF, quantile bisection
  __main__.py       # CLI entry: python -m ms_var_prediction
notebooks/          # backtest.ipynb — main analysis
outputs/            # CSV results (pass/fail per ticker per test)
yfinance_cache/     # Local cache to avoid re-downloading price data
```

## Configuration
Settings live in `.env` (loaded by Pydantic in `config.py`):
```
START_DATE="2010-01-01"
END_DATE="2026-01-01"
alpha=0.05
window_shape=250
```

## Running
```bash
# Full CLI backtest (saves to outputs/)
python -m ms_var_prediction

# Interactive analysis
jupyter notebook notebooks/backtest.ipynb
```

## Key Results (from backtest.ipynb)
- MSM beats GMM on **independence tests** — captures regime clustering better
- GMM beats MSM on **unconditional coverage** (Kupiec/binomial)
- Conclusion: MS model better at temporal clustering; GMM better at raw frequency calibration

## Performance Notes
- `utils.py` and `models/loglikelyhood.py` use **Numba JIT** — first call will be slow (compilation)
- yfinance data is cached under `yfinance_cache/`

## Code Quality
```bash
pre-commit run --all-files   # black + ruff
mypy src/                    # strict typing enforced
```

## Dependencies
Managed with Poetry (`pyproject.toml`). Core: numpy, pandas, scipy, scikit-learn, numba, yfinance, matplotlib.
