# Experiment artifacts — MSM vs GMM VaR on the S&P 500

Backtest results for the 8-config sweep. Only `results/` is committed; the large
`features/` files are kept out of git (see Notes).

## Run configuration
- **Models × states:** `MSM` and `GMM`, each with `n_components ∈ {2, 3, 4, 5}` → 8 configs.
- **Period:** `2010-01-01 → 2026-01-01` (exclusive). 2010 is warm-up (window = 252
  trading days), so results cover **15 full years (2011–2025)**.
- **Universe:** the **421** S&P 500 constituents with full price history over the range,
  ranked by average dollar volume.
- **Alphas:** `0.05` and `0.01`. Seed fixed at 42.
- Produced by `backtester run`.

## Layout
```
artifacts/
├── README.md                    # this file
└── results/                     # per-stock pass/fail tables (committed)
    └── {model}_n{N}_a{alpha}.csv
```
16 files: `{msm,gmm} × n{2,3,4,5} × a{0.05,0.01}`, e.g. `msm_n2_a0.05.csv`.

## File schema
Each **`results/{model}_n{N}_a{alpha}.csv`** is a **per-stock** table:
- rows: the 421 tickers (index column `ticker`);
- columns: `binomial_passed, independence_simple_passed, kupiec_passed,
  christoffersen_independence_passed, christoffersen_conditional_passed`;
- entries: `true` / `false` — whether that stock passes the test at that alpha.

Aggregate pass-rates (e.g. the notebook's summary tables) are derived by counting
`true` values per column.

## Notes
- `features/` (the per-window fitted parameters, ~4.6 GB) is **not committed**. It is
  regenerable with `backtester run --model … --features-path …` and can then be
  re-scored at any alpha to reproduce these `results/` files.
