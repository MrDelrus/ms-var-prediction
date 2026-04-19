# Refactoring Plan

## Goal
Clean project with W&B experiment tracking, config-driven multi-model experiments, warm-start support.

## Order of Work
1. **Tests** — pytest with synthetic fixtures, covering models / pipeline / backtester
2. **W&B integration** — tracker connector, warm-start params on models, run scripts
3. **configs/ migration** — move .env → configs/default.toml, update config.py

---

## Phase 1 — Tests ✅ / ⬜

| File | What it tests |
|------|--------------|
| `tests/conftest.py` | Fixtures: synthetic returns, pre-fit MSM & GMM |
| `tests/test_models.py` | MSM/GMM fit, predict, get/load fitted params |
| `tests/test_var_tests.py` | kupiec, christoffersen, binomial, independence |
| `tests/test_pipeline.py` | rolling_returns_and_var, Backtester.test() |
| `tests/test_config.py` | Settings load from TOML |

---

## Phase 2 — W&B Integration ⬜

### New files
- `src/ms_var_prediction/tracking/wandb_tracker.py` — `WandBTracker` class  
- `scripts/run_experiment.py` — runs one named experiment from a TOML suite  
- `scripts/run_suite.py` — reads suite TOML, runs all experiments under one W&B run  

### Model changes
Both `MarkovSwitchingVaR` and `GaussianMixtureVaR` get:
- `get_fitted_params() -> dict` — serialisable dict of all fitted parameters
- `load_fitted_params(params: dict) -> None` — restores fitted state without re-fitting

### WandBTracker API
```python
tracker = WandBTracker(project, entity, run_name, config)
tracker.log_window_row(model, window_start, window_end)  # accumulates per-window params
tracker.log_model_results(model_name, results_df)         # CSV artifact
tracker.log_summary(all_results: dict[str, pd.DataFrame]) # HTML summary table
tracker.finish()
```

### Experiment TOML structure (`configs/experiments/<suite_name>.toml`)
```toml
[experiments.NAME]
name = "NAME"
model_type = "msm" | "gmm"

[experiments.NAME.hyperparams]
n_states = 2
optimizer_method = "powell"

[experiments.NAME.experiment_params]
start_date = "2010-01-01"
end_date  = "2026-01-01"
alpha = 0.05
window_shape = 250

[experiments.NAME.logging]
log_window_csv   = true
log_summary      = true
log_model_csv    = true
```

---

## Phase 3 — configs/ Migration ⬜

```
configs/
  default.toml        # replaces .env — global defaults
  experiments/        # per-suite experiment definitions
    baseline.toml
```

`config.py` updated to load from `configs/default.toml` via `tomllib` (stdlib Python 3.11+).  
`.env` kept temporarily with deprecation comment, then deleted.

---

## Status
- [x] Phase 1 tests
- [x] Phase 2 warm-start model methods
- [x] Phase 2 WandBTracker
- [x] Phase 2 scripts
- [x] Phase 3 configs migration

## How to run tests
```bash
poetry run python -m pytest tests/ -v
```

## How to run experiments
```bash
# Single experiment:
poetry run python scripts/run_experiment.py \
    --suite configs/experiments/baseline.toml \
    --experiment msm_2state \
    --wandb-project var-research \
    --wandb-entity YOUR_ENTITY \
    --tickers SPY AAPL

# Full suite (single W&B run):
poetry run python scripts/run_suite.py \
    --suite configs/experiments/baseline.toml \
    --wandb-project var-research \
    --wandb-entity YOUR_ENTITY \
    --tickers SPY AAPL
```

## W&B credentials
Set `WANDB_API_KEY` env var or run `wandb login` once.  
Entity / project are passed as CLI flags — fill in when ready.
