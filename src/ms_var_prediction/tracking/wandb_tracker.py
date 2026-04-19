"""
W&B experiment tracker for VaR models.

Usage
-----
tracker = WandBTracker(project="var-project", entity="my-entity", run_name="suite-v1")
tracker.start(config={"alpha": 0.05})

# Inside rolling loop:
tracker.log_window_row(model, experiment_name, window_start="2020-01-01", window_end="2021-01-01")

# After a full model run:
tracker.log_model_results(experiment_name="msm_2state", results_df=df)

# After all experiments in the suite:
tracker.log_summary(all_results={"msm_2state": df1, "gmm_base": df2})
tracker.finish()
"""

from __future__ import annotations

import io
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import wandb
from sklearn.base import BaseEstimator


class WandBTracker:
    def __init__(
        self,
        project: str,
        entity: str | None = None,
        run_name: str | None = None,
    ) -> None:
        self._project = project
        self._entity = entity
        self._run_name = run_name
        self._run: wandb.sdk.wandb_run.Run | None = None
        # Accumulates per-window rows keyed by experiment name
        self._window_rows: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, config: dict[str, Any] | None = None) -> "WandBTracker":
        self._run = wandb.init(
            project=self._project,
            entity=self._entity,
            name=self._run_name,
            config=config or {},
            reinit=True,
        )
        return self

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    # ------------------------------------------------------------------
    # Per-window param logging
    # ------------------------------------------------------------------

    def log_window_row(
        self,
        model: BaseEstimator,
        experiment_name: str,
        window_start: str | date,
        window_end: str | date,
    ) -> None:
        """Append one row (one window position) to the per-experiment window table."""
        if not hasattr(model, "get_fitted_params"):
            raise TypeError(
                f"{type(model).__name__} does not implement get_fitted_params()"
            )

        params = model.get_fitted_params()
        row: dict[str, Any] = {
            "start_date": str(window_start),
            "end_date": str(window_end),
        }

        # Flatten nested params (lists/arrays → individual columns)
        for key, value in params.items():
            if isinstance(value, (list, np.ndarray)):
                arr = np.asarray(value).flatten()
                for i, v in enumerate(arr):
                    row[f"{key}_{i}"] = float(v)
            elif isinstance(value, (int, float, str)):
                row[key] = value

        self._window_rows.setdefault(experiment_name, []).append(row)

    def flush_window_csv(self, experiment_name: str) -> None:
        """Upload accumulated window rows as a CSV artifact for one experiment."""
        rows = self._window_rows.get(experiment_name)
        if not rows:
            return
        df = pd.DataFrame(rows)
        self._log_csv_artifact(df, artifact_name=f"{experiment_name}_window_params")
        self._window_rows[experiment_name] = []

    # ------------------------------------------------------------------
    # Per-model results (full backtest CSV)
    # ------------------------------------------------------------------

    def log_model_results(self, experiment_name: str, results_df: pd.DataFrame) -> None:
        """Upload the full per-ticker backtest results as a CSV artifact."""
        self._log_csv_artifact(results_df, artifact_name=f"{experiment_name}_results")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def log_summary(
        self,
        all_results: dict[str, tuple[pd.DataFrame, dict[str, Any]]],
    ) -> None:
        """
        Log aggregated summary to W&B.

        Parameters
        ----------
        all_results : dict[experiment_name -> (results_df, hyperparams_dict)]
        """
        rows = []
        for exp_name, (df, hyperparams) in all_results.items():
            row: dict[str, Any] = {"experiment": exp_name}
            row.update({f"hp_{k}": v for k, v in hyperparams.items()})
            for col in df.columns:
                row[f"metric_{col}_pass_rate"] = float(df[col].mean())
            rows.append(row)

        summary_df = pd.DataFrame(rows).set_index("experiment")
        html = self._summary_html(summary_df)

        artifact = wandb.Artifact(name="experiment_summary", type="summary")
        with artifact.new_file("summary.html", mode="w") as f:
            f.write(html)
        if self._run is not None:
            self._run.log_artifact(artifact)
            self._run.log({"summary_table": wandb.Html(html)})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_csv_artifact(self, df: pd.DataFrame, artifact_name: str) -> None:
        if self._run is None:
            return
        buf = io.StringIO()
        df.to_csv(buf)
        artifact = wandb.Artifact(name=artifact_name, type="dataset")
        with artifact.new_file(f"{artifact_name}.csv", mode="w") as f:
            f.write(buf.getvalue())
        self._run.log_artifact(artifact)

    @staticmethod
    def _summary_html(df: pd.DataFrame) -> str:
        hp_cols = [c for c in df.columns if c.startswith("hp_")]
        metric_cols = [c for c in df.columns if c.startswith("metric_")]

        def _fmt(val: Any) -> str:
            if isinstance(val, float):
                return f"{val:.3f}"
            return str(val)

        header_hp = "".join(f"<th>{c[3:]}</th>" for c in hp_cols)
        header_metric = "".join(f"<th>{c[7:]}</th>" for c in metric_cols)
        header = (
            f"<tr><th>experiment</th>{header_hp}"
            f'<th style="border-left:3px solid #333"></th>'
            f"{header_metric}</tr>"
        )

        body_rows = []
        for exp, row in df.iterrows():
            hp_cells = "".join(f"<td>{_fmt(row[c])}</td>" for c in hp_cols)
            metric_cells = "".join(f"<td>{_fmt(row[c])}</td>" for c in metric_cols)
            body_rows.append(
                f"<tr><td><b>{exp}</b></td>{hp_cells}"
                f'<td style="border-left:3px solid #333"></td>'
                f"{metric_cells}</tr>"
            )

        return f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: monospace; font-size: 13px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
  th {{ background: #f0f0f0; font-weight: bold; }}
  tr:hover {{ background: #f9f9f9; }}
</style>
</head>
<body>
<h2>Experiment Summary</h2>
<table>
<thead>{header}</thead>
<tbody>{"".join(body_rows)}</tbody>
</table>
</body>
</html>"""
