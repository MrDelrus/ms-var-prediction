"""
VaR vs returns plots.

Usage
-----
from ms_var_prediction.plotting import plot_var

plot_var(
    "SPY",
    {
        "MSM 3-state": "outputs/var_series/msm_3state/SPY.csv",
        "GMM 3-comp":  "outputs/var_series/gmm_3comp/SPY.csv",
    },
    save_path="outputs/plots/SPY_comparison.png",
)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Distinct, colourblind-friendly palette for VaR lines
_VAR_COLOURS = ["#e06c00", "#9b30d9", "#1a9e3f", "#c0392b", "#0077b6"]

_RETURNS_COLOUR = "#555555"  # neutral dark-grey for returns

# Larger canvas for the full-history per-ticker panels.
_BIG_FIGSIZE = (20, 7)
_REGIME_FIGSIZE = (16, 5)


def plot_var_frame(
    ticker: str,
    df: pd.DataFrame,
    alpha: float,
    save_path: Union[str, Path, None] = None,
    figsize: tuple[int, int] = (16, 6),
) -> plt.Figure:
    """
    Plot realised returns and the predicted VaR series for a single ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol, used for the title.
    df : pd.DataFrame
        Must contain columns ``date``, ``realized_return`` and ``var``.
    alpha : float
        VaR significance level, shown in the legend.
    save_path : str or Path, optional
        If given, saves the figure to this path.
    """
    dates = pd.to_datetime(df["date"])
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        dates,
        df["realized_return"],
        color=_RETURNS_COLOUR,
        alpha=0.7,
        linewidth=0.9,
        label="Realised return",
        zorder=2,
    )
    ax.plot(
        dates,
        df["var"],
        color=_VAR_COLOURS[0],
        alpha=0.8,
        linewidth=1.6,
        label=f"Predicted VaR α={alpha:.2f}",
        zorder=3,
    )

    ax.axhline(0, color="#aaaaaa", linewidth=0.6, linestyle="--", zorder=1)
    ax.set_title(f"{ticker} — Returns vs Predicted VaR", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


def plot_var(
    ticker: str,
    series_paths: dict[str, Union[str, Path]],
    save_path: Union[str, Path, None] = None,
    figsize: tuple[int, int] = (16, 6),
) -> plt.Figure:
    """
    Plot returns and one or more VaR series for a single ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol, used only for the title.
    series_paths : dict[model_name -> csv_path]
        Each CSV must have columns: date (index), return, var, alpha.
    save_path : str or Path, optional
        If given, saves the figure to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)

    returns_plotted = False
    for i, (model_name, path) in enumerate(series_paths.items()):
        df = pd.read_csv(path, index_col="date", parse_dates=True)

        if not returns_plotted:
            ax.plot(
                df.index,
                df["return"],
                color=_RETURNS_COLOUR,
                alpha=0.7,
                linewidth=0.9,
                label="Returns",
                zorder=2,
            )
            returns_plotted = True

        alpha_val = df["alpha"].iloc[0]
        colour = _VAR_COLOURS[i % len(_VAR_COLOURS)]
        ax.plot(
            df.index,
            df["var"],
            color=colour,
            alpha=0.5,
            linewidth=1.6,
            label=f"{model_name}  VaR α={alpha_val:.2f}",
            zorder=3,
        )

    ax.axhline(0, color="#aaaaaa", linewidth=0.6, linestyle="--", zorder=1)
    ax.set_title(f"{ticker} — Returns vs VaR", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=9,
        framealpha=0.85,
    )
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    return fig


# ---------------------------------------------------------------------------
# Per-ticker model-state panels (model-agnostic: MSM and GMM share these)
# ---------------------------------------------------------------------------


def _regime_arrays(params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalise one window's fitted params to ``(means, stds, probs)`` per regime.

    Handles both models transparently — MSM stores ``mus/sigmas/state_probs``,
    GMM stores ``means/covariances/weights`` (its ``covariances`` are already
    standard deviations). Regimes are sorted by std ascending, so index 0 is
    always the calm regime and index n-1 the most volatile, which keeps the
    per-regime curves stable across windows despite label switching.
    """
    means = np.asarray(params.get("mus", params.get("means")), dtype=float)
    stds = np.asarray(params.get("sigmas", params.get("covariances")), dtype=float)
    probs = np.asarray(params.get("state_probs", params.get("weights")), dtype=float)
    order = np.argsort(stds)
    return means[order], stds[order], probs[order]


def build_regime_timeseries(feature_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-date table of regime parameters for one ticker.

    ``feature_rows`` must have ``date`` and ``params`` (JSON) columns. Returns a
    frame sorted by date with ``mean_k``, ``std_k`` and ``prob_k`` columns for
    each regime ``k`` (0 = calm, sorted by volatility).
    """
    records: list[dict[str, Any]] = []
    for date, raw in zip(feature_rows["date"], feature_rows["params"]):
        means, stds, probs = _regime_arrays(json.loads(raw))
        rec: dict[str, Any] = {"date": date}
        for k in range(len(means)):
            rec[f"mean_{k}"] = means[k]
            rec[f"std_{k}"] = stds[k]
            rec[f"prob_{k}"] = probs[k]
        records.append(rec)
    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def plot_returns_and_var(
    ticker: str,
    returns: pd.Series,
    predictions: pd.DataFrame,
    alpha: float,
    save_path: Union[str, Path, None] = None,
    figsize: tuple[int, int] = _BIG_FIGSIZE,
) -> plt.Figure:
    """
    Full-history returns with the predicted-VaR line overlaid.

    ``returns`` is a datetime-indexed series spanning the whole range (warm-up
    year included); ``predictions`` has ``date`` and ``var`` columns and only
    covers the fitted dates, so the VaR line grows as more years are fit.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        returns.index,
        returns.to_numpy(),
        color=_RETURNS_COLOUR,
        alpha=0.7,
        linewidth=0.9,
        label="Returns",
        zorder=2,
    )
    ax.plot(
        pd.to_datetime(predictions["date"]),
        predictions["var"],
        color=_VAR_COLOURS[0],
        alpha=0.8,
        linewidth=1.6,
        label=f"Predicted VaR α={alpha:.2f}",
        zorder=3,
    )
    ax.axhline(0, color="#aaaaaa", linewidth=0.6, linestyle="--", zorder=1)
    ax.set_title(f"{ticker} — returns & predicted VaR (full history)", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def _n_regimes(ts: pd.DataFrame) -> int:
    return sum(c.startswith("std_") for c in ts.columns)


def _regime_label(k: int, n: int) -> str:
    if k == 0:
        return f"regime {k} (calm)"
    if k == n - 1:
        return f"regime {k} (turbulent)"
    return f"regime {k}"


def plot_regime_metric(
    ticker: str,
    ts: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    save_path: Union[str, Path, None] = None,
    figsize: tuple[int, int] = _REGIME_FIGSIZE,
) -> plt.Figure:
    """One curve per regime for ``metric`` ('std' or 'mean') over time."""
    n = _n_regimes(ts)
    dates = pd.to_datetime(ts["date"])
    fig, ax = plt.subplots(figsize=figsize)
    for k in range(n):
        ax.plot(
            dates,
            ts[f"{metric}_{k}"],
            color=_VAR_COLOURS[k % len(_VAR_COLOURS)],
            alpha=0.85,
            linewidth=1.5,
            label=_regime_label(k, n),
            zorder=3,
        )
    ax.set_title(f"{ticker} — {title}", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=n, fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_regime_prob(
    ticker: str,
    ts: pd.DataFrame,
    save_path: Union[str, Path, None] = None,
    figsize: tuple[int, int] = _REGIME_FIGSIZE,
) -> plt.Figure:
    """Probability mass in the calm (lowest-volatility) regime over time."""
    dates = pd.to_datetime(ts["date"])
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        dates,
        ts["prob_0"],
        color=_VAR_COLOURS[0],
        alpha=0.85,
        linewidth=1.5,
        label="P(calm regime)",
        zorder=3,
    )
    ax.axhline(0.5, color="#aaaaaa", linewidth=0.6, linestyle="--", zorder=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{ticker} — probability of the calm regime", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig
