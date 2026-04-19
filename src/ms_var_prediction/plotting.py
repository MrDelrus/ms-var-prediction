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

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

# Distinct, colourblind-friendly palette for VaR lines
_VAR_COLOURS = ["#e06c00", "#9b30d9", "#1a9e3f", "#c0392b", "#0077b6"]

_RETURNS_COLOUR = "#555555"  # neutral dark-grey for returns


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
