"""Tests for plot_var — uses temporary CSV files, no real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")

from ms_var_prediction.plotting import plot_var


@pytest.fixture
def var_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-02", periods=100, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "return": rng.normal(0, 0.01, 100),
            "var": rng.normal(-0.02, 0.005, 100),
            "alpha": 0.05,
        }
    ).set_index("date")
    path = tmp_path / "SPY.csv"
    df.to_csv(path)
    return path


class TestPlotVar:
    def test_returns_figure(self, var_csv: Path) -> None:
        import matplotlib.pyplot as plt

        fig = plot_var("SPY", {"Test model": var_csv})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, var_csv: Path, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        save_path = tmp_path / "plots" / "out.png"
        fig = plot_var("SPY", {"Test model": var_csv}, save_path=save_path)
        assert save_path.exists()
        plt.close(fig)

    def test_multiple_series(self, var_csv: Path) -> None:
        import matplotlib.pyplot as plt

        fig = plot_var(
            "SPY",
            {"Model A": var_csv, "Model B": var_csv},
        )
        ax = fig.axes[0]
        assert len(ax.lines) >= 3  # returns + 2 VaR lines + zero line
        plt.close(fig)
