from pathlib import Path

from ms_var_prediction.config import Settings


def _write_toml(path: Path, content: str) -> None:
    path.write_text(content)


class TestSettings:
    def test_loads_from_default_toml(self) -> None:
        s = Settings()
        assert s.START_DATE == "2010-01-01"
        assert s.END_DATE == "2026-01-01"
        assert s.ALPHA == 0.05
        assert s.WINDOW_SHAPE == 250

    def test_loads_custom_toml(self, tmp_path: Path) -> None:
        cfg = tmp_path / "custom.toml"
        cfg.write_text(
            '[data]\nstart_date = "2020-01-01"\nend_date = "2023-01-01"\n'
            "[backtest]\nalpha = 0.01\nwindow_shape = 100\n"
        )
        s = Settings(config_path=cfg)
        assert s.START_DATE == "2020-01-01"
        assert s.END_DATE == "2023-01-01"
        assert s.ALPHA == 0.01
        assert s.WINDOW_SHAPE == 100

    def test_alpha_is_float(self) -> None:
        s = Settings()
        assert isinstance(s.ALPHA, float)

    def test_window_shape_is_int(self) -> None:
        s = Settings()
        assert isinstance(s.WINDOW_SHAPE, int)
