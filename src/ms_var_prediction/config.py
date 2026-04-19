import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
_DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.toml"


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


class Settings:
    def __init__(self, config_path: Path = _DEFAULT_CONFIG) -> None:
        raw = _load_toml(config_path)
        self.START_DATE: str = raw["data"]["start_date"]
        self.END_DATE: str = raw["data"]["end_date"]
        self.ALPHA: float = float(raw["backtest"]["alpha"])
        self.WINDOW_SHAPE: int = int(raw["backtest"]["window_shape"])


settings = Settings()
