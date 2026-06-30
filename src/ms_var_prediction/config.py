import os
import tomllib
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Location of the *installed package*. In a source checkout PROJECT_ROOT is the
# repo root; once installed as a wheel it points inside site-packages, so it must
# only be used to locate resources that are bundled with the package.
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Where runtime data (caches, outputs) is read/written. Anchored to the current
# working directory so the tool behaves the same from a source checkout or an
# installed wheel; override with the MS_VAR_DATA_DIR environment variable.
_data_env = os.environ.get("MS_VAR_DATA_DIR")
DATA_DIR = Path(_data_env) if _data_env else Path.cwd()
OUTPUT_DIR = DATA_DIR / "outputs"

# configs/ lives outside the package and is NOT bundled in the wheel, so this
# path only resolves in a source checkout. Settings falls back to _DEFAULTS when
# the file is absent (e.g. when running from an installed wheel).
_DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.toml"

_DEFAULTS = {
    "start_date": "2010-01-01",
    "end_date": "2026-01-01",
    "alpha": 0.05,
    "window_shape": 250,
}


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


class Settings:
    def __init__(self, config_path: Path = _DEFAULT_CONFIG) -> None:
        if Path(config_path).exists():
            raw = _load_toml(config_path)
            self.START_DATE: str = raw["data"]["start_date"]
            self.END_DATE: str = raw["data"]["end_date"]
            self.ALPHA: float = float(raw["backtest"]["alpha"])
            self.WINDOW_SHAPE: int = int(raw["backtest"]["window_shape"])
        else:
            self.START_DATE = str(_DEFAULTS["start_date"])
            self.END_DATE = str(_DEFAULTS["end_date"])
            self.ALPHA = float(_DEFAULTS["alpha"])
            self.WINDOW_SHAPE = int(_DEFAULTS["window_shape"])


settings = Settings()
