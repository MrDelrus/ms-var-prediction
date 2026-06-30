import os
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
