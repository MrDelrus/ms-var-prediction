from pydantic_settings import BaseSettings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


class Settings(BaseSettings):
    START_DATE: str
    END_DATE: str
    ALPHA: float
    WINDOW_SHAPE: int

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
