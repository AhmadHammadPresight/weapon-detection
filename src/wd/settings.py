"""Runtime settings, read once from environment variables (scripts/start.sh sets them)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    detector: str
    device: str
    sample_fps: float
    cooldown_seconds: float

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def alerts_dir(self) -> Path:
        return self.data_dir / "alerts"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "alerts.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        data_dir=Path(_env("WD_DATA_DIR", str(REPO_ROOT / "data"))).resolve(),
        detector=_env("WD_DETECTOR", "rfdetr"),
        device=_env("WD_DEVICE", "mps"),
        sample_fps=float(_env("WD_SAMPLE_FPS", "2")),
        cooldown_seconds=float(_env("WD_COOLDOWN_SECONDS", "10")),
    )
