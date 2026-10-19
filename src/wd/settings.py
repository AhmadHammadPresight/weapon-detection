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
    # Videos only: consecutive sampled frames a weapon must be seen in before it raises an alert.
    video_min_frames: int

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
        # 3 frames in a row at 2 frames per second: the weapon stays in view for 1 to 1.5 s. Chosen by reasoning
        # (a brandished weapon is held for seconds, false detections tend to flicker), not tuned on any footage.
        video_min_frames=int(_env("WD_VIDEO_MIN_FRAMES", "3")),
    )
