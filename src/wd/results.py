"""The committed evaluation results under `results/`, written by scripts/evaluate.sh and read by the app."""

from __future__ import annotations

import json
from pathlib import Path

from wd.settings import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"
EVALUATION_PATH = RESULTS_DIR / "evaluation.json"
FALSE_ALARMS_PATH = RESULTS_DIR / "false_alarms.json"


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


def alert_threshold(detector: str) -> float:
    """The alert threshold scripts/evaluate.sh chose for `detector` on the training images."""
    evaluation = read_json(EVALUATION_PATH)
    if not evaluation or detector not in evaluation.get("detectors", {}):
        raise RuntimeError(f"no threshold for {detector!r} in {EVALUATION_PATH.name}; run scripts/evaluate.sh first")
    return float(evaluation["detectors"][detector]["threshold"])
