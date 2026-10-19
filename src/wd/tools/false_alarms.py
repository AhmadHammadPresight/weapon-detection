"""False-alarm check: run the demo detector over real CCTV clips that contain no weapons.

    .venv/bin/python -m wd.tools.false_alarms <clips folder>     (or set WD_CLIPS_DIR)

The clips are the 11 public clips of the Arabic video search PoC (its scripts/fetch_footage.sh downloads them into
that project's data/footage/clips/). They are only read, never copied. Frames are sampled and grouped into alerts
exactly as the app does, with the threshold chosen on the training images, so every alert here is a false alarm.
Writes results/false_alarms.json.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from wd.alerts import AlertGrouper, alerting
from wd.detectors import DETECTORS, load_detector
from wd.media import VIDEO_SUFFIXES, iter_frames, probe
from wd.results import FALSE_ALARMS_PATH, alert_threshold, write_json
from wd.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="?", type=Path, default=os.environ.get("WD_CLIPS_DIR"))
    parser.add_argument("--detector", choices=DETECTORS, default=None)
    args = parser.parse_args()
    if args.clips is None:
        raise SystemExit("give the clips folder as an argument or in WD_CLIPS_DIR")
    clips_dir = Path(args.clips).expanduser().resolve()
    clips = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if not clips:
        raise SystemExit(f"no video clips in {clips_dir}")
    settings = get_settings()
    name = args.detector or settings.detector
    threshold = alert_threshold(name)
    detector = load_detector(name, settings.models_dir, settings.device)

    rows = []
    for clip in clips:
        info = probe(clip)
        grouper = AlertGrouper(threshold, settings.cooldown_seconds)
        frames = alerting_frames = 0
        top = 0.0
        started = time.monotonic()
        for t, frame in iter_frames(clip, settings.sample_fps, info):
            detections = detector.detect(frame)
            frames += 1
            alerting_frames += bool(alerting(detections, threshold))
            top = max([top] + [d.confidence for d in detections])
            grouper.add(t, detections)
        row = {
            "clip": clip.name,
            "seconds": round(info.duration, 1),
            "resolution": f"{info.width}x{info.height}",
            "frames_checked": frames,
            "alerting_frames": alerting_frames,
            "false_alarms": len(grouper.alerts),
            "by_class": {label: sum(a.label == label for a in grouper.alerts) for label in ("gun", "knife")},
            "alerts": [
                {"label": a.label, "at_seconds": a.best_at, "confidence": round(a.confidence, 3), "frames": a.frames}
                for a in grouper.alerts
            ],
            "top_confidence": round(top, 3),
            "processing_seconds": round(time.monotonic() - started, 1),
        }
        rows.append(row)
        print(f"  {clip.name}: {row['false_alarms']} false alarms ({alerting_frames}/{frames} frames over the threshold)")

    write_json(
        FALSE_ALARMS_PATH,
        {
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "detector": name,
            "threshold": threshold,
            "sample_fps": settings.sample_fps,
            "cooldown_seconds": settings.cooldown_seconds,
            "note": "Real CCTV clips with no weapons in them: every alert is a false alarm.",
            "clips": rows,
            "total": {
                "clips": len(rows),
                "seconds": round(sum(r["seconds"] for r in rows), 1),
                "frames_checked": sum(r["frames_checked"] for r in rows),
                "alerting_frames": sum(r["alerting_frames"] for r in rows),
                "false_alarms": sum(r["false_alarms"] for r in rows),
                "clips_with_false_alarms": sum(r["false_alarms"] > 0 for r in rows),
            },
        },
    )
    print(f"wrote {FALSE_ALARMS_PATH.relative_to(FALSE_ALARMS_PATH.parents[1])}")


if __name__ == "__main__":
    main()
