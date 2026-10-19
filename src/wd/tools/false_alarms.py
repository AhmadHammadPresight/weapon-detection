"""False-alarm check: run the demo detector over real CCTV clips that contain no weapons.

    .venv/bin/python -m wd.tools.false_alarms <clips folder>     (or set WD_CLIPS_DIR)

The clips are the 11 public clips of the Arabic video search PoC (its scripts/fetch_footage.sh downloads them into
that project's data/footage/clips/). They are only read, never copied. Frames are sampled and grouped into alerts
exactly as the app does, with the thresholds chosen on the training images, so every alert here is a false alarm.
The counts for the earlier rules (one shared threshold, single-frame alerts) are kept beside the app's rule, from
the same detections, for a fair before/after. Writes results/false_alarms.json.
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
from wd.dataset import CLASSES
from wd.results import EVALUATION_PATH, FALSE_ALARMS_PATH, read_json, write_json
from wd.settings import get_settings

APP_RULE = "class_thresholds_persistence"  # what the app does with uploaded videos


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
    evaluation = read_json(EVALUATION_PATH) or {}
    result = evaluation.get("detectors", {}).get(name)
    if not result:
        raise SystemExit(f"no thresholds for {name!r} in {EVALUATION_PATH.name}; run the test scoring first")
    detector = load_detector(name, settings.models_dir, settings.device)
    # Every rule is counted from the same detections, so the before/after comparison is like for like.
    rules = {
        "one_threshold_single_frame": (result["single_threshold"], 1),
        "one_threshold_persistence": (result["single_threshold"], settings.video_min_frames),
        "class_thresholds_single_frame": (result["thresholds"], 1),
        "class_thresholds_persistence": (result["thresholds"], settings.video_min_frames),
    }

    rows = []
    for clip in clips:
        info = probe(clip)
        groupers = {rule: AlertGrouper(t, settings.cooldown_seconds, n) for rule, (t, n) in rules.items()}
        alerting_frames = dict.fromkeys(rules, 0)
        frames = 0
        top = 0.0
        started = time.monotonic()
        for t, frame in iter_frames(clip, settings.sample_fps, info):
            detections = detector.detect(frame)
            frames += 1
            top = max([top] + [d.confidence for d in detections])
            for rule, grouper in groupers.items():
                alerting_frames[rule] += bool(alerting(detections, rules[rule][0]))
                grouper.add(t, detections)
        final = groupers[APP_RULE]
        row = {
            "clip": clip.name,
            "seconds": round(info.duration, 1),
            "resolution": f"{info.width}x{info.height}",
            "frames_checked": frames,
            "false_alarms": len(final.alerts),
            "by_class": {label: sum(a.label == label for a in final.alerts) for label in CLASSES},
            "alerts": [
                {"label": a.label, "at_seconds": a.best_at, "confidence": round(a.confidence, 3), "frames": a.frames}
                for a in final.alerts
            ],
            "by_rule": {
                rule: {"false_alarms": len(g.alerts), "frames_over_threshold": alerting_frames[rule]}
                for rule, g in groupers.items()
            },
            "top_confidence": round(top, 3),
            "processing_seconds": round(time.monotonic() - started, 1),
        }
        rows.append(row)
        counts = ", ".join(f"{rule} {r['false_alarms']}" for rule, r in row["by_rule"].items())
        print(f"  {clip.name}: {row['false_alarms']} false alarms ({counts})")

    write_json(
        FALSE_ALARMS_PATH,
        {
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "detector": name,
            "model": result["model"],
            "thresholds": result["thresholds"],
            "single_threshold": result["single_threshold"],
            "sample_fps": settings.sample_fps,
            "cooldown_seconds": settings.cooldown_seconds,
            "video_min_frames": settings.video_min_frames,
            "app_rule": APP_RULE,
            "rules": {rule: {"thresholds": t, "min_frames": n} for rule, (t, n) in rules.items()},
            "note": "Real CCTV clips with no weapons in them: every alert is a false alarm.",
            "clips": rows,
            "total": {
                "clips": len(rows),
                "seconds": round(sum(r["seconds"] for r in rows), 1),
                "frames_checked": sum(r["frames_checked"] for r in rows),
                "false_alarms": sum(r["false_alarms"] for r in rows),
                "clips_with_false_alarms": sum(r["false_alarms"] > 0 for r in rows),
                "by_rule": {
                    rule: {
                        "false_alarms": sum(r["by_rule"][rule]["false_alarms"] for r in rows),
                        "clips_with_false_alarms": sum(r["by_rule"][rule]["false_alarms"] > 0 for r in rows),
                        "frames_over_threshold": sum(r["by_rule"][rule]["frames_over_threshold"] for r in rows),
                    }
                    for rule in rules
                },
            },
        },
    )
    print(f"wrote {FALSE_ALARMS_PATH.relative_to(FALSE_ALARMS_PATH.parents[1])}")


if __name__ == "__main__":
    main()
