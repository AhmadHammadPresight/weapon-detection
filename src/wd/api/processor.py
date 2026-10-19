"""Background processing of uploads: run the detector, group detections into alerts, save boxed frames.

One worker thread handles uploads in order, so the detector runs on one image at a time.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from wd.alerts import Alert, AlertGrouper, alert_time
from wd.api import db
from wd.boxes import rounded
from wd.detectors import Detector
from wd.media import draw_boxes, iter_frames, probe, read_image
from wd.settings import Settings

log = logging.getLogger("wd.processor")


class Processor:
    def __init__(self, settings: Settings, load: Callable[[], tuple[Detector, dict[str, float]]]) -> None:
        self.settings = settings
        self._load = load
        self.detector: Detector | None = None
        self.thresholds: dict[str, float] | None = None
        self.state = "loading"  # loading | ready | error
        self.error: str | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="upload-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)

    def submit(self, upload_id: str) -> None:
        self._queue.put(upload_id)

    def _conn(self):
        return db.connect(self.settings.db_path)

    def _run(self) -> None:
        try:
            self.detector, self.thresholds = self._load()
            self.state = "ready"
        except Exception as exc:  # the UI shows this; uploads stay queued until a restart fixes it
            log.exception("could not load the detector")
            self.state, self.error = "error", str(exc)
            return
        with self._conn() as conn:
            for upload in db.unfinished_uploads(conn):
                self._queue.put(upload["id"])
        while (upload_id := self._queue.get()) is not None:
            try:
                self._process(upload_id)
            except Exception as exc:
                log.exception("upload %s failed", upload_id)
                with self._conn() as conn:
                    db.update_upload(conn, upload_id, status="failed", error=str(exc))

    def _process(self, upload_id: str) -> None:
        with self._conn() as conn:
            upload = db.get_upload(conn, upload_id)
            if upload is None or upload["status"] == "done":
                return
            # Persistence applies to videos only; a still image has one frame, so one detection is enough.
            min_frames = self.settings.video_min_frames if upload["kind"] == "video" else 1
            db.update_upload(conn, upload_id, status="processing", progress=0, frames=0, error=None,
                             detector=self.detector.name, thresholds=json.dumps(self.thresholds), min_frames=min_frames)
        path = self.settings.uploads_dir / upload["stored"]
        start = datetime.fromisoformat(upload["start_time"])
        grouper = AlertGrouper(self.thresholds, self.settings.cooldown_seconds, min_frames)
        frames = 0
        if upload["kind"] == "image":
            source = [(0.0, read_image(path))]
            expected = 1
        else:
            info = probe(path)
            source = iter_frames(path, self.settings.sample_fps, info)
            expected = max(1, int(info.duration * self.settings.sample_fps))
        for t, frame in source:
            detections = self.detector.detect(frame)
            frames += 1
            for change in grouper.add(t, detections):
                self._save_alert(upload, change.alert, frame, start)
            if frames % 5 == 0 or upload["kind"] == "image":
                with self._conn() as conn:
                    db.update_upload(conn, upload_id, progress=min(0.99, frames / expected), frames=frames)
        with self._conn() as conn:
            for alert in grouper.alerts:  # final frame counts
                conn.execute("UPDATE alerts SET frames = ? WHERE id = ?", (alert.frames, self._alert_id(upload, alert)))
            db.update_upload(conn, upload_id, status="done", progress=1, frames=frames)

    @staticmethod
    def _alert_id(upload: dict, alert: Alert) -> str:
        return f"{upload['id']}-{alert.key}"

    def _save_alert(self, upload: dict, alert: Alert, frame, start: datetime) -> None:
        alert_id = self._alert_id(upload, alert)
        image_name = f"{alert_id}.jpg"
        self.settings.alerts_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.settings.alerts_dir / f"{alert_id}.partial.jpg"
        draw_boxes(frame, alert.detections).save(tmp, quality=88)
        tmp.replace(self.settings.alerts_dir / image_name)
        with self._conn() as conn:
            db.upsert_alert(
                conn,
                id=alert_id,
                upload_id=upload["id"],
                camera=upload["camera"],
                label=alert.label,
                confidence=round(alert.confidence, 4),
                at_time=alert_time(start, alert.best_at).isoformat(timespec="seconds"),
                offset_seconds=round(alert.best_at, 2),
                frames=alert.frames,
                image=image_name,
                boxes=[{"label": d.label, "confidence": round(d.confidence, 4), "box": rounded(d.box)} for d in alert.detections],
            )


def remove_files(directory: Path, names: list[str]) -> None:
    for name in names:
        (directory / name).unlink(missing_ok=True)
