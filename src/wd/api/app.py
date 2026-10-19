"""Web API and UI server.

Run it with `uvicorn wd.api.app:create_app --factory` (scripts/start.sh does). It serves the web UI, the JSON API
under `/api`, the boxed alert frames under `/media/alerts`, and the committed test images under `/dataset`.
The detector loads in the background, so the page is usable while the model is still loading.
"""

from __future__ import annotations

import re
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from wd.api import db
from wd.api.processor import Processor, remove_files
from wd.detectors import load_detector
from wd.media import IMAGE_SUFFIXES, VIDEO_SUFFIXES
from wd.results import alert_thresholds
from wd.settings import REPO_ROOT, Settings, get_settings

CAMERA_ID = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
WEB_DIR = REPO_ROOT / "web"


class RevalidatedStaticFiles(StaticFiles):
    """Static files the browser must revalidate (ETag) on every load, so UI updates show at once."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def default_loader(settings: Settings):
    def load():
        return load_detector(settings.detector, settings.models_dir, settings.device), alert_thresholds(settings.detector)

    return load


def parse_start_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(422, "start_time must be a date and time such as 2026-10-19T14:30:00") from exc
    if parsed.tzinfo is not None:
        raise HTTPException(422, "start_time is the camera's local time, without a time zone")
    return parsed


def create_app(settings: Settings | None = None, processor: Processor | None = None) -> FastAPI:
    settings = settings or get_settings()
    db.init_db(settings.db_path)
    for directory in (settings.uploads_dir, settings.alerts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    processor = processor or Processor(settings, default_loader(settings))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        processor.start()
        yield
        processor.stop()

    app = FastAPI(title="Weapon detection", docs_url=None, redoc_url=None, lifespan=lifespan)

    def conn():
        return db.connect(settings.db_path)

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": processor.state,
            "detector": settings.detector,
            "thresholds": processor.thresholds,
            "error": processor.error,
            "sample_fps": settings.sample_fps,
            "cooldown_seconds": settings.cooldown_seconds,
            "video_min_frames": settings.video_min_frames,
        }

    @app.post("/api/uploads", status_code=201)
    def upload(
        file: UploadFile = File(...),
        camera_id: str = Form(...),
        start_time: str = Form(...),
    ) -> dict:
        camera_id = camera_id.strip()
        if not CAMERA_ID.match(camera_id):
            raise HTTPException(422, "camera_id: 1-40 letters, digits, '.', '_' or '-'")
        start = parse_start_time(start_time)
        filename = Path(file.filename or "upload").name
        suffix = Path(filename).suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            kind = "image"
        elif suffix in VIDEO_SUFFIXES:
            kind = "video"
        else:
            raise HTTPException(415, f"unsupported file type {suffix or '(none)'}; upload an image or a video")
        upload_id = uuid.uuid4().hex[:12]
        stored = f"{upload_id}{suffix}"
        with (settings.uploads_dir / stored).open("wb") as out:
            shutil.copyfileobj(file.file, out)
        with conn() as c:
            db.insert_upload(
                c, id=upload_id, kind=kind, filename=filename, stored=stored, camera=camera_id,
                start_time=start.isoformat(timespec="seconds"),
            )
            row = db.get_upload(c, upload_id)
        processor.submit(upload_id)
        return row

    @app.get("/api/uploads")
    def uploads() -> list[dict]:
        with conn() as c:
            return db.list_uploads(c)

    @app.get("/api/uploads/{upload_id}")
    def get_upload(upload_id: str) -> dict:
        with conn() as c:
            row = db.get_upload(c, upload_id)
        if row is None:
            raise HTTPException(404, "upload not found")
        return row

    @app.get("/api/alerts")
    def alerts(upload_id: str | None = None) -> list[dict]:
        with conn() as c:
            return db.list_alerts(c, upload_id)

    @app.delete("/api/alerts", status_code=204)
    def clear_alerts() -> None:
        with conn() as c:
            if db.unfinished_uploads(c):
                raise HTTPException(409, "an upload is still being processed")
            files = db.clear(c)
        remove_files(settings.uploads_dir, files)
        remove_files(settings.alerts_dir, files)

    app.mount("/media/alerts", StaticFiles(directory=settings.alerts_dir), name="alerts")
    app.mount("/", RevalidatedStaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
