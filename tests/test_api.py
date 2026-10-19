import io
import json
import subprocess
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wd.api.app import create_app
from wd.api.processor import Processor
from wd.detectors import Detection
from wd.settings import Settings


class FakeDetector:
    name = "fake"

    def detect(self, image):
        # A gun wherever the image has a bright pixel in its top-left corner.
        return [Detection("gun", 0.9, (1, 1, 8, 8))] if image[0, 0].mean() > 128 else [Detection("gun", 0.2, (1, 1, 8, 8))]


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, detector="fake", device="cpu", sample_fps=2, cooldown_seconds=10, video_min_frames=3)
    processor = Processor(settings, lambda: (FakeDetector(), {"gun": 0.5, "knife": 0.5}))
    with TestClient(create_app(settings, processor)) as c:
        yield c


def png(bright: bool) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((16, 16, 3), 255 if bright else 0, dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def send(client, data: bytes, name="frame.png", camera="CAM-01", start="2026-10-19T14:30:00"):
    return client.post("/api/uploads", files={"file": (name, data)}, data={"camera_id": camera, "start_time": start})


def wait_done(client, upload_id):
    for _ in range(200):
        row = client.get(f"/api/uploads/{upload_id}").json()
        if row["status"] in ("done", "failed"):
            return row
        time.sleep(0.02)
    raise AssertionError("upload not processed")


def test_an_image_with_a_weapon_raises_an_alert_with_camera_time_and_boxed_frame(client):
    response = send(client, png(bright=True))
    assert response.status_code == 201
    row = wait_done(client, response.json()["id"])
    assert row["status"] == "done" and json.loads(row["thresholds"]) == {"gun": 0.5, "knife": 0.5}
    assert row["min_frames"] == 1  # a still image needs one detection
    alerts = client.get("/api/alerts").json()
    assert len(alerts) == 1
    alert = alerts[0]
    assert (alert["label"], alert["camera"], alert["at_time"]) == ("gun", "CAM-01", "2026-10-19T14:30:00")
    assert alert["confidence"] == 0.9 and alert["boxes"][0]["box"] == [1, 1, 8, 8]
    frame = client.get(f"/media/alerts/{alert['image']}")
    assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"


def test_an_image_below_the_threshold_raises_no_alert(client):
    row = wait_done(client, send(client, png(bright=False)).json()["id"])
    assert row["status"] == "done"
    assert client.get("/api/alerts").json() == []
    assert client.get("/api/uploads").json()[0]["alerts"] == 0


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"camera": "bad camera!"}, 422),
        ({"start": "yesterday"}, 422),
        ({"start": "2026-10-19T14:30:00+03:00"}, 422),
        ({"name": "notes.txt"}, 415),
    ],
)
def test_bad_uploads_are_refused(client, kwargs, status):
    assert send(client, png(bright=True), **kwargs).status_code == status


def test_clear_removes_uploads_alerts_and_files(client, tmp_path):
    wait_done(client, send(client, png(bright=True)).json()["id"])
    assert client.delete("/api/alerts").status_code == 204
    assert client.get("/api/alerts").json() == [] and client.get("/api/uploads").json() == []
    assert not any((tmp_path / "alerts").iterdir()) and not any((tmp_path / "uploads").iterdir())


def test_health_and_pages(client):
    for _ in range(100):
        if client.get("/api/health").json()["status"] == "ready":
            break
        time.sleep(0.02)
    assert client.get("/api/health").json()["thresholds"] == {"gun": 0.5, "knife": 0.5}
    assert "upload-form" in client.get("/").text


def video(tmp_path, bright_seconds: list[int], seconds: int = 6) -> bytes:
    """An MJPEG video, one second per entry, bright (a 'gun' for the fake detector) where listed."""
    frames = tmp_path / "frames"
    frames.mkdir(parents=True)
    for i in range(seconds):
        Image.fromarray(np.full((16, 16, 3), 255 if i in bright_seconds else 0, dtype=np.uint8)).save(frames / f"{i:03d}.png")
    out = tmp_path / "clip.avi"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-framerate", "1", "-i", str(frames / "%03d.png"), "-c:v", "mjpeg", "-r", "2", str(out)],
        check=True,
    )
    return out.read_bytes()


def test_a_video_needs_the_weapon_in_three_frames_in_a_row(client, tmp_path):
    # Bright for one second only: 2 sampled frames at 2 fps, below the 3 the persistence rule needs.
    row = wait_done(client, send(client, video(tmp_path / "a", [2]), name="short.avi").json()["id"])
    assert (row["status"], row["min_frames"]) == ("done", 3)
    assert client.get("/api/alerts").json() == []
    # Bright for two seconds: 4 sampled frames in a row, so one alert.
    row = wait_done(client, send(client, video(tmp_path / "b", [2, 3]), name="long.avi").json()["id"])
    alerts = client.get("/api/alerts", params={"upload_id": row["id"]}).json()
    assert [a["label"] for a in alerts] == ["gun"]
