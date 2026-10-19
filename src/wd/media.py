"""Image and video input: read an uploaded still, sample frames from a video with FFmpeg, draw boxes.

Video decoding follows the Arabic video search PoC: FFmpeg asks for VideoToolbox (the Apple media engine)
and falls back to software decoding by itself for codecs it does not handle.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from wd.detectors import Detection

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mpg", ".mpeg", ".webm", ".ts"}
COLOURS = {"gun": (230, 40, 40), "knife": (255, 150, 0)}


class MediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    width: int
    height: int


def read_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as im:
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except OSError as exc:
        raise MediaError(f"could not read the image {path.name}: {exc}") from exc


def probe(path: Path) -> VideoInfo:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:stream_side_data=rotation:format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaError(f"ffprobe failed for {path.name}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise MediaError(f"no video stream in {path.name}")
    width, height = int(streams[0]["width"]), int(streams[0]["height"])
    for side in streams[0].get("side_data_list") or []:
        if abs(int(side.get("rotation", 0) or 0)) % 180 == 90:
            width, height = height, width  # FFmpeg auto-rotates on decode
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    return VideoInfo(duration=duration, width=width, height=height)


def iter_frames(path: Path, sample_fps: float, info: VideoInfo | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (seconds from start, RGB frame) at `sample_fps` frames per second."""
    info = info or probe(path)
    frame_bytes = info.width * info.height * 3
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-hwaccel", "videotoolbox",
        "-i", str(path), "-an", "-sn",
        "-vf", f"fps={sample_fps}", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    index = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield index / sample_fps, np.frombuffer(buf, dtype=np.uint8).reshape(info.height, info.width, 3)
            index += 1
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        returncode = proc.wait()
    if returncode != 0 and index == 0:
        raise MediaError(f"ffmpeg could not decode {path.name}: {stderr.strip()}")


def draw_boxes(image: np.ndarray, detections: list[Detection]) -> Image.Image:
    """The frame with a box around each detection, red for a gun and orange for a knife.

    No text is drawn into the frame: the page shows the class and confidence in the operator's language.
    """
    im = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(im)
    width = max(3, round(max(im.size) / 250))
    for det in detections:
        x1, y1, x2, y2 = det.box
        pad = width  # keep the outline just outside the weapon
        draw.rectangle((x1 - pad, y1 - pad, x2 + pad, y2 + pad), outline=COLOURS.get(det.label, (60, 140, 255)), width=width)
    return im
