"""Box helpers: label conversion, clipping and overlap.

Every box in this project is (x1, y1, x2, y2) in pixels of the original image.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, width: int, height: int) -> Box:
    """A YOLO label box (centre and size, 0-1 of the image) as a pixel box clipped to the image."""
    for name, value in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"YOLO {name} must be within 0-1, got {value}")
    if width <= 0 or height <= 0:
        raise ValueError(f"image size must be positive, got {width}x{height}")
    box = ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)
    return clip(box, width, height)


def parse_yolo_line(line: str) -> tuple[int, float, float, float, float]:
    """One `class cx cy w h` label line."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"a YOLO label line has 5 fields, got {len(parts)}: {line!r}")
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def clip(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )


def area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes; 0 when either box is empty."""
    inter = area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def rounded(box: Box, digits: int = 1) -> list[float]:
    return [round(float(v), digits) for v in box]
