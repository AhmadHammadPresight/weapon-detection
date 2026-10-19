"""Alert logic: which detections raise an alert, and how a video's frames are grouped into alerts.

A detection counts when its confidence reaches its class's alert threshold (chosen on the training images).

Persistence (videos): a weapon must be seen in `min_frames` consecutive sampled frames, each time near where it was
in the frame before, before it raises an alert. A real weapon stays in view for seconds, while most false detections
flicker on single frames. Still images use `min_frames=1`, so a single detection raises an alert.

Grouping: nearby boxes of the same class in one frame are one weapon, and a confirmed weapon opens a new alert unless
an alert for the same class was seen within the cooldown window, in which case it joins that alert (one alert per
event). Any sighting of the class, even one not yet confirmed, keeps that window open. Each alert keeps its strongest frame, which is the boxed frame shown to the operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from wd.boxes import Box
from wd.detectors import Detection
from wd.metrics import Thresholds, threshold_for

# Two boxes are "near" when their centres are at most this many box lengths apart (the longer side of the larger
# box): a weapon can move about its own size between two samples, but not jump across the frame.
NEAR_FACTOR = 1.0


def near(a: Box, b: Box) -> bool:
    centre_a = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    centre_b = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    reach = NEAR_FACTOR * max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    return (centre_a[0] - centre_b[0]) ** 2 + (centre_a[1] - centre_b[1]) ** 2 <= reach**2


def alerting(detections: list[Detection], thresholds: Thresholds) -> list[Detection]:
    """The detections at or above their class's alert threshold, strongest first."""
    hits = (d for d in detections if d.confidence >= threshold_for(d.label, thresholds))
    return sorted(hits, key=lambda d: d.confidence, reverse=True)


@dataclass
class Alert:
    label: str
    first_seen: float  # seconds from the start of the upload (the start of the confirming streak)
    last_seen: float
    best_at: float  # the moment of the strongest frame
    confidence: float  # the strongest confidence for this class in the alert
    detections: list[Detection]  # confirmed detections of every class in the strongest frame
    frames: int = 1  # frames in which this alert's class was confirmed
    key: int = field(default=-1)  # caller's id for the alert, set by AlertGrouper


@dataclass(frozen=True)
class Streak:
    """One weapon followed through consecutive sampled frames."""

    detection: Detection  # its detection in the latest frame
    frames: int  # consecutive frames it has been seen in, up to now
    start: float  # when the streak began


@dataclass(frozen=True)
class Change:
    kind: str  # "new" or "update" (a stronger frame replaced the alert's frame)
    alert: Alert


class AlertGrouper:
    """Feed every sampled frame in time order; get back the alerts they open or strengthen."""

    def __init__(self, thresholds: Thresholds, cooldown_seconds: float, min_frames: int = 1) -> None:
        if min_frames < 1:
            raise ValueError("min_frames must be at least 1")
        self.thresholds = thresholds
        self.cooldown = cooldown_seconds
        self.min_frames = min_frames
        self.alerts: list[Alert] = []
        self._open: dict[str, Alert] = {}
        self._streaks: list[Streak] = []

    def _follow(self, t: float, hits: list[Detection]) -> list[Streak]:
        """Continue or start a streak for each weapon in this frame; streaks not seen in it end."""
        weapons: list[Detection] = []
        for hit in hits:  # strongest first; a nearby weaker box of the same class is the same weapon
            if not any(w.label == hit.label and near(w.box, hit.box) for w in weapons):
                weapons.append(hit)
        previous, streaks = list(self._streaks), []
        for weapon in weapons:
            match = max(
                (s for s in previous if s.detection.label == weapon.label and near(s.detection.box, weapon.box)),
                key=lambda s: s.frames,
                default=None,
            )
            if match is not None:
                previous.remove(match)
                streaks.append(Streak(weapon, match.frames + 1, match.start))
            else:
                streaks.append(Streak(weapon, 1, t))
        self._streaks = streaks
        return [s for s in streaks if s.frames >= self.min_frames]

    def add(self, t: float, detections: list[Detection]) -> list[Change]:
        hits = alerting(detections, self.thresholds)
        confirmed = self._follow(t, hits)
        shown = [s.detection for s in confirmed]  # the boxes drawn on this frame if it becomes an alert's frame
        changes = []
        for label in sorted({s.detection.label for s in confirmed}):
            strongest = max(s.detection.confidence for s in confirmed if s.detection.label == label)
            alert = self._open.get(label)
            if alert is not None and t - alert.last_seen <= self.cooldown:
                alert.last_seen = t
                alert.frames += 1
                if strongest > alert.confidence:
                    alert.confidence, alert.best_at, alert.detections = strongest, t, shown
                    changes.append(Change("update", alert))
                continue
            start = min(s.start for s in confirmed if s.detection.label == label)
            alert = Alert(label, start, t, t, strongest, shown, key=len(self.alerts))
            self.alerts.append(alert)
            self._open[label] = alert
            changes.append(Change("new", alert))
        # Any sighting of a class, confirmed or not, keeps its open alert going: a weapon that flickers between
        # confirmed runs is still the same event, not a new alert.
        for label in {h.label for h in hits}:
            alert = self._open.get(label)
            if alert is not None and t - alert.last_seen <= self.cooldown:
                alert.last_seen = t
        return changes


def alert_time(start: datetime, offset_seconds: float) -> datetime:
    """Camera-local wall-clock time of a moment in an upload."""
    return start + timedelta(seconds=offset_seconds)
