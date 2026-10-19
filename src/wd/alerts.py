"""Alert logic: which detections raise an alert, and how a video's frames are grouped into alerts.

A detection raises an alert when its confidence reaches the alert threshold (chosen on the training images).
In a video the same weapon shows up in many sampled frames, so frames are grouped per class: a detection opens a
new alert unless an alert for the same class was seen within the cooldown window, in which case it joins that alert.
Each alert keeps its strongest frame, which is the boxed frame shown to the operator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from wd.detectors import Detection


def alerting(detections: list[Detection], threshold: float) -> list[Detection]:
    """The detections that raise an alert, strongest first."""
    return sorted((d for d in detections if d.confidence >= threshold), key=lambda d: d.confidence, reverse=True)


@dataclass
class Alert:
    label: str
    first_seen: float  # seconds from the start of the upload
    last_seen: float
    best_at: float  # the moment of the strongest frame
    confidence: float  # the strongest confidence for this class in the alert
    detections: list[Detection]  # alerting detections of every class in the strongest frame
    frames: int = 1
    key: int = field(default=-1)  # caller's id for the alert, set by AlertGrouper


@dataclass(frozen=True)
class Change:
    kind: str  # "new" or "update" (a stronger frame replaced the alert's frame)
    alert: Alert


class AlertGrouper:
    """Feed sampled frames in time order; get back the alerts they open or strengthen."""

    def __init__(self, threshold: float, cooldown_seconds: float) -> None:
        self.threshold = threshold
        self.cooldown = cooldown_seconds
        self.alerts: list[Alert] = []
        self._open: dict[str, Alert] = {}

    def add(self, t: float, detections: list[Detection]) -> list[Change]:
        hits = alerting(detections, self.threshold)
        changes = []
        for label in sorted({d.label for d in hits}):
            strongest = max(d.confidence for d in hits if d.label == label)
            alert = self._open.get(label)
            if alert is not None and t - alert.last_seen <= self.cooldown:
                alert.last_seen = t
                alert.frames += 1
                if strongest > alert.confidence:
                    alert.confidence, alert.best_at, alert.detections = strongest, t, hits
                    changes.append(Change("update", alert))
                continue
            alert = Alert(label, t, t, t, strongest, hits, key=len(self.alerts))
            self.alerts.append(alert)
            self._open[label] = alert
            changes.append(Change("new", alert))
        return changes


def alert_time(start: datetime, offset_seconds: float) -> datetime:
    """Camera-local wall-clock time of a moment in an upload."""
    return start + timedelta(seconds=offset_seconds)
