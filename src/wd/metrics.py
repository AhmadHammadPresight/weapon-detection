"""Detection scores: precision and recall at a threshold, and AP at IoU 0.5, per class and overall.

A prediction is correct when it has the right class and overlaps an unclaimed true box of that class by IoU >= 0.5;
predictions are matched strongest first, and each true box can be claimed once.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from wd.boxes import Box, iou

IOU_MATCH = 0.5
THRESHOLD_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 .. 0.95


@dataclass(frozen=True)
class Pred:
    image: str
    label: str
    confidence: float
    box: Box


@dataclass(frozen=True)
class GroundTruth:
    image: str
    label: str
    box: Box


def match(preds: list[Pred], truths: list[GroundTruth], iou_match: float = IOU_MATCH) -> list[bool]:
    """Whether each prediction is a true positive, listed strongest prediction first."""
    claimed: set[int] = set()
    outcome = []
    for pred in sorted(preds, key=lambda p: p.confidence, reverse=True):
        best, best_iou = None, iou_match
        for index, truth in enumerate(truths):
            if index in claimed or truth.image != pred.image or truth.label != pred.label:
                continue
            overlap = iou(pred.box, truth.box)
            if overlap >= best_iou:
                best, best_iou = index, overlap
        if best is not None:
            claimed.add(best)
        outcome.append(best is not None)
    return outcome


def average_precision(preds: list[Pred], truths: list[GroundTruth]) -> float:
    """All-point interpolated AP (the area under the precision envelope), for one class."""
    if not truths:
        return 0.0
    hits = match(preds, truths)
    tp = fp = 0
    recalls, precisions = [0.0], [1.0]
    for hit in hits:
        tp += hit
        fp += not hit
        recalls.append(tp / len(truths))
        precisions.append(tp / (tp + fp))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    return sum((recalls[i] - recalls[i - 1]) * precisions[i] for i in range(1, len(recalls)))


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


Thresholds = float | Mapping[str, float]  # one alert threshold for every class, or one per class


def threshold_for(label: str, thresholds: Thresholds) -> float:
    return thresholds if isinstance(thresholds, (int, float)) else thresholds[label]


def counts_at(preds: list[Pred], truths: list[GroundTruth], thresholds: Thresholds) -> Counts:
    kept = [p for p in preds if p.confidence >= threshold_for(p.label, thresholds)]
    tp = sum(match(kept, truths))
    return Counts(tp=tp, fp=len(kept) - tp, fn=len(truths) - tp)


def choose_threshold(preds: list[Pred], truths: list[GroundTruth], grid: tuple[float, ...] = THRESHOLD_GRID) -> float:
    """The grid threshold with the best F1 over the given predictions; the middle one when several tie."""
    scored = [(round(counts_at(preds, truths, t).f1, 6), t) for t in grid]
    best = max(f1 for f1, _ in scored)
    tied = [t for f1, t in scored if f1 == best]
    return tied[(len(tied) - 1) // 2]


def choose_class_thresholds(preds: list[Pred], truths: list[GroundTruth], classes: tuple[str, ...]) -> dict[str, float]:
    """One threshold per class, each the best F1 for that class alone (its false detections included)."""
    return {
        cls: choose_threshold([p for p in preds if p.label == cls], [t for t in truths if t.label == cls])
        for cls in classes
    }


def score(preds: list[Pred], truths: list[GroundTruth], classes: tuple[str, ...], thresholds: Thresholds) -> dict:
    """Precision, recall and AP50 per class, plus the same pooled over all classes and mAP50."""
    out: dict = {"thresholds": {cls: threshold_for(cls, thresholds) for cls in classes}, "per_class": {}}
    for cls in classes:
        p = [x for x in preds if x.label == cls]
        t = [x for x in truths if x.label == cls]
        c = counts_at(p, t, thresholds)
        out["per_class"][cls] = {
            "true_boxes": len(t),
            "tp": c.tp,
            "fp": c.fp,
            "fn": c.fn,
            "precision": round(c.precision, 3),
            "recall": round(c.recall, 3),
            "ap50": round(average_precision(p, t), 3),
        }
    c = counts_at(preds, truths, thresholds)
    out["overall"] = {
        "true_boxes": len(truths),
        "tp": c.tp,
        "fp": c.fp,
        "fn": c.fn,
        "precision": round(c.precision, 3),
        "recall": round(c.recall, 3),
        "map50": round(sum(out["per_class"][cls]["ap50"] for cls in classes) / len(classes), 3),
    }
    return out
