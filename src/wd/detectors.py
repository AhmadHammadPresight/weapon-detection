"""The two weapon detectors compared in this PoC, behind one small interface.

- `rfdetr`: RF-DETR Medium (Apache-2.0) fine-tuned on the 20 training images for gun and knife (scripts/train.sh).
- `owlv2`: OWLv2 base (Apache-2.0), open-vocabulary, no training, prompted with text for gun and knife.

Both return every box above a low floor score, so the evaluation can rank them; the alert threshold is applied later.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from wd.boxes import Box, iou

# Boxes below this score are dropped before anything else; the alert threshold is chosen on training data.
SCORE_FLOOR = 0.05
DETECTORS = ("rfdetr", "owlv2")

# COCO-pretrained starting point. A bare file name: rfdetr keeps it in its own cache (RF_HOME, by default
# ~/.roboflow/models), downloads it when missing and checks its MD5, which the pinned rfdetr version fixes.
RFDETR_BASE_WEIGHTS = "rf-detr-medium.pth"
RFDETR_FINETUNED = "rfdetr-weapons/checkpoint_best_total.pth"

OWLV2_MODEL = "google/owlv2-base-patch16-ensemble"
OWLV2_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"  # pinned Hugging Face commit
# Several phrasings per class; each box takes its best-matching phrase, then the phrase's class.
OWLV2_PROMPTS = {
    "a photo of a gun": "gun",
    "a photo of a handgun": "gun",
    "a photo of a pistol": "gun",
    "a photo of a knife": "knife",
}


@dataclass(frozen=True)
class Detection:
    label: str  # "gun" or "knife"
    confidence: float
    box: Box  # x1, y1, x2, y2 in pixels


class Detector(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Weapons in one RGB image (H x W x 3, uint8)."""
        ...


def suppress_overlaps(detections: list[Detection], threshold: float = 0.5) -> list[Detection]:
    """Keep the strongest of boxes that overlap by more than `threshold` IoU, whatever their class."""
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(iou(det.box, k.box) <= threshold for k in kept):
            kept.append(det)
    return kept


class RFDETRWeapons:
    name = "rfdetr"

    def __init__(self, checkpoint: Path, device: str) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"no fine-tuned RF-DETR checkpoint at {checkpoint}; run scripts/train.sh first")
        from rfdetr import RFDETRMedium

        self._model = RFDETRMedium.from_checkpoint(str(checkpoint), device=device)
        self._labels = list(self._model.class_names)
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[Detection]:
        with self._lock:
            result = self._model.predict(image, threshold=SCORE_FLOOR)
        out = []
        for box, class_id, conf in zip(result.xyxy, result.class_id, result.confidence, strict=True):
            # The classification head has one slot more than the classes (no object); it is not a weapon.
            if 0 <= int(class_id) < len(self._labels):
                out.append(Detection(self._labels[int(class_id)], float(conf), tuple(float(v) for v in box)))
        return suppress_overlaps(out)


class OWLv2Weapons:
    name = "owlv2"

    def __init__(self, device: str) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._torch = torch
        self._device = device
        self._processor = Owlv2Processor.from_pretrained(OWLV2_MODEL, revision=OWLV2_REVISION)
        self._model = Owlv2ForObjectDetection.from_pretrained(OWLV2_MODEL, revision=OWLV2_REVISION).to(device).eval()
        self._prompts = list(OWLV2_PROMPTS)
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        torch = self._torch
        with self._lock, torch.inference_mode():
            inputs = self._processor(text=[self._prompts], images=image, return_tensors="pt").to(self._device)
            outputs = self._model(**inputs)
            # OWLv2 pads the image to a square, so boxes are scaled back with the longer side.
            side = max(height, width)
            results = self._processor.post_process_grounded_object_detection(
                outputs, threshold=SCORE_FLOOR, target_sizes=torch.tensor([[side, side]])
            )[0]
        out = []
        for box, score, index in zip(results["boxes"].tolist(), results["scores"].tolist(), results["labels"].tolist(), strict=True):
            x1, y1, x2, y2 = box
            clipped = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                out.append(Detection(OWLV2_PROMPTS[self._prompts[index]], float(score), clipped))
        return suppress_overlaps(out)


def load_detector(name: str, models_dir: Path, device: str) -> Detector:
    if name == "rfdetr":
        return RFDETRWeapons(models_dir / RFDETR_FINETUNED, device)
    if name == "owlv2":
        return OWLv2Weapons(device)
    raise ValueError(f"unknown detector {name!r}; choose one of {DETECTORS}")
