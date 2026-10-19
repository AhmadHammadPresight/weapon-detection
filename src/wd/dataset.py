"""The fixed train/test split, committed under `dataset/` and listed in `dataset/manifest.json`.

- 20 training and 20 test images with weapons, from the ATM robbery set (source "atm").
- 10 weapon-free training images ("negatives", label "none", no boxes) from the Simuletic surveillance set
  (source "grounding"), so the model also learns what is not a weapon.

`wd.tools.build_split` wrote the manifest once from the Kaggle downloads; everything else reads only the copy here.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from wd.boxes import Box
from wd.settings import REPO_ROOT

DATASET_DIR = REPO_ROOT / "dataset"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
CLASSES = ("gun", "knife")
SPLITS = ("train", "test")
NEGATIVE = "none"  # the label of a weapon-free image
SOURCES = ("atm", "grounding")
# `handgun_robbery_var10_img3.png` -> scene "handgun", variant "handgun_robbery_var10"
NAME = re.compile(r"^(?P<variant>(?P<scene>handgun|knife)_robbery_var\d+)_img\d+\.png$")
SCENE_CLASS = {"handgun": "gun", "knife": "knife"}


@dataclass(frozen=True)
class Truth:
    label: str
    box: Box


@dataclass(frozen=True)
class Sample:
    split: str
    source: str  # "atm" (weapons) or "grounding" (weapon-free negatives)
    file: str  # relative to dataset/
    source_file: str  # relative to the Kaggle dataset folder
    variant: str  # the scene: ATM scene variant, or the surveillance set's scene name for a negative
    label: str  # the scene's weapon class, or "none" for a negative
    width: int
    height: int
    sha256: str
    boxes: tuple[Truth, ...]

    @property
    def path(self) -> Path:
        return DATASET_DIR / self.file

    @property
    def name(self) -> str:
        return Path(self.file).name


@dataclass(frozen=True)
class Manifest:
    seed: int
    samples: tuple[Sample, ...]
    meta: dict

    def split(self, name: str) -> list[Sample]:
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r}")
        return [s for s in self.samples if s.split == name]


def parse_name(filename: str) -> tuple[str, str]:
    """(variant, weapon class) of an ATM image file name."""
    match = NAME.match(filename)
    if not match:
        raise ValueError(f"not an ATM robbery image name: {filename!r}")
    return match["variant"], SCENE_CLASS[match["scene"]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    data = json.loads(path.read_text())
    samples = tuple(
        Sample(
            split=item["split"],
            source=item["source"],
            file=item["file"],
            source_file=item["source_file"],
            variant=item["variant"],
            label=item["label"],
            width=int(item["width"]),
            height=int(item["height"]),
            sha256=item["sha256"],
            boxes=tuple(Truth(b["label"], tuple(float(v) for v in b["box"])) for b in item["boxes"]),
        )
        for item in data["images"]
    )
    meta = {k: v for k, v in data.items() if k != "images"}
    return Manifest(seed=int(data["seed"]), samples=samples, meta=meta)


def check_manifest(manifest: Manifest, per_split: int = 20, negatives: int = 10, check_files: bool = True) -> list[str]:
    """Every rule the split must keep; returns the broken ones (empty when the split is sound)."""
    problems = []
    negative_samples = [s for s in manifest.samples if s.label == NEGATIVE]
    if len(negative_samples) != negatives:
        problems.append(f"{len(negative_samples)} weapon-free images, expected {negatives}")
    for s in negative_samples:
        if s.split != "train" or s.source != "grounding" or s.boxes:
            problems.append(f"{s.name}: a weapon-free image must be a training image from the surveillance set, without boxes")
        if s.file != f"images/negatives/{s.name}":
            problems.append(f"{s.name}: a weapon-free image belongs in images/negatives/")
    for split in SPLITS:
        samples = [s for s in manifest.split(split) if s.label != NEGATIVE]
        if len(samples) != per_split:
            problems.append(f"{split} has {len(samples)} images, expected {per_split}")
        for cls in CLASSES:
            count = sum(s.label == cls for s in samples)
            if count != per_split // 2:
                problems.append(f"{split} has {count} {cls} images, expected {per_split // 2}")
    train_variants = {s.variant for s in manifest.split("train") if s.source == "atm"}
    test_variants = {s.variant for s in manifest.split("test")}
    if shared := train_variants & test_variants:
        problems.append(f"scene variants in both train and test: {sorted(shared)}")
    names = [s.name for s in manifest.samples]
    if len(set(names)) != len(names):
        problems.append("an image is listed twice")
    for s in manifest.samples:
        if s.source not in SOURCES:
            problems.append(f"{s.name}: unknown source {s.source!r}")
        if s.label == NEGATIVE:
            pass
        elif s.source != "atm":
            problems.append(f"{s.name}: weapon images come from the ATM set only")
        else:
            variant, label = parse_name(s.name)
            if (variant, label) != (s.variant, s.label):
                problems.append(f"{s.name}: variant or class does not match its file name")
            if not s.boxes:
                problems.append(f"{s.name}: no weapon box")
            if s.file != f"images/{s.split}/{s.name}":
                problems.append(f"{s.name}: belongs in images/{s.split}/")
        for truth in s.boxes:
            x1, y1, x2, y2 = truth.box
            if truth.label not in CLASSES:
                problems.append(f"{s.name}: unknown class {truth.label!r}")
            if not (0 <= x1 < x2 <= s.width and 0 <= y1 < y2 <= s.height):
                problems.append(f"{s.name}: box {truth.box} is outside the {s.width}x{s.height} image")
        if check_files:
            if not s.path.is_file():
                problems.append(f"{s.file} is missing")
            elif sha256(s.path) != s.sha256:
                problems.append(f"{s.file} does not match its recorded SHA-256")
    return problems
