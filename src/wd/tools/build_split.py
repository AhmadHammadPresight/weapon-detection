"""Pick the fixed split from the two Simuletic Kaggle datasets and copy it into `dataset/`.

Run once by the maintainer; the result is committed, so nothing reads the Kaggle downloads afterwards:

    .venv/bin/python -m wd.tools.build_split <ATM robbery dataset folder> <surveillance VLM dataset folder>

The source folders are only read. The pick is deterministic for a given seed and sources:
- an image is eligible when its labels hold exactly one weapon box, of the class its scene name says
  (a few knife scenes carry a gun label or two knives; those are left out rather than guessed);
- per weapon class, the scene variants are shuffled with the seed: the first 10 give the training images
  and the next 10 the test images, one image per variant, so no test scene is ever seen in training;
- 10 weapon-free training images ("negatives") come from the surveillance set's "safe" images, whose dataset.json
  answer says no weapon is there: 5 from its knife scenes and 5 from its firearm scenes, drawn with their own seed.
  They teach the model what is not a weapon; they are synthetic and share nothing with the real CCTV clips.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from wd.boxes import parse_yolo_line, rounded, yolo_to_xyxy
from wd.dataset import CLASSES, DATASET_DIR, MANIFEST_PATH, NEGATIVE, check_manifest, load_manifest, parse_name, sha256

SEED = 20261019
PER_CLASS = 10
# Class ids in the dataset's labels.txt: 0 offender, 1 victim, 2 gun, 3 knife. People are ignored.
WEAPON_IDS = {2: "gun", 3: "knife"}
NEGATIVES_PER_FAMILY = 5
SOURCES = {
    "atm": {
        "name": "CCTV ATM Robbery Detection Dataset: Gun & Knife",
        "publisher": "Simuletic",
        "url": "https://www.kaggle.com/datasets/simuletic/cctv-atm-robbery-detection-dataset-gun-and-knife",
        "used_for": "the 20 training and 20 test images with weapons",
    },
    "grounding": {
        "name": "Surveillance VLM: Weapon & Knife Detection Dataset",
        "publisher": "Simuletic",
        "url": "https://www.kaggle.com/datasets/simuletic/surveillance-vlm-weapon-and-knife-detection-dataset",
        "used_for": "the 10 weapon-free training images (negatives)",
    },
}
SAFE_ANSWER = re.compile(r"^No (knife|firearm or weapon) detected\.$")
SAFE_FAMILIES = ("knifes_safe_", "weapons_safe_")


def natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def weapon_boxes(label_file: Path, width: int, height: int) -> list[dict]:
    boxes = []
    for line in label_file.read_text().splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, w, h = parse_yolo_line(line)
        if class_id in WEAPON_IDS:
            boxes.append({"label": WEAPON_IDS[class_id], "box": rounded(yolo_to_xyxy(cx, cy, w, h, width, height))})
    return boxes


def pick(source: Path, seed: int = SEED, per_class: int = PER_CLASS) -> list[dict]:
    eligible: dict[str, dict[str, list[dict]]] = {cls: defaultdict(list) for cls in CLASSES}
    for image in sorted((source / "images").glob("*.png"), key=lambda p: natural_key(p.name)):
        variant, label = parse_name(image.name)
        label_file = source / "labels" / f"{image.stem}.txt"
        if not label_file.is_file():
            continue
        with Image.open(image) as im:
            width, height = im.size
        boxes = weapon_boxes(label_file, width, height)
        if len(boxes) != 1 or boxes[0]["label"] != label:
            continue
        eligible[label][variant].append(
            {"name": image.name, "variant": variant, "label": label, "width": width, "height": height, "boxes": boxes}
        )
    rng = random.Random(seed)
    chosen = []
    for cls in CLASSES:
        variants = sorted(eligible[cls], key=natural_key)
        if len(variants) < 2 * per_class:
            raise SystemExit(f"only {len(variants)} eligible {cls} scene variants; need {2 * per_class}")
        rng.shuffle(variants)
        for split, group in (("train", variants[:per_class]), ("test", variants[per_class : 2 * per_class])):
            for variant in sorted(group, key=natural_key):
                item = rng.choice(eligible[cls][variant])
                chosen.append({**item, "split": split})
    return chosen


def pick_negatives(source: Path, seed: int = SEED, per_family: int = NEGATIVES_PER_FAMILY) -> list[dict]:
    """Weapon-free images: the surveillance set's images whose first answer says no weapon is there."""
    entries = json.loads((source / "dataset.json").read_text())
    safe = sorted(
        (e["image"] for e in entries if SAFE_ANSWER.match(e["conversations"][1]["value"].strip())), key=natural_key
    )
    rng = random.Random(seed + 1)  # its own sequence, so the ATM pick above stays the same
    chosen = []
    for family in SAFE_FAMILIES:
        members = [name for name in safe if name.startswith(family)]
        if len(members) < per_family:
            raise SystemExit(f"only {len(members)} safe {family}* images; need {per_family}")
        for name in sorted(rng.sample(members, per_family), key=natural_key):
            with Image.open(source / "images" / name) as im:
                width, height = im.size
            scene = re.sub(r"_\d+\.png$", "", name)
            chosen.append({"name": name, "variant": scene, "width": width, "height": height})
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="the unzipped ATM robbery dataset folder (holds images/ and labels/)")
    parser.add_argument("negatives_source", type=Path, help="the unzipped surveillance VLM dataset folder (holds dataset.json)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    negatives_source = args.negatives_source.expanduser().resolve()

    chosen = pick(source, args.seed)
    negatives = pick_negatives(negatives_source, args.seed)
    for split in ("train", "test", "negatives"):
        target = DATASET_DIR / "images" / split
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
    images = []
    for item in sorted(chosen, key=lambda i: (i["split"] != "train", natural_key(i["name"]))):
        relative = f"images/{item['split']}/{item['name']}"
        shutil.copyfile(source / "images" / item["name"], DATASET_DIR / relative)
        images.append(
            {
                "split": item["split"],
                "source": "atm",
                "file": relative,
                "source_file": f"images/{item['name']}",
                "variant": item["variant"],
                "label": item["label"],
                "width": item["width"],
                "height": item["height"],
                "sha256": sha256(DATASET_DIR / relative),
                "boxes": item["boxes"],
            }
        )
    for item in negatives:
        relative = f"images/negatives/{item['name']}"
        shutil.copyfile(negatives_source / "images" / item["name"], DATASET_DIR / relative)
        images.append(
            {
                "split": "train",
                "source": "grounding",
                "file": relative,
                "source_file": f"images/{item['name']}",
                "variant": item["variant"],
                "label": NEGATIVE,
                "width": item["width"],
                "height": item["height"],
                "sha256": sha256(DATASET_DIR / relative),
                "boxes": [],
            }
        )
    manifest = {
        "sources": SOURCES,
        "seed": args.seed,
        "classes": list(CLASSES),
        "rule": (
            "Eligible: exactly one weapon box, of the class the scene name says. Per class, scene variants are "
            "shuffled with the seed; the first 10 give one training image each and the next 10 one test image each. "
            "Negatives: 5 safe knife-scene and 5 safe firearm-scene images of the surveillance set, drawn with seed + 1."
        ),
        "boxes": "pixels of the original image, [x1, y1, x2, y2], converted from the YOLO labels",
        "images": images,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1) + "\n")
    problems = check_manifest(load_manifest())
    if problems:
        raise SystemExit("the split breaks its rules:\n" + "\n".join(problems))
    print(f"wrote {len(images)} images and {MANIFEST_PATH.relative_to(DATASET_DIR.parent)}")


if __name__ == "__main__":
    main()
