"""Pick the fixed 20 + 20 split from the Kaggle ATM robbery dataset and copy it into `dataset/`.

Run once by the maintainer; the result is committed, so nothing reads the Kaggle download afterwards:

    .venv/bin/python -m wd.tools.build_split <path to CCTV_ATM_Robbery_Knife_Gun_Detection_Dataset>

The source folder is only read. The pick is deterministic for a given seed and source:
- an image is eligible when its labels hold exactly one weapon box, of the class its scene name says
  (a few knife scenes carry a gun label or two knives; those are left out rather than guessed);
- per weapon class, the scene variants are shuffled with the seed: the first 10 give the training images
  and the next 10 the test images, one image per variant, so no test scene is ever seen in training.
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
from wd.dataset import CLASSES, DATASET_DIR, MANIFEST_PATH, check_manifest, load_manifest, parse_name, sha256

SEED = 20261019
PER_CLASS = 10
# Class ids in the dataset's labels.txt: 0 offender, 1 victim, 2 gun, 3 knife. People are ignored.
WEAPON_IDS = {2: "gun", 3: "knife"}
SOURCE = {
    "name": "CCTV ATM Robbery Detection Dataset: Gun & Knife",
    "publisher": "Simuletic",
    "url": "https://www.kaggle.com/datasets/simuletic/cctv-atm-robbery-detection-dataset-gun-and-knife",
}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="the unzipped Kaggle dataset folder (holds images/ and labels/)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()

    chosen = pick(source, args.seed)
    for split in ("train", "test"):
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
    manifest = {
        "source": SOURCE,
        "seed": args.seed,
        "classes": list(CLASSES),
        "rule": (
            "Eligible: exactly one weapon box, of the class the scene name says. Per class, scene variants are "
            "shuffled with the seed; the first 10 give one training image each and the next 10 one test image each."
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
