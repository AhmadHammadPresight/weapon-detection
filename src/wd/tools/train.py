"""Fine-tune RF-DETR Medium on the 20 training images for two classes, gun and knife.

    .venv/bin/python -m wd.tools.train        (scripts/train.sh runs this)

The run is seeded. There is no separate validation set: with 20 images every one is needed for training, so the
"validation" images RF-DETR asks for are the training images again. The checkpoint choice therefore only ever sees
training images, and the test images stay unseen until scripts/evaluate.sh.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

from wd.dataset import CLASSES, load_manifest
from wd.detectors import RFDETR_BASE_WEIGHTS, RFDETR_FINETUNED
from wd.settings import get_settings

SEED = 7
EPOCHS = 60
BATCH_SIZE = 4
LEARNING_RATE = 1e-4


def write_coco(split_dir, samples) -> None:
    """A Roboflow-style COCO folder (images plus _annotations.coco.json) for the given samples."""
    split_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for image_id, sample in enumerate(samples):
        shutil.copyfile(sample.path, split_dir / sample.name)
        images.append({"id": image_id, "file_name": sample.name, "width": sample.width, "height": sample.height})
        for truth in sample.boxes:
            x1, y1, x2, y2 = truth.box
            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": image_id,
                    "category_id": CLASSES.index(truth.label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
    categories = [{"id": i, "name": name, "supercategory": "weapon"} for i, name in enumerate(CLASSES)]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default=None, help="mps (default from WD_DEVICE) or cpu")
    args = parser.parse_args()
    settings = get_settings()
    device = args.device or settings.device

    train = load_manifest().split("train")
    work = settings.data_dir / "train-coco"
    if work.exists():
        shutil.rmtree(work)
    for split in ("train", "valid"):
        write_coco(work / split, train)

    output = (settings.models_dir / RFDETR_FINETUNED).parent
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    from rfdetr import RFDETRMedium

    model = RFDETRMedium(pretrain_weights=RFDETR_BASE_WEIGHTS, device=device)
    started = time.monotonic()
    model.train(
        dataset_dir=str(work),
        output_dir=str(output),
        epochs=args.epochs,
        batch_size=BATCH_SIZE,
        grad_accum_steps=1,
        lr=LEARNING_RATE,
        seed=SEED,
        device=device,
        num_workers=0,
        tensorboard=False,
        progress_bar=None,
        checkpoint_interval=args.epochs,
    )
    checkpoint = settings.models_dir / RFDETR_FINETUNED
    if not checkpoint.is_file():
        raise SystemExit(f"training finished but wrote no {checkpoint.name}; see the log above")
    print(f"trained in {time.monotonic() - started:.0f}s: {checkpoint}")


if __name__ == "__main__":
    main()
