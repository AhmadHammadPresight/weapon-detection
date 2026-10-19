"""Score both detectors on the 20 test images and write results/evaluation.json.

    .venv/bin/python -m wd.tools.evaluate        (scripts/evaluate.sh runs this)

For each detector, the alert thresholds are chosen on the training images only (best F1 on a 0.05 grid; one per
class, plus one shared threshold kept for comparison), then applied unchanged to the test images. The training
images are the 20 weapon images and the 10 weapon-free ones, or the weapon images only with --without-negatives.
The test images are never used to choose anything.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from wd.boxes import rounded
from wd.dataset import CLASSES, NEGATIVE, Sample, check_manifest, load_manifest
from wd.detectors import DETECTORS, OWLV2_MODEL, OWLV2_PROMPTS, OWLV2_REVISION, SCORE_FLOOR, Detector, load_detector
from wd.media import read_image
from wd.metrics import IOU_MATCH, GroundTruth, Pred, choose_class_thresholds, choose_threshold, counts_at, score
from wd.results import EVALUATION_PATH, read_json, write_json
from wd.settings import get_settings
from wd.tools.train import TRAINING_RECORD

def model_note(name: str, models_dir: Path) -> str:
    if name == "owlv2":
        return f"{OWLV2_MODEL}@{OWLV2_REVISION[:10]}, no training, prompts: {', '.join(OWLV2_PROMPTS)}"
    run = read_json(models_dir / TRAINING_RECORD) or {}
    images = f"{run.get('weapon_images', '?')} weapon images and {run.get('negative_images', '?')} weapon-free images"
    return f"RF-DETR Medium, COCO-pretrained, fine-tuned on {images} ({run.get('epochs', '?')} epochs, seed {run.get('seed', '?')})"


def run(detector: Detector, samples: list[Sample]) -> tuple[list[Pred], list[float]]:
    preds, seconds = [], []
    for sample in samples:
        image = read_image(sample.path)
        started = time.perf_counter()
        detections = detector.detect(image)
        seconds.append(time.perf_counter() - started)
        preds += [Pred(sample.file, d.label, d.confidence, d.box) for d in detections]
    return preds, seconds


def truths(samples: list[Sample]) -> list[GroundTruth]:
    return [GroundTruth(s.file, t.label, t.box) for s in samples for t in s.boxes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detectors", nargs="+", choices=DETECTORS, default=list(DETECTORS))
    parser.add_argument(
        "--without-negatives", action="store_true",
        help="choose thresholds on the 20 weapon images only (for a model trained with --without-negatives)",
    )
    args = parser.parse_args()
    settings = get_settings()
    manifest = load_manifest()
    if problems := check_manifest(manifest):
        raise SystemExit("the split breaks its rules:\n" + "\n".join(problems))
    train, test = manifest.split("train"), manifest.split("test")
    if args.without_negatives:
        train = [s for s in train if s.label != NEGATIVE]

    results = {}
    for name in args.detectors:
        print(f"==> {name}")
        detector = load_detector(name, settings.models_dir, settings.device)
        detector.detect(read_image(train[0].path))  # warm-up, so timings are steady
        train_preds, _ = run(detector, train)
        # Both chosen on the training images only: one shared threshold (the first version of this PoC) and one
        # threshold per class (what the app uses now).
        single = choose_threshold(train_preds, truths(train))
        thresholds = choose_class_thresholds(train_preds, truths(train), CLASSES)
        test_preds, seconds = run(detector, test)
        negatives = [s for s in train if not s.boxes]
        result = {
            "model": model_note(name, settings.models_dir),
            "thresholds": thresholds,
            "single_threshold": single,
            "train": score(train_preds, truths(train), CLASSES, thresholds),
            "train_negatives_with_alerts": sum(
                any(p.image == s.file and p.confidence >= thresholds[p.label] for p in train_preds) for s in negatives
            ),
            "train_negatives": len(negatives),
            "test": score(test_preds, truths(test), CLASSES, thresholds),
            "test_single_threshold": score(test_preds, truths(test), CLASSES, single),
            "seconds_per_image": round(statistics.median(seconds), 3),
            "per_image": {
                s.file: asdict(counts_at([p for p in test_preds if p.image == s.file], truths([s]), thresholds))
                for s in test
            },
            "predictions": {
                s.file: [
                    {"label": p.label, "confidence": round(p.confidence, 3), "box": rounded(p.box)}
                    for p in sorted(test_preds, key=lambda p: p.confidence, reverse=True)
                    if p.image == s.file
                ]
                for s in test
            },
        }
        results[name] = result
        overall, per_class = result["test"]["overall"], result["test"]["per_class"]
        print(
            f"  thresholds {thresholds} (one shared: {single}), from training images; test: precision {overall['precision']}, "
            f"recall {overall['recall']}, mAP50 {overall['map50']} "
            f"(gun AP50 {per_class['gun']['ap50']}, knife AP50 {per_class['knife']['ap50']}); "
            f"{result['seconds_per_image']}s per image"
        )
        del detector

    write_json(
        EVALUATION_PATH,
        {
            "generated": datetime.now(UTC).isoformat(timespec="seconds"),
            "note": "Small-sample PoC result on 20 synthetic test images; not a performance claim.",
            "iou_match": IOU_MATCH,
            "score_floor": SCORE_FLOOR,
            "threshold_rule": "best F1 on the training images (one per class; and one shared for comparison), 0.05 grid, middle of ties",
            "test_images": [
                {"file": s.file, "label": s.label, "variant": s.variant, "width": s.width, "height": s.height}
                for s in test
            ],
            "truth": {s.file: [{"label": t.label, "box": list(t.box)} for t in s.boxes] for s in test},
            "environment": {
                "machine": platform.machine(),
                "macos": platform.mac_ver()[0],
                "device": settings.device,
                "torch": version("torch"),
                "rfdetr": version("rfdetr"),
                "transformers": version("transformers"),
            },
            "detectors": results,
        },
    )
    print(f"wrote {EVALUATION_PATH.relative_to(EVALUATION_PATH.parents[1])}")


if __name__ == "__main__":
    main()
