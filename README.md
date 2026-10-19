# Weapon detection PoC

A small proof of concept that finds guns and knives in CCTV stills and videos, draws a box around each weapon and raises an alert with the camera, the time and the boxed frame.
It runs fully on a Mac (Apple Silicon), with local models only and no hosted APIs.
It is a separate project from the Arabic video search PoC and runs beside it (this app on port 8490, the video search demo on 8470 and 8480).

The numbers below are a small-sample PoC result on synthetic images, not a performance claim.

## What it does

- **Alerts page** (`/`): upload an image or a video with a camera ID and the recording's start time.
  Images are checked whole; videos are sampled at 2 frames per second.
  A weapon found at or above its class's alert threshold raises an alert that shows the boxed frame, the class (gun or knife), the confidence, the camera and the camera-local time.
  In a video, a weapon must also be seen in 3 sampled frames in a row (about 1 to 1.5 seconds), near the same place each time, before it raises an alert; sightings of the same class less than 10 seconds apart are one alert, which keeps its strongest frame.
- Arabic is the default, right to left, with a switch to English; one language is shown at a time, and every string lives in one table per language (`web/i18n.js`).
- `demo-samples/` holds ready-to-use demo files (test images the model gets right, weapon-free images and an 18-second real clip that raise no alert), described in `demo-samples/README.md`.

## Run it on this Mac

To run the demo on another Mac without training, with the ready-made model zip, follow [docs/DEMO-GUIDE.md](docs/DEMO-GUIDE.md).

If setup, training and evaluation have already been run in this folder:

```bash
scripts/start.sh     # starts the app in the background; prints http://127.0.0.1:8490
scripts/stop.sh      # stops it
```

`WD_PORT` picks another port.
The app keeps uploads, alerts and logs under `data/` (git-ignored); "Clear all" on the alerts page deletes them.

## Run from a clean Mac

On an Apple Silicon Mac with macOS 26 and Homebrew, from a fresh clone of this repo.
Times are from an M4 Pro with 48 GB; the download times depend on the network.

| Step | Command | Rough time |
| --- | --- | --- |
| 1. Install FFmpeg and uv if missing, the locked Python environment and the pinned model weights | `scripts/setup.sh` | 2 min, plus about 1 GB of weight downloads the first time on a Mac (RF-DETR Medium 400 MB, OWLv2 620 MB) |
| 2. Fine-tune RF-DETR on the 30 training images (seeded) | `scripts/train.sh` | 10 to 25 min (slower when another app is using the GPU) |
| 3. Score both detectors on the 20 test images, and count false alarms on the real clips | `scripts/evaluate.sh <clips folder>` | 2 min |
| 4. Start the app | `scripts/start.sh` | 15 s |
| 5. Run the unit tests (optional) | `.venv/bin/pytest` | 5 s |

The clips folder for step 3 is the `data/footage/clips/` folder of a clone of the Arabic video search project, filled by that project's `scripts/setup.sh` and `scripts/fetch_footage.sh`.
It can also be given as `WD_CLIPS_DIR`.
Without it, step 3 only scores the test images and leaves `results/false_alarms.json` as committed.

Nothing reads the original Kaggle downloads: the images and the split manifest are committed in `dataset/`.
Model weights go to the standard user caches (`~/.cache/huggingface` for OWLv2, `~/.roboflow/models` for RF-DETR, which `RF_HOME` can move), so a second clone on the same Mac skips the downloads; a Mac without them downloads them in step 1.
Every version is pinned: Python packages by `uv.lock`, OWLv2 by a Hugging Face commit, and the RF-DETR base weights by the MD5 that the locked `rfdetr` version checks.

The first version (stages 1 and 2 below) can be rebuilt with `scripts/train.sh --without-negatives`, then `.venv/bin/python -m wd.tools.evaluate --without-negatives` and `.venv/bin/python -m wd.tools.false_alarms <clips folder>`; its committed record is in `results/v1/`.

### How close a rerun gets

Steps 2 and 3 rebuild the fine-tuned weights and `results/evaluation.json` from scratch.
Training is seeded, but PyTorch on the Apple GPU (MPS) is not bit-for-bit deterministic, so a rerun gives slightly different weights.
Expect the test scores to stay within about one image of the committed ones (with 10 images per class, one image moves a per-class precision or recall by 0.1) and the false-alarm count to move by a few.
OWLv2 has no training step, so its scores repeat exactly on the same Mac.

Measured on the first version (before the weapon-free images were added): a fresh clone of this repo, run through steps 1 to 5 on the same Mac (with the weights already in the user caches), took 2 min for setup, 9 min for training, 2 min for evaluation and 6 s to start, and all tests passed.
Its scores against the first committed run:

| | First committed run | Fresh-clone rerun |
| --- | --- | --- |
| RF-DETR gun: precision / recall / AP@50 | 0.90 / 0.90 / 0.90 | 1.00 / 0.90 / 0.98 |
| RF-DETR knife: precision / recall / AP@50 | 0.90 / 0.90 / 0.98 | 0.90 / 0.90 / 0.87 |
| RF-DETR all: precision / recall / mAP@50 | 0.90 / 0.90 / 0.94 | 0.95 / 0.90 / 0.93 |
| RF-DETR alert threshold (from training images) | 0.45 | 0.45 |
| OWLv2 all: precision / recall / mAP@50 | 0.79 / 0.75 / 0.77 | 0.79 / 0.75 / 0.77 (identical) |
| Single-frame false alarms on the 11 real clips | 24 in 9 clips | 24 in 9 clips (per clip within 2) |

The current model (with the weapon-free images) was trained once; a retrain should land as close, but its zero false alarms on the clips could become one or two.

## The data and the split

Weapon images: the Simuletic "CCTV ATM Robbery Detection Dataset: Gun & Knife" on Kaggle, about 237 synthetic CCTV-style frames of robberies at ATMs, in 48 scene variants, with YOLO boxes for offender, victim, gun and knife.
Only the gun and knife boxes are used.
The Kaggle page says the images are "100% computer-generated"; it does not say with which tool.

Weapon-free images: 10 "safe" images (people without weapons) from the Simuletic "Surveillance VLM: Weapon & Knife Detection Dataset", the same publisher's synthetic outdoor and corridor CCTV scenes.
They are used only as training negatives, so the model learns what is not a weapon; they share nothing with the real clips used for the false-alarm check.

The split is fixed and recorded in `dataset/manifest.json` (file names, sources, SHA-256, pixel boxes), built once by `wd.tools.build_split` with seed 20261019:

- An ATM image is eligible when its labels hold exactly one weapon box, of the class its scene name says.
  Six images that break this (three knife scenes labelled as gun, two with two knives, one with two guns) are left out rather than guessed.
- Per class, the ATM scene variants are shuffled with the seed: the first 10 give one training image each and the next 10 one test image each.
- So there are 20 training images (10 gun, 10 knife) and 20 test images (10 gun, 10 knife) from 40 different scene variants, and no test scene has any image in training.
- The 10 weapon-free training images are 5 from the surveillance set's safe knife scenes and 5 from its safe firearm scenes (seed + 1), in `dataset/images/negatives/`.

What this split does not cover: all 48 ATM variants are the same kind of picture (a high camera over an ATM, two people, the weapon held at chest height near the middle of the frame, in a handful of ATM settings that recur across variants).
The test therefore checks new scenes in the same style, not new places, camera angles or real footage.
The only real footage in this PoC is the 11 weapon-free clips used for the false-alarm check.

## Models and alert rules

- **RF-DETR Medium, fine-tuned** (Apache-2.0, the same detector family the video search PoC uses for people): COCO-pretrained, fine-tuned for two classes on the 30 training images (60 epochs, batch 4, seed 7).
  With so few images there is no separate validation set; the checkpoint choice only sees training images.
- **OWLv2 base patch16 ensemble, no training** (Apache-2.0): open-vocabulary, prompted with "a photo of a gun / handgun / pistol" (gun) and "a photo of a knife" (knife).

Alert thresholds: one per class, each the best F1 for that class on the training images only (0.05 grid, the middle of tied values), and applied unchanged everywhere else.
For comparison, one shared threshold is chosen the same way.
Nothing is tuned on the test images or on the real clips.

Persistence (videos only): 3 consecutive sampled frames, chosen by reasoning, not on any footage: at 2 frames per second it means the weapon stays in view for 1 to 1.5 seconds, as a brandished weapon does, while most false detections flicker on one or two frames.
A detection continues a streak only when its box is near the box of the frame before (centres within one box length), and nearby boxes in one frame count as one weapon.
Once an alert is open, any sighting of its class within 10 seconds, confirmed or not, keeps it the same event.
Still images keep the single-frame rule.

## Results

### What each change did

Test scores are RF-DETR on the 20 test images; false alarms are on the 11 real weapon-free clips (about 20 minutes).
Each step adds one change to the one before; the counts come from `results/v1/` (first model) and `results/` (current model).

| Step | Test precision / recall / mAP@50 | False alarms on the real clips |
| --- | --- | --- |
| 1. First version: 20 weapon images, one threshold (0.45), alert on a single frame | 0.90 / 0.90 / 0.94 | 24 (in 9 clips) |
| 2. + persistence: 3 frames in a row (videos only) | 0.90 / 0.90 / 0.94 (stills are unaffected) | 7 (in 5 clips) |
| 3. + 10 weapon-free training images (retrained), one threshold (0.45) | 0.95 / 0.90 / 0.95 | 0 (2 in 2 clips with single-frame alerts) |
| 4. + a threshold per class (gun 0.50, knife 0.45): the current version | 0.95 / 0.90 / 0.95 | 0 |

- Persistence and the weapon-free images did the work: both cut false alarms, and together they reach zero on these clips.
  The retrained model also scores slightly higher on the test images (one fewer false detection).
- Per-class thresholds changed nothing here: on the training images both classes score perfectly over a wide range, so the rule lands on 0.50 for gun and 0.45 for knife, and the knife bar did not become stricter.
  They are kept because they cost nothing and let each class be set on its own once there is more data.
- A first draft of the persistence rule split one flickering event into several alerts (one clip went from 1 to 3); it was fixed by letting any sighting keep an open alert going, before these numbers were taken.

### Scores on the 20 test images (current version)

A detection is a hit when its class is right and it overlaps the true box with IoU of at least 0.5.
Precision and recall are at the per-class alert thresholds; AP@50 ranks every detection above a 0.05 floor.

| Detector | Class | Precision | Recall | AP@50 | Hits / false / missed |
| --- | --- | --- | --- | --- | --- |
| RF-DETR fine-tuned (gun 0.50, knife 0.45) | gun | 1.00 | 0.90 | 0.90 | 9 / 0 / 1 |
| | knife | 0.90 | 0.90 | 0.99 | 9 / 1 / 1 |
| | **all** | **0.95** | **0.90** | **mAP 0.95** | 18 / 1 / 2 |
| OWLv2, no training (gun 0.45, knife 0.15) | gun | 1.00 | 1.00 | 1.00 | 10 / 0 / 0 |
| | knife | 0.50 | 0.60 | 0.54 | 6 / 6 / 4 |
| | **all** | **0.73** | **0.80** | **mAP 0.77** | 16 / 6 / 4 |

With one shared threshold (0.25), OWLv2 was at precision 0.79, recall 0.75; its per-class knife threshold of 0.15 finds one more knife but lets in two more false ones.
Speed on the Apple GPU of an M4 Pro: RF-DETR 0.04 s per image, OWLv2 0.24 s per image.
With 10 test images per class, one image is 0.1 of a per-class precision or recall, so differences of 0.1 are within noise.

### Detector chosen for the demo: fine-tuned RF-DETR

It scores higher overall on the same 20 test images (mAP@50 0.95 against 0.77), mostly because OWLv2 misses or mislabels many of the knives, and it is about six times faster, which matters for video.
OWLv2 is better on guns here (10 out of 10), so neither result is a strong claim with this little data.

### False alarms on real CCTV, clip by clip (fine-tuned RF-DETR)

The 11 real clips of the Arabic video search PoC (about 20 minutes in total, from MEVA, CAVIAR, NVIDIA and Pexels sources) contain no weapons, so every alert on them is a false alarm.
They were run exactly as the app runs uploads: 2 frames per second, the training-chosen thresholds, alerts grouped with a 10-second cooldown.
Every column is counted from the same detections, so the comparison is like for like.

| Clip | Length (s) | Frames checked | 1. First version | 2. + persistence | Retrained, single frame | Now |
| --- | --- | --- | --- | --- | --- | --- |
| caviar-leftbag-atchair | 45 | 90 | 2 | 2 | 0 | 0 |
| caviar-leftbag-pickedup | 54 | 109 | 1 | 1 | 1 | 0 |
| meva-g331-waiting-0307 | 300 | 601 | 7 | 1 | 1 | 0 |
| meva-g331-waiting-0315 | 130 | 260 | 2 | 0 | 0 | 0 |
| meva-g420-stairs | 150 | 300 | 0 | 0 | 0 | 0 |
| meva-g421-cafe | 130 | 260 | 4 | 1 | 0 | 0 |
| meva-g638-entrance | 130 | 260 | 4 | 2 | 0 | 0 |
| nvidia-071-c0644 | 120 | 240 | 1 | 0 | 0 | 0 |
| nvidia-071-c0647 | 120 | 240 | 2 | 0 | 0 | 0 |
| pexels-mall-corridor | 12 | 24 | 1 | 0 | 0 | 0 |
| pexels-mall-high-angle | 31 | 63 | 0 | 0 | 0 | 0 |
| **Total** | **1223** | **2447** | **24** | **7** | **2** | **0** |

All of the first version's false alarms were "knife" (hands, phones and carried objects); there was never a false gun.
With the current model, only 3 of the 2447 frames have any detection over the threshold, and none lasts 3 frames.
The full per-clip record is in `results/false_alarms.json` and `results/v1/false_alarms.json`; OWLv2 was not run on the clips.

### Limits

- 30 training and 20 test images, all synthetic: these scores say the approach works on this data, not how it would do on a client's cameras.
- The test scenes are new arrangements of the same ATM settings, so the test is easier than real deployment.
- Zero false alarms on 20 minutes of 11 clips is encouraging, not a guarantee; other cameras and scenes will produce some.
- Real footage was only used to count false alarms; there is no real footage with weapons in this PoC, so real-world recall is unknown.
  Persistence also means a weapon in view for less than about 1 second in a video raises no alert.
- A useful next step would be real or more varied weapon footage and more hard negatives (phones, tools, hands) for training.

## Licences

- **Data: conflicting licence, treat as non-commercial.**
  For both Simuletic image sets, the Kaggle licence field says CC BY-NC-SA 4.0, while the description says CC BY 4.0.
  Under CC BY-NC-SA these images, and the RF-DETR weights fine-tuned on them, may not be used for commercial purposes, and a demo to a commercial client can count as commercial use.
  Until Simuletic confirms CC BY 4.0 in writing, this PoC is cleared for internal, non-commercial demos only.
  Details and the citation are in `ATTRIBUTION.md`.
- **Models:** RF-DETR Medium and OWLv2 are Apache-2.0; no AGPL code (such as Ultralytics YOLO) is used.
- **Real clips:** the 11 clips are read from the video search project, never copied here; their licences are in that project's `footage/starter.json`.
  The one 18-second trim in `demo-samples/` is from MEVA (CC BY 4.0), attributed in `demo-samples/README.md`.

## Layout

```
dataset/        the committed split: 40 weapon images, 10 weapon-free images, manifest.json
demo-samples/   ready-to-use demo files and their README
docs/           DEMO-GUIDE.md: run the demo with the ready-made model, and what to show
results/        evaluation.json and false_alarms.json (current), and v1/ (the first version)
scripts/        setup, train, evaluate, start, stop
src/wd/         boxes, split, detectors, metrics, alert logic, media helpers
src/wd/api/     FastAPI app, SQLite storage, background processing
src/wd/tools/   build_split, download_models, train, evaluate, false_alarms
web/            plain HTML/CSS/JS, no build step; i18n.js holds every string
tests/          unit tests (box conversion, split manifest, alert and persistence logic, metrics, API)
```
