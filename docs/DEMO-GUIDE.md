# Demo guide: run the weapon detection demo without training

This guide gets the demo running on your own Mac with a ready-made model, so you do not need to train anything.
You only type a few commands in Terminal; no AI knowledge is needed.

## What you need

- A Mac with an Apple chip (M1 or later) on a recent macOS (it was tested on macOS 26).
- Homebrew, the Mac package installer (https://brew.sh); the setup script uses it to install FFmpeg and uv if they are missing.
- About 5 GB of free disk space and an internet connection for the first setup.
- The file `wd-demo-data.zip` (about 124 MB), which holds the trained model; Ahmad sends it to you separately.
- The link to this project's repository, also from Ahmad.

## Set it up (once)

1. Open Terminal (press Command + Space, type "Terminal", press Enter).

2. Download the project and go into its folder:

   ```bash
   git clone <repository link> weapon-detection
   cd weapon-detection
   ```

3. Run the setup script:

   ```bash
   scripts/setup.sh
   ```

   It installs the Python environment and downloads some model files.
   It takes a few minutes, longer on a slow connection (about 1 GB of downloads the first time).
   It ends with "setup done".

4. Move `wd-demo-data.zip` into the `weapon-detection` folder, then unpack it from Terminal, inside that folder:

   ```bash
   unzip wd-demo-data.zip
   ```

   Do not unpack it by double-clicking it in Finder.
   Finder puts the files in a new separate folder, and the app will not find the model there.
   The `unzip` command puts the model exactly where the app looks for it: `data/models/rfdetr-weapons/checkpoint_best_total.pth`.
   To check, run `ls data/models/rfdetr-weapons/`; it should list `checkpoint_best_total.pth`.

## Start the demo

```bash
scripts/start.sh
```

When it prints "Open http://127.0.0.1:8490", open that address in your browser.
The page is in Arabic; the button in the top corner switches to English and back.
The green text next to it says the model is ready.

## A short demo script

The images below are in the project folder, under `dataset/images/test/`; pick them with the file button on the page.
They are test images: the model never saw them during training.

1. **A gun.**
   Upload `handgun_robbery_var12_img1.png` with camera ID `ATM-07`.
   An alert appears within a second or two: the frame with a red box around the gun, the label "gun", about 93% confidence, the camera and the time.

2. **A knife.**
   Upload `knife_robbery_var27_img1.png` with camera ID `ATM-03`.
   An alert appears with an orange box around the knife, about 92% confidence.

3. **One real CCTV clip without weapons.**
   Upload a short real CCTV video (MP4 or MOV), with any camera ID.
   The app checks 2 frames per second and shows its progress in the "Uploads" table.
   If the Arabic video search demo is set up on your Mac, its clips are in that project's `data/footage/clips/` folder; `pexels-mall-corridor.mp4` (12 seconds) is a good one.
   A clip with no weapon should ideally say "No weapons", but expect that it may raise a false knife alarm: on `pexels-mall-corridor.mp4` the model marks something on a passer-by as a knife, at 48% confidence, 8 seconds in.
   Use this to show the limits honestly (see below).

4. **The results page.**
   Open "Test results" at the top.
   It shows how the trained model (RF-DETR) and a model used without training (OWLv2) scored on the same 20 test images, the false-alarm check on 11 real clips, and every test image with the true box (dashed green) and what the model found.

To start over, "Clear all" on the alerts page deletes every upload and alert.

## What to say about the results

- The model learned from only 20 images, and all of them are computer-generated (synthetic) ATM robbery scenes.
- On 20 other synthetic test images from new scenes it found 18 of the 20 weapons, with 2 false detections.
  That shows the approach works on this kind of picture, not how it would do on a client's real cameras.
- On about 20 minutes of real CCTV with no weapons in it, it raised 24 false alarms, all of them "knife" (hands, phones and carried objects), and no false guns.
  This is the main limit, and the expected one for so little and such artificial training data.
- The way to improve it is real or more varied weapon footage, plus "hard negatives" (phones, tools, hands) for training.
- The data licence is unclear (the Kaggle page says both "non-commercial" and "commercial use allowed").
  Until the data owner confirms, treat this as an internal demo, not a demo for a commercial client.

## Stop the demo

```bash
scripts/stop.sh
```

## If something goes wrong

- **"the model failed to load ... run scripts/train.sh first":** the model file is not in place; redo step 4 (`unzip wd-demo-data.zip` inside the project folder).
- **The port is busy:** start it on another port, for example `WD_PORT=8495 scripts/start.sh`, and open `http://127.0.0.1:8495`.
- **More detail:** the app writes its log to `data/logs/app.log`.

## Rebuilding the model instead of using the zip

The zip only saves the training step.
To build the model yourself (about 10 to 20 minutes), skip step 4 and run, after setup:

```bash
scripts/train.sh
scripts/evaluate.sh
scripts/start.sh
```

The README has the full steps, the data split and the detailed results.
