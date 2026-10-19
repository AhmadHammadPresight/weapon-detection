# Demo guide: run the weapon detection demo without training

This guide gets the demo running on your own Mac with a ready-made model, so you do not need to train anything.
You only type a few commands in Terminal; no AI knowledge is needed.

## What you need

- A Mac with an Apple chip (M1 or later) on a recent macOS (it was tested on macOS 26).
- Homebrew, the Mac package installer (https://brew.sh); the setup script uses it to install FFmpeg and uv if they are missing.
- About 5 GB of free disk space and an internet connection for the first setup.
- The file `wd-demo-data.zip` (about 124 MB), which holds the trained model; the project team sends it to you separately.
- The link to this project's repository, also from the project team.

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

## The safe demo path

Everything you need is in the `demo-samples` folder of the project; pick the files with the file button on the page.
None of these files was used to train the model.
Show them in this order (`demo-samples/README.md` describes each one):

1. **A gun:** `handgun_robbery_var12_img1.png`, camera ID `ATM-07`.
   Within a second or two an alert appears: the frame with a red box around the gun, "gun", about 94% confidence, the camera and the time.
2. **A knife:** `knife_robbery_var7_img5.png`, camera ID `ATM-03`.
   An alert appears with an orange box around the knife, about 95% confidence.
3. **Nothing to see:** `weapons_safe_Scene4_12.png`, a person crossing a car park, camera ID `CAR-PARK-1`.
   No alert; the "Uploads" table says "No weapons".
4. **Real CCTV, staying quiet:** `meva-bus-station-waiting-room.mp4`, 18 seconds of a real bus station waiting room, camera ID `G331`.
   The app checks 2 frames per second and shows its progress in the "Uploads" table; after a few seconds it says "No weapons".

The other files in `demo-samples` (two more guns, two more knives, one more weapon-free image) work the same way if you have time.
To start over, "Clear all" on the alerts page deletes every upload and alert.

If a false alarm appears (for example on a video of your own), say:
"This is a first version trained on 30 computer-generated images, and a security operator would confirm each alert; more real training footage is the next step."

## What to say about the results

- The model learned from 30 computer-generated images: 20 ATM robbery scenes with a gun or a knife, and 10 scenes of people without weapons.
- On 20 other test images from new scenes it found 18 of the 20 weapons, with one false detection.
  That shows the approach works on this kind of picture, not how it would do on a client's real cameras.
- On about 20 minutes of real CCTV with no weapons in it, the first version raised 24 false alarms, all "knife".
  Two changes brought that to 0: teaching the model with weapon-free images, and alerting on a video only when the weapon stays in view for 3 frames in a row (about 1 to 1.5 seconds).
  Zero on 20 minutes of footage is encouraging, not a guarantee.
- There is no real footage with weapons in this demo, so how well it finds real weapons on real cameras is still unknown; real or more varied weapon footage is the next step.
- The data licence is unclear (the Kaggle pages say both "non-commercial" and "commercial use allowed").
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
To build the model yourself (about 10 to 25 minutes), skip step 4 and run, after setup:

```bash
scripts/train.sh
scripts/evaluate.sh
scripts/start.sh
```

The README has the full steps, the data split and the detailed results.
