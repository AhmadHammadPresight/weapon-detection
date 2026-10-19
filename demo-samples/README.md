# Demo samples

Ready-to-use files for showing the demo, picked after the final training run.
None of them was used in training: the weapon images come from the test split, and the weapon-free images and the clip were never shown to the model.
Each was checked with the final model; the confidences below are what it gave.

Show them in this order (the demo guide, `docs/DEMO-GUIDE.md`, has the full script).

| File | What it shows | What the app should do |
| --- | --- | --- |
| `handgun_robbery_var12_img1.png` | A robbery at a street ATM, the gun pointed at the customer (synthetic) | One "gun" alert, about 94% |
| `handgun_robbery_var20_img5.png` | A robbery at an indoor ATM with a blue screen (synthetic) | One "gun" alert, about 95% |
| `handgun_robbery_var32_img5.png` | A robbery at an ATM inside a glass lobby (synthetic) | One "gun" alert, about 94% |
| `knife_robbery_var7_img5.png` | A knife held to a customer at a street ATM at night (synthetic) | One "knife" alert, about 95% |
| `knife_robbery_var27_img1.png` | A knife at an indoor ATM with a blue screen (synthetic) | One "knife" alert, about 93% |
| `knife_robbery_var41_img2.png` | A knife pointed at a customer holding a wallet, street ATM (synthetic) | One "knife" alert, about 93% |
| `knifes_safe_Knife_scenario_12.png` | A person walking down an indoor corridor, no weapon (synthetic CCTV) | No alert ("No weapons") |
| `weapons_safe_Scene4_12.png` | A person walking across a car park, no weapon (synthetic CCTV) | No alert ("No weapons") |
| `meva-bus-station-waiting-room.mp4` | 18 seconds of real CCTV in a bus station waiting room: people waiting, one holding a phone, an ATM in the corner; no weapon | No alert ("No weapons"); the highest score in any frame is about 0.28, far below the alert threshold |

The images are 1 to 1.6 MB each and the clip is about 9 MB.

## Sources and licences

- **ATM robbery images** (`handgun_*`, `knife_*`): Simuletic, "CCTV ATM Robbery Detection Dataset: Gun & Knife", https://www.kaggle.com/datasets/simuletic/cctv-atm-robbery-detection-dataset-gun-and-knife.
- **Weapon-free images** (`*_safe_*`): Simuletic, "Surveillance VLM: Weapon & Knife Detection Dataset", https://www.kaggle.com/datasets/simuletic/surveillance-vlm-weapon-and-knife-detection-dataset.
- Both image sets are computer-generated, and their licence is unclear: the Kaggle licence field says CC BY-NC-SA 4.0 (non-commercial) while the descriptions say CC BY 4.0.
  Until the publisher confirms, use them for internal, non-commercial demos only (see `ATTRIBUTION.md`).
- **The clip** (`meva-bus-station-waiting-room.mp4`): "Multiview Extended Video with Activities" (MEVA) dataset by Kitware Inc. and the Intelligence Advanced Research Projects Activity (IARPA), licensed under CC BY 4.0 (http://creativecommons.org/licenses/by/4.0/), https://mevadata.org/.
  It is camera G331 on 2018-03-15, from the source file `2018-03-15.15-50-00.15-55-00.bus.G331.r13.avi`, seconds 55 to 73, trimmed and re-encoded without audio.
