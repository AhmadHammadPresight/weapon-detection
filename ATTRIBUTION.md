# Attribution and licences

## Images in `dataset/`

The 40 images in `dataset/images/` are unmodified copies from one Kaggle dataset:

- **Name:** CCTV ATM Robbery Detection Dataset: Gun & Knife
- **Publisher:** Simuletic (https://simuletic.com)
- **URL:** https://www.kaggle.com/datasets/simuletic/cctv-atm-robbery-detection-dataset-gun-and-knife
- **Used here:** 40 of its 237 images, listed with their source file names and SHA-256 in `dataset/manifest.json`.
  The weapon boxes in the manifest are converted from the dataset's YOLO labels (offender and victim labels are not used).

How the images were made, quoted from the Kaggle page:

> "This data is 100% computer-generated. No real individuals were recorded, staged, monitored, or harmed in the creation of this project."

> "Generated entirely using synthetic pipelines. Contains zero real human likenesses or sensitive biometrics"

The page does not say which tool generated them.

### Licence: conflicting, treat as non-commercial

The Kaggle page gives two different licences for the same dataset (checked on 2026-10-19):

- The dataset's licence field on Kaggle says **CC BY-NC-SA 4.0** (Attribution, NonCommercial, ShareAlike).
- The description text on the same page says: "License: CC BY 4.0. You are free to share, adapt, and build upon this data for academic research or commercial production pipelines, provided appropriate attribution credit is given to Simuletic."

Plainly: under the licence field, these images (and a model fine-tuned on them) **may not be used for commercial purposes**, and a demo to a commercial client can count as commercial use.
ShareAlike would also require adapted material to carry the same licence.
Until Simuletic confirms CC BY 4.0 in writing, treat the data as CC BY-NC-SA 4.0: fine for an internal, non-commercial proof of concept, not cleared for a commercial client demo.

Requested citation:

```
@dataset{simuletic_atm_robbery_threat_2026,
  author = {Simuletic Team},
  title = {Simuletic Synthetic ATM Robbery Threat Detection Dataset},
  year = {2026},
  url = {https://simuletic.com}
}
```

### Not used

The Simuletic "Surveillance VLM: Weapon & Knife Detection Dataset" (https://www.kaggle.com/datasets/simuletic/surveillance-vlm-weapon-and-knife-detection-dataset) is not used: no images, labels or results from it are in this repo.

## Real CCTV clips used for the false-alarm check

The false-alarm check reads the 11 public clips of the Arabic video search PoC from that project's folder; they are not copied into this repo.
Their sources and licences are recorded in that project's `footage/starter.json`.

## Models

| Model | Licence | Source |
| --- | --- | --- |
| RF-DETR Medium (COCO-pretrained, fine-tuned here) | Apache-2.0 | https://github.com/roboflow/rf-detr |
| OWLv2 base patch16 ensemble | Apache-2.0 | https://huggingface.co/google/owlv2-base-patch16-ensemble |

No AGPL code or weights (such as Ultralytics YOLO) are used.
The fine-tuned RF-DETR weights are built from the images above, so the data licence note applies to them too.
