"""Fetch the model weights once, during setup; everything after runs with the hub offline.

- RF-DETR Medium COCO weights (Apache-2.0) into rfdetr's cache (RF_HOME, by default ~/.roboflow/models);
  rfdetr checks their MD5, pinned by the locked rfdetr version.
- OWLv2 base patch16 ensemble (Apache-2.0) at the pinned revision, into the Hugging Face cache.
"""

from __future__ import annotations

from huggingface_hub import snapshot_download

from wd.detectors import OWLV2_MODEL, OWLV2_REVISION, RFDETR_BASE_WEIGHTS


def main() -> None:
    from rfdetr import RFDETRMedium
    from rfdetr.assets.model_weights import get_model_cache_dir

    RFDETRMedium(pretrain_weights=RFDETR_BASE_WEIGHTS, device="cpu")
    print(f"RF-DETR Medium ready: {get_model_cache_dir()}/{RFDETR_BASE_WEIGHTS}")
    path = snapshot_download(
        OWLV2_MODEL, revision=OWLV2_REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt"]
    )
    print(f"OWLv2 ready: {OWLV2_MODEL}@{OWLV2_REVISION[:10]} in {path}")


if __name__ == "__main__":
    main()
