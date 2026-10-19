#!/usr/bin/env bash
# Fine-tune RF-DETR Medium on the 20 committed training images (seeded) into data/models/rfdetr-weapons/.
# About 45 minutes on the Apple GPU of an M4 Pro. Log: data/logs/train.log.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -x .venv/bin/python ]] || { echo "train: run scripts/setup.sh first" >&2; exit 1; }
mkdir -p data/logs
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
# Run the rare PyTorch op that has no Apple GPU kernel on the CPU instead of failing.
export PYTORCH_ENABLE_MPS_FALLBACK=1
.venv/bin/python -m wd.tools.train "$@" 2>&1 | tee data/logs/train.log
