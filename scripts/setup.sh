#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac: FFmpeg, uv, the locked Python environment and the pinned model weights.
# Safe to run again; weights already in their caches are only checked.
set -euo pipefail
cd "$(dirname "$0")/.."

die() { echo "setup: $*" >&2; exit 1; }

[[ "$(uname -s)/$(uname -m)" == "Darwin/arm64" ]] || die "this PoC needs an Apple Silicon Mac (Darwin/arm64)"

if ! command -v ffmpeg >/dev/null || ! command -v uv >/dev/null; then
  command -v brew >/dev/null || die "install Homebrew first (https://brew.sh), then run this script again"
  command -v ffmpeg >/dev/null || brew install ffmpeg
  command -v uv >/dev/null || brew install uv
fi

echo "==> Python environment (uv.lock, Python $(cat .python-version))"
uv sync --frozen

echo "==> Model weights (RF-DETR Medium about 400 MB, OWLv2 base about 620 MB, the first time)"
TQDM_DISABLE=1 HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/python -m wd.tools.download_models

echo "setup done. Next: scripts/train.sh, then scripts/evaluate.sh, then scripts/start.sh"
