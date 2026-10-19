#!/usr/bin/env bash
# Score both detectors on the 20 test images (results/evaluation.json), then, when a folder of the video search
# PoC's clips is given (argument or WD_CLIPS_DIR), count false alarms on them (results/false_alarms.json).
#   scripts/evaluate.sh [clips folder]
# Get the clips on a new Mac with the Arabic video search project's scripts/fetch_footage.sh;
# they land in that project's data/footage/clips/.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -x .venv/bin/python ]] || { echo "evaluate: run scripts/setup.sh first" >&2; exit 1; }
mkdir -p data/logs
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
CLIPS="${1:-${WD_CLIPS_DIR:-}}"

.venv/bin/python -m wd.tools.evaluate 2>&1 | tee data/logs/evaluate.log
if [[ -n "$CLIPS" ]]; then
  .venv/bin/python -m wd.tools.false_alarms "$CLIPS" 2>&1 | tee data/logs/false_alarms.log
else
  echo "evaluate: no clips folder given, so the false-alarm check was skipped (results/false_alarms.json unchanged)"
fi
