#!/usr/bin/env bash
# Start the web app natively in the background on 127.0.0.1 (port 8490 unless WD_PORT says otherwise).
# Log: data/logs/app.log. scripts/stop.sh stops it.
set -euo pipefail
cd "$(dirname "$0")/.."

export WD_DATA_DIR="${WD_DATA_DIR:-$(pwd)/data}"
PORT="${WD_PORT:-8490}"
# Nothing leaves the Mac: models come from the local caches, no hub calls, no telemetry.
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_OFFLINE=1 TQDM_DISABLE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

RUN="$WD_DATA_DIR/run"
LOGS="$WD_DATA_DIR/logs"
mkdir -p "$RUN" "$LOGS"
[[ -x .venv/bin/python ]] || { echo "start: run scripts/setup.sh first" >&2; exit 1; }
[[ -f results/evaluation.json ]] || { echo "start: run scripts/evaluate.sh first (the alert threshold comes from it)" >&2; exit 1; }

if [[ -f "$RUN/app.pid" ]] && kill -0 "$(cat "$RUN/app.pid")" 2>/dev/null; then
  echo "app already running (pid $(cat "$RUN/app.pid")) on http://127.0.0.1:$(cat "$RUN/app.port" 2>/dev/null || echo "$PORT")"
  exit 0
fi
nohup .venv/bin/python -m uvicorn wd.api.app:create_app --factory --host 127.0.0.1 --port "$PORT" >>"$LOGS/app.log" 2>&1 &
echo $! >"$RUN/app.pid"
echo "$PORT" >"$RUN/app.port"

for _ in $(seq 180); do
  if status="$(curl -fsS "http://127.0.0.1:$PORT/api/health" 2>/dev/null)"; then
    case "$status" in
      *'"status":"ready"'*) echo "Open http://127.0.0.1:$PORT"; exit 0 ;;
      *'"status":"error"'*) echo "start: the model failed to load: $status" >&2; exit 1 ;;
    esac
  fi
  kill -0 "$(cat "$RUN/app.pid")" 2>/dev/null || { echo "start: the app exited; see data/logs/app.log" >&2; tail -20 "$LOGS/app.log" >&2; exit 1; }
  sleep 1
done
echo "start: the model is not ready after 180s; see data/logs/app.log" >&2
exit 1
