#!/usr/bin/env bash
# Stop the app started by scripts/start.sh. An upload interrupted mid-way is checked again on the next start.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${WD_DATA_DIR:-$(pwd)/data}/run"
pidfile="$RUN/app.pid"
[[ -f "$pidfile" ]] || { echo "app not running"; exit 0; }
pid="$(cat "$pidfile")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
  echo "app stopped"
fi
rm -f "$pidfile" "$RUN/app.port"
