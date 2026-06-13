#!/usr/bin/env bash
# Mechanical heartbeat for the autonomous champion lab — NO LLM required.
#
# Every ~9 min it: reaps finished evolution runs, evaluates their checkpoints
# across scales, launches the next queued experiments, refreshes the board
# page, and (only when the leaderboard/queue actually changed) commits + pushes
# wiki/. It is the SOLE git driver, so an LLM /loop can enqueue experiments and
# write narrative without ever racing on push.
#
# Detached (start_new_session) so it survives the interactive session ending —
# this is the "your loop will be there when my limit ends" piece.
#
# Start:  nohup setsid python/scripts/lab_heartbeat.sh >/dev/null 2>&1 &
# Stop:   touch python/lab/STOP     (clean stop after current tick)
#         or kill the process.
set -u

cd "$(dirname "$0")/../.." || exit 1          # repo root
REPO="$(pwd)"
PY="$REPO/python/.venv/bin/python"
LAB="$REPO/python/lab"
LOG="$LAB/heartbeat.log"
INTERVAL="${LAB_HEARTBEAT_INTERVAL:-540}"     # seconds between ticks
mkdir -p "$LAB"

echo "# heartbeat start $(date -u +%FT%TZ) pid=$$ interval=${INTERVAL}s" >> "$LOG"

hash_state() {
  "$PY" - <<'PYEOF' 2>/dev/null || echo none
import hashlib, pathlib
b = b""
for n in ("leaderboard.json", "queue.jsonl", "running.json"):
    p = pathlib.Path("python/lab") / n
    if p.exists():
        b += p.read_bytes()
print(hashlib.md5(b).hexdigest() if b else "none")
PYEOF
}

while true; do
  [ -f "$LAB/STOP" ] && { echo "# STOP $(date -u +%FT%TZ)" >> "$LOG"; rm -f "$LAB/STOP"; break; }
  ts=$(date -u +%FT%TZ)
  ( cd "$REPO/python" && "$PY" scripts/champion_lab.py tick ) >> "$LOG" 2>&1

  new=$(hash_state)
  old=$(cat "$LAB/.state_hash" 2>/dev/null || echo init)
  if [ "$new" != "$old" ]; then
    ( cd "$REPO/python" && "$PY" scripts/champion_lab.py render-wiki ) >> "$LOG" 2>&1
    if ! git -C "$REPO" diff --quiet -- wiki/ 2>/dev/null; then
      git -C "$REPO" add wiki/ >> "$LOG" 2>&1
      git -C "$REPO" commit -q -m "champion lab: board + narrative [$ts]

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" >> "$LOG" 2>&1
      git -C "$REPO" pull --rebase --autostash -q origin main >> "$LOG" 2>&1
      git -C "$REPO" push -q origin main >> "$LOG" 2>&1 && \
        echo "[$ts] pushed board update" >> "$LOG"
    fi
    echo "$new" > "$LAB/.state_hash"
  fi
  sleep "$INTERVAL"
done
