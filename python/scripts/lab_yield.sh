#!/usr/bin/env bash
# lab_yield.sh — mechanical "yield the machine, then reclaim it" controller.
#
# NO LLM / NO API cost. A detached sibling of lab_heartbeat.sh. It watches how
# much of the box is being used by things OTHER than the flux lab (i.e. Jason's
# other-game trainers). When that external load climbs high and stays there, it
# PAUSES the lab — without losing any work:
#     touch lab/STOP            (heartbeat stops launching / evaluating)
#     kill -STOP <flux venv py> (TRUE suspend of the in-flight evolution — frozen
#                                mid-generation, resumes exactly where it was)
# Then it keeps watching. Once external load has dropped back to baseline for 90
# continuous minutes (the big trainer finished), it RESUMES the lab:
#     kill -CONT <flux venv py> (thaw the evolution)
#     relaunch lab_heartbeat.sh
#
# "external" = sum of %CPU of every process NOT running the flux venv python,
# divided by core count -> a machine-wide %. This is robust even when the box is
# saturated, because it measures the trainer's processes directly rather than
# inferring from aggregate idle. The lab is excluded in BOTH states, so the same
# number works for "is the trainer here" and "has the trainer gone".
#
# Manual override (dead-reliable — use these if you'd rather not trust the auto
# thresholds): touch lab/YIELD to pause now, lab/UNYIELD to resume now.
# Stop the controller itself: touch lab/YIELD_STOP  (leaves the lab as-is).
#
# Start: cd /Users/jason/code/flux && nohup python/scripts/lab_yield.sh >/dev/null 2>&1 & disown
set -u

cd "$(dirname "$0")/../.." || exit 1          # repo root
REPO="$(pwd)"
LAB="$REPO/python/lab"
LOG="$LAB/yield.log"
STATE="$LAB/yield.state"                       # "running" | "paused"
VENV='flux/python/.venv/bin/python'            # flux-lab process signature
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 8)
mkdir -p "$LAB"

PERIOD=${LAB_YIELD_PERIOD:-60}                 # sample cadence (s)
BUSY_PCT=${LAB_YIELD_BUSY:-60}                 # external% that means "a big trainer is here"
IDLE_PCT=${LAB_YIELD_IDLE:-50}                 # external% at/below which the box is "back to baseline"
BUSY_NEED=${LAB_YIELD_BUSY_MIN:-3}             # minutes of sustained busy before pausing
IDLE_NEED=${LAB_YIELD_IDLE_MIN:-90}            # minutes of sustained baseline before resuming

log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

# machine-wide % CPU used by everything that is NOT the flux venv python
external_pct(){
  ps -A -o %cpu=,command= 2>/dev/null | awk -v v="$VENV" -v n="$NCPU" \
    '$0 !~ v {s+=$1} END{ printf "%d", (s/n)+0.5 }'
}
lab_pids(){ pgrep -f "$VENV" 2>/dev/null; }
hb_pids(){ pgrep -f lab_heartbeat.sh 2>/dev/null; }

do_pause(){
  log "PAUSE  external sustained >= ${BUSY_PCT}% -> suspending lab"
  touch "$LAB/STOP"                            # belt: a fresh heartbeat sees this and exits at once
  local p h; p=$(lab_pids); h=$(hb_pids)
  [ -n "$p" ] && kill -STOP $p 2>/dev/null     # freeze the evolution mid-generation
  [ -n "$h" ] && kill -STOP $h 2>/dev/null     # freeze the heartbeat too: stays alive (pgrep finds it,
  echo paused > "$STATE"                       #   so the loop's health check won't relaunch it) but inert
  log "PAUSE  done; suspended $(echo $p | wc -w | tr -d ' ') lab + $(echo $h | wc -w | tr -d ' ') hb procs"
}

do_resume(){
  log "RESUME external <= ${IDLE_PCT}% for ${IDLE_NEED}m -> thawing lab"
  local p h; h=$(hb_pids); [ -n "$h" ] && kill -CONT $h 2>/dev/null   # thaw heartbeat first
  p=$(lab_pids); [ -n "$p" ] && kill -CONT $p 2>/dev/null             # thaw the evolution
  rm -f "$LAB/STOP"                            # clear stop flag so the thawed heartbeat keeps ticking
  hb_pids >/dev/null 2>&1 || \
    ( cd "$REPO" && LAB_HEARTBEAT_INTERVAL=540 nohup python/scripts/lab_heartbeat.sh >/dev/null 2>&1 & disown )
  echo running > "$STATE"
  log "RESUME done; heartbeat $(hb_pids | head -1 || echo '?')"
}

[ -f "$STATE" ] || echo running > "$STATE"
busy_since=""; idle_since=""
log "controller start pid=$$ ncpu=$NCPU pause>=${BUSY_PCT}%/${BUSY_NEED}m resume<=${IDLE_PCT}%/${IDLE_NEED}m"

while true; do
  [ -f "$LAB/YIELD_STOP" ] && { log "YIELD_STOP -> exit"; rm -f "$LAB/YIELD_STOP"; break; }
  st=$(cat "$STATE" 2>/dev/null || echo running)
  now=$(date +%s)

  # manual overrides win immediately
  if [ -f "$LAB/YIELD" ];   then rm -f "$LAB/YIELD";   if [ "$st" = running ]; then log "manual YIELD";   do_pause;  st=paused;  idle_since=""; fi; fi
  if [ -f "$LAB/UNYIELD" ]; then rm -f "$LAB/UNYIELD"; if [ "$st" = paused  ]; then log "manual UNYIELD"; do_resume; st=running; busy_since=""; fi; fi

  ext=$(external_pct); ext=${ext:-0}

  if [ "$st" = running ]; then
    if [ "$ext" -ge "$BUSY_PCT" ]; then
      busy_since="${busy_since:-$now}"
      [ $(( now - busy_since )) -ge $(( BUSY_NEED * 60 )) ] && { do_pause; idle_since=""; busy_since=""; st=paused; }
    else
      busy_since=""
    fi
  else
    # paused: keep STOP asserted and re-suspend anything that slipped through (e.g.
    # a run a relaunched heartbeat tried to start), then track baseline idleness.
    touch "$LAB/STOP"
    p=$(lab_pids); [ -n "$p" ] && kill -STOP $p 2>/dev/null
    h=$(hb_pids);  [ -n "$h" ] && kill -STOP $h 2>/dev/null
    if [ "$ext" -le "$IDLE_PCT" ]; then
      idle_since="${idle_since:-$now}"
      [ $(( now - idle_since )) -ge $(( IDLE_NEED * 60 )) ] && { do_resume; busy_since=""; idle_since=""; st=running; }
    else
      idle_since=""
    fi
  fi

  bs=$([ -n "$busy_since" ] && echo $(( (now-busy_since)/60 ))m || echo -)
  is=$([ -n "$idle_since" ] && echo $(( (now-idle_since)/60 ))m || echo -)
  log "st=$st ext=${ext}% busy=${bs} idle=${is}"
  sleep "$PERIOD"
done
