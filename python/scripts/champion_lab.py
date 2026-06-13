#!/usr/bin/env python
"""Autonomous champion lab — persistent experiment queue + scheduler.

Drives the flux v2 "greatest champion" campaign. Designed to be poked by a
10-minute /loop: each `tick` reaps finished background runs, evaluates their
checkpoints across scales vs a fixed baseline, ranks a leaderboard, and
launches the next queued experiments — keeping <=MAX_EVOLVE evolution runs and
<=MAX_EVAL evals alive at once. All state lives on disk under python/lab/, so a
fresh agent (after a context reset or a new cron firing) reconstructs the whole
campaign from files.

Commands:
  tick              reap, evaluate, launch, then print status   (the heartbeat)
  status            print current state, no changes
  seed              enqueue the initial experiment batch (idempotent-ish)
  add '<json>'      enqueue one experiment spec
  board             print the leaderboard as markdown (for the wiki)

Experiment spec (queue.jsonl line), e.g.:
  {"id":"ms_ring0","ring":0,"label":"multi-scale Ring 0 (transfer)",
   "args":["--radii","7,12,20","--dead-frac","0.22","--generations","150",
           "--pop","12","--boards","18","--workers","6"]}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PYDIR = REPO / "python"
LAB = PYDIR / "lab"
RUNS = LAB / "runs"
PY = sys.executable

# --- campaign constants ------------------------------------------------------
MAX_EVOLVE = 2          # concurrent evolution runs (each uses ~6 workers)
MAX_EVAL = 3            # concurrent eval-suite jobs (each ~1 core)
EVAL_PAIRS = 20         # matched pairs per radius in the leaderboard eval
EVAL_RADII = "7,12,20"  # scales the leaderboard scores across
BASELINE = "lightning_sum_throttled"   # every candidate scored vs this
EVAL_SEED = 20260613    # fixed → all candidates face the same eval boards

Q = LAB / "queue.jsonl"
EVQ = LAB / "eval_queue.jsonl"
RUN = LAB / "running.json"
LB = LAB / "leaderboard.json"
EVENTS = LAB / "events.jsonl"


# --- io helpers --------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _write_jsonl(p: Path, items: list) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(x) + "\n" for x in items))
    tmp.replace(p)


def _append_jsonl(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(obj) + "\n")


def _load_json(p: Path, default):
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return default


def _save_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(p)


def _event(kind: str, **extra) -> None:
    rec = {"ts": _now(), "event": kind, **extra}
    _append_jsonl(EVENTS, rec)
    print(f"  [{kind}] " + " ".join(f"{k}={v}" for k, v in extra.items()))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


# --- launching ---------------------------------------------------------------

def _spawn(cmd: list, logpath: Path) -> int:
    logpath.parent.mkdir(parents=True, exist_ok=True)
    logf = open(logpath, "a")
    logf.write(f"\n# launch {_now()}: {' '.join(cmd)}\n")
    logf.flush()
    proc = subprocess.Popen(
        cmd, cwd=str(PYDIR), stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True,   # detach so it survives this process exiting
    )
    return proc.pid


def _launch_evolve(exp: dict) -> dict:
    ckpt = RUNS / f"{exp['id']}.json"
    log = RUNS / f"{exp['id']}.log"
    seed = int(exp.get("seed", zlib.crc32(exp["id"].encode()) % 100000))
    cmd = [PY, "scripts/evolve_champion.py",
           "--ring", str(exp["ring"]),
           "--out", str(ckpt), "--seed", str(seed),
           *exp.get("args", [])]
    pid = _spawn(cmd, log)
    _event("evolve_start", id=exp["id"], pid=pid, label=exp.get("label", ""))
    return {"type": "evolve", "id": exp["id"], "ring": exp["ring"], "pid": pid,
            "ckpt": str(ckpt), "log": str(log), "label": exp.get("label", ""),
            "args": exp.get("args", []), "started": _now()}


def _launch_eval(item: dict) -> dict:
    ckpt = Path(item["ckpt"])
    log = RUNS / f"{item['id']}.eval.log"
    cmd = [PY, "scripts/evolve_champion.py",
           "--ring", str(item["ring"]), "--resume", str(ckpt),
           "--confirm-pairs", str(EVAL_PAIRS),
           "--eval-suite", "--eval-radii", EVAL_RADII,
           "--eval-opponent", BASELINE, "--seed", str(EVAL_SEED)]
    pid = _spawn(cmd, log)
    _event("eval_start", id=item["id"], pid=pid)
    return {"type": "eval", "id": item["id"], "ring": item["ring"], "pid": pid,
            "ckpt": str(ckpt), "log": str(log), "label": item.get("label", ""),
            "args": item.get("args", []), "started": _now()}


def _ingest_eval(job: dict) -> None:
    """Parse the EVALJSON line from a finished eval log → leaderboard entry."""
    log = Path(job["log"])
    suite = None
    if log.exists():
        for line in log.read_text().splitlines():
            if line.startswith("EVALJSON:"):
                try:
                    suite = json.loads(line[len("EVALJSON:"):])
                except json.JSONDecodeError:
                    suite = None
    if suite is None:
        _event("eval_failed", id=job["id"], reason="no EVALJSON in log")
        return
    lb = _load_json(LB, [])
    lb = [e for e in lb if e["id"] != job["id"]]   # replace prior entry
    genome = _read_genome(job.get("ckpt"))
    lb.append({
        "id": job["id"], "ring": job["ring"], "label": job.get("label", ""),
        "args": job.get("args", []),
        "score": suite["score"], "mean_win": suite["mean_win_rate"],
        "worst_win": suite["worst_win_rate"],
        "per_radius": [{"r": e["radius"], "win": e["win_rate"],
                        "coh": [e["coherent_cand"], e["coherent_opp"]],
                        "p": e["sign_p"]} for e in suite["per_radius"]],
        "genome": genome, "evaluated_at": _now(), "ckpt": job.get("ckpt"),
    })
    lb.sort(key=lambda e: -e["score"])
    _save_json(LB, lb)
    rank = [e["id"] for e in lb].index(job["id"]) + 1
    _event("eval_done", id=job["id"], score=round(suite["score"], 3),
           mean=round(suite["mean_win_rate"], 3),
           worst=round(suite["worst_win_rate"], 3), rank=f"{rank}/{len(lb)}")


def _read_genome(ckpt) -> dict:
    if not ckpt or not Path(ckpt).exists():
        return {}
    blob = _load_json(Path(ckpt), {})
    names = blob.get("names", [])
    vec = blob.get("best_vector", [])
    return {n: round(float(v), 4) for n, v in zip(names, vec)}


# --- tick --------------------------------------------------------------------

def tick() -> None:
    """Concurrency-safe heartbeat — a file lock guards the read-modify-write so
    the mechanical heartbeat and an LLM loop can both poke it without racing."""
    import fcntl
    LAB.mkdir(parents=True, exist_ok=True)
    lockf = open(LAB / ".tick.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another tick in progress; skipping")
        return
    try:
        _tick_body()
    finally:
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()


def _tick_body() -> None:
    print(f"== champion-lab tick {_now()} ==")
    running = _load_json(RUN, [])
    queue = _load_jsonl(Q)
    evq = _load_jsonl(EVQ)

    # 1. reap
    still, reaped = [], []
    for r in running:
        (still if _pid_alive(int(r["pid"])) else reaped).append(r)
    for r in reaped:
        if r["type"] == "evolve":
            if Path(r["ckpt"]).exists():
                evq.append(r)        # queue an eval of its checkpoint
                _event("evolve_done", id=r["id"], action="queued_eval")
            else:
                _event("evolve_crashed", id=r["id"], note="no checkpoint")
        else:  # eval
            _ingest_eval(r)

    # 2. fill eval slots (evals are cheap; drain first so the board stays fresh)
    n_eval = sum(1 for r in still if r["type"] == "eval")
    while n_eval < MAX_EVAL and evq:
        still.append(_launch_eval(evq.pop(0)))
        n_eval += 1

    # 3. fill evolve slots
    n_evolve = sum(1 for r in still if r["type"] == "evolve")
    while n_evolve < MAX_EVOLVE and queue:
        still.append(_launch_evolve(queue.pop(0)))
        n_evolve += 1

    _save_json(RUN, still)
    _write_jsonl(Q, queue)
    _write_jsonl(EVQ, evq)
    status(running=still, queue=queue, evq=evq)


def status(running=None, queue=None, evq=None) -> None:
    running = running if running is not None else _load_json(RUN, [])
    queue = queue if queue is not None else _load_jsonl(Q)
    evq = evq if evq is not None else _load_jsonl(EVQ)
    lb = _load_json(LB, [])
    print(f"\nrunning ({len(running)}):")
    for r in running:
        gen = _last_gen(r) if r["type"] == "evolve" else ""
        print(f"  {r['type']:>6} {r['id']:<22} pid={r['pid']} {gen}")
    print(f"queue: {len(queue)} evolve pending, {len(evq)} eval pending")
    print(f"\nleaderboard (top 8 of {len(lb)}, baseline {BASELINE}=0.50):")
    print(f"  {'#':>2} {'score':>6} {'mean':>5} {'worst':>5}  {'id':<22} per-radius win%")
    for i, e in enumerate(lb[:8], 1):
        pr = " ".join(f"R{p['r']}:{p['win']*100:.0f}" for p in e["per_radius"])
        print(f"  {i:>2} {e['score']:>6.3f} {e['mean_win']*100:>4.0f}% "
              f"{e['worst_win']*100:>4.0f}%  {e['id']:<22} {pr}")


def _last_gen(r: dict) -> str:
    log = Path(r["log"])
    if not log.exists():
        return ""
    last = ""
    for line in log.read_text().splitlines():
        if line.startswith("gen "):
            last = line.split("|")[0].strip()
    return last


# --- board (markdown for the wiki) -------------------------------------------

def board() -> None:
    lb = _load_json(LB, [])
    running = _load_json(RUN, [])
    queue = _load_jsonl(Q)
    evq = _load_jsonl(EVQ)
    print(f"_Updated {_now()} · baseline `{BASELINE}` = 0.50 · "
          f"{len(running)} running, {len(queue)+len(evq)} pending._\n")
    if not lb:
        print("No evaluated candidates yet.\n")
    else:
        print("| # | score | mean | worst | id | per-radius win% | key genome shift |")
        print("|---|---|---|---|---|---|---|")
        for i, e in enumerate(lb[:12], 1):
            pr = " ".join(f"R{p['r']}={p['win']*100:.0f}%" for p in e["per_radius"])
            g = e.get("genome", {})
            shift = _genome_highlight(g, e["ring"])
            print(f"| {i} | {e['score']:.3f} | {e['mean_win']*100:.0f}% | "
                  f"{e['worst_win']*100:.0f}% | `{e['id']}` | {pr} | {shift} |")
    print()
    if running:
        print("**Running:** " + ", ".join(
            f"`{r['id']}` ({_last_gen(r) or r['type']})" for r in running))
    if queue:
        print("\n**Queued:** " + ", ".join(f"`{q['id']}`" for q in queue))


_WIKI_HEADER = """---
title: v2 champion lab — autonomous campaign board
kind: topic
first_seen: 2026-06-13
last_updated: AUTO
status: active
---

## What this is

The live board for the autonomous "greatest champion" campaign
([[v2-beat-the-solver-plan]]). A persistent queue + scheduler
(`python/scripts/champion_lab.py`) keeps <=2 evolution runs and a few eval
jobs alive at once, evaluates every finished checkpoint across scales
(R={radii}) vs the fixed baseline `{baseline}`, and ranks them below. A
10-minute /loop pokes `champion_lab.py tick`, refreshes this page, proposes
new experiments when the queue runs low, and commits + pushes.

**Score** = 0.5·mean-win-rate + 0.5·worst-scale-win-rate (so a champion that
wins at R=7 but loses at R=20 can't top a transfer-robust one). Baseline
self-play = 0.50. A score clearly above 0.50 whose worst-scale win rate is
also above 0.50 is a genuine, transfer-robust improvement over the champion —
the campaign's target. Promotion to a named solver still goes through Todd
(`eval_solvers.py`), never the lab's internal score.

Raw state lives under `python/lab/` (gitignored, durable on disk); this page
is the committed, human-readable projection.

## Live leaderboard
"""


def render_wiki(path: str) -> None:
    p = Path(path)
    head = (_WIKI_HEADER
            .replace("AUTO", _now())
            .replace("{radii}", EVAL_RADII)
            .replace("{baseline}", BASELINE))
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        board()
    body = buf.getvalue()
    p.write_text(head + "\n" + body + "\n"
                 "Related: [[v2-beat-the-solver-plan]], [[v2-grand-research-plan]], "
                 "[[v2-todd-measurement-lab]].\n")
    print(f"wrote {path}")


def _genome_highlight(g: dict, ring: int) -> str:
    if not g:
        return ""
    # champion reference values
    champ0 = {"gamma": 0.85, "weak_bonus": 1.0, "expand_bonus": 0.6,
              "defense_bonus": 0.0, "fanout_eps": 0.05, "throttle": 1.0}
    ref = champ0 if ring == 0 else {}
    bits = []
    for k, v in g.items():
        if k in ref and abs(v - ref[k]) > 0.08:
            bits.append(f"{k}={v:g}")
    return ", ".join(bits[:4])


# --- seed --------------------------------------------------------------------

def seed() -> None:
    """Initial experiment batch. Methodical sweep toward a transfer-robust
    champion. Cheap, high-signal experiments first; the loop adds more."""
    common = ["--pop", "12", "--workers", "6", "--replay-every", "10"]
    batch = [
        {"id": "ms_ring0", "ring": 0,
         "label": "multi-scale Ring 0 (the transfer fix)",
         "args": ["--radii", "7,12,20", "--dead-frac", "0.22",
                  "--generations", "150", "--boards", "18", *common]},
        {"id": "ring0_r20", "ring": 0,
         "label": "Ring 0 tuned purely at R=20 (isolate scale)",
         "args": ["--radii", "20", "--dead-frac", "0.22",
                  "--generations", "120", "--boards", "16", *common]},
        {"id": "ms_ring0_fluid", "ring": 0,
         "label": "multi-scale Ring 0 under fluid edges (EDGE_ALPHA via solver)",
         "args": ["--radii", "7,12,20", "--dead-frac", "0.22", "--edge-alpha",
                  "0.05", "--generations", "150", "--boards", "18", *common]},
        {"id": "ring0_p12", "ring": 0,
         "label": "Ring 0 at R=12 P=12 (multi-enemy / dithering regime)",
         "args": ["--radii", "12", "--dead-frac", "0.22", "--num-players", "12",
                  "--generations", "120", "--boards", "16", *common]},
        {"id": "ms_ring1", "ring": 1,
         "label": "multi-scale Ring 1 (richer policy, where it may have headroom)",
         "args": ["--radii", "12,20", "--dead-frac", "0.22",
                  "--generations", "150", "--boards", "18", *common]},
        {"id": "ring1_r12_noanchor", "ring": 1,
         "label": "Ring 1 at R=12, no anchor (let capacity explore)",
         "args": ["--radii", "12", "--dead-frac", "0.22", "--anchor-coef", "0.0",
                  "--generations", "150", "--boards", "18", *common]},
        # second wave — fill scale gaps, push past the defense-on champion,
        # and probe where the richer Ring 1 policy might finally have headroom.
        {"id": "ring0_r12", "ring": 0,
         "label": "Ring 0 at R=12 (mid-scale baseline)",
         "args": ["--radii", "12", "--dead-frac", "0.22",
                  "--generations", "120", "--boards", "16", *common]},
        {"id": "ms_ring0_vs_evolver0", "ring": 0,
         "label": "multi-scale Ring 0 vs the defense-on evolve_r0 (beat the stronger baseline)",
         "args": ["--radii", "7,12,20", "--dead-frac", "0.22",
                  "--opponent", "evolve_r0",
                  "--generations", "150", "--boards", "18", *common]},
        {"id": "ms_ring0_wide", "ring": 0,
         "label": "multi-scale Ring 0, more boards (lower CRN noise)",
         "args": ["--radii", "7,12,20", "--dead-frac", "0.22",
                  "--generations", "150", "--boards", "27", *common]},
        {"id": "ms_ring0_dead40", "ring": 0,
         "label": "multi-scale Ring 0 on sparse 40%-dead boards",
         "args": ["--radii", "7,12,20", "--dead-frac", "0.40",
                  "--generations", "150", "--boards", "18", *common]},
        {"id": "ms_ring0_p12", "ring": 0,
         "label": "multi-scale Ring 0, P=12 (scale + many seats)",
         "args": ["--radii", "7,12,20", "--dead-frac", "0.22", "--num-players", "12",
                  "--generations", "150", "--boards", "18", *common]},
        {"id": "ring0_r20_p12", "ring": 0,
         "label": "Ring 0 at R=20 P=12 (big multi-enemy)",
         "args": ["--radii", "20", "--dead-frac", "0.22", "--num-players", "12",
                  "--generations", "120", "--boards", "14", *common]},
        {"id": "ms_ring1_anchor", "ring": 1,
         "label": "multi-scale Ring 1 with light anchor (vs the no-anchor twin)",
         "args": ["--radii", "12,20", "--dead-frac", "0.22", "--anchor-coef", "0.02",
                  "--generations", "150", "--boards", "18", *common]},
        {"id": "ms_ring1_p12", "ring": 1,
         "label": "multi-scale Ring 1, P=12 (dithering + scale — Ring 1's best shot)",
         "args": ["--radii", "12,20", "--dead-frac", "0.22", "--num-players", "12",
                  "--generations", "150", "--boards", "18", *common]},
        {"id": "ring1_r20_noanchor", "ring": 1,
         "label": "Ring 1 at R=20, no anchor",
         "args": ["--radii", "20", "--dead-frac", "0.22", "--anchor-coef", "0.0",
                  "--generations", "150", "--boards", "16", *common]},
        {"id": "ms_ring1_fluid", "ring": 1,
         "label": "multi-scale Ring 1 under fluid edges",
         "args": ["--radii", "12,20", "--dead-frac", "0.22", "--edge-alpha", "0.05",
                  "--generations", "150", "--boards", "18", *common]},
    ]
    existing = {e["id"] for e in _load_jsonl(Q)}
    done = {e["id"] for e in _load_json(LB, [])}
    running = {r["id"] for r in _load_json(RUN, [])}
    added = 0
    for exp in batch:
        if exp["id"] in existing or exp["id"] in done or exp["id"] in running:
            continue
        _append_jsonl(Q, exp)
        added += 1
    _event("seed", added=added, total=len(batch))


def add(spec_json: str) -> None:
    exp = json.loads(spec_json)
    assert "id" in exp and "ring" in exp and "args" in exp, "need id/ring/args"
    _append_jsonl(Q, exp)
    _event("enqueue", id=exp["id"], label=exp.get("label", ""))


def promote(args_str: str) -> None:
    """promote <candidate_id> [champion_name] — copy a lab checkpoint into
    checkpoints/evolve/promoted/, where run_v2_solver auto-registers it as a
    solver (usable by Todd AND as an evolution opponent). The loop should only
    call this AFTER a tighter confirm eval clears the bar."""
    import shutil
    parts = args_str.split()
    cid = parts[0]
    name = parts[1] if len(parts) > 1 else cid
    src = RUNS / f"{cid}.json"
    if not src.exists():
        raise SystemExit(f"no checkpoint for '{cid}' at {src}")
    dst = REPO / "python" / "checkpoints" / "evolve" / "promoted" / f"{name}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    lb = _load_json(LB, [])
    for e in lb:
        if e["id"] == cid:
            e["promoted_as"] = name
    _save_json(LB, lb)
    _event("promote", id=cid, name=name, dst=str(dst.relative_to(REPO)))


def prioritize(ids_csv: str) -> None:
    """Move the named experiment ids to the FRONT of the queue (preserving
    their given order), so high-value directions run next. flock-safe."""
    import fcntl
    LAB.mkdir(parents=True, exist_ok=True)
    lockf = open(LAB / ".tick.lock", "w")
    fcntl.flock(lockf, fcntl.LOCK_EX)
    try:
        want = [s for s in ids_csv.split(",") if s.strip()]
        q = _load_jsonl(Q)
        front = [e for w in want for e in q if e["id"] == w]
        rest = [e for e in q if e["id"] not in set(want)]
        _write_jsonl(Q, front + rest)
        _event("prioritize", moved=len(front), order=",".join(e["id"] for e in front))
    finally:
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()


# --- main --------------------------------------------------------------------

def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "tick":
        tick()
    elif cmd == "status":
        status()
    elif cmd == "seed":
        seed()
        status()
    elif cmd == "add":
        add(sys.argv[2])
    elif cmd == "prioritize":
        prioritize(sys.argv[2])
    elif cmd == "promote":
        promote(" ".join(sys.argv[2:]))
    elif cmd == "board":
        board()
    elif cmd == "render-wiki":
        render_wiki(sys.argv[2] if len(sys.argv) > 2
                    else str(REPO / "wiki/topics/v2-champion-lab.md"))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
