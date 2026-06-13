"""Evolve a flux v2 champion — Rings 0 and 1 of the beat-the-solver plan.

Self-contained (μ, λ) evolution strategy, no external optimizer deps, fully
offline. Fitness is matched-pair performance against a frozen opponent
(default: the reigning champion `lightning_sum_throttled`) on a FIXED set of
deterministic boards — common random numbers, so candidate comparisons sit
far below the 6pp seat-bias noise floor that plagues fresh-board win rates.

Ring 0: the champion's own scalar knobs (gamma, weak/expand/defense bonuses,
        fanout_eps, throttle) — the class trivially contains the champion.
Ring 1: the field-policy genome from flux_v2.ring1 (neural intrinsic +
        readout + per-cell throttle around the frozen Bellman operator) —
        initialized at the champion exactly (Gate 2 parity is tested in
        tests/test_v2_ring1_parity.py).

Per-game score: 1 / 0 for a decisive win / loss, 0.5 for timeout, plus a
small cell-share margin term — both stalemate pathologies (waste-bound and
pressure-shy) read 0% on wins alone, so the margin supplies gradient where
win rate is flat.

Usage (screen world, overnight-friendly):
    .venv/bin/python scripts/evolve_champion.py --ring 0 \
        --generations 200 --pop 12 --boards 12 --workers 8

    .venv/bin/python scripts/evolve_champion.py --ring 1 \
        --generations 300 --pop 16 --boards 12 --workers 8

Watch:  tail -F /tmp/flux-evolve-r{0,1}.log
Resume: --resume python/checkpoints/evolve/<file>.json
Holdout confirm (fresh boards, Wilson CI):
    .venv/bin/python scripts/evolve_champion.py --ring 0 \
        --resume python/checkpoints/evolve/ring0.json --confirm-pairs 40

See wiki/topics/v2-beat-the-solver-plan.md (Ring 0 / Ring 1 / gates).
Promotion to "new champion" goes through Todd's official protocol
(scripts/eval_solvers.py), never on ES internal fitness.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flux_v2 import ACTION_NOOP, apply_actions, tick
from flux_v2.replay import append_index, state_to_frame
from flux_v2.ring1 import (
    GENOME_NAMES as R1_NAMES,
    champion_vector,
    clear_field_caches,
    genome_bounds,
    make_field_solver,
)
from flux_v2.solver_lightning import lightning_solver_actions
from flux_v2.state import copy_state

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_DIR = REPO_ROOT / "python" / "checkpoints" / "evolve"

# ---------------------------------------------------------------------------
# Ring 0 genome
# ---------------------------------------------------------------------------

R0_NAMES = ("gamma", "weak_bonus", "expand_bonus", "defense_bonus",
            "fanout_eps", "throttle")
R0_CHAMPION = np.array([0.85, 1.0, 0.6, 0.0, 0.05, 1.0], dtype=np.float64)
R0_LO = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
R0_HI = np.array([0.985, 3.0, 3.0, 3.0, 0.6, 4.49], dtype=np.float64)


def make_ring0_solver(vec: np.ndarray):
    gamma, weak, expand, defense, eps, thr = [float(x) for x in vec]
    throttle = int(np.clip(round(thr), 1, 6))

    def _solver(state, seat, rng=None):
        return lightning_solver_actions(
            state, seat, rng=rng, mode="sum",
            gamma=gamma, weak_bonus=weak, expand_bonus=expand,
            defense_bonus=defense, fanout_eps=eps, throttle=throttle,
        )

    return _solver


def ring_spec(ring: int):
    """(names, x0, lo, hi, make_solver) per ring."""
    if ring == 0:
        return R0_NAMES, R0_CHAMPION.copy(), R0_LO, R0_HI, make_ring0_solver
    if ring == 1:
        lo, hi = genome_bounds()
        return R1_NAMES, champion_vector(), lo, hi, make_field_solver
    raise ValueError(f"unknown ring {ring}")


def describe_diff(ring: int, vec: np.ndarray, tol: float = 1e-9) -> str:
    names, x0, _, _, _ = ring_spec(ring)
    parts = [
        f"{n}={float(v):.3g}"
        for n, v, c in zip(names, vec, x0)
        if abs(float(v) - float(c)) > tol
    ]
    return " ".join(parts) if parts else "(= champion)"


# ---------------------------------------------------------------------------
# Match runner (CRN matched pairs vs frozen opponent)
# ---------------------------------------------------------------------------

_BOARD_CACHE: dict = {}


def _get_board(seed: int, cfg: dict):
    key = (seed, cfg["radius"], cfg["num_players"], cfg["num_dead_cells"],
           cfg["connect_mode"])
    hit = _BOARD_CACHE.get(key)
    if hit is None:
        from scripts.run_v2_solver import _build_initial_state
        rng = np.random.default_rng(seed)
        state0, _dead = _build_initial_state(
            cfg["radius"], cfg["num_players"], cfg["num_dead_cells"], rng,
            connect_mode=cfg["connect_mode"],
        )
        _BOARD_CACHE[key] = state0
        hit = state0
    return hit


def _cells_of(state, seats: list[int]) -> int:
    return int(np.isin(state.owner, seats).sum())


def _play(state0, cand_fn, opp_fn, cand_seats: set[int], cfg: dict,
          game_seed, record_stride: int = 0):
    """One game: candidate on cand_seats, opponent elsewhere.

    Returns (score, share, decisive, ticks, frames, final_state):
      score 1.0/0.0 decisive win/loss for the candidate, 0.5 timeout;
      share = candidate cell share among player-owned cells at the end.
    """
    clear_field_caches()
    state = copy_state(state0)
    rng = np.random.default_rng(game_seed)
    P = cfg["num_players"]
    ai_period = cfg["ai_period_ticks"]
    edge_alpha = cfg["edge_alpha"]
    frames = [state_to_frame(state)] if record_stride else []
    for t in range(1, cfg["max_ticks"] + 1):
        if t % ai_period == 0:
            combined = np.full(state.N, ACTION_NOOP, dtype=np.int32)
            owner = state.owner
            for seat in range(P):
                fn = cand_fn if seat in cand_seats else opp_fn
                acts = fn(state, seat, rng=rng)
                mask = owner == seat
                combined[mask] = acts[mask]
            state = apply_actions(state, combined)
        state = tick(state, edge_alpha=edge_alpha)
        if record_stride and t % record_stride == 0:
            frames.append(state_to_frame(state))
        cells = np.bincount(state.owner[state.owner >= 0], minlength=P)
        if (cells > 0).sum() <= 1:
            if record_stride:
                frames.append(state_to_frame(state))
            break
    cells = np.bincount(state.owner[state.owner >= 0], minlength=P)
    alive = int((cells > 0).sum())
    cand_cells = int(cells[sorted(cand_seats)].sum())
    total = int(cells.sum())
    share = cand_cells / total if total > 0 else 0.5
    decisive = alive <= 1
    if decisive:
        score = 1.0 if cand_cells > 0 else 0.0
    else:
        score = 0.5
    return score, share, decisive, state.tick, frames, state


def _eval_vector(vec: np.ndarray, ring: int, board_seeds, cfg: dict) -> dict:
    """CRN fitness of one genome over all boards (2 seat-swapped halves each)."""
    _, _, _, _, make_solver = ring_spec(ring)
    cand_fn = make_solver(np.asarray(vec, dtype=np.float64))
    from scripts.run_v2_solver import _solver_instance
    P = cfg["num_players"]
    even = set(range(0, P, 2))
    odd = set(range(1, P, 2))
    scores, shares, decisives = [], [], 0
    for bs in board_seeds:
        state0 = _get_board(int(bs), cfg)
        for half, seats in ((0, even), (1, odd)):
            opp_fn = _solver_instance(cfg["opponent"])
            score, share, decisive, _, _, _ = _play(
                state0, cand_fn, opp_fn, seats, cfg,
                game_seed=[int(bs), half],
            )
            scores.append(score)
            shares.append(share)
            decisives += int(decisive)
    n = len(scores)
    score_mean = float(np.mean(scores))
    margin_mean = float(np.mean(shares)) - 0.5
    fitness = score_mean + cfg["dom_coef"] * margin_mean
    # Champion anchor: penalize drift from the champion in normalized param
    # space. A regularizer against CRN overfitting — keeps the genome near the
    # known-good champion unless the games really support moving (the
    # teacher-lock-in vs deviation tension from the plan). Reported separately
    # so score_mean/margin stay the true game outcomes.
    anchor_pen = 0.0
    anchor_coef = cfg.get("anchor_coef", 0.0)
    if anchor_coef > 0.0:
        names0, x0, lo, hi, _ = ring_spec(ring)
        rng_span = np.maximum(hi - lo, 1e-9)
        anchor_pen = float(np.mean(((np.asarray(vec) - x0) / rng_span) ** 2))
        fitness -= anchor_coef * anchor_pen
    return {
        "fitness": fitness,
        "score_mean": score_mean,
        "margin_mean": margin_mean,
        "anchor_pen": anchor_pen,
        "decisive_frac": decisives / n,
        "games": n,
    }


def _eval_worker(payload: dict) -> dict:
    """Top-level worker for multiprocessing (spawn-safe)."""
    res = _eval_vector(
        np.asarray(payload["vec"], dtype=np.float64),
        payload["ring"], payload["board_seeds"], payload["cfg"],
    )
    res["idx"] = payload["idx"]
    return res


# ---------------------------------------------------------------------------
# Checkpoint / replay / confirm
# ---------------------------------------------------------------------------


def _save_ckpt(path: Path, ring, names, mean, sigma_mult, best_vec, best_fit,
               gen, board_seeds, cfg, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "ring": ring,
        "names": list(names),
        "mean": [float(x) for x in mean],
        "sigma_mult": float(sigma_mult),
        "best_vector": [float(x) for x in best_vec],
        "best_fitness": float(best_fit),
        "generation": int(gen),
        "board_seeds": [int(s) for s in board_seeds],
        "config": cfg,
        "history": history[-500:],
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(blob, indent=1))
    tmp.replace(path)


def _emit_replay(best_vec, ring, board_seeds, cfg, gen):
    from scripts.run_v2_solver import (
        DEFAULT_OUT_DIR, _solver_instance, write_replay,
    )
    _, _, _, _, make_solver = ring_spec(ring)
    cand_fn = make_solver(best_vec)
    opp_fn = _solver_instance(cfg["opponent"])
    P = cfg["num_players"]
    state0 = _get_board(int(board_seeds[0]), cfg)
    stride = 10
    score, share, decisive, ticks, frames, state = _play(
        state0, cand_fn, opp_fn, set(range(0, P, 2)), cfg,
        game_seed=[int(board_seeds[0]), 0], record_stride=stride,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    ea = float(cfg["edge_alpha"])
    alpha_tag = f"_ea{ea:g}".replace(".", "p")
    name = f"evolve_r{ring}_gen{gen:04d}{alpha_tag}_{stamp}.flxr"
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    seats = [
        f"evolve_r{ring}" if s % 2 == 0 else cfg["opponent"] for s in range(P)
    ]
    cells = np.bincount(state.owner[state.owner >= 0], minlength=P)
    metadata = {
        "kind": "evolve_v2",
        "model": f"evolve_r{ring}_gen{gen}",
        "ruleset": "v2-pressure" if ea >= 1.0 else f"v2-fluid-{ea:g}",
        "edge_alpha": ea,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "seats": seats,
        "seed": int(board_seeds[0]),
        "radius": int(cfg["radius"]),
        "num_players": P,
        "num_dead_cells": int(cfg["num_dead_cells"]),
        "ai_period_ticks": int(cfg["ai_period_ticks"]),
        "max_ticks": int(cfg["max_ticks"]),
        "record_stride": stride,
        "winner": int(cells.argmax()) if decisive and cells.max() > 0 else -1,
        "ticks": int(ticks),
        "generation": int(gen),
        "iteration": int(gen),
        "genome": {n: float(v) for n, v in zip(ring_spec(ring)[0], best_vec)},
        "score_vs_opponent": score,
        "cell_share": share,
    }
    path = DEFAULT_OUT_DIR / name
    write_replay(path, frames, cfg["radius"], P, state.N, stride, metadata)
    entry = {k: metadata.get(k) for k in (
        "saved_at", "kind", "model", "ruleset", "edge_alpha", "seats", "seed",
        "winner", "ticks", "generation", "iteration", "radius", "num_players",
        "num_dead_cells",
    )}
    entry["file"] = name
    append_index(DEFAULT_OUT_DIR, entry)
    return name, score


def _wilson(wins: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.5, 0.0, 1.0
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, center - radius, center + radius


def _confirm(best_vec, ring, cfg, pairs: int, seed: int):
    """Holdout matched pairs on FRESH boards (not the CRN training set)."""
    rng = np.random.default_rng(seed + 777_777)
    seeds = rng.integers(0, 2**31 - 1, size=pairs, dtype=np.int64)
    _, _, _, _, make_solver = ring_spec(ring)
    cand_fn = make_solver(best_vec)
    from scripts.run_v2_solver import _solver_instance
    P = cfg["num_players"]
    wins = losses = timeouts = coh_c = coh_o = split = unresolved = 0
    t0 = time.time()
    for i, bs in enumerate(seeds):
        state0 = _get_board(int(bs), cfg)
        halves = []
        for half, seats in ((0, set(range(0, P, 2))), (1, set(range(1, P, 2)))):
            opp_fn = _solver_instance(cfg["opponent"])
            score, _, decisive, _, _, _ = _play(
                state0, cand_fn, opp_fn, seats, cfg, game_seed=[int(bs), half],
            )
            halves.append((score, decisive))
            if not decisive:
                timeouts += 1
            elif score >= 1.0:
                wins += 1
            else:
                losses += 1
        s1, d1 = halves[0]
        s2, d2 = halves[1]
        if not (d1 and d2):
            unresolved += 1
        elif s1 >= 1.0 and s2 >= 1.0:
            coh_c += 1
        elif s1 <= 0.0 and s2 <= 0.0:
            coh_o += 1
        else:
            split += 1
        if (i + 1) % 10 == 0 or i + 1 == pairs:
            print(f"  [{i+1}/{pairs}] cand wins={wins} losses={losses} "
                  f"timeouts={timeouts} | coherent cand={coh_c} opp={coh_o} "
                  f"split={split} unresolved={unresolved} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    decided = wins + losses
    p, lo, hi = _wilson(wins, max(decided, 1))
    print(f"\n== holdout confirm: ring{ring} best vs {cfg['opponent']} ==")
    print(f"  decided games: {decided} | candidate {p:.1%} "
          f"[{lo:.1%}, {hi:.1%}] Wilson 95%")
    nc = coh_c + coh_o
    if nc:
        ps = sum(math.comb(nc, k) for k in range(max(coh_c, coh_o), nc + 1))
        pval = min(2 * ps * (0.5 ** nc), 1.0)
        print(f"  coherent pairs: cand {coh_c} / opp {coh_o} "
              f"(split {split}, unresolved {unresolved}) sign-test p≈{pval:.4f}")
    print("  (promotion still goes through Todd: scripts/eval_solvers.py)")


# ---------------------------------------------------------------------------
# ES loop
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", type=int, choices=(0, 1), default=0)
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--boards", type=int, default=12,
                    help="CRN matched-pair boards per evaluation (2 games each)")
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--num-players", type=int, default=6)
    ap.add_argument("--num-dead-cells", type=int, default=15)
    ap.add_argument("--ai-period-ticks", type=int, default=5)
    ap.add_argument("--max-ticks", type=int, default=3000)
    ap.add_argument("--edge-alpha", type=float, default=1.0)
    ap.add_argument("--connect-mode", choices=("retry", "carve"), default="retry")
    ap.add_argument("--opponent", type=str, default="lightning_sum_throttled")
    ap.add_argument("--dom-coef", type=float, default=0.2,
                    help="weight of the cell-share margin term in fitness")
    ap.add_argument("--anchor-coef", type=float, default=0.0,
                    help="regularizer toward the champion (normalized L2); "
                         "fights CRN overfitting, esp. for ring 1")
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="initial mutation scale as a fraction of each "
                         "param's range (0 = auto: 0.10 ring0, 0.04 ring1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = auto (cpu-2, max 8); 1 = in-process")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--replay-every", type=int, default=10,
                    help="emit a best-vs-opponent replay every N generations "
                         "(0 = off)")
    ap.add_argument("--confirm-pairs", type=int, default=0,
                    help="run a fresh-board holdout eval of the current best "
                         "and exit (use with --resume)")
    args = ap.parse_args()

    names, x0, lo, hi, _ = ring_spec(args.ring)
    cfg = {
        "radius": args.radius, "num_players": args.num_players,
        "num_dead_cells": args.num_dead_cells,
        "ai_period_ticks": args.ai_period_ticks, "max_ticks": args.max_ticks,
        "edge_alpha": args.edge_alpha, "connect_mode": args.connect_mode,
        "opponent": args.opponent, "dom_coef": args.dom_coef,
        "anchor_coef": args.anchor_coef,
    }

    out = Path(args.out) if args.out else DEFAULT_CKPT_DIR / f"ring{args.ring}.json"
    rng = np.random.default_rng(args.seed)
    mean = x0.copy()
    sigma_mult = 1.0
    best_vec = x0.copy()
    best_fit = -np.inf
    start_gen = 0
    history: list[dict] = []
    board_seeds = rng.integers(0, 2**31 - 1, size=args.boards, dtype=np.int64)

    if args.resume:
        blob = json.loads(Path(args.resume).read_text())
        assert blob["ring"] == args.ring, "checkpoint ring mismatch"
        mean = np.array(blob["mean"], dtype=np.float64)
        sigma_mult = float(blob["sigma_mult"])
        best_vec = np.array(blob["best_vector"], dtype=np.float64)
        best_fit = float(blob["best_fitness"])
        start_gen = int(blob["generation"])
        board_seeds = np.array(blob["board_seeds"], dtype=np.int64)
        cfg = blob["config"]
        history = blob.get("history", [])
        out = Path(args.resume)
        print(f"resumed gen {start_gen}, best {best_fit:.4f}: "
              f"{describe_diff(args.ring, best_vec)}")

    if args.confirm_pairs > 0:
        _confirm(best_vec, args.ring, cfg, args.confirm_pairs, args.seed)
        return

    sigma_frac = args.sigma if args.sigma > 0 else (0.10 if args.ring == 0 else 0.04)
    sigma0 = sigma_frac * (hi - lo)
    lam = max(args.pop, 4)
    mu = max(lam // 4, 2)
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()

    workers = args.workers
    if workers <= 0:
        import os
        workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    pool = None
    if workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(workers)

    print(f"evolve ring{args.ring} vs {cfg['opponent']} | pop {lam} mu {mu} "
          f"| boards {len(board_seeds)} (CRN, {2*len(board_seeds)} games/cand) "
          f"| R={cfg['radius']} P={cfg['num_players']} dead={cfg['num_dead_cells']} "
          f"ticks={cfg['max_ticks']} ea={cfg['edge_alpha']:g} | workers {workers}")
    print(f"checkpoint: {out}")

    es_rng = np.random.default_rng(args.seed + 1000 + start_gen)
    try:
        for gen in range(start_gen + 1, args.generations + 1):
            t0 = time.time()
            cands = [mean.copy()]
            for _ in range(lam - 1):
                z = es_rng.standard_normal(len(mean))
                cands.append(np.clip(mean + sigma_mult * sigma0 * z, lo, hi))

            payloads = [
                {"vec": c.tolist(), "ring": args.ring, "idx": i,
                 "board_seeds": [int(s) for s in board_seeds], "cfg": cfg}
                for i, c in enumerate(cands)
            ]
            if pool is not None:
                results = pool.map(_eval_worker, payloads)
            else:
                results = [_eval_worker(p) for p in payloads]
            results.sort(key=lambda r: r["idx"])
            fits = np.array([r["fitness"] for r in results])

            order = np.argsort(-fits)
            gen_best_i = int(order[0])
            gen_best_fit = float(fits[gen_best_i])
            improved = gen_best_fit > best_fit
            if improved:
                best_fit = gen_best_fit
                best_vec = cands[gen_best_i].copy()

            # Rank-weighted recombination over this gen's elites PLUS the
            # best-ever genome (CRN makes its cached fitness comparable) —
            # anchors the mean so one bad generation can't walk it into a
            # losing region of a cliff-y fitness landscape.
            elite_pool = [(float(fits[i]), cands[i]) for i in order[:mu]]
            if np.isfinite(best_fit):
                elite_pool.append((best_fit, best_vec))
            elite_pool.sort(key=lambda p: -p[0])
            new_mean = np.zeros_like(mean)
            for wi, (_, e) in zip(w, elite_pool[:mu]):
                new_mean += wi * e
            mean = np.clip(new_mean, lo, hi)
            sigma_mult = min(max(sigma_mult * (1.05 if improved else 0.97),
                                 0.05), 3.0)

            r0 = results[0]  # unmutated mean — baseline drift check
            gb = results[gen_best_i]
            history.append({
                "gen": gen, "best_fit": gen_best_fit,
                "best_ever": best_fit,
                "mean_fit": float(fits.mean()),
                "mean_self": float(r0["fitness"]),
                "decisive": gb["decisive_frac"],
                "sigma": sigma_mult,
            })
            print(
                f"gen {gen:4d} | best {gen_best_fit:+.4f} "
                f"(ever {best_fit:+.4f}) | pop mean {fits.mean():+.4f} | "
                f"mean-genome {r0['fitness']:+.4f} | "
                f"win% {gb['score_mean']:.0%} dec% {gb['decisive_frac']:.0%} | "
                f"sigma {sigma_mult:.2f} | {time.time()-t0:.1f}s",
                flush=True,
            )
            if improved:
                print(f"      new best: {describe_diff(args.ring, best_vec)}",
                      flush=True)
            _save_ckpt(out, args.ring, names, mean, sigma_mult, best_vec,
                       best_fit, gen, board_seeds, cfg, history)
            if args.replay_every and gen % args.replay_every == 0:
                try:
                    rname, rscore = _emit_replay(
                        best_vec, args.ring, board_seeds, cfg, gen)
                    print(f"      replay → public/v2/replays/{rname} "
                          f"(score {rscore:g})", flush=True)
                except Exception as e:  # replay is decorative; never kill the run
                    print(f"      replay emit failed: {e}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted — checkpoint saved at last completed generation")
    finally:
        if pool is not None:
            pool.terminate()

    print(f"\nbest ever {best_fit:+.4f}: {describe_diff(args.ring, best_vec)}")
    print(f"checkpoint: {out}")
    print("next: holdout confirm with "
          f"--resume {out} --confirm-pairs 40, then Todd's eval_solvers.py")


if __name__ == "__main__":
    main()
