"""Gate 0 — quantify the information gap behind the BC/PPO failures.

The beat-the-solver plan diagnoses the 2026-05-16 behavior-cloning failures
as REPRESENTATIONAL: the champion's per-edge decision is a function of a
global Bellman potential field that a 3-hop GNN over local channels cannot
compute, so cloning was doomed before optimization dynamics enter. This probe
tests that claim directly and cheaply, isolated from any training dynamics.

Method (pure numpy, no sklearn, CPU-only):
  1. Run champion (lightning_sum_throttled) self-play; capture decision states.
  2. For every owned cell's valid alive outflow slot, the label is the
     champion's POST-THROTTLE desired-mask bit — the real structural decision.
  3. Build three nested edge-feature sets and fit a logistic probe on each:
       L0  : the exact 9 node channels the failed GNN saw, for source + dest,
             plus the edge's own pressure                       (0-hop)
       L3  : L0 + 1/2/3-hop mean-pooled node channels for src+dst — the SAME
             receptive field the 3-hop GNN had                  (3-hop)
       POT : L3 + the champion's potential field (pot[c], pot[d], gap, rank,
             margin-to-max-friendly)                            (global)
  4. Train/test split BY GAME (held-out games — no leakage). Report AUC, F1,
     balanced accuracy per feature set, overall and on relay edges alone.

Prediction if the diagnosis holds: L0 ≈ L3 (3 hops can't reconstruct the
global field) and POT jumps toward ~1.0. The L3→POT gap is the information
gap, quantified — and the reason Ring 1 hands the network the field instead
of asking it to relearn one.

Usage:
    .venv/bin/python scripts/gate0_probe.py --games 8 --radius 7 \
        --num-players 6 --num-dead-cells 15 --max-ticks 3000 --tick-sample 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from flux_v2 import (
    ACTION_NOOP, DEAD, K, MAX_EDGE, MAX_STRENGTH, NEUTRAL, apply_actions, tick,
)
from flux_v2.solver_vec import (
    _actions_from_desired,
    _gradient_relay_desired,
    _throttle_desired,
    compute_potential,
)
from flux_v2.solver_lightning import lightning_solver_actions
from flux_v2.state import OPPOSITE_SLOT
from scripts.run_v2_solver import _build_initial_state

# Champion hyperparameters (lightning_sum_throttled).
G_GAMMA, G_WEAK, G_EXPAND, G_EPS, G_THROTTLE = 0.85, 1.0, 0.6, 0.05, 1


# ---------------------------------------------------------------------------
# Per-cell node channels — exactly the 9 the failed GNN saw (ppo.build_features)
# ---------------------------------------------------------------------------


def _node_channels(state, seat: int) -> np.ndarray:
    """(N, 9) seat-relative node features matching flux_v2.ppo.build_features."""
    owner = state.owner
    strength = state.strength
    N = state.N
    nb = state.neighbors
    valid = nb >= 0
    nb_safe = np.where(valid, nb, 0)

    is_dead = owner == DEAD
    not_dead = ~is_dead
    is_mine = (owner == seat) & not_dead
    is_neutral = (owner == NEUTRAL) & not_dead
    is_enemy = (owner != seat) & (owner >= 0)

    strength_norm = strength / MAX_STRENGTH
    outflow_count = state.outflow.sum(axis=1) / float(K)

    ep = state.edge_pressure                                    # (N, K)
    opp = OPPOSITE_SLOT
    # inbound pressure on slot k = edge_pressure[neighbor, opposite_slot]
    pressure_in_slot = ep[nb_safe, opp[None, :]] * valid       # (N, K)
    owner_d = owner[nb_safe]
    friendly_d = valid & (owner_d == seat)
    pressure_in_friendly = (pressure_in_slot * friendly_d).sum(axis=1) / MAX_EDGE
    pressure_in_total = pressure_in_slot.sum(axis=1) / MAX_EDGE
    pressure_in_enemy = pressure_in_total - pressure_in_friendly
    pressure_out = (state.outflow * ep).sum(axis=1) / MAX_EDGE * is_mine

    return np.stack([
        strength_norm, is_mine.astype(np.float64), is_enemy.astype(np.float64),
        is_neutral.astype(np.float64), is_dead.astype(np.float64),
        outflow_count, pressure_in_friendly, pressure_in_enemy, pressure_out,
    ], axis=1)


def _pool(feats: np.ndarray, nb_safe: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """One round of neighbor mean-pooling (GCN-style), valid neighbors only."""
    gathered = feats[nb_safe]                                  # (N, K, F)
    m = valid[:, :, None]
    summed = (gathered * m).sum(axis=1)
    count = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    return summed / count


# ---------------------------------------------------------------------------
# Example collection from champion self-play
# ---------------------------------------------------------------------------


def precompute(state, seat, pot=None):
    """All per-state arrays both the collector and the deployed probe need.
    Computing pot once and sharing it guarantees train/deploy features match."""
    owner = state.owner
    nb = state.neighbors
    valid = nb >= 0
    nb_safe = np.where(valid, nb, 0)
    if pot is None:
        pot = compute_potential(state, seat, gamma=G_GAMMA, weak_bonus=G_WEAK,
                                expand_bonus=G_EXPAND, mode="sum")
    base = _node_channels(state, seat)
    p1 = _pool(base, nb_safe, valid)
    p2 = _pool(p1, nb_safe, valid)
    p3 = _pool(p2, nb_safe, valid)
    ep = state.edge_pressure / MAX_EDGE
    potmax = max(float(pot.max()), 1e-9)
    pot_norm = pot / potmax
    owner_d = owner[nb_safe]
    friendly_d = valid & (owner_d == seat)
    has_friend = friendly_d.any(axis=1)
    # max friendly-neighbor pot; cells with no friendly neighbor fall back to
    # their own pot so the margin feature is 0 there (no relay anyway) — avoids
    # a -1e30 sentinel overflowing the normalized margin.
    max_friend = np.where(
        has_friend, np.max(np.where(friendly_d, pot[nb_safe], -np.inf), axis=1), pot,
    )
    pot_d_all = np.where(valid, pot[nb_safe], -1e30)
    rank_norm = (pot_d_all[:, :, None] < pot_d_all[:, None, :]).sum(axis=2) / float(K)
    return dict(owner=owner, valid=valid, nb_safe=nb_safe, pot=pot, base=base,
                p1=p1, p2=p2, p3=p3, ep=ep, pot_norm=pot_norm,
                max_friend=max_friend, rank_norm=rank_norm, potmax=potmax)


def feature_tensors(pre, seat):
    """(L0, L3, POT) each (N, K, dim), plus the valid-alive edge mask (N, K).
    Edge feature order is identical to the per-row construction it replaces."""
    base, nb_safe, valid = pre["base"], pre["nb_safe"], pre["valid"]
    N = base.shape[0]
    alive = valid & (pre["owner"][nb_safe] != DEAD)            # (N, K)

    def src(a):   # (N, F) -> (N, K, F)
        return np.broadcast_to(a[:, None, :], (N, K, a.shape[1]))

    def dst(a):   # (N, F) -> (N, K, F) gathered at each slot's neighbor
        return a[nb_safe]

    ep = pre["ep"][:, :, None]                                 # (N, K, 1)
    L0 = np.concatenate([src(base), dst(base), ep], axis=-1)
    L3 = np.concatenate([
        src(base), dst(base), src(pre["p1"]), dst(pre["p1"]),
        src(pre["p2"]), dst(pre["p2"]), src(pre["p3"]), dst(pre["p3"]), ep,
    ], axis=-1)
    pn = pre["pot_norm"]
    pot_feats = np.stack([
        np.broadcast_to(pn[:, None], (N, K)),
        pn[nb_safe],
        pn[nb_safe] - pn[:, None],
        pre["rank_norm"],
        (pre["max_friend"][:, None] - pre["pot"][nb_safe]) / pre["potmax"],
        (pre["owner"][nb_safe] == seat).astype(np.float64),   # relay-gate flag
    ], axis=-1)
    POT = np.concatenate([L3, pot_feats], axis=-1)
    return L0, L3, POT, alive


def collect(games, radius, num_players, num_dead, max_ticks, ai_period,
            tick_sample, seed, max_examples, per_game_cap):
    rng_master = np.random.default_rng(seed)
    rows_L0, rows_L3, rows_POT, labels, gid, is_relay = [], [], [], [], [], []
    total = 0
    for g in range(games):
        board_seed = int(rng_master.integers(0, 2**31 - 1))
        state, _ = _build_initial_state(
            radius, num_players, num_dead, np.random.default_rng(board_seed),
        )
        ai_count = 0
        game_count = 0
        for t in range(1, max_ticks + 1):
            if t % ai_period == 0:
                sample_now = (ai_count % tick_sample == 0) and (game_count < per_game_cap)
                ai_count += 1
                per_seat = []
                for seat in range(num_players):
                    acts = lightning_solver_actions(
                        state, seat, rng=np.random.default_rng([board_seed, t, seat]),
                        mode="sum", gamma=G_GAMMA, weak_bonus=G_WEAK,
                        expand_bonus=G_EXPAND, fanout_eps=G_EPS, throttle=G_THROTTLE,
                    )
                    per_seat.append(acts)
                    if not sample_now:
                        continue
                    mine = state.owner == seat
                    if not mine.any():
                        continue
                    pot = compute_potential(
                        state, seat, gamma=G_GAMMA, weak_bonus=G_WEAK,
                        expand_bonus=G_EXPAND, mode="sum",
                    )
                    attack, relay = _gradient_relay_desired(state, seat, pot, G_EPS)
                    desired = _throttle_desired(
                        state, attack | relay, attack, pot, G_THROTTLE,
                    )
                    pre = precompute(state, seat, pot=pot)
                    L0t, L3t, POTt, alive = feature_tensors(pre, seat)
                    sel = alive & mine[:, None]                # (N, K)
                    rows_L0.append(L0t[sel])
                    rows_L3.append(L3t[sel])
                    rows_POT.append(POTt[sel])
                    labels.append(desired[sel].astype(np.float64))
                    gid.append(np.full(int(sel.sum()), g))
                    is_relay.append((relay & ~attack)[sel].astype(np.float64))
                    total += int(sel.sum())
                    game_count += int(sel.sum())
                combined = per_seat[0].copy()
                for seat in range(1, num_players):
                    mk = state.owner == seat
                    combined[mk] = per_seat[seat][mk]
                state = apply_actions(state, combined)
            state = tick(state)
            cells = np.bincount(state.owner[state.owner >= 0], minlength=num_players)
            if (cells > 0).sum() <= 1:
                break
            if total >= max_examples:
                break
        if total >= max_examples:
            break
    return (np.concatenate(rows_L0), np.concatenate(rows_L3),
            np.concatenate(rows_POT), np.concatenate(labels),
            np.concatenate(gid), np.concatenate(is_relay))


# ---------------------------------------------------------------------------
# Logistic regression (numpy) + metrics
# ---------------------------------------------------------------------------


def fit_logreg(X, y, l2=1e-3, iters=400, lr=0.5):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    Xs = (X - mu) / sd
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    # class-balanced weights so the sparse positive class isn't ignored
    pos = max(y.sum(), 1.0)
    neg = max((1 - y).sum(), 1.0)
    sw = np.where(y > 0.5, n / (2 * pos), n / (2 * neg))
    for _ in range(iters):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        gz = sw * (p - y)
        gw = Xs.T @ gz / n + l2 * w
        gb = gz.mean()
        w -= lr * gw
        b -= lr * gb
    return (w, b, mu, sd)


def predict(model, X):
    w, b, mu, sd = model
    z = ((X - mu) / sd) @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def auc(y, p):
    # Mann-Whitney U via rank of positives.
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y > 0.5
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def metrics_at(y, p, thr=0.5):
    yhat = (p >= thr).astype(np.float64)
    tp = float(((yhat == 1) & (y == 1)).sum())
    fp = float(((yhat == 1) & (y == 0)).sum())
    fn = float(((yhat == 0) & (y == 1)).sum())
    tn = float(((yhat == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    spec = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (rec + spec)
    return f1, bal_acc


def make_probe_solver(model, kind):
    """Wrap a fitted logistic probe as a solver(state, seat, rng) — threshold
    the per-edge desired probability at 0.5, then serialize through the
    champion's own picker. This is a learned-from-scratch *local* clone."""
    w, b, mu, sd = model

    def _solver(state, seat, rng=None):
        mine = state.owner == seat
        if not mine.any():
            return np.full(state.N, ACTION_NOOP, dtype=np.int32)
        pre = precompute(state, seat)
        L0t, L3t, POTt, alive = feature_tensors(pre, seat)
        X = {"L0": L0t, "L3": L3t, "POT": POTt}[kind]
        N, Kk, dim = X.shape
        z = ((X.reshape(N * Kk, dim) - mu) / sd) @ w + b
        prob = (1.0 / (1.0 + np.exp(-z))).reshape(N, Kk)
        desired = (prob >= 0.5) & alive & mine[:, None]
        return _actions_from_desired(mine, desired, state.outflow.astype(np.bool_), rng)

    return _solver


def deploy_eval(radius, args, pairs):
    """Capstone: do the per-edge probes actually PLAY? Fit L3 and POT clones,
    then play them head-to-head vs the champion on fresh boards. The corrected
    Gate 0 thesis predicts they lose badly despite ~0.93–0.95 per-edge AUC —
    error compounding + the independent-threshold interface break the policy."""
    from scripts.evolve_champion import _get_board, _play
    from scripts.run_v2_solver import _solver_instance

    num_dead = int(round(args.dead_frac * (3 * radius * radius + 3 * radius + 1)))
    max_ticks = max(args.max_ticks, 400 * radius // 7)
    _, L3, POT, y, _, _ = collect(
        args.games, radius, args.num_players, num_dead, max_ticks,
        args.ai_period_ticks, args.tick_sample, args.seed, args.max_examples,
        args.per_game_cap,
    )
    models = {"L3 (3-hop local)": fit_logreg(L3, y),
              "POT (+global field)": fit_logreg(POT, y)}

    cfg = {"radius": radius, "num_players": args.num_players,
           "num_dead_cells": num_dead, "connect_mode": "retry",
           "ai_period_ticks": args.ai_period_ticks, "edge_alpha": 1.0,
           "max_ticks": max_ticks}
    rng = np.random.default_rng(args.seed + 424242)
    seeds = rng.integers(0, 2**31 - 1, size=pairs, dtype=np.int64)
    P = args.num_players
    even, odd = set(range(0, P, 2)), set(range(1, P, 2))

    print(f"\n  deploy probes as solvers vs champion — R={radius}, {pairs} pairs "
          f"({2*pairs} games), fresh boards:")
    for label, model in models.items():
        cand = make_probe_solver(model, "L3" if label.startswith("L3") else "POT")
        wins = losses = timeouts = 0
        for bs in seeds:
            state0 = _get_board(int(bs), cfg)
            for half, seats in ((0, even), (1, odd)):
                opp = _solver_instance("lightning_sum_throttled")
                score, _, decisive, _, _, _ = _play(
                    state0, cand, opp, seats, cfg, game_seed=[int(bs), half],
                )
                if not decisive:
                    timeouts += 1
                elif score >= 1.0:
                    wins += 1
                else:
                    losses += 1
        decided = max(wins + losses, 1)
        print(f"    {label:<22} decisive win rate {wins/decided:>5.1%} "
              f"({wins}W {losses}L {timeouts}T)")
    print("    champion-vs-champion baseline is ~50%. A clone near baseline "
          "would mean per-edge AUC transferred to play; far below means it didn't.")


def probe_one(radius, args):
    """Collect + fit + evaluate at one radius. Returns the metric dict."""
    # Dead count and game length scale with board area / diameter so each
    # radius is a comparable "same rules, bigger board" condition.
    num_dead = int(round(args.dead_frac * (3 * radius * radius + 3 * radius + 1)))
    max_ticks = max(args.max_ticks, 400 * radius // 7)
    L0, L3, POT, y, gid, relay = collect(
        args.games, radius, args.num_players, num_dead,
        max_ticks, args.ai_period_ticks, args.tick_sample, args.seed,
        args.max_examples, args.per_game_cap,
    )
    games = np.unique(gid)
    n_test = max(1, len(games) // 3)
    test_games = set(games[-n_test:].tolist())
    te = np.array([g in test_games for g in gid])
    tr = ~te
    out = {"radius": radius, "n": len(y), "pos": float(y.mean()),
           "games": len(games), "dead": num_dead}
    for key, X in (("L0", L0), ("L3", L3), ("POT", POT)):
        model = fit_logreg(X[tr], y[tr])
        p = predict(model, X[te])
        rmask = relay[te] > 0.5
        sub = rmask | (y[te] < 0.5)
        f1, bal = metrics_at(y[te], p)
        out[key] = {
            "auc": auc(y[te], p), "f1": f1, "bal": bal,
            "relay_auc": auc(y[te][sub], p[sub]) if rmask.sum() else float("nan"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=str, default="5,7,12,20",
                    help="comma-separated board radii to sweep")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--num-players", type=int, default=6)
    ap.add_argument("--dead-frac", type=float, default=0.22,
                    help="dead-cell fraction (scales dead count with area)")
    ap.add_argument("--max-ticks", type=int, default=2500)
    ap.add_argument("--ai-period-ticks", type=int, default=5)
    ap.add_argument("--tick-sample", type=int, default=4,
                    help="keep 1 of every N champion AI decisions")
    ap.add_argument("--per-game-cap", type=int, default=10000,
                    help="max edge examples per game (spreads across games)")
    ap.add_argument("--max-examples", type=int, default=140000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--deploy-pairs", type=int, default=0,
                    help="if >0, also deploy the L3/POT probes as solvers and "
                         "play that many matched pairs vs the champion (capstone)")
    args = ap.parse_args()

    radii = [int(r) for r in args.radii.split(",")]
    print(f"gate0 information-gap probe — champion self-play, P={args.num_players}, "
          f"dead~{args.dead_frac:.0%}, radii {radii}")
    print("predict the champion's post-throttle desired-mask bit per edge; "
          "compare receptive fields.\n")
    print(f"  {'R':>3} {'examples':>9} {'pos%':>5} {'gms':>4}   "
          f"{'L0_AUC':>7} {'L3_AUC':>7} {'POT_AUC':>7}   "
          f"{'gap(L3->POT)':>12}  {'relay L3->POT':>14}")
    print("  " + "-" * 92)
    rows = []
    t0 = time.time()
    for r in radii:
        res = probe_one(r, args)
        rows.append(res)
        gap = res["POT"]["auc"] - res["L3"]["auc"]
        print(f"  {r:>3} {res['n']:>9,} {res['pos']*100:>4.0f}% {res['games']:>4}   "
              f"{res['L0']['auc']:>7.3f} {res['L3']['auc']:>7.3f} "
              f"{res['POT']['auc']:>7.3f}   {gap:>+12.3f}  "
              f"{res['L3']['relay_auc']:>6.3f}->{res['POT']['relay_auc']:.3f}",
              flush=True)
    print()
    gaps = [(res["radius"], res["POT"]["auc"] - res["L3"]["auc"]) for res in rows]
    small, big = gaps[0][1], gaps[-1][1]
    print(f"  L3->POT AUC gap: R={gaps[0][0]} {small:+.3f}  ->  "
          f"R={gaps[-1][0]} {big:+.3f}   (Δgap {big-small:+.3f})")
    if big > small + 0.02 and big > 0.04:
        print("  RECEPTIVE-FIELD CONFIRMED: the information gap GROWS with board")
        print("  size. On big boards the 3-hop field can't reconstruct the global")
        print("  potential, so pot becomes decisive — exactly why Ring 1 hands the")
        print("  network the field rather than asking it to relearn one.")
    elif big > 0.03:
        print("  pot carries real non-redundant signal at all scales; gap is")
        print("  scale-dependent. Per-row relay AUC isolates the pot-critical half.")
    else:
        print("  gap modest across scales — inspect F1/balAcc and the playability")
        print("  argument (0.9 per-edge AUC still compounds into an unplayable clone).")
    if args.deploy_pairs > 0:
        deploy_eval(radii[0], args, args.deploy_pairs)
    print(f"\n  ({time.time()-t0:.0f}s)  see "
          "wiki/topics/v2-beat-the-solver-plan.md — Gate 0")


if __name__ == "__main__":
    main()
