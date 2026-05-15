"""Vectorized v2 solver families.

One file, all modes, no per-cell Python loops. Replaces the per-cell loops
that used to live in solver_lightning.py and solver.py.

Public entry points:
  bfs_actions(state, seat, rng) -> (N,) int32
  lightning_actions(state, seat, rng, *, mode=..., **kwargs) -> (N,) int32

Each mode reduces to: compute a (N, K) bool `desired` mask, hand off to
`_actions_from_desired` for the SET-missing > CLEAR-stale > NOOP picker.

Tie-break: when multiple slots qualify (e.g. several missing attack slots),
the picker rotates the slot search order by a per-cell offset. With an RNG,
that offset is random per cell; without one, it's zero (slot-0 bias).

The behavioral differences from the per-cell-loop version are intentional
and documented in the wiki; the relevant ones are:
  * RNG draw schedule changes (replays from older runs don't bit-replay).
  * The relay rule selects "all friendly slots whose pot is within
    `fanout_eps` of the cell's max friendly pot AND strictly above
    pot[c]". The old code was path-dependent through an incremental
    running-best variable; the new behavior is cleaner.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit

from .state import (
    ACTION_CLEAR_BASE,
    ACTION_NOOP,
    ACTION_SET_BASE,
    DEAD,
    K,
    MAX_EDGE,
    MAX_STRENGTH,
    NEUTRAL,
    OPPOSITE_SLOT,
    State,
)


# ---------------------------------------------------------------------------
# Shared geometry helpers (vectorized).
# ---------------------------------------------------------------------------


def _neighbor_owner_grid(state: State) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nb_safe, nb_valid, nb_owner): (N, K) shapes.
    nb_owner[c, k] = owner[nb[c, k]] (or NEUTRAL where invalid)."""
    nb = state.neighbors
    nb_safe = np.maximum(nb, 0)
    nb_valid = nb >= 0
    nb_owner = np.where(nb_valid, state.owner[nb_safe], NEUTRAL)
    return nb_safe, nb_valid, nb_owner


def _inbound_enemy_pressure(state: State, seat: int) -> np.ndarray:
    """Vectorized (N,) sum of inbound edge_pressure from enemy seats."""
    N = state.N
    nb_safe, nb_valid, nb_owner = _neighbor_owner_grid(state)
    ep = state.edge_pressure                                # (N, K)
    # ep[d, opp_k] for d = nb[c, k]
    opp = OPPOSITE_SLOT                                     # (K,)
    pressure_grid = ep[nb_safe, opp[None, :]]               # (N, K)
    is_enemy_slot = nb_valid & (nb_owner != seat) & (nb_owner >= 0)
    return (pressure_grid * is_enemy_slot.astype(np.float32)).sum(axis=1)


# ---------------------------------------------------------------------------
# Potential-field computation (already vectorized — kept here so the new
# solver doesn't import the old module).
# ---------------------------------------------------------------------------


def compute_potential_live(
    state: State,
    seat: int,
    gamma: float = 0.85,
    weak_bonus: float = 1.0,
    expand_bonus: float = 0.6,
    defense_bonus: float = 0.0,
    flow_weight: float = 1.0,
    strength_weight: float = 0.5,
) -> np.ndarray:
    """Live-field potential proxy. Skips iterated value-iteration.

    Returns (N,) float32. Four additive parts:

      * intrinsic — same source field as `compute_potential` (high on
        weak enemies / neutrals).
      * one-hop neighbor average of intrinsic, discounted by `gamma`.
        Gives frontline-adjacent friendlies some pull.
      * strength_signal — friendly cells get `strength_weight * (1 -
        strength/MAX_STRENGTH)`. Sets the *rear → front* gradient
        from cell state alone, no iteration needed. Full backline
        cells have low pot, drained frontline cells have high pot,
        so relay always points toward where pressure is needed.
      * flow_signal — normalized total of inbound + outbound
        edge_pressure at each cell. Encodes "this cell is on an
        active pressure pathway." Under fluid physics
        (`EDGE_ALPHA < 1`) this field has been propagating for many
        ticks, so it captures multi-hop information without iter.

    Strategically useful only when edge_pressure carries time-
    integrated signal (i.e., when `EDGE_ALPHA < 1`) and the
    strength gradient bootstraps the relay direction. Under
    snap-to-target rules the live field is just this-tick's spill
    share and the strength gradient may be misaligned (full backline
    cells under snap still need to relay forward even though they
    have low pot here) — pick `sum` / `sum_long` for the original
    rules.
    """
    N = state.N
    owner = state.owner
    strength = state.strength
    nb_safe, nb_valid, _ = _neighbor_owner_grid(state)

    is_enemy = (owner >= 0) & (owner != seat)
    is_neutral = owner == NEUTRAL
    is_mine = owner == seat
    is_dead = owner == DEAD

    intrinsic = np.zeros(N, dtype=np.float32)
    if is_enemy.any():
        intrinsic[is_enemy] = (
            weak_bonus * (1.0 - strength[is_enemy] / MAX_STRENGTH)
        ).clip(0.0, weak_bonus)
    intrinsic[is_neutral] = expand_bonus
    if defense_bonus > 0.0 and is_mine.any():
        inbound = _inbound_enemy_pressure(state, seat)
        threat = (inbound / MAX_EDGE).clip(0.0, 1.0)
        intrinsic[is_mine] = defense_bonus * threat[is_mine]
    intrinsic[is_dead] = 0.0

    nbr_dead = (owner[nb_safe] == DEAD) & nb_valid
    nbr_ok = nb_valid & ~nbr_dead
    nbr_ok_f = nbr_ok.astype(np.float32)

    # One-hop neighbor average of intrinsic.
    deg_self = nbr_ok_f.sum(axis=1)
    safe_deg = np.where(deg_self > 0, deg_self, 1.0)
    one_hop = (intrinsic[nb_safe] * nbr_ok_f).sum(axis=1) / safe_deg

    # Friendly strength gradient: empty cells > full cells. Pulls relay
    # from rear to front without needing the pressure field to have
    # propagated yet (bootstraps from game start).
    strength_signal = np.where(
        is_mine,
        (1.0 - strength / MAX_STRENGTH).clip(0.0, 1.0),
        0.0,
    ).astype(np.float32)

    # Flow signal from live edge_pressure: inbound + outbound, normalized.
    ep = state.edge_pressure
    opp = OPPOSITE_SLOT[None, :]
    inbound_grid = ep[nb_safe, opp] * nbr_ok_f
    inbound_total = inbound_grid.sum(axis=1)
    outbound_total = (ep * nbr_ok_f).sum(axis=1)
    norm = max(2.0 * K * MAX_EDGE, 1.0)
    flow_signal = (inbound_total + outbound_total) / norm

    pot = (
        intrinsic
        + gamma * one_hop
        + strength_weight * strength_signal
        + flow_weight * flow_signal
    )
    pot = pot.astype(np.float32)
    pot[is_dead] = 0.0
    return pot


_OPPOSITE_SLOT_ARR_VEC = np.asarray(OPPOSITE_SLOT, dtype=np.int32)


@njit(cache=True, fastmath=True)
def _compute_potential_core(
    owner: np.ndarray,
    intrinsic: np.ndarray,
    neighbors: np.ndarray,
    edge_pressure: np.ndarray,
    opposite_slot: np.ndarray,
    gamma: float,
    max_iter: int,
    tol: float,
    mode_id: int,
    dead_id: int,
) -> np.ndarray:
    """JIT'd value iteration for compute_potential.

    mode_id: 0=max, 1=sum, 2=sum_pw.
    Returns (N,) float32 pot field. Same algorithm as the numpy version,
    just per-cell explicit loops so Numba can vectorize/optimize.
    """
    N = owner.shape[0]
    K = neighbors.shape[1]
    pot = intrinsic.copy()
    new_pot = np.empty_like(pot)

    # Precompute per-cell degree (count of non-DEAD non-off-grid neighbors).
    deg = np.zeros(N, dtype=np.int32)
    for c in range(N):
        n = 0
        for k in range(K):
            d = neighbors[c, k]
            if d < 0:
                continue
            if owner[d] == dead_id:
                continue
            n += 1
        deg[c] = n

    for _ in range(max_iter):
        max_diff = 0.0
        if mode_id == 0:  # max
            for c in range(N):
                if owner[c] == dead_id:
                    new_pot[c] = 0.0
                    continue
                max_nbr = 0.0
                for k in range(K):
                    d = neighbors[c, k]
                    if d < 0:
                        continue
                    if owner[d] == dead_id:
                        continue
                    if pot[d] > max_nbr:
                        max_nbr = pot[d]
                v = intrinsic[c]
                gm = gamma * max_nbr
                if gm > v:
                    v = gm
                d_abs = v - pot[c]
                if d_abs < 0.0:
                    d_abs = -d_abs
                if d_abs > max_diff:
                    max_diff = d_abs
                new_pot[c] = v
        elif mode_id == 1:  # sum (uniform)
            for c in range(N):
                if owner[c] == dead_id:
                    new_pot[c] = 0.0
                    continue
                contrib = 0.0
                for k in range(K):
                    d = neighbors[c, k]
                    if d < 0:
                        continue
                    dd = deg[d]
                    if dd == 0:
                        continue
                    if owner[d] == dead_id:
                        continue
                    contrib += pot[d] / dd
                v = intrinsic[c] + gamma * contrib
                d_abs = v - pot[c]
                if d_abs < 0.0:
                    d_abs = -d_abs
                if d_abs > max_diff:
                    max_diff = d_abs
                new_pot[c] = v
        else:  # 2: sum_pw (pressure-weighted)
            for c in range(N):
                if owner[c] == dead_id:
                    new_pot[c] = 0.0
                    continue
                contrib = 0.0
                for k in range(K):
                    d = neighbors[c, k]
                    if d < 0:
                        continue
                    if owner[d] == dead_id:
                        continue
                    # weight = ep[d, opp(k)] / sum_k' ep[d, k']  (or 1/deg(d) fallback)
                    opp_k = opposite_slot[k]
                    ep_dc = edge_pressure[d, opp_k]
                    total = 0.0
                    for kk in range(K):
                        total += edge_pressure[d, kk]
                    if total > 1e-9:
                        w = ep_dc / total
                    else:
                        dd = deg[d]
                        w = 1.0 / dd if dd > 0 else 0.0
                    contrib += pot[d] * w
                v = intrinsic[c] + gamma * contrib
                d_abs = v - pot[c]
                if d_abs < 0.0:
                    d_abs = -d_abs
                if d_abs > max_diff:
                    max_diff = d_abs
                new_pot[c] = v

        # swap pot ↔ new_pot
        tmp = pot
        pot = new_pot
        new_pot = tmp
        if max_diff < tol:
            break
    return pot


def compute_potential(
    state: State,
    seat: int,
    gamma: float = 0.85,
    weak_bonus: float = 1.0,
    expand_bonus: float = 0.6,
    defense_bonus: float = 0.0,
    max_iter: int = 32,
    tol: float = 1e-4,
    mode: str = "max",
) -> np.ndarray:
    """Return (N,) float32 potential field for `seat`. Modes:
    'max': pot[c] = max(intrinsic[c], gamma * max neighbor pot)
    'sum': pot[c] = intrinsic[c] + gamma * uniform sum of neighbor pots
    'sum_pw': sum but weighted by current edge pressure ratios
    """
    N = state.N
    owner = state.owner
    strength = state.strength

    is_enemy = (owner >= 0) & (owner != seat)
    is_neutral = owner == NEUTRAL
    is_mine = owner == seat
    is_dead = owner == DEAD

    intrinsic = np.zeros(N, dtype=np.float32)
    if is_enemy.any():
        intrinsic[is_enemy] = (
            weak_bonus * (1.0 - strength[is_enemy] / MAX_STRENGTH)
        ).clip(0.0, weak_bonus)
    intrinsic[is_neutral] = expand_bonus
    if defense_bonus > 0.0 and is_mine.any():
        inbound = _inbound_enemy_pressure(state, seat)
        threat = (inbound / MAX_EDGE).clip(0.0, 1.0)
        intrinsic[is_mine] = defense_bonus * threat[is_mine]
    intrinsic[is_dead] = 0.0

    mode_id = {"max": 0, "sum": 1, "sum_pw": 2}.get(mode, -1)
    if mode_id < 0:
        raise ValueError(f"unknown mode {mode!r}")

    pot = _compute_potential_core(
        np.ascontiguousarray(owner, dtype=np.int32),
        np.ascontiguousarray(intrinsic, dtype=np.float32),
        np.ascontiguousarray(state.neighbors, dtype=np.int32),
        np.ascontiguousarray(state.edge_pressure, dtype=np.float32),
        _OPPOSITE_SLOT_ARR_VEC,
        float(gamma),
        int(max_iter),
        float(tol),
        int(mode_id),
        int(DEAD),
    )
    return pot.astype(np.float32, copy=False)




# ---------------------------------------------------------------------------
# Action picker — the heart of the vectorization.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _picker_core(
    is_mine: np.ndarray,
    desired: np.ndarray,
    cur: np.ndarray,
    offsets: np.ndarray,
    action_set_base: int,
    action_clear_base: int,
    action_noop: int,
) -> np.ndarray:
    """Explicit-loop picker. For each owned cell c, walk slots starting at
    `offsets[c]` (mod K) and emit the first SET-missing slot (preferred) or
    CLEAR-stale slot (fallback). NOOP if neither exists.
    """
    N = is_mine.shape[0]
    Kn = desired.shape[1]
    actions = np.full(N, action_noop, dtype=np.int32)
    for c in range(N):
        if not is_mine[c]:
            continue
        offset = offsets[c]
        # First pass: any SET-missing slot? Walk in rotated order.
        chosen = -1
        for j in range(Kn):
            k = offset + j
            if k >= Kn:
                k -= Kn
            if desired[c, k] and not cur[c, k]:
                chosen = k
                break
        if chosen >= 0:
            actions[c] = action_set_base + chosen
            continue
        for j in range(Kn):
            k = offset + j
            if k >= Kn:
                k -= Kn
            if cur[c, k] and not desired[c, k]:
                chosen = k
                break
        if chosen >= 0:
            actions[c] = action_clear_base + chosen
    return actions


def _actions_from_desired(
    is_mine: np.ndarray,
    desired: np.ndarray,
    cur: np.ndarray,
    rng: Optional[np.random.Generator],
) -> np.ndarray:
    """Build (N,) int32 action array from a per-cell (N, K) desired mask.

    For each owned cell:
      1. If any slot is "missing" (desired & not current) → SET it.
      2. Else if any slot is "stale" (current & not desired) → CLEAR it.
      3. Else NOOP.

    Tie-breaking: a per-cell rotation offset rotates the slot search order
    so we don't always pick the lowest slot index.
    """
    N = is_mine.shape[0]
    if rng is not None:
        offsets = rng.integers(0, K, size=N).astype(np.int32)
    else:
        offsets = np.zeros(N, dtype=np.int32)
    actions = _picker_core(
        np.ascontiguousarray(is_mine, dtype=np.bool_),
        np.ascontiguousarray(desired, dtype=np.bool_),
        np.ascontiguousarray(cur, dtype=np.bool_),
        offsets,
        int(ACTION_SET_BASE),
        int(ACTION_CLEAR_BASE),
        int(ACTION_NOOP),
    )
    return actions


# ---------------------------------------------------------------------------
# Per-cell "neighbor classification" helpers.
# ---------------------------------------------------------------------------


def _classify_slots(
    state: State, seat: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (nb_safe, nb_valid, nb_owner, is_friendly, is_attack):
      is_friendly[c, k]: slot k of c points at a friendly non-dead non-self cell.
      is_attack[c, k]:  slot k of c points at a non-friendly non-dead cell.
    All shapes are (N, K) bool except nb_owner which is (N, K) int.
    """
    nb_safe, nb_valid, nb_owner = _neighbor_owner_grid(state)
    is_friendly = nb_valid & (nb_owner == seat)
    is_attack = nb_valid & (nb_owner != seat) & (nb_owner != DEAD)
    return nb_safe, nb_valid, nb_owner, is_friendly, is_attack


def _relay_mask(
    pot: np.ndarray, nb_safe: np.ndarray, is_friendly: np.ndarray,
    fanout_eps: float,
) -> np.ndarray:
    """Friendly slots whose pot is within `fanout_eps` of the cell's max
    friendly-slot pot AND strictly greater than pot[c]."""
    N, _ = is_friendly.shape
    NEG_INF = np.float32(-1e30)
    pot_at_slot = np.where(is_friendly, pot[nb_safe], NEG_INF)        # (N, K)
    max_per_cell = pot_at_slot.max(axis=1)                            # (N,)
    has_any_friendly = is_friendly.any(axis=1)                        # (N,)
    near_max = (max_per_cell[:, None] - pot_at_slot) <= fanout_eps    # (N, K)
    above_self = pot_at_slot > pot[:, None]                           # (N, K)
    return has_any_friendly[:, None] & near_max & above_self & is_friendly


# ---------------------------------------------------------------------------
# BFS solver (replaces solver.solver_actions per-cell loop).
# ---------------------------------------------------------------------------


def _frontier_distance(state: State, seat: int) -> np.ndarray:
    """Vectorized iterative BFS over owned cells from the frontier.
    Distance through owned cells to the nearest non-friendly non-dead cell.

    Implementation: Bellman-Ford-style relaxation over (N, K). Iterates at
    most diameter of the friendly subgraph. For our board sizes (≤4000)
    this is cheaper than queue management.
    """
    N = state.N
    owner = state.owner
    nb_safe, nb_valid, nb_owner = _neighbor_owner_grid(state)
    is_mine = owner == seat

    BIG = np.int32(10_000)
    dist = np.full(N, BIG, dtype=np.int32)

    if not is_mine.any():
        return dist

    # Seed: friendly cells with at least one non-friendly non-dead neighbor.
    is_frontier_target = nb_valid & (nb_owner != seat) & (nb_owner != DEAD)
    seed = is_mine & is_frontier_target.any(axis=1)
    dist[seed] = 0

    # Iterate. Max iterations = friendly subgraph diameter; cap at N.
    for _ in range(N):
        prev = dist
        # For each non-frontier owned cell c: dist[c] = min over friendly nbrs of dist[d]+1
        # Build (N, K) candidate distances from friendly neighbors.
        nbr_dist = np.where(
            is_mine[nb_safe] & nb_valid,
            dist[nb_safe].astype(np.int32),
            BIG,
        )
        cand = nbr_dist.min(axis=1) + 1
        cand = np.minimum(cand, BIG)
        new_dist = np.minimum(prev, cand)
        new_dist = np.where(is_mine, new_dist, BIG)
        new_dist[seed] = 0
        if np.array_equal(new_dist, prev):
            break
        dist = new_dist
    return dist


def bfs_actions(
    state: State, seat: int, rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Vectorized BFS solver.

    For each owned cell:
      attack: slot points at enemy/neutral (non-dead).
      relay:  slot points at friendly cell with smaller frontier distance,
              EXCLUDING dead-end MAX-sink friendlies (MAX strength + 0 outflows).
      priority: SET missing attack > SET missing relay > CLEAR stale.
    """
    N = state.N
    owner = state.owner
    outflow = state.outflow

    is_mine = owner == seat
    if not is_mine.any():
        return np.full(N, ACTION_NOOP, dtype=np.int32)

    nb_safe, nb_valid, nb_owner, is_friendly, is_attack = _classify_slots(state, seat)
    dist = _frontier_distance(state, seat)

    # Friendly neighbor closer to frontier?
    BIG = np.int32(10_000)
    nbr_dist = np.where(is_friendly, dist[nb_safe], BIG)
    is_closer = nbr_dist < dist[:, None]                      # (N, K)

    # Dead-end MAX sink filter on the friendly neighbor.
    num_active = outflow.sum(axis=1).astype(np.int32)
    nbr_is_dead_end = (
        is_friendly
        & (state.strength[nb_safe] >= MAX_STRENGTH)
        & (num_active[nb_safe] == 0)
    )
    relay = is_friendly & is_closer & ~nbr_is_dead_end

    # Phase 1: missing attack. Phase 2: missing relay. Then clear stale.
    # We collapse the two SET phases by building a desired = attack | relay
    # mask, with attack treated as higher priority via a tweak: if a cell has
    # any missing attack slot we ignore relay slots for the set step.
    desired = is_attack | relay
    cur = outflow.astype(np.bool_)

    needs_set_attack = is_attack & ~cur
    has_attack_missing = needs_set_attack.any(axis=1)

    # When attack-missing, drop relay from desired so the picker chooses an
    # attack slot for the SET. When no attack missing, picker handles relay
    # normally.
    desired_for_set = np.where(has_attack_missing[:, None], is_attack, desired)
    # For the clear phase we still want to clear toward attack|relay.
    # Use a two-pass: compute SET first, then CLEAR against `desired`.
    set_mask = desired_for_set & ~cur
    clear_mask = cur & ~desired
    return _picker_two_pass(is_mine, set_mask, clear_mask, rng)


def _picker_two_pass(
    is_mine: np.ndarray,
    set_mask: np.ndarray,
    clear_mask: np.ndarray,
    rng: Optional[np.random.Generator],
) -> np.ndarray:
    """Same as _actions_from_desired but accepts the two phase masks
    directly (caller has already imposed any priority ordering).
    Delegates to the JIT'd _picker_core; set_mask and clear_mask are
    the pre-computed desired-vs-cur masks."""
    N = is_mine.shape[0]
    if rng is not None:
        offsets = rng.integers(0, K, size=N).astype(np.int32)
    else:
        offsets = np.zeros(N, dtype=np.int32)
    return _picker_core_split(
        np.ascontiguousarray(is_mine, dtype=np.bool_),
        np.ascontiguousarray(set_mask, dtype=np.bool_),
        np.ascontiguousarray(clear_mask, dtype=np.bool_),
        offsets,
        int(ACTION_SET_BASE),
        int(ACTION_CLEAR_BASE),
        int(ACTION_NOOP),
    )


@njit(cache=True)
def _picker_core_split(
    is_mine: np.ndarray,
    set_mask: np.ndarray,
    clear_mask: np.ndarray,
    offsets: np.ndarray,
    action_set_base: int,
    action_clear_base: int,
    action_noop: int,
) -> np.ndarray:
    """Like _picker_core but takes the two phase masks directly (BFS
    solver uses this because it has its own SET priority logic)."""
    N = is_mine.shape[0]
    Kn = set_mask.shape[1]
    actions = np.full(N, action_noop, dtype=np.int32)
    for c in range(N):
        if not is_mine[c]:
            continue
        offset = offsets[c]
        chosen = -1
        for j in range(Kn):
            k = offset + j
            if k >= Kn:
                k -= Kn
            if set_mask[c, k]:
                chosen = k
                break
        if chosen >= 0:
            actions[c] = action_set_base + chosen
            continue
        for j in range(Kn):
            k = offset + j
            if k >= Kn:
                k -= Kn
            if clear_mask[c, k]:
                chosen = k
                break
        if chosen >= 0:
            actions[c] = action_clear_base + chosen
    return actions


# ---------------------------------------------------------------------------
# Lightning solver — all modes vectorized.
# ---------------------------------------------------------------------------


def _gradient_relay_desired(
    state: State, seat: int,
    pot: np.ndarray, fanout_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (attack, relay) (N, K) bool masks for the gradient-relay rule
    used by max / sum / sum_pw modes."""
    nb_safe, nb_valid, nb_owner, is_friendly, is_attack = _classify_slots(state, seat)
    relay = _relay_mask(pot, nb_safe, is_friendly, fanout_eps)
    return is_attack, relay


def _wave_gate(
    state: State, seat: int, wave_frac: float,
    attack: np.ndarray, relay: np.ndarray, gate_attack: bool,
) -> np.ndarray:
    """Combined desired mask for wave-gated modes."""
    is_mine = state.owner == seat
    below = (state.strength < wave_frac * MAX_STRENGTH) & is_mine        # (N,)
    if gate_attack:
        desired_fired = attack | relay
        desired_gated = np.zeros_like(attack)
    else:
        desired_fired = attack | relay
        desired_gated = attack
    return np.where(below[:, None], desired_gated, desired_fired)


def _loop_desired(
    state: State, seat: int, curl_dir: int,
) -> np.ndarray:
    """Structural loop curl: even-k slots get set when slot k and slot
    (k+curl_dir)%K both point at friendlies. Frontier always attacks."""
    _, _, _, is_friendly, is_attack = _classify_slots(state, seat)
    # For each even k in {0,2,4}, partner = (k+curl_dir)%K. Build (N, K) loop mask.
    loop = np.zeros_like(is_friendly)
    for k in (0, 2, 4):
        kk = (k + curl_dir) % K
        loop[:, k] = is_friendly[:, k] & is_friendly[:, kk]
    return is_attack | loop


def _attn_desired(
    state: State, seat: int,
    gamma: float, weak_bonus: float, expand_bonus: float,
    deep_threshold: float, relay_thresh: float,
    build_release_frac: float,
) -> np.ndarray:
    """Two-head attn solver: gradient attack head blended with even-k loop
    head, weighted by frontier distance through friendly cells.
    """
    N = state.N
    pot = compute_potential(
        state, seat, gamma=gamma, weak_bonus=weak_bonus,
        expand_bonus=expand_bonus, defense_bonus=0.0, mode="max",
    )
    nb_safe, nb_valid, nb_owner, is_friendly, is_attack = _classify_slots(state, seat)
    is_mine = state.owner == seat

    # Frontier-distance BFS through friendly cells, distance to any
    # non-friendly-alive (enemy or neutral) cell.
    not_friendly_alive = nb_valid & (nb_owner != seat) & (nb_owner != DEAD)
    INF = np.float32(1e9)
    fdist = np.full(N, INF, dtype=np.float32)
    seed = is_mine & not_friendly_alive.any(axis=1)
    fdist[seed] = 0.0
    iters = int(2 * deep_threshold) + 4
    for _ in range(iters):
        prev = fdist
        nbr_friendly = is_friendly & is_mine[nb_safe]
        cand_per_slot = np.where(nbr_friendly, fdist[nb_safe] + 1.0, INF)
        cand = cand_per_slot.min(axis=1)
        new_fdist = np.where(is_mine, np.minimum(prev, cand), prev)
        if np.array_equal(new_fdist, prev):
            break
        fdist = new_fdist

    alpha = np.clip(fdist / max(deep_threshold, 1e-6), 0.0, 1.0).astype(np.float32)

    # ATTACK head scores per friendly slot: max(0, pot[d] - pot[c]).
    pot_at_slot = pot[nb_safe]
    attack_score = np.where(
        is_friendly, np.maximum(0.0, pot_at_slot - pot[:, None]), 0.0,
    ).astype(np.float32)
    amax = attack_score.max(axis=1, keepdims=True)
    safe_amax = np.where(amax > 0, amax, 1.0)
    attack_score = np.where(amax > 0, attack_score / safe_amax, attack_score)

    # LOOP head: even-k & friend k+1 friend.
    loop_score = np.zeros_like(attack_score)
    for k in (0, 2, 4):
        kk = (k + 1) % K
        loop_score[:, k] = (is_friendly[:, k] & is_friendly[:, kk]).astype(np.float32)

    # Build-release gate on the LOOP head only.
    if build_release_frac > 0.0:
        strength_frac = state.strength / MAX_STRENGTH
        loop_scale = (strength_frac >= build_release_frac).astype(np.float32)
        loop_score = loop_score * loop_scale[:, None]

    combined = (1.0 - alpha)[:, None] * attack_score + alpha[:, None] * loop_score
    has_any_friendly = is_friendly.any(axis=1)
    cmax = combined.max(axis=1)
    safe_cmax = np.where(cmax > 0, cmax, 1.0)
    thresh = np.maximum(relay_thresh * cmax, 0.15)
    above_thresh = (combined >= thresh[:, None]) & is_friendly & (cmax[:, None] > 0)
    relay = has_any_friendly[:, None] & above_thresh
    return is_attack | relay


def _chase_desired(state: State, seat: int, panic_threshold: float) -> np.ndarray:
    """Counter-attack: always attack frontier; relay toward any friendly
    neighbor whose inbound enemy pressure exceeds the panic threshold.
    """
    N = state.N
    nb_safe, nb_valid, nb_owner, is_friendly, is_attack = _classify_slots(state, seat)
    inbound = _inbound_enemy_pressure(state, seat)
    threat_thresh = panic_threshold * MAX_EDGE
    friend_under_threat = is_friendly & (inbound[nb_safe] > threat_thresh)
    return is_attack | friend_under_threat


def _random_actions(
    state: State, seat: int, rng: Optional[np.random.Generator],
) -> np.ndarray:
    """Random per-cell action."""
    if rng is None:
        rng = np.random.default_rng()
    N = state.N
    owner = state.owner
    nb_safe, nb_valid, nb_owner, is_friendly, is_attack = _classify_slots(state, seat)
    valid_slot = nb_valid & (nb_owner != DEAD)
    is_mine = owner == seat

    # 30% noop, otherwise 70% SET / 30% CLEAR on a uniformly chosen valid slot.
    actions = np.full(N, ACTION_NOOP, dtype=np.int32)
    if not is_mine.any():
        return actions

    r = rng.random(N).astype(np.float32)
    do_action = (r >= 0.3) & is_mine
    # Choose a valid slot per cell.
    valid_count = valid_slot.sum(axis=1).astype(np.int32)
    has_valid = valid_count > 0
    do_action = do_action & has_valid
    # Pick slot: rotate by offset; first True after rotation is our slot.
    offsets = rng.integers(0, K, size=N).astype(np.int32)
    slot_idx = np.arange(K, dtype=np.int32)
    rot = (slot_idx[None, :] + offsets[:, None]) % K
    valid_rot = np.take_along_axis(valid_slot, rot, axis=1)
    first = valid_rot.argmax(axis=1)
    chosen_slot = ((first + offsets) % K).astype(np.int32)

    r2 = rng.random(N).astype(np.float32)
    set_choice = r2 < 0.7
    actions = np.where(
        do_action & set_choice,
        ACTION_SET_BASE + chosen_slot,
        actions,
    )
    actions = np.where(
        do_action & ~set_choice,
        ACTION_CLEAR_BASE + chosen_slot,
        actions,
    )
    return actions


def _flood_desired(state: State, seat: int) -> np.ndarray:
    """Set every valid non-dead slot."""
    _, nb_valid, nb_owner, _, _ = _classify_slots(state, seat)
    return nb_valid & (nb_owner != DEAD)


# ---------------------------------------------------------------------------
# Top-level mode dispatch.
# ---------------------------------------------------------------------------


def lightning_actions(
    state: State,
    seat: int,
    rng: Optional[np.random.Generator] = None,
    gamma: float = 0.85,
    weak_bonus: float = 1.0,
    expand_bonus: float = 0.6,
    defense_bonus: float = 0.0,
    fanout_eps: float = 0.05,
    mode: str = "max",
    curl_dir: int = 1,
    **mode_kwargs,
) -> np.ndarray:
    """Vectorized dispatch for all lightning modes."""
    N = state.N
    owner = state.owner
    is_mine = owner == seat
    if not is_mine.any():
        return np.full(N, ACTION_NOOP, dtype=np.int32)

    cur = state.outflow.astype(np.bool_)

    # Pure modes that don't need a potential field.
    if mode == "loop":
        desired = _loop_desired(state, seat, curl_dir=curl_dir)
        return _actions_from_desired(is_mine, desired, cur, rng)
    if mode == "vortex":
        desired = _loop_desired(state, seat, curl_dir=-1)
        return _actions_from_desired(is_mine, desired, cur, rng)
    if mode == "flood":
        desired = _flood_desired(state, seat)
        return _actions_from_desired(is_mine, desired, cur, rng)
    if mode == "random":
        return _random_actions(state, seat, rng)
    if mode == "chase":
        desired = _chase_desired(
            state, seat,
            panic_threshold=mode_kwargs.pop("panic_threshold", 0.3),
        )
        return _actions_from_desired(is_mine, desired, cur, rng)
    if mode == "attn":
        desired = _attn_desired(
            state, seat,
            gamma=gamma, weak_bonus=weak_bonus, expand_bonus=expand_bonus,
            deep_threshold=mode_kwargs.pop("deep_threshold", 2.0),
            relay_thresh=mode_kwargs.pop("relay_thresh", 0.5),
            build_release_frac=mode_kwargs.pop("build_release_frac", 0.0),
        )
        return _actions_from_desired(is_mine, desired, cur, rng)
    if mode == "attn_release":
        desired = _attn_desired(
            state, seat,
            gamma=gamma, weak_bonus=weak_bonus, expand_bonus=expand_bonus,
            deep_threshold=mode_kwargs.pop("deep_threshold", 2.0),
            relay_thresh=mode_kwargs.pop("relay_thresh", 0.5),
            build_release_frac=mode_kwargs.pop("build_release_frac", 0.7),
        )
        return _actions_from_desired(is_mine, desired, cur, rng)
    if mode == "attn_slam":
        desired = _attn_desired(
            state, seat,
            gamma=gamma, weak_bonus=weak_bonus, expand_bonus=expand_bonus,
            deep_threshold=mode_kwargs.pop("deep_threshold", 2.0),
            relay_thresh=mode_kwargs.pop("relay_thresh", 0.5),
            build_release_frac=mode_kwargs.pop("build_release_frac", 0.95),
        )
        return _actions_from_desired(is_mine, desired, cur, rng)

    # Potential-field-based modes.
    if mode in ("max", "sum", "sum_pw"):
        pot = compute_potential(
            state, seat, gamma=gamma, weak_bonus=weak_bonus,
            expand_bonus=expand_bonus, defense_bonus=defense_bonus, mode=mode,
        )
        attack, relay = _gradient_relay_desired(state, seat, pot, fanout_eps)
        return _actions_from_desired(is_mine, attack | relay, cur, rng)

    if mode == "live":
        # One-pass live-field proxy. Skips the 32-iter Bellman solve.
        # Designed for fluid (EDGE_ALPHA < 1.0) rules where edge_pressure
        # is time-integrated; the strength_signal term bootstraps the
        # relay gradient at game start before pressure has propagated.
        pot = compute_potential_live(
            state, seat, gamma=gamma, weak_bonus=weak_bonus,
            expand_bonus=expand_bonus, defense_bonus=defense_bonus,
            flow_weight=mode_kwargs.pop("flow_weight", 1.0),
            strength_weight=mode_kwargs.pop("strength_weight", 0.5),
        )
        attack, relay = _gradient_relay_desired(state, seat, pot, fanout_eps)
        return _actions_from_desired(is_mine, attack | relay, cur, rng)

    if mode in ("sum_wave", "max_wave", "wave_keep_attack"):
        pot_mode = "sum" if mode in ("sum_wave", "wave_keep_attack") else "max"
        wave_frac = mode_kwargs.pop("wave_frac", 0.6)
        pot = compute_potential(
            state, seat, gamma=gamma, weak_bonus=weak_bonus,
            expand_bonus=expand_bonus, defense_bonus=0.0, mode=pot_mode,
        )
        attack, relay = _gradient_relay_desired(state, seat, pot, fanout_eps)
        gate_attack = (mode != "wave_keep_attack")
        desired = _wave_gate(state, seat, wave_frac, attack, relay, gate_attack)
        return _actions_from_desired(is_mine, desired, cur, rng)

    if mode in ("pulse", "pulse_stagger"):
        period = mode_kwargs.pop("period", 200)
        duty = mode_kwargs.pop("duty", 0.5)
        stagger = (mode == "pulse_stagger") or mode_kwargs.pop("stagger", False)
        cycle_pos = int(state.tick) % period
        global_fire = cycle_pos >= int(period * (1.0 - duty))
        pot = compute_potential(
            state, seat, gamma=gamma, weak_bonus=weak_bonus,
            expand_bonus=expand_bonus, defense_bonus=0.0, mode="sum",
        )
        attack, relay = _gradient_relay_desired(state, seat, pot, fanout_eps)
        if stagger:
            # Even-index cells fire when global_fire, odd cells fire when not.
            cell_idx = np.arange(N, dtype=np.int32)
            cell_fire = ((cell_idx % 2 == 0) == global_fire)
        else:
            cell_fire = np.full(N, global_fire, dtype=np.bool_)
        cell_fire = cell_fire & is_mine
        desired = np.where(cell_fire[:, None], attack | relay,
                           np.zeros_like(attack))
        return _actions_from_desired(is_mine, desired, cur, rng)

    raise ValueError(f"unknown mode {mode!r}")
