"""Lightning solver for flux v2 — purely local potential-field heuristic.

No global computation, no priority queue. Each cell stores a single scalar
"potential" computed by local diffusion from intrinsic sources, then sets
outflows toward the steepest-uphill neighbor. Lightning behavior emerges:
when one enemy is much weaker than the rest, its gradient dominates the
network and pressure naturally channels toward it. When it dies, the next
attractor takes over automatically.

Per-cell intrinsic source (read fresh each AI tick):
    enemy cell        weak_bonus * (1 - strength/MAX_STRENGTH)
    neutral cell      expand_bonus  (small constant pull)
    friendly cell     defense_bonus * (inbound_enemy_pressure / MAX_EDGE)
                      (threatened back-line cells glow → relays swerve to save)
    dead cell         0  (walls don't conduct)

Three diffusion modes (see v2-edge-loop-emergence wiki):

    mode="max" (original)
        pot[c] = max(intrinsic[c], gamma * max(pot[d] for d in non-dead nbrs))
        Tree-like field; each cell inherits from one steepest parent. Loop
        patterns (a→b→c→a) are forbidden by construction.

    mode="sum"
        pot[c] = intrinsic[c] + gamma * sum_d ((1/deg(d)) * pot[d])
        Bellman value-iteration on a uniform stochastic transition. Cycles
        self-reinforce as a geometric series — the closed-form "future
        residual" of pressure circulating forever, no rollout required.

    mode="sum_pw" (sum, pressure-weighted)
        pot[c] = intrinsic[c] + gamma * sum_d (w[d→c] * pot[d])
        where w[d→c] = edge_pressure[d→c] / total_outflow[d]
                       (uniform 1/deg(d) fallback when total_outflow[d]≈0)
        Rich-get-richer: cycles in the *current* edge_pressure field
        amplify themselves through the diffusion.

Action rule (per owned cell):
    desired-on slots = those pointing at the steepest-uphill neighbor(s)
                       (pot[d] strictly greater than pot[c])
    one action this tick:  SET a missing on-slot > CLEAR a stale slot > NOOP
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .solver import _pick
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


def _inbound_enemy_pressure(state: State, seat: int) -> np.ndarray:
    """Sum of edge_pressure on inbound edges from enemy seats. (N,) float32."""
    N = state.N
    owner = state.owner
    nb = state.neighbors
    ep = state.edge_pressure

    inbound = np.zeros(N, dtype=np.float32)
    for k in range(K):
        d_ids = nb[:, k]
        valid = d_ids >= 0
        if not valid.any():
            continue
        opp = int(OPPOSITE_SLOT[k])
        valid_d = d_ids[valid]
        press = np.zeros(N, dtype=np.float32)
        press[valid] = ep[valid_d, opp]
        d_owner = np.full(N, NEUTRAL, dtype=np.int32)
        d_owner[valid] = owner[valid_d]
        is_enemy_src = valid & (d_owner != seat) & (d_owner >= 0)
        inbound += press * is_enemy_src.astype(np.float32)
    return inbound


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
    """Return (N,) float32 potential field for `seat`. All reads are local —
    each iteration only inspects each cell's 6 neighbors.

    mode in {"max", "sum", "sum_pw"} — see module docstring.
    """
    N = state.N
    owner = state.owner
    strength = state.strength
    nb = state.neighbors

    intrinsic = np.zeros(N, dtype=np.float32)

    is_enemy = (owner >= 0) & (owner != seat)
    if is_enemy.any():
        intrinsic[is_enemy] = (
            weak_bonus * (1.0 - strength[is_enemy] / MAX_STRENGTH)
        ).clip(0.0, weak_bonus)

    is_neutral = owner == NEUTRAL
    intrinsic[is_neutral] = expand_bonus

    is_mine = owner == seat
    if is_mine.any():
        inbound = _inbound_enemy_pressure(state, seat)
        threat = (inbound / MAX_EDGE).clip(0.0, 1.0)
        intrinsic[is_mine] = defense_bonus * threat[is_mine]

    is_dead = owner == DEAD
    intrinsic[is_dead] = 0.0

    pot = intrinsic.copy()
    nb_safe = np.maximum(nb, 0)
    nb_valid = (nb >= 0)
    nbr_dead = (owner[nb_safe] == DEAD) & nb_valid
    nbr_ok = nb_valid & ~nbr_dead                                # (N, K) bool
    nbr_ok_f = nbr_ok.astype(np.float32)

    if mode == "max":
        for _ in range(max_iter):
            nbr_pot = pot[nb_safe] * nbr_ok_f                    # (N, K)
            max_nbr = nbr_pot.max(axis=1)
            new_pot = np.maximum(intrinsic, gamma * max_nbr)
            new_pot[is_dead] = 0.0
            diff = np.abs(new_pot - pot).max()
            pot = new_pot
            if diff < tol:
                break
        return pot

    if mode not in ("sum", "sum_pw"):
        raise ValueError(f"unknown mode {mode!r} (max|sum|sum_pw)")

    # Uniform per-cell distribution weight: each cell d distributes its pot
    # equally among its non-dead neighbors. deg_self[d] = #valid out-nbrs of d.
    # Row-sum of the operator over d→neighbors is ≤ 1, so γ < 1 guarantees a
    # contraction.
    deg_self = nbr_ok_f.sum(axis=1)                              # (N,)
    safe_deg = np.where(deg_self > 0, deg_self, 1.0)
    uniform_per_edge_d = np.where(deg_self > 0, 1.0 / safe_deg, 0.0)  # (N,)
    uniform_w = uniform_per_edge_d[nb_safe] * nbr_ok_f           # (N, K)

    if mode == "sum":
        weights = uniform_w
    else:  # sum_pw
        ep = state.edge_pressure                                 # (N, K) float32
        # For each c, slot k points at d = nb_safe[c, k]. The edge d→c uses
        # slot OPPOSITE_SLOT[k] on d, so pressure d→c = ep[d, opp[k]].
        opp = OPPOSITE_SLOT[None, :]                             # (1, K)
        ep_dc = ep[nb_safe, opp]                                 # (N, K)
        out_total_d = ep.sum(axis=1)                             # (N,)
        denom_dc = np.maximum(out_total_d[nb_safe], 1e-9)        # (N, K)
        pw = ep_dc / denom_dc
        # Fallback to uniform when d has no current outflow.
        has_flow = out_total_d[nb_safe] > 1e-9                   # (N, K)
        weights = np.where(has_flow, pw, uniform_w) * nbr_ok_f

    for _ in range(max_iter):
        contrib = (pot[nb_safe] * weights).sum(axis=1)
        new_pot = intrinsic + gamma * contrib
        new_pot[is_dead] = 0.0
        diff = np.abs(new_pot - pot).max()
        pot = new_pot
        if diff < tol:
            break
    return pot


def _attn_actions(
    state: State,
    seat: int,
    rng: Optional[np.random.Generator],
    deep_threshold: float = 2.0,
    gamma: float = 0.85,
    weak_bonus: float = 1.0,
    expand_bonus: float = 0.6,
    relay_thresh: float = 0.5,
    build_release_frac: float = 0.0,
) -> np.ndarray:
    """Two-head attention solver. Each owned cell c blends two heads:

      ATTACK head: max-pot gradient (lightning's original max mode).
                   attack_score[k] ∝ max(0, pot[nb[c,k]] - pot[c]).
      LOOP head:   structural even-k curl (lightning_loop's mode).
                   loop_score[k] = 1 iff k ∈ {0,2,4} and both slots k and k+1
                                   point at friendlies.

    Per-cell mixing weight α(c) = clip(frontier_dist[c] / deep_threshold, 0, 1)
    where frontier_dist is BFS distance through friendly cells to the nearest
    non-friendly-alive (enemy/neutral) cell.

      α(c) = 0 at frontier      → pure ATTACK head (gradient relay)
      α(c) = 1 deep interior    → pure LOOP head (triskelion)
      α(c) intermediate         → blend: the loops "tilt" forward as the
                                  frontier gets closer, so backline storage
                                  gradually leaks toward shots-on-goal.

    Combined per-slot score = (1-α) * attack_score + α * loop_score.
    Forced-attack rule still applies on every non-friendly non-dead slot.
    Friendly-slot relay activates when combined ≥ max(0.5 · combined.max,
    `relay_thresh`).
    """
    N = state.N
    owner = state.owner
    nb = state.neighbors
    outflow = state.outflow

    actions = np.full(N, ACTION_NOOP, dtype=np.int32)
    is_mine = owner == seat
    if not is_mine.any():
        return actions

    pot = compute_potential(
        state, seat,
        gamma=gamma, weak_bonus=weak_bonus,
        expand_bonus=expand_bonus, defense_bonus=0.0, mode="max",
    )

    # Frontier-distance BFS through friendly cells.
    nb_safe = np.maximum(nb, 0)
    nb_valid = nb >= 0
    not_friendly_alive = (owner != seat) & (owner != DEAD)
    INF = np.float32(1e9)
    fdist = np.full(N, INF, dtype=np.float32)
    # Seed: friendly cell adjacent to any non-friendly-alive cell → 0.
    for k in range(K):
        v = nb_valid[:, k]
        d_safe = nb_safe[:, k]
        attacky_nbr = v & not_friendly_alive[d_safe]
        fdist = np.where(is_mine & attacky_nbr, 0.0, fdist)
    # Relax.
    for _ in range(int(2 * deep_threshold) + 4):
        prev = fdist.copy()
        cand = np.full(N, INF, dtype=np.float32)
        for k in range(K):
            v = nb_valid[:, k]
            d_safe = nb_safe[:, k]
            d_is_friendly = v & is_mine[d_safe]
            this = np.where(d_is_friendly, fdist[d_safe] + 1.0, INF)
            cand = np.minimum(cand, this)
        fdist = np.where(is_mine, np.minimum(fdist, cand), fdist)
        if np.array_equal(fdist, prev):
            break

    alpha_arr = np.clip(fdist / max(deep_threshold, 1e-6), 0.0, 1.0).astype(np.float32)

    owned = np.where(is_mine)[0]
    for c in owned:
        c = int(c)
        a = float(alpha_arr[c])
        attack_score = np.zeros(K, dtype=np.float32)
        loop_score = np.zeros(K, dtype=np.float32)
        is_friendly_slot = np.zeros(K, dtype=np.bool_)
        force_attack = np.zeros(K, dtype=np.bool_)
        for k in range(K):
            d = int(nb[c, k])
            if d < 0:
                continue
            od = int(owner[d])
            if od == DEAD:
                continue
            if od != seat:
                force_attack[k] = True
            else:
                is_friendly_slot[k] = True
                if pot[d] > pot[c]:
                    attack_score[k] = pot[d] - pot[c]
        for k in (0, 2, 4):
            kk = (k + 1) % K
            if is_friendly_slot[k] and is_friendly_slot[kk]:
                loop_score[k] = 1.0
        amax = attack_score.max()
        if amax > 0:
            attack_score = attack_score / amax
        # Build-and-release gate: when the cell's strength is below
        # `build_release_frac * MAX_STRENGTH`, suppress the LOOP component
        # (storage relays in the deep interior) but keep the ATTACK
        # component (forward relays toward weak enemies via the field).
        # Frontier cells (α≈0) are unaffected because attack dominates;
        # deep cells (α≈1) suppress relays until they fill. The result is
        # stepped waves: cells fill, then join the storage network, then
        # contribute to the loop substrate.
        if build_release_frac > 0.0:
            strength_frac = float(state.strength[c]) / MAX_STRENGTH
            loop_scale = 1.0 if strength_frac >= build_release_frac else 0.0
        else:
            loop_scale = 1.0
        combined = (1.0 - a) * attack_score + a * loop_score * loop_scale
        desired = force_attack.copy()
        cmax = combined.max() if is_friendly_slot.any() else 0.0
        if cmax > 0:
            thresh = max(relay_thresh * cmax, 0.15)
            desired |= (combined >= thresh) & is_friendly_slot
        cur = outflow[c]
        missing = np.where(desired & ~cur)[0]
        if missing.size:
            actions[c] = ACTION_SET_BASE + _pick(missing, rng)
            continue
        stale = np.where(cur & ~desired)[0]
        if stale.size:
            actions[c] = ACTION_CLEAR_BASE + _pick(stale, rng)
            continue
    return actions


def _chase_actions(
    state: State, seat: int, rng: Optional[np.random.Generator],
    panic_threshold: float = 0.3,
) -> np.ndarray:
    """Counter-attack solver. Each owned cell:
      1. Frontier rule (always attack non-friendly).
      2. Inspect inbound edge_pressure from non-friendly neighbors. If
         the strongest incoming threat exceeds `panic_threshold * MAX_EDGE`,
         add a counter-attack outflow toward THAT direction (already covered
         by the frontier rule, but emphasizes the choice).
      3. For each friendly slot, if the friendly target has incoming threat
         > panic_threshold, set outflow toward that friend (reinforce).
    """
    N = state.N
    owner = state.owner
    nb = state.neighbors
    outflow = state.outflow
    edge_pressure = state.edge_pressure
    actions = np.full(N, ACTION_NOOP, dtype=np.int32)
    is_mine = owner == seat
    if not is_mine.any():
        return actions

    # Precompute inbound non-friendly pressure per cell.
    inbound_enemy = np.zeros(N, dtype=np.float32)
    for k in range(K):
        d_ids = nb[:, k]
        valid = d_ids >= 0
        if not valid.any():
            continue
        opp = int(OPPOSITE_SLOT[k])
        valid_d = d_ids[valid]
        press = np.zeros(N, dtype=np.float32)
        press[valid] = edge_pressure[valid_d, opp]
        d_owner = np.full(N, NEUTRAL, dtype=np.int32)
        d_owner[valid] = owner[valid_d]
        is_threat_src = valid & (d_owner != seat) & (d_owner >= 0)
        inbound_enemy += press * is_threat_src.astype(np.float32)

    threat_thresh = panic_threshold * MAX_EDGE
    owned = np.where(is_mine)[0]
    for c in owned:
        c = int(c)
        desired = np.zeros(K, dtype=np.bool_)
        for k in range(K):
            d = int(nb[c, k])
            if d < 0:
                continue
            od = int(owner[d])
            if od == DEAD:
                continue
            if od != seat:
                # Frontier: always set toward non-friendly.
                desired[k] = True
            else:
                # Friendly relay: reinforce if friend is under threat.
                if inbound_enemy[d] > threat_thresh:
                    desired[k] = True
        cur = outflow[c]
        missing = np.where(desired & ~cur)[0]
        if missing.size:
            actions[c] = ACTION_SET_BASE + _pick(missing, rng)
            continue
        stale = np.where(cur & ~desired)[0]
        if stale.size:
            actions[c] = ACTION_CLEAR_BASE + _pick(stale, rng)
    return actions


def _flood_actions(
    state: State, seat: int, rng: Optional[np.random.Generator],
) -> np.ndarray:
    """Flood: every owned cell sets all six outflows toward non-dead
    neighbors. No gating, no field, no relays — just maximal output in
    all valid directions. Baseline of 'always fire'.

    Note: the v2 reducer's no-bidirectional-friendly invariant will
    clear one side of every friendly edge, so the steady state has
    each friendly edge owned by the higher-cell-index side.
    """
    N = state.N
    owner = state.owner
    nb = state.neighbors
    outflow = state.outflow
    actions = np.full(N, ACTION_NOOP, dtype=np.int32)
    is_mine = owner == seat
    if not is_mine.any():
        return actions
    owned = np.where(is_mine)[0]
    for c in owned:
        c = int(c)
        desired = np.zeros(K, dtype=np.bool_)
        for k in range(K):
            d = int(nb[c, k])
            if d < 0:
                continue
            od = int(owner[d])
            if od == DEAD:
                continue
            desired[k] = True
        cur = outflow[c]
        missing = np.where(desired & ~cur)[0]
        if missing.size:
            actions[c] = ACTION_SET_BASE + _pick(missing, rng)
            continue
        stale = np.where(cur & ~desired)[0]
        if stale.size:
            actions[c] = ACTION_CLEAR_BASE + _pick(stale, rng)
    return actions


def _random_actions(
    state: State, seat: int, rng: Optional[np.random.Generator],
) -> np.ndarray:
    """Pick a random action per owned cell. Baseline of 'no policy'."""
    if rng is None:
        rng = np.random.default_rng()
    N = state.N
    owner = state.owner
    actions = np.full(N, ACTION_NOOP, dtype=np.int32)
    is_mine = owner == seat
    owned = np.where(is_mine)[0]
    for c in owned:
        c = int(c)
        nb_c = state.neighbors[c]
        # Valid slots: non-off-board, non-dead.
        valid = []
        for k in range(K):
            d = int(nb_c[k])
            if d < 0:
                continue
            if int(owner[d]) == DEAD:
                continue
            valid.append(k)
        if not valid:
            continue
        # Half chance noop, otherwise random SET/CLEAR on a random valid slot.
        r = rng.random()
        if r < 0.3:
            continue  # noop
        slot = int(rng.choice(valid))
        if rng.random() < 0.7:
            actions[c] = ACTION_SET_BASE + slot
        else:
            actions[c] = ACTION_CLEAR_BASE + slot
    return actions


def _loop_actions(
    state: State, seat: int, rng: Optional[np.random.Generator], curl_dir: int = 1,
) -> np.ndarray:
    """Structural curl rule — produces directed 3-loops on every friendly
    hex triangle of one parity. Independent of any potential field.

    Geometry: on a hex grid, neighbors at slots k and k+1 (mod 6) are
    mutually adjacent, forming a triangle with c. Every such triangle has a
    fixed slot-parity: from each of its three corners, the slot pair used
    is either both-even-k or both-odd-k. By restricting the rule to k ∈
    {0,2,4} (curl_dir=+1) or k ∈ {0,2,4} with kk=k-1 (curl_dir=-1), every
    set outflow goes from a slot to its (k+curl_dir) partner, and the
    opposite (back-edge) slot — which would be (k+3) — is always odd and
    therefore never set. No bidirectional collisions, the v2 "no friendly
    bidirectional flow" invariant in `step.apply_actions` never has to
    clear anything, and a clean CCW (or CW) directed 3-cycle survives on
    every even-parity friendly triangle.

    Frontier cells still attack every non-friendly slot (air-breakdown
    rule). Loops form selectively across the friendly interior where both
    slots of the (k, k+curl_dir) pair are friendly.

    curl_dir = +1 for CCW (the default), -1 for CW.
    """
    N = state.N
    owner = state.owner
    nb = state.neighbors
    outflow = state.outflow

    actions = np.full(N, ACTION_NOOP, dtype=np.int32)
    is_mine = owner == seat
    if not is_mine.any():
        return actions

    even_slots = (0, 2, 4)

    owned = np.where(is_mine)[0]
    for c in owned:
        c = int(c)
        attack = np.zeros(K, dtype=np.bool_)
        is_friendly_slot = np.zeros(K, dtype=np.bool_)
        for k in range(K):
            d = int(nb[c, k])
            if d < 0:
                continue
            od = int(owner[d])
            if od == DEAD:
                continue
            if od == seat:
                is_friendly_slot[k] = True
            else:
                attack[k] = True
        # Loop relay: only even-k slots, where slot k and slot (k+curl_dir)
        # are both friendly. Restricting to even k guarantees the
        # back-edge (slot k+3, always odd) is never set on the destination,
        # so the bidirectional-friendly invariant never triggers.
        loop_relay = np.zeros(K, dtype=np.bool_)
        for k in even_slots:
            kk = (k + curl_dir) % K
            if is_friendly_slot[k] and is_friendly_slot[kk]:
                loop_relay[k] = True
        desired = attack | loop_relay
        cur = outflow[c]
        missing = np.where(desired & ~cur)[0]
        if missing.size:
            actions[c] = ACTION_SET_BASE + _pick(missing, rng)
            continue
        stale = np.where(cur & ~desired)[0]
        if stale.size:
            actions[c] = ACTION_CLEAR_BASE + _pick(stale, rng)
            continue
    return actions


def _sum_wave_actions(
    state: State, seat: int, rng: Optional[np.random.Generator],
    gamma: float, weak_bonus: float, expand_bonus: float,
    wave_frac: float,
) -> np.ndarray:
    return _wave_actions(state, seat, rng, gamma=gamma, weak_bonus=weak_bonus,
                          expand_bonus=expand_bonus, wave_frac=wave_frac, pot_mode="sum")


def _pulse_actions(
    state: State, seat: int, rng: Optional[np.random.Generator],
    gamma: float, weak_bonus: float, expand_bonus: float,
    period: int, duty: float,
    stagger: bool = False,
) -> np.ndarray:
    """Globally synchronized pulse: read state.tick, switch all owned cells
    between CHARGE (clear outflows, accumulate strength) and FIRE (sum-mode)
    phases together. Period=200 default, duty=0.5 means 100 ticks charge then
    100 ticks fire. Visual: the whole board's territory pulses in unison.

    If stagger=True, split owned cells by index parity — even cells fire on
    one half of the cycle, odd cells on the other — so half the territory
    is always firing. Tests whether pulse failure is about charge time or
    about synchronization specifically.
    """
    N = state.N
    owner = state.owner
    nb = state.neighbors
    outflow = state.outflow
    actions = np.full(N, ACTION_NOOP, dtype=np.int32)
    is_mine = owner == seat
    if not is_mine.any():
        return actions

    cycle_pos = int(state.tick) % period
    global_fire = cycle_pos >= int(period * (1.0 - duty))

    pot = compute_potential(state, seat, gamma=gamma, weak_bonus=weak_bonus,
                            expand_bonus=expand_bonus, mode="sum")
    owned = np.where(is_mine)[0]
    for c in owned:
        c_int = int(c)
        if stagger:
            # Even cells fire when global_fire is True; odd cells fire when False.
            # So at any given tick, half the territory is firing.
            cell_fire = global_fire if (c_int % 2 == 0) else (not global_fire)
        else:
            cell_fire = global_fire

        if not cell_fire:
            cur = outflow[c_int]
            stale = np.where(cur)[0]
            if stale.size:
                actions[c_int] = ACTION_CLEAR_BASE + _pick(stale, rng)
            continue
        c = c_int
        attack = np.zeros(K, dtype=np.bool_)
        relay = np.zeros(K, dtype=np.bool_)
        my_pot = pot[c]
        best_friendly_pot = my_pot
        best_friendly_slots: list[int] = []
        for k in range(K):
            d = int(nb[c, k])
            if d < 0:
                continue
            od = int(owner[d])
            if od == DEAD:
                continue
            if od != seat:
                attack[k] = True
                continue
            pd = pot[d]
            if pd > best_friendly_pot + 0.05:
                best_friendly_pot = pd
                best_friendly_slots = [k]
            elif abs(pd - best_friendly_pot) <= 0.05 and pd > my_pot:
                best_friendly_slots.append(k)
        for k in best_friendly_slots:
            relay[k] = True
        desired = attack | relay
        cur = outflow[c]
        missing = np.where(desired & ~cur)[0]
        if missing.size:
            actions[c] = ACTION_SET_BASE + _pick(missing, rng)
            continue
        stale = np.where(cur & ~desired)[0]
        if stale.size:
            actions[c] = ACTION_CLEAR_BASE + _pick(stale, rng)
    return actions


def _wave_actions(
    state: State, seat: int, rng: Optional[np.random.Generator],
    gamma: float, weak_bonus: float, expand_bonus: float,
    wave_frac: float, pot_mode: str = "sum",
    gate_attack: bool = True,
) -> np.ndarray:
    """`pot_mode`-mode but each cell only fires when strength ≥ wave_frac * MAX.
    Sub-threshold cells clear all outflows so they charge up before
    discharging. Visual: territory pulses — pressure builds up, then
    each cell snaps to its `pot_mode`-mode action when full.

    If gate_attack=False, frontier (attack) outflows are kept even when
    below threshold — only relays are gated. The frontier never goes
    offline; only interior charging is throttled.
    """
    from .state import MAX_STRENGTH
    N = state.N
    owner = state.owner
    nb = state.neighbors
    outflow = state.outflow
    actions = np.full(N, ACTION_NOOP, dtype=np.int32)
    is_mine = owner == seat
    if not is_mine.any():
        return actions

    pot = compute_potential(state, seat, gamma=gamma, weak_bonus=weak_bonus,
                            expand_bonus=expand_bonus, mode=pot_mode)
    thresh = wave_frac * MAX_STRENGTH

    owned = np.where(is_mine)[0]
    for c in owned:
        c = int(c)
        s_c = float(state.strength[c])
        cur = outflow[c]
        # Always compute attack/relay first so the gate logic can pick subsets.
        attack = np.zeros(K, dtype=np.bool_)
        relay = np.zeros(K, dtype=np.bool_)
        my_pot = pot[c]
        best_friendly_pot = my_pot
        best_friendly_slots: list[int] = []
        for k in range(K):
            d = int(nb[c, k])
            if d < 0:
                continue
            od = int(owner[d])
            if od == DEAD:
                continue
            if od != seat:
                attack[k] = True
                continue
            pd = pot[d]
            if pd > best_friendly_pot + 0.05:
                best_friendly_pot = pd
                best_friendly_slots = [k]
            elif abs(pd - best_friendly_pot) <= 0.05 and pd > my_pot:
                best_friendly_slots.append(k)
        for k in best_friendly_slots:
            relay[k] = True

        if s_c < thresh:
            if gate_attack:
                # Below threshold and gating everything: clear all outflows.
                desired = np.zeros(K, dtype=np.bool_)
            else:
                # Below threshold but keep frontier (attack) outflows. Drop relays only.
                desired = attack
        else:
            desired = attack | relay

        missing = np.where(desired & ~cur)[0]
        if missing.size:
            actions[c] = ACTION_SET_BASE + _pick(missing, rng)
            continue
        stale = np.where(cur & ~desired)[0]
        if stale.size:
            actions[c] = ACTION_CLEAR_BASE + _pick(stale, rng)
    return actions


def lightning_solver_actions(
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
    """Return (N,) int32 actions for `seat`. Cells not owned by `seat` get NOOP.
    Owned cells emit the one action that nudges their outflow toward the
    steepest-uphill neighbor in the local potential field.

    mode in {"max", "sum", "sum_pw", "loop"} selects the operator.
    The "loop" mode ignores the potential field entirely and uses the
    structural curl rule (see _loop_actions docstring) to produce 3-loops
    on every friendly triangle. curl_dir = +1 CCW (default), -1 CW.
    """
    if mode == "loop":
        return _loop_actions(state, seat, rng, curl_dir=curl_dir)
    if mode == "vortex":
        # CW (curl_dir=-1) vs default CCW. Same shape, opposite spin.
        return _loop_actions(state, seat, rng, curl_dir=-1)
    if mode == "attn":
        return _attn_actions(
            state, seat, rng,
            gamma=gamma, weak_bonus=weak_bonus, expand_bonus=expand_bonus,
            **mode_kwargs,
        )
    if mode == "attn_release":
        return _attn_actions(
            state, seat, rng,
            gamma=gamma, weak_bonus=weak_bonus, expand_bonus=expand_bonus,
            build_release_frac=mode_kwargs.pop("build_release_frac", 0.7),
            **mode_kwargs,
        )
    if mode == "attn_slam":
        return _attn_actions(
            state, seat, rng,
            gamma=gamma, weak_bonus=weak_bonus, expand_bonus=expand_bonus,
            build_release_frac=mode_kwargs.pop("build_release_frac", 0.95),
            **mode_kwargs,
        )
    if mode == "flood":
        return _flood_actions(state, seat, rng)
    if mode == "random":
        return _random_actions(state, seat, rng)
    if mode == "chase":
        return _chase_actions(state, seat, rng)
    if mode == "sum_wave":
        return _sum_wave_actions(
            state, seat, rng,
            gamma=gamma, weak_bonus=weak_bonus, expand_bonus=expand_bonus,
            wave_frac=mode_kwargs.pop("wave_frac", 0.6),
        )
    if mode == "max_wave":
        return _wave_actions(
            state, seat, rng,
            gamma=gamma, weak_bonus=weak_bonus, expand_bonus=expand_bonus,
            wave_frac=mode_kwargs.pop("wave_frac", 0.6), pot_mode="max",
        )
    if mode == "wave_keep_attack":
        # Wave gate but frontier (attack) outflows are kept even when charging.
        return _wave_actions(
            state, seat, rng,
            gamma=gamma, weak_bonus=weak_bonus, expand_bonus=expand_bonus,
            wave_frac=mode_kwargs.pop("wave_frac", 0.6), pot_mode="sum",
            gate_attack=False,
        )
    if mode == "pulse":
        return _pulse_actions(
            state, seat, rng,
            gamma=gamma, weak_bonus=weak_bonus, expand_bonus=expand_bonus,
            period=mode_kwargs.pop("period", 200),
            duty=mode_kwargs.pop("duty", 0.5),
            stagger=mode_kwargs.pop("stagger", False),
        )
    if mode == "pulse_stagger":
        return _pulse_actions(
            state, seat, rng,
            gamma=gamma, weak_bonus=weak_bonus, expand_bonus=expand_bonus,
            period=mode_kwargs.pop("period", 200),
            duty=mode_kwargs.pop("duty", 0.5),
            stagger=True,
        )

    N = state.N
    owner = state.owner
    nb = state.neighbors
    outflow = state.outflow

    actions = np.full(N, ACTION_NOOP, dtype=np.int32)
    is_mine = owner == seat
    if not is_mine.any():
        return actions

    pot = compute_potential(
        state, seat,
        gamma=gamma, weak_bonus=weak_bonus,
        expand_bonus=expand_bonus, defense_bonus=defense_bonus,
        mode=mode,
    )

    owned = np.where(is_mine)[0]
    for c in owned:
        c = int(c)
        # Frontier slots: any non-friendly, non-dead neighbor — always attack.
        # This is the air-breakdown rule: pressure discharges at any exposed
        # boundary. No "is this the weakest" check; if you touch it, you
        # pump pressure into it.
        attack = np.zeros(K, dtype=np.bool_)
        relay = np.zeros(K, dtype=np.bool_)
        my_pot = pot[c]
        best_friendly_pot = my_pot
        best_friendly_slots: list[int] = []
        for k in range(K):
            d = int(nb[c, k])
            if d < 0:
                continue
            od = int(owner[d])
            if od == DEAD:
                continue
            if od != seat:
                attack[k] = True
                continue
            # Friendly: candidate for relay if strictly uphill.
            pd = pot[d]
            if pd > best_friendly_pot + fanout_eps:
                best_friendly_pot = pd
                best_friendly_slots = [k]
            elif abs(pd - best_friendly_pot) <= fanout_eps and pd > my_pot:
                best_friendly_slots.append(k)
        for k in best_friendly_slots:
            relay[k] = True
        desired = attack | relay

        cur = outflow[c]
        missing = np.where(desired & ~cur)[0]
        if missing.size:
            actions[c] = ACTION_SET_BASE + _pick(missing, rng)
            continue
        stale = np.where(cur & ~desired)[0]
        if stale.size:
            actions[c] = ACTION_CLEAR_BASE + _pick(stale, rng)
            continue
    return actions
