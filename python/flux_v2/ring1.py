"""Ring 1 field policy — neural boundary conditions on the frozen Bellman
operator.

Implements the Ring 1 class from wiki/topics/v2-beat-the-solver-plan.md:

    actions = picker ∘ throttle_θ ∘ readout_θ ∘ Bellman ∘ intrinsic_θ

The global propagation (`_compute_potential_core`, sum mode) and the picker
(`_actions_from_desired`) are reused unchanged from `solver_vec`. The genome
parameterizes only what feeds the field (linear intrinsic over per-cell
features) and what reads it (relay/attack gates, slot ranking, per-cell
throttle count).

The class STRICTLY CONTAINS the current champion `lightning_sum_throttled`:
`champion_vector()` reproduces its actions bit-for-bit (same pot trajectory,
same desired mask, same picker RNG consumption). That property is the Gate 2
reproduction test in tests/test_v2_ring1_parity.py — failure there is an
implementation bug by definition.

Caches: like `compute_potential`, the Bellman solve warm-starts from the
previous AI tick's field. Call `clear_field_caches()` between games (the
evolve harness does) so games stay deterministic and `id()` reuse across
boards can't contaminate warm starts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .solver_vec import (
    _OPPOSITE_SLOT_ARR_VEC,
    _actions_from_desired,
    _compute_potential_core,
    _inbound_enemy_pressure,
    _pot_warm_cache,
)
from .state import (
    ACTION_NOOP,
    DEAD,
    K,
    MAX_EDGE,
    MAX_STRENGTH,
    NEUTRAL,
    State,
)

# ---------------------------------------------------------------------------
# Genome layout
# ---------------------------------------------------------------------------

GENOME_NAMES: tuple[str, ...] = (
    # field
    "gamma",            # Bellman discount (champion 0.85)
    # intrinsic_θ — linear weights over per-cell features
    "w_weak",           # is_enemy * (1 - strength/MAX)        (champion 1.0)
    "w_enemy",          # is_enemy                              (champion 0.0)
    "w_expand",         # is_neutral                            (champion 0.6)
    "w_defense",        # is_mine * clip(inbound_enemy/MAX_EDGE) (champion 0.0)
    "w_enemy_strong",   # is_enemy * (strength/MAX)             (champion 0.0)
    "w_mine",           # is_mine                               (champion 0.0)
    # readout_θ — relay rule
    "relay_margin",     # relay needs pot[d] > pot[c] + margin  (champion 0.0)
    "fanout_eps",       # relay needs max_friendly - pot[d] <= eps (champion 0.05)
    # readout_θ — attack gate (attack slot kept iff gate > 0)
    "atk_bias",         # constant                              (champion 1.0 → always)
    "atk_pot",          # × potn[d] (pot normalized to [0,1])   (champion 0.0)
    "atk_weak",         # × (1 - strength[d]/MAX)               (champion 0.0)
    # throttle_θ — slot ranking for the per-cell top-k cap.
    # NOTE: the hand champion's attack-tier ranking is VESTIGIAL — it adds
    # BIG=1e30 to pot[d] in float32, which saturates, so all attack slots tie
    # and its scan picks the lowest slot index (a fixed eastward bias!).
    # Champion-init therefore sets rank_pot_attack=0 (flat attack tier, ties
    # → lowest k) and rank_pot_relay=1 (relay genuinely ranked by pot).
    # Evolution can turn rank_pot_attack into the ranking the original code
    # intended but never actually computed.
    "rank_tier",        # bonus for attack-tier slots           (champion 1000)
    "rank_pot_attack",  # × potn[d] on attack slots             (champion 0.0)
    "rank_pot_relay",   # × potn[d] on relay slots              (champion 1.0)
    "rank_press",       # × edge_pressure[c,k]/MAX_EDGE         (champion 0.0)
    "rank_weak",        # × is_enemy[d]*(1 - strength[d]/MAX)   (champion 0.0)
    # throttle_θ — per-cell outflow cap (rounded, clipped to [1, K])
    "thr_base",         # base count                            (champion 1.0)
    "thr_frontier",     # added on frontier cells               (champion 0.0)
)

GENOME_DIM = len(GENOME_NAMES)
_IDX = {n: i for i, n in enumerate(GENOME_NAMES)}

# Ring 2: Ring 1 plus three opponent-field weights. The opponent field is a
# SECOND sum-Bellman potential built from the enemy bloc's intrinsic (where
# the union of all live non-me seats wants to go). No hand solver uses it.
# The three extra weights are appended after the base 19 so a Ring 2 genome
# reduces EXACTLY to Ring 1 when they are all zero (champion2 = champion ++ 0).
GENOME2_NAMES: tuple[str, ...] = GENOME_NAMES + (
    "w_opp_def",   # defend my cells the enemy field covets (intrinsic, pre-solve)
    "atk_opp",     # attack gate × opp_pot[d] (prefer/avoid contested targets)
    "rank_opp",    # throttle ranking × opp_pot[d]
)
GENOME2_DIM = len(GENOME2_NAMES)
_IDX2 = {n: i for i, n in enumerate(GENOME2_NAMES)}


def champion_vector() -> np.ndarray:
    """The genome that reproduces `lightning_sum_throttled` exactly."""
    v = np.zeros(GENOME_DIM, dtype=np.float64)
    v[_IDX["gamma"]] = 0.85
    v[_IDX["w_weak"]] = 1.0
    v[_IDX["w_expand"]] = 0.6
    v[_IDX["fanout_eps"]] = 0.05
    v[_IDX["atk_bias"]] = 1.0
    v[_IDX["rank_tier"]] = 1000.0
    v[_IDX["rank_pot_relay"]] = 1.0
    v[_IDX["thr_base"]] = 1.0
    return v


def genome_bounds() -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) per-param clip bounds for evolution."""
    lo = np.zeros(GENOME_DIM, dtype=np.float64)
    hi = np.zeros(GENOME_DIM, dtype=np.float64)

    def set_b(name: str, a: float, b: float) -> None:
        lo[_IDX[name]] = a
        hi[_IDX[name]] = b

    set_b("gamma", 0.55, 0.985)
    set_b("w_weak", -1.0, 3.0)
    set_b("w_enemy", -2.0, 2.0)
    set_b("w_expand", -1.0, 3.0)
    set_b("w_defense", -1.0, 3.0)
    set_b("w_enemy_strong", -2.0, 2.0)
    set_b("w_mine", -1.0, 1.0)
    set_b("relay_margin", -0.2, 0.5)
    set_b("fanout_eps", 0.0, 0.6)
    set_b("atk_bias", -2.0, 2.0)
    set_b("atk_pot", -3.0, 3.0)
    set_b("atk_weak", -3.0, 3.0)
    set_b("rank_tier", -1000.0, 1000.0)
    set_b("rank_pot_attack", -3.0, 3.0)
    set_b("rank_pot_relay", -3.0, 3.0)
    set_b("rank_press", -3.0, 3.0)
    set_b("rank_weak", -3.0, 3.0)
    set_b("thr_base", 0.5, 4.49)
    set_b("thr_frontier", -2.0, 3.0)
    return lo, hi


def champion2_vector() -> np.ndarray:
    """Ring 2 champion-init: the Ring 1 champion padded with 3 zero opp
    weights. With the opp weights zero, `field_policy2_actions` skips the
    opponent field entirely and is bit-identical to the champion."""
    v = np.zeros(GENOME2_DIM, dtype=np.float64)
    v[:GENOME_DIM] = champion_vector()
    return v


def genome2_bounds() -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) for the Ring 2 genome: Ring 1 bounds extended with the three
    opponent-weight bounds."""
    lo1, hi1 = genome_bounds()
    lo = np.zeros(GENOME2_DIM, dtype=np.float64)
    hi = np.zeros(GENOME2_DIM, dtype=np.float64)
    lo[:GENOME_DIM] = lo1
    hi[:GENOME_DIM] = hi1
    lo[_IDX2["w_opp_def"]], hi[_IDX2["w_opp_def"]] = -2.0, 2.0
    lo[_IDX2["atk_opp"]], hi[_IDX2["atk_opp"]] = -3.0, 3.0
    lo[_IDX2["rank_opp"]], hi[_IDX2["rank_opp"]] = -3.0, 3.0
    return lo, hi


def describe(vector: np.ndarray) -> str:
    """Compact human-readable genome string (deviations from champion bolded
    by a * marker)."""
    champ = champion_vector()
    parts = []
    for i, name in enumerate(GENOME_NAMES):
        mark = "*" if abs(float(vector[i]) - float(champ[i])) > 1e-9 else ""
        parts.append(f"{name}={float(vector[i]):.3g}{mark}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Field computation (same warm-start semantics as compute_potential)
# ---------------------------------------------------------------------------

_field_warm_cache: dict = {}
# Separate warm-start cache for the opponent field so its solve can't collide
# with the seat's own field cache (same id(neighbors)/seat, different field).
_opp_warm_cache: dict = {}
_WARM_ITERS = 4
_COLD_ITERS = 32
_TOL = 1e-4


def clear_field_caches() -> None:
    """Reset warm-start caches (ours and the solver's) — call between games."""
    _field_warm_cache.clear()
    _opp_warm_cache.clear()
    _pot_warm_cache.clear()


def _potential_from_intrinsic(
    state: State, seat: int, intrinsic: np.ndarray, gamma: float,
    cache: dict = _field_warm_cache, tag: str = "",
) -> np.ndarray:
    # `tag` distinguishes the seat's own field ("") from the opponent field
    # ("opp") so they warm-start from separate caches and never collide.
    cache_key = (id(state.neighbors), int(seat), tag)
    prev = cache.get(cache_key)
    if prev is not None and prev.shape == intrinsic.shape:
        init_pot = prev
        iters = _WARM_ITERS
    else:
        init_pot = intrinsic
        iters = _COLD_ITERS
    pot = _compute_potential_core(
        np.ascontiguousarray(state.owner, dtype=np.int32),
        np.ascontiguousarray(intrinsic, dtype=np.float32),
        np.ascontiguousarray(init_pot, dtype=np.float32),
        np.ascontiguousarray(state.neighbors, dtype=np.int32),
        np.ascontiguousarray(state.edge_pressure, dtype=np.float32),
        _OPPOSITE_SLOT_ARR_VEC,
        float(gamma),
        int(iters),
        float(_TOL),
        1,  # mode_id: sum
        int(DEAD),
    ).astype(np.float32, copy=False)
    if len(cache) > 256:
        cache.clear()
    cache[cache_key] = pot
    return pot


# ---------------------------------------------------------------------------
# Opponent field (Ring 2) — where the enemy bloc wants to go
# ---------------------------------------------------------------------------


def opp_intrinsic(
    state: State, seat: int,
    weak_bonus: float = 1.0, expand_bonus: float = 0.6,
) -> np.ndarray:
    """Enemy-bloc intrinsic for `seat`: the source term for a sum-Bellman
    field representing where the union of all live non-`seat` seats wants to
    flow. My weak cells are their attack targets; neutral cells are their
    expansion targets.

      MY cells:       weak_bonus * clip(1 - my_strength/MAX_STRENGTH, 0, ..)
                      (clipped to [0, weak_bonus] when weak_bonus >= 0)
      NEUTRAL cells:  expand_bonus
      ENEMY/other:    0
      DEAD:           0
    """
    owner = state.owner
    strength = state.strength
    N = state.N
    is_mine = owner == seat
    is_neutral = owner == NEUTRAL
    intrinsic = np.zeros(N, dtype=np.float32)
    if is_mine.any() and weak_bonus != 0.0:
        term = weak_bonus * (1.0 - strength[is_mine] / MAX_STRENGTH)
        if weak_bonus >= 0.0:
            term = term.clip(0.0, weak_bonus)
        intrinsic[is_mine] = term
    intrinsic[is_neutral] = np.float32(expand_bonus)
    # DEAD cells are already 0 and _compute_potential_core forces them to 0.
    return intrinsic


def opp_potential(
    state: State, seat: int, gamma: float,
    weak_bonus: float = 1.0, expand_bonus: float = 0.6,
) -> np.ndarray:
    """Sum-Bellman propagation of `opp_intrinsic`, using the same frozen
    operator as the seat's own field but a SEPARATE warm-start cache (tag
    "opp") so it cannot contaminate the seat's pot trajectory."""
    intrinsic = opp_intrinsic(state, seat, weak_bonus, expand_bonus)
    return _potential_from_intrinsic(
        state, seat, intrinsic, gamma, cache=_opp_warm_cache, tag="opp",
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def field_policy_actions(
    state: State,
    seat: int,
    rng: Optional[np.random.Generator] = None,
    vector: Optional[np.ndarray] = None,
    *,
    opp_pot: Optional[np.ndarray] = None,
    w_opp_def: float = 0.0,
    atk_opp: float = 0.0,
    rank_opp: float = 0.0,
) -> np.ndarray:
    """Ring 1 policy: (N,) int32 actions for `seat` from genome `vector`.

    Ring 2 extension (all optional, fully guarded): `opp_pot` is the enemy
    bloc's normalized sum-Bellman field (see `opp_potential`). When
    `opp_pot is None` OR a given opp weight is 0.0, that code path is
    byte-identical to Ring 1 — so `champion2_vector()` reproduces the
    champion bit-for-bit. The opp weights are passed as kwargs (NOT appended
    to `vector`) so the positional `_IDX` map over GENOME_NAMES is unchanged.
    """
    g = champion_vector() if vector is None else np.asarray(vector, dtype=np.float64)
    owner = state.owner
    strength = state.strength
    N = state.N
    is_mine = owner == seat
    if not is_mine.any():
        return np.full(N, ACTION_NOOP, dtype=np.int32)

    is_enemy = (owner >= 0) & (owner != seat)
    is_neutral = owner == NEUTRAL
    is_dead = owner == DEAD

    # ---- intrinsic_θ ------------------------------------------------------
    # The two champion terms mirror compute_potential's exact float32 ops so
    # the champion genome stays bit-identical; extras are guarded so they
    # contribute nothing (not even rounding) when their weight is zero.
    intrinsic = np.zeros(N, dtype=np.float32)
    w_weak = float(g[_IDX["w_weak"]])
    if is_enemy.any() and w_weak != 0.0:
        term = w_weak * (1.0 - strength[is_enemy] / MAX_STRENGTH)
        if w_weak >= 0.0:
            term = term.clip(0.0, w_weak)
        intrinsic[is_enemy] = term
    intrinsic[is_neutral] = float(g[_IDX["w_expand"]])
    w_enemy = float(g[_IDX["w_enemy"]])
    if w_enemy != 0.0:
        intrinsic[is_enemy] += np.float32(w_enemy)
    w_es = float(g[_IDX["w_enemy_strong"]])
    if w_es != 0.0:
        intrinsic[is_enemy] += (w_es * (strength[is_enemy] / MAX_STRENGTH)).astype(np.float32)
    w_def = float(g[_IDX["w_defense"]])
    if w_def != 0.0 and is_mine.any():
        inbound = _inbound_enemy_pressure(state, seat)
        threat = (inbound / MAX_EDGE).clip(0.0, 1.0)
        intrinsic[is_mine] += (w_def * threat[is_mine]).astype(np.float32)
    w_mine = float(g[_IDX["w_mine"]])
    if w_mine != 0.0:
        intrinsic[is_mine] += np.float32(w_mine)
    # Ring 2: defend cells the enemy bloc covets (added to the intrinsic
    # BEFORE the seat's Bellman solve). Guarded so Ring 1 is untouched.
    if opp_pot is not None and w_opp_def != 0.0 and is_mine.any():
        intrinsic[is_mine] += (w_opp_def * opp_pot[is_mine]).astype(np.float32)
    intrinsic[is_dead] = 0.0

    # ---- Bellman (frozen operator) -----------------------------------------
    pot = _potential_from_intrinsic(state, seat, intrinsic, float(g[_IDX["gamma"]]))

    # ---- slot classification ----------------------------------------------
    nb = state.neighbors
    valid = nb >= 0
    nb_safe = np.where(valid, nb, 0)
    owner_d = owner[nb_safe]
    alive_d = valid & (owner_d != DEAD)
    friendly = alive_d & (owner_d == seat)
    attackable = alive_d & (owner_d != seat)

    pot_d32 = pot[nb_safe]                       # (N, K) float32
    pot_c32 = pot[:, None]
    pot_d = pot_d32.astype(np.float64)
    pmax = float(pot.max())
    potn_d = pot_d / pmax if pmax > 0.0 else np.zeros_like(pot_d)

    s_norm_d = (strength.astype(np.float64) / MAX_STRENGTH)[nb_safe]
    weak_d = 1.0 - s_norm_d

    # ---- readout_θ: relay --------------------------------------------------
    # Comparisons mirror _gradient_relay_core's float32 arithmetic exactly
    # (f32 subtract, then promote to f64 against the eps scalar) so the
    # champion genome reproduces the champion bit-for-bit.
    relay_margin = float(g[_IDX["relay_margin"]])
    fanout_eps = float(g[_IDX["fanout_eps"]])
    if relay_margin == 0.0:
        uphill = pot_d32 > pot_c32
    else:
        uphill = pot_d > pot_c32.astype(np.float64) + relay_margin
    max_friendly32 = np.max(
        np.where(friendly, pot_d32, np.float32(-1.0e30)), axis=1,
    )
    relay = (
        friendly
        & uphill
        & ((max_friendly32[:, None] - pot_d32).astype(np.float64) <= fanout_eps)
    )

    # ---- readout_θ: attack gate --------------------------------------------
    # Ring 2: opp_pot[d] at the neighbor, gathered once and reused below.
    opp_d = None
    if opp_pot is not None and (atk_opp != 0.0 or rank_opp != 0.0):
        opp_d = opp_pot[nb_safe].astype(np.float64)  # (N, K)
    gate = (
        float(g[_IDX["atk_bias"]])
        + float(g[_IDX["atk_pot"]]) * potn_d
        + float(g[_IDX["atk_weak"]]) * weak_d
    )
    if opp_d is not None and atk_opp != 0.0:
        gate = gate + atk_opp * opp_d
    attack = attackable & (gate > 0.0)

    desired = attack | relay

    # ---- throttle_θ ---------------------------------------------------------
    frontier = attackable.any(axis=1)
    count = np.clip(
        np.rint(
            float(g[_IDX["thr_base"]])
            + float(g[_IDX["thr_frontier"]]) * frontier.astype(np.float64)
        ),
        1, K,
    ).astype(np.int64)
    need = desired.sum(axis=1)
    if (need > count).any():
        enemy_d = alive_d & (owner_d >= 0) & (owner_d != seat)
        score = (
            float(g[_IDX["rank_tier"]]) * attack.astype(np.float64)
            + float(g[_IDX["rank_pot_attack"]]) * potn_d * attack
            + float(g[_IDX["rank_pot_relay"]]) * potn_d * relay
            + float(g[_IDX["rank_press"]])
            * (state.edge_pressure.astype(np.float64) / MAX_EDGE)
            + float(g[_IDX["rank_weak"]]) * weak_d * enemy_d.astype(np.float64)
        )
        # Ring 2: bias the throttle ranking toward/away from cells the enemy
        # bloc covets. Guarded so Ring 1's score is byte-identical.
        if opp_d is not None and rank_opp != 0.0:
            score = score + rank_opp * opp_d
        score = np.where(desired, score, -np.inf)
        order = np.argsort(-score, axis=1, kind="stable")
        rank = np.empty_like(order)
        np.put_along_axis(
            rank, order,
            np.broadcast_to(np.arange(K, dtype=order.dtype), (N, K)).copy(),
            axis=1,
        )
        desired = desired & (rank < count[:, None])

    cur = state.outflow.astype(np.bool_)
    return _actions_from_desired(is_mine, desired, cur, rng)


def make_field_solver(vector: np.ndarray):
    """Wrap a genome vector as a solver(state, seat, rng) callable."""
    vec = np.asarray(vector, dtype=np.float64).copy()

    def _solver(state: State, seat: int, rng=None) -> np.ndarray:
        return field_policy_actions(state, seat, rng=rng, vector=vec)

    return _solver


# ---------------------------------------------------------------------------
# Ring 2 policy — Ring 1 plus opponent-field awareness
# ---------------------------------------------------------------------------


def field_policy2_actions(
    state: State,
    seat: int,
    rng: Optional[np.random.Generator] = None,
    vector: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Ring 2 policy: (N,) int32 actions for `seat` from a GENOME2 vector.

    Splits the vector into the 19 base Ring 1 params + 3 opponent params.
    The opponent field is computed ONLY if at least one opp param is nonzero
    (otherwise opp_pot=None for zero cost and bit-exact Ring 1 behavior).
    """
    g = champion2_vector() if vector is None else np.asarray(vector, dtype=np.float64)
    base = g[:GENOME_DIM]
    w_opp_def = float(g[_IDX2["w_opp_def"]])
    atk_opp = float(g[_IDX2["atk_opp"]])
    rank_opp = float(g[_IDX2["rank_opp"]])

    opp_pot = None
    if w_opp_def != 0.0 or atk_opp != 0.0 or rank_opp != 0.0:
        gamma = float(base[_IDX["gamma"]])
        weak_bonus = float(base[_IDX["w_weak"]])
        expand_bonus = float(base[_IDX["w_expand"]])
        pot = opp_potential(state, seat, gamma, weak_bonus, expand_bonus)
        pmax = float(pot.max())
        opp_pot = (pot / pmax) if pmax > 0.0 else np.zeros_like(pot)

    return field_policy_actions(
        state, seat, rng=rng, vector=base,
        opp_pot=opp_pot, w_opp_def=w_opp_def, atk_opp=atk_opp, rank_opp=rank_opp,
    )


def make_field2_solver(vector: np.ndarray):
    """Wrap a GENOME2 vector as a solver(state, seat, rng) callable."""
    vec = np.asarray(vector, dtype=np.float64).copy()

    def _solver(state: State, seat: int, rng=None) -> np.ndarray:
        return field_policy2_actions(state, seat, rng=rng, vector=vec)

    return _solver


__all__ = [
    "GENOME_NAMES",
    "GENOME_DIM",
    "champion_vector",
    "genome_bounds",
    "describe",
    "clear_field_caches",
    "field_policy_actions",
    "make_field_solver",
    # Ring 2
    "GENOME2_NAMES",
    "GENOME2_DIM",
    "champion2_vector",
    "genome2_bounds",
    "opp_intrinsic",
    "opp_potential",
    "field_policy2_actions",
    "make_field2_solver",
]
