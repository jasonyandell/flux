"""1-step K-candidate lookahead action selection.

A pragmatic stand-in for MCTS that fits the existing rollout loop with minimal
plumbing. At each AI tick:

  1. Forward the current policy to get logits at the current state.
  2. Sample K joint actions from the policy (Gumbel-max).
  3. For each candidate, simulate one AI-period of step_batched_regen
     (assuming all other seats also use this candidate — i.e. the seats are
     evaluated as a cooperative team for this one tick of lookahead).
  4. Evaluate the value head at each candidate's post-state.
  5. Per game, pick the candidate with the highest sum-of-seat values.

The policy's *log-prob of the chosen action* (under the original logits) is
recorded as the behaviour-policy log_prob, so PPO can train as usual with
the off-policy lookahead acting as the data-collection policy.

Cost: K candidate forwards + K simulations per AI tick. With K=4, that's a
~4× slowdown on the rollout phase. Update phase is unchanged.
"""
from __future__ import annotations

import mlx.core as mx

from .mlx_batch import build_flows_from_actions
from .mlx_step_regen import step_batched_regen


def _sample_one(logits: mx.array, rng_key: mx.array) -> tuple[mx.array, mx.array]:
    """One Gumbel-max sample from logits. Returns (actions, new_rng)."""
    rng_key, sub = mx.random.split(rng_key)
    gumbel = -mx.log(-mx.log(mx.random.uniform(shape=logits.shape, key=sub) + 1e-9) + 1e-9)
    actions = (logits + gumbel).argmax(axis=-1).astype(mx.int32)
    return actions, rng_key


def lookahead_action_select(
    model,
    owner: mx.array,           # (G, N) int32
    strength: mx.array,        # (G, N) float32
    passthrough: mx.array,     # (G, N) float32
    neighbors: mx.array,       # (N * STRIDE,) int32
    dead_mask: mx.array,       # (G, N) bool
    live_mask: mx.array,       # (G,) bool
    num_players: int,
    rng_key: mx.array,
    K: int = 4,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Pick K candidate actions, simulate one step, return best per game.

    Returns:
      best_actions:  (G, S, N) int32 — chosen joint action per game
      root_logits:   (G, S, N, A) float32 — policy logits at the *current* state
                     (so callers can compute log-probs under the behaviour policy)
      root_value:    (G, S) float32 — value head at the current state
      rng_key:       updated rng key
    """
    G, N = owner.shape
    P = num_players

    # Root forward — we need both logits (for sampling + log-prob) and value
    # (for the existing rollout to log).
    root_logits, root_value = model(owner, strength, neighbors, P, dead_mask)
    mx.eval(root_logits, root_value)

    # Sample K candidate joint actions.
    candidates: list[mx.array] = []
    for _ in range(K):
        act_k, rng_key = _sample_one(root_logits, rng_key)
        candidates.append(act_k)

    # For each candidate, simulate one step and evaluate the post-state value.
    cand_values: list[mx.array] = []
    for act_k in candidates:
        fs, fd, fp, fv = build_flows_from_actions(act_k, owner, neighbors, dead_mask)
        new_owner, new_strength, *_ = step_batched_regen(
            owner, strength, fs, fd, fp, fv,
            live_mask, P, 0.1, passthrough, neighbors,
        )
        # One-step value: sum across seats. (For self-play we'd want per-seat
        # selection, but this is the simpler team-utilitarian version.)
        _, v = model(new_owner, new_strength, neighbors, P, dead_mask)
        cand_values.append(v.sum(axis=-1))  # (G,)

    # Stack: (K, G) — pick best k per game.
    values_stacked = mx.stack(cand_values, axis=0)
    best_k = values_stacked.argmax(axis=0)                        # (G,)

    # Gather the corresponding action per game. mx.take_along_axis on the K
    # dim needs the index broadcast to (1, G, S, N).
    actions_stacked = mx.stack(candidates, axis=0)                # (K, G, S, N)
    S = actions_stacked.shape[2]
    idx = best_k.reshape(1, G, 1, 1)
    idx = mx.broadcast_to(idx, (1, G, S, N)).astype(mx.int32)
    best_actions = mx.take_along_axis(actions_stacked, idx, axis=0).reshape(G, S, N)

    return best_actions.astype(mx.int32), root_logits, root_value, rng_key
