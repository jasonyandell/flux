"""MLX implementation of the per-cell NN forward pass.

Same architecture as the NumPy reference (91 → 32 tanh → 19, ~3.5k weights),
but vectorized over all cells in a single matmul. Float32 throughout (MLX
default). No parity claim against NumPy/JS — under the new architecture the
browser plays recorded replays, not deterministic re-simulations.
"""
from __future__ import annotations

import mlx.core as mx

from .state import MAX_STRENGTH

NN_IN_DIM = 91
NN_HIDDEN = 32
NN_OUT_DIM = 19
NEIGHBOR_STRIDE = 18

WEIGHTS_PER_GENOME = (
    NN_IN_DIM * NN_HIDDEN
    + NN_HIDDEN
    + NN_HIDDEN * NN_OUT_DIM
    + NN_OUT_DIM
)  # 3571

W1_OFF = 0
B1_OFF = NN_IN_DIM * NN_HIDDEN
W2_OFF = B1_OFF + NN_HIDDEN
B2_OFF = W2_OFF + NN_HIDDEN * NN_OUT_DIM


def unpack_weights(genome: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Slice the flat genome into (W1, b1, W2, b2)."""
    W1 = genome[W1_OFF:B1_OFF].reshape(NN_IN_DIM, NN_HIDDEN)
    b1 = genome[B1_OFF:W2_OFF]
    W2 = genome[W2_OFF:B2_OFF].reshape(NN_HIDDEN, NN_OUT_DIM)
    b2 = genome[B2_OFF:]
    return W1, b1, W2, b2


def build_features(
    owner: mx.array,      # (N,) int32, -1 = neutral
    strength: mx.array,   # (N,) float32
    neighbors: mx.array,  # (N * NEIGHBOR_STRIDE,) int32, -1 for empty slot
    player: int,
) -> mx.array:
    """Returns (N, 91) feature tensor matching the NumPy reference layout:
       [0]      own_strength / MAX
       [1..90]  18 neighbor groups of 5: (nb_strength/MAX, is_friend_a,
                is_friend_b, is_enemy, is_neutral)
    Empty slots and out-of-bounds neighbors are treated as neutral with
    nb_strength = 0.
    """
    N = strength.shape[0]
    self_feat = (strength / MAX_STRENGTH).reshape(N, 1)

    nb_ids = neighbors.reshape(N, NEIGHBOR_STRIDE)
    nb_valid = nb_ids >= 0
    safe_ids = mx.maximum(nb_ids, 0)
    nb_strength = strength[safe_ids] / MAX_STRENGTH
    nb_strength = mx.where(nb_valid, nb_strength, mx.zeros_like(nb_strength))
    nb_owner = owner[safe_ids]
    is_friend = nb_valid & (nb_owner == player)
    is_enemy = nb_valid & (nb_owner != player) & (nb_owner != -1)
    # Neutral slot: empty OR neutral cell (owner == -1).
    is_neutral = ~nb_valid | (nb_owner == -1)
    fa = is_friend.astype(mx.float32)
    fb = fa  # NumPy reference duplicates the friend channel (intentional artifact)
    en = is_enemy.astype(mx.float32)
    nu = is_neutral.astype(mx.float32)
    # Stack per-neighbor 5-tuple along a new axis, then flatten the (STRIDE, 5)
    # block into 90 features.
    nb_block = mx.stack([nb_strength, fa, fb, en, nu], axis=2)  # (N, STRIDE, 5)
    nb_flat = nb_block.reshape(N, NEIGHBOR_STRIDE * 5)
    return mx.concatenate([self_feat, nb_flat], axis=1)  # (N, 91)


def nn_infer_all_cells(
    genome: mx.array,
    owner: mx.array,
    strength: mx.array,
    neighbors: mx.array,
    player: int,
) -> mx.array:
    """Forward pass for every cell on the board for one player.
    Returns (N,) int32 — argmax action index (0..17 = neighbor slot, 18 = noop).
    """
    W1, b1, W2, b2 = unpack_weights(genome)
    features = build_features(owner, strength, neighbors, player)  # (N, 91)
    h = mx.tanh(features @ W1 + b1)                                # (N, 32)
    out = h @ W2 + b2                                              # (N, 19)
    return out.argmax(axis=1).astype(mx.int32)


def random_genome(rng_key: mx.array, std: float = 0.5) -> mx.array:
    """Sample a fresh genome from N(0, std^2). Float32."""
    return mx.random.normal(shape=(WEIGHTS_PER_GENOME,), scale=std, key=rng_key)
