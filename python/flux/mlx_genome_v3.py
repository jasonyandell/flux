"""V3 NN: 2-layer GCN-style graph neural network policy.

Architecture: per-(seat, cell) input → message-passing layer 1 → ReLU →
message-passing layer 2 → ReLU → per-cell linear → 19 action logits.

Each message-passing layer:
    H'[c] = relu(W_self @ H[c] + W_neigh @ mean(H[neighbor] for neighbor in cell.neighbors) + b)

The "neighbor" set is the game's distance-≤2 graph (18 cells, padded with -1
for boundaries — same as v1's NEIGHBOR_STRIDE). The point of v3 isn't a wider
receptive field per se (v2 already gave us that). It's *coordination*: two
rounds of MP let cell c's representation absorb info that flowed through its
neighbors, which is the structural fingerprint of chains and loops in this
game (a chain of 3 friendlies sharing flows is a 2-hop coordination pattern).

Per-(game, seat, cell) features (4 channels):
    [strength/MAX, is_mine, is_enemy, is_neutral_or_empty]

Weights laid out flat in a per-genome float32 vector (length
WEIGHTS_PER_GENOME_V3 = 2995). Same layout pattern as v1/v2.
"""
from __future__ import annotations


# Feature / layer sizes.
NN_IN_DIM_V3 = 4                 # strength_norm + is_mine + is_enemy + is_neutral
NN_HIDDEN_V3 = 32
NN_OUT_DIM_V3 = 19               # same action space (18 graph neighbors + noop)

# Slice offsets for unpacking the flat genome.
_W1_SELF_LEN = NN_IN_DIM_V3 * NN_HIDDEN_V3              # 128
_W1_NEIGH_LEN = NN_IN_DIM_V3 * NN_HIDDEN_V3             # 128
_B1_LEN = NN_HIDDEN_V3                                   # 32
_W2_SELF_LEN = NN_HIDDEN_V3 * NN_HIDDEN_V3              # 1024
_W2_NEIGH_LEN = NN_HIDDEN_V3 * NN_HIDDEN_V3             # 1024
_B2_LEN = NN_HIDDEN_V3                                   # 32
_W_OUT_LEN = NN_HIDDEN_V3 * NN_OUT_DIM_V3               # 608
_B_OUT_LEN = NN_OUT_DIM_V3                               # 19

OFF_W1_SELF = 0
OFF_W1_NEIGH = OFF_W1_SELF + _W1_SELF_LEN
OFF_B1 = OFF_W1_NEIGH + _W1_NEIGH_LEN
OFF_W2_SELF = OFF_B1 + _B1_LEN
OFF_W2_NEIGH = OFF_W2_SELF + _W2_SELF_LEN
OFF_B2 = OFF_W2_NEIGH + _W2_NEIGH_LEN
OFF_W_OUT = OFF_B2 + _B2_LEN
OFF_B_OUT = OFF_W_OUT + _W_OUT_LEN

WEIGHTS_PER_GENOME_V3 = OFF_B_OUT + _B_OUT_LEN          # 2995
