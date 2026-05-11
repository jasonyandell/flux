from __future__ import annotations
import math
from typing import Callable, List, Tuple
import numpy as np
from .state import Action, GameState, MAX_STRENGTH, Flow

NN_IN_DIM = 91
NN_HIDDEN = 32
NN_OUT_DIM = 19
NEIGHBOR_STRIDE = 18

WEIGHTS_PER_GENOME = (
    NN_IN_DIM * NN_HIDDEN + NN_HIDDEN + NN_HIDDEN * NN_OUT_DIM + NN_OUT_DIM
)
# = 91*32 + 32 + 32*19 + 19 = 3571

Genome = np.ndarray  # float32 array, length WEIGHTS_PER_GENOME


def build_neighbor_table(state: GameState) -> np.ndarray:
    """Per-cell neighbor index table sorted by (x, y) position for determinism.
    Length cells * NEIGHBOR_STRIDE. -1 = empty slot. Returns int32 array.
    """
    n_nodes = len(state.nodes)
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for e in state.edges:
        adj[e.a].append(e.b)
        adj[e.b].append(e.a)

    out = np.full(n_nodes * NEIGHBOR_STRIDE, -1, dtype=np.int32)
    for i in range(n_nodes):
        lst = list(adj[i])
        # JS: list.sort((a,b) => pa.x !== pb.x ? pa.x - pb.x : pa.y - pb.y)
        # Stable sort by (x, y).
        lst.sort(key=lambda nid: (state.nodes[nid].pos.x, state.nodes[nid].pos.y))
        for j in range(min(len(lst), NEIGHBOR_STRIDE)):
            out[i * NEIGHBOR_STRIDE + j] = lst[j]
    return out


def gaussian(rand: Callable[[], float]) -> float:
    """Box-Muller, mean=0 std=1. Pure. Result is an f64 Python float."""
    u = 0.0
    v = 0.0
    while u == 0.0:
        u = rand()
    while v == 0.0:
        v = rand()
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def random_genome(rand: Callable[[], float], std: float = 0.5) -> Genome:
    """Returns a Float32Array-equivalent genome. Each weight is f32-rounded
    (matches JS `g[i] = gaussian(rand) * std` into a Float32Array)."""
    g = np.empty(WEIGHTS_PER_GENOME, dtype=np.float32)
    for i in range(WEIGHTS_PER_GENOME):
        g[i] = gaussian(rand) * std  # numpy rounds the f64 RHS to f32 on store
    return g


def mutate(parent: Genome, rand: Callable[[], float], sigma: float = 0.05) -> Genome:
    child = np.empty(parent.shape, dtype=np.float32)
    for i in range(parent.size):
        child[i] = float(parent[i]) + gaussian(rand) * sigma
    return child


def nn_infer_cell(
    genome: Genome,
    state: GameState,
    cell_id: int,
    player: int,
    neighbors: np.ndarray,
) -> int:
    """Forward pass for a single cell. Bit-exact match for the JS Float32Array
    semantics: h and out are float32, genome reads round-trip through f32, but
    each accumulation expression evaluates in f64 (Python float) before being
    stored back to float32."""
    W1off = 0
    b1off = NN_IN_DIM * NN_HIDDEN
    W2off = b1off + NN_HIDDEN
    b2off = W2off + NN_HIDDEN * NN_OUT_DIM

    h = np.empty(NN_HIDDEN, dtype=np.float32)
    for j in range(NN_HIDDEN):
        h[j] = genome[b1off + j]  # already f32

    f0 = state.nodes[cell_id].strength / MAX_STRENGTH  # f64
    for j in range(NN_HIDDEN):
        # JS: h[j] += f0 * genome[W1off + j]
        # The RHS multiplication is f64 * f64_from_f32; sum with current f32-read.
        h[j] = float(h[j]) + f0 * float(genome[W1off + j])

    for n in range(NEIGHBOR_STRIDE):
        nb_id = int(neighbors[cell_id * NEIGHBOR_STRIDE + n])
        s_feat = 0.0
        is_friend_a = 0.0
        is_friend_b = 0.0
        is_enemy = 0.0
        is_neutral = 0.0
        if nb_id >= 0:
            s_feat = state.nodes[nb_id].strength / MAX_STRENGTH
            o = state.nodes[nb_id].owner
            if o == player:
                is_friend_a = 1.0
                is_friend_b = 1.0
            elif o is None:
                is_neutral = 1.0
            else:
                is_enemy = 1.0
        else:
            is_neutral = 1.0
        feat_base = 1 + n * 5
        for j in range(NN_HIDDEN):
            # Match JS sum order exactly:
            #   sFeat*w0 + isFriendA*w1 + isFriendB*w2 + isEnemy*w3 + isNeutral*w4
            # Evaluated left-to-right in f64, then added to h[j] (which is read
            # from f32 → f64), final result stored back as f32.
            w0 = float(genome[W1off + (feat_base + 0) * NN_HIDDEN + j])
            w1 = float(genome[W1off + (feat_base + 1) * NN_HIDDEN + j])
            w2 = float(genome[W1off + (feat_base + 2) * NN_HIDDEN + j])
            w3 = float(genome[W1off + (feat_base + 3) * NN_HIDDEN + j])
            w4 = float(genome[W1off + (feat_base + 4) * NN_HIDDEN + j])
            add = (
                s_feat * w0
                + is_friend_a * w1
                + is_friend_b * w2
                + is_enemy * w3
                + is_neutral * w4
            )
            h[j] = float(h[j]) + add

    for j in range(NN_HIDDEN):
        x = float(h[j])
        e1 = math.exp(x)
        e2 = math.exp(-x)
        h[j] = (e1 - e2) / (e1 + e2)

    out = np.empty(NN_OUT_DIM, dtype=np.float32)
    for k in range(NN_OUT_DIM):
        out[k] = genome[b2off + k]
    for j in range(NN_HIDDEN):
        hj = float(h[j])
        for k in range(NN_OUT_DIM):
            out[k] = float(out[k]) + hj * float(genome[W2off + j * NN_OUT_DIM + k])

    best_k = 0
    best_v = float(out[0])
    for k in range(1, NN_OUT_DIM):
        v = float(out[k])
        if v > best_v:
            best_v = v
            best_k = k
    return best_k


def ai_think(
    state: GameState,
    player: int,
    genome: Genome,
    neighbors: np.ndarray,
) -> List[Action]:
    """Port of `aiThink` from src/gpu/evolved.ts (genome-injected variant)."""
    actions: List[Action] = []
    desired_by_src: dict[int, int] = {}
    for cell in range(len(state.nodes)):
        if state.nodes[cell].owner != player:
            continue
        a = nn_infer_cell(genome, state, cell, player, neighbors)
        if a < 0 or a >= NEIGHBOR_STRIDE:
            continue
        dst = int(neighbors[cell * NEIGHBOR_STRIDE + a])
        if dst < 0:
            continue
        desired_by_src[cell] = dst

    # Cancel existing flows from this player whose dst doesn't match desired;
    # drop matching ones from desired map to avoid double-toggling.
    for f in state.flows:
        if f.player != player:
            continue
        desired = desired_by_src.get(f.src)
        if desired is None or desired != f.dst:
            actions.append(Action(kind="toggleFlow", src=f.src, dst=f.dst, player=player))
        else:
            del desired_by_src[f.src]

    # Add remaining desired flows (JS Map preserves insertion order, Python dict same).
    for src, dst in desired_by_src.items():
        actions.append(Action(kind="toggleFlow", src=src, dst=dst, player=player))

    return actions
