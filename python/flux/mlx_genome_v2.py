"""V2 NN: 3-hop vision (36 neighbor slots), 181 → 32 → 19.

Same hidden width and output space as v1 — the only structural change is the
wider input. Output is still 18 graph-neighbor flow targets + 1 noop; flows
travel via the existing graph edges (hex-distance ≤ 2), not the vision ball.

Genome layout: flat float32, length WEIGHTS_PER_GENOME_V2 = 6451.
"""
from __future__ import annotations

from .vision import STRIDE_V2

NN_IN_DIM_V2 = 1 + STRIDE_V2 * 5      # 181
NN_HIDDEN_V2 = 32
NN_OUT_DIM_V2 = 19

WEIGHTS_PER_GENOME_V2 = (
    NN_IN_DIM_V2 * NN_HIDDEN_V2
    + NN_HIDDEN_V2
    + NN_HIDDEN_V2 * NN_OUT_DIM_V2
    + NN_OUT_DIM_V2
)  # 181*32 + 32 + 32*19 + 19 = 6451

W1_OFF_V2 = 0
B1_OFF_V2 = NN_IN_DIM_V2 * NN_HIDDEN_V2
W2_OFF_V2 = B1_OFF_V2 + NN_HIDDEN_V2
B2_OFF_V2 = W2_OFF_V2 + NN_HIDDEN_V2 * NN_OUT_DIM_V2
