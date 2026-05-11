"""Parity scenario: deterministic 100-tick run with NN-driven actions.

Prints state hashes every 10 ticks. The JS counterpart at
`python/tests/dump_reference.ts` runs the IDENTICAL scenario and must produce
the same hashes.

Run:
    uv run python tests/test_parity.py
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path

# Allow running as a script from the python/ dir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flux import (
    make_initial_state,
    step,
    apply_action,
    mulberry32,
    random_genome,
    build_neighbor_table,
    ai_think,
)


def state_hash(state) -> str:
    h = hashlib.sha256()
    h.update(b"NODES")
    for n in state.nodes:
        # JSON-style with explicit owner sentinel -1.
        owner = -1 if n.owner is None else n.owner
        s_bytes = repr(float(n.strength)).encode()
        h.update(f"{n.id}|{owner}|".encode())
        h.update(s_bytes)
        h.update(b";")
    h.update(b"FLOWS")
    for f in state.flows:
        h.update(f"{f.src}-{f.dst}-{f.player};".encode())
    h.update(b"TICK")
    h.update(str(state.tick).encode())
    return h.hexdigest()[:16]


def run(verbose: bool = True) -> list[tuple[int, str]]:
    state = make_initial_state(radius=9, distance=2, num_players=4)
    rng = mulberry32(42)
    # Larger std to push the NN past the trivial-no-op regime so it produces
    # diverse argmax actions and exercises full step + flow lifecycle.
    genome = random_genome(rng, std=2.0)
    nb = build_neighbor_table(state)

    results: list[tuple[int, str]] = []
    # Initial hash
    h = state_hash(state)
    if verbose:
        print(f"tick 0   hash {h}")
    results.append((0, h))

    # A couple of manual seed actions to exercise apply_action toggle logic.
    # Player 0 owns cell 10 at start; edges (10,0), (10,1), (10,11), (10,21)
    # exist. We force a manual flow before the NN takes over so step processes
    # a real flow on tick 1.
    from flux import Action  # noqa: WPS433

    seed_actions = [
        Action(kind="toggleFlow", src=10, dst=11, player=0),
        Action(kind="toggleFlow", src=10, dst=21, player=0),
    ]
    for a in seed_actions:
        state = apply_action(state, a)

    for t in range(1, 101):
        # Each tick: each player picks actions via NN, applies them, then step.
        for p in range(state.num_players):
            actions = ai_think(state, p, genome, nb)
            for a in actions:
                state = apply_action(state, a)
        # On tick 50, manually toggle a flow to exercise the apply_action path.
        if t == 50:
            state = apply_action(
                state, Action(kind="toggleFlow", src=10, dst=11, player=0)
            )
        state = step(state, 0.1)
        if t % 10 == 0:
            h = state_hash(state)
            if verbose:
                print(f"tick {t:3d} hash {h}")
            results.append((t, h))

    return results


if __name__ == "__main__":
    run()
