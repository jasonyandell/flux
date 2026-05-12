"""Evolution loop for flux genomes on MLX.

Dumb-first variant: one game at a time, randomized seat roster from the
population each game, fitness averaged over plays. Selection happens in
batches every `selection_every_games` games. No batching across games yet —
the full-sized game already takes "every few seconds" on Apple Silicon, which
is the user-facing replay drop cadence.

Future paths from here:
- Batch games across the population for higher throughput
- Swap NEAT mutation for PPO / policy gradient (game_loop.play_game stays as
  the rollout interface; only the update step changes)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from .game_loop import GameConfig, RecordConfig, play_batch_games, weights_per_genome
from .replay import ReplayHeader, ReplayWriter
from .state import GameState


@dataclass
class EvolutionConfig:
    pop_size: int = 24
    games_per_batch: int = 24            # G parallel games per evolution step
    init_std: float = 0.5
    mutation_sigma: float = 0.05
    elites: int = 6
    tournament_k: int = 3
    selection_every_batches: int = 1     # selection step every N batches
    min_plays_for_selection: int = 1
    game: GameConfig = field(default_factory=GameConfig)


@dataclass
class EvolutionState:
    population: list[mx.array]            # length pop_size, each (W,)
    fitness: np.ndarray                   # (pop_size,) running mean
    plays: np.ndarray                     # (pop_size,) games played
    generation: int = 0
    games_played: int = 0
    best_fitness: float = float("-inf")
    all_time_best_fitness: float = float("-inf")
    all_time_best_genome: Optional[mx.array] = None


def make_initial_evolution_state(
    config: EvolutionConfig, rng_seed: int
) -> EvolutionState:
    W = weights_per_genome(config.game.model_kind)
    key = mx.random.key(rng_seed)
    pop: list[mx.array] = []
    for _ in range(config.pop_size):
        key, sub = mx.random.split(key)
        g = mx.random.normal(shape=(W,), scale=config.init_std, key=sub)
        mx.eval(g)
        pop.append(g)
    return EvolutionState(
        population=pop,
        fitness=np.full(config.pop_size, -np.inf, dtype=np.float64),
        plays=np.zeros(config.pop_size, dtype=np.int64),
    )


def _tournament_pick(rng: np.random.Generator, fitness: np.ndarray, k: int) -> int:
    """Return the index of the best of k random picks (with replacement)."""
    idxs = rng.integers(0, len(fitness), size=k)
    best = int(idxs[0])
    for i in idxs[1:]:
        if fitness[int(i)] > fitness[best]:
            best = int(i)
    return best


def _mutate(parent: mx.array, sigma: float, rng_key: mx.array) -> tuple[mx.array, mx.array]:
    rng_key, sub = mx.random.split(rng_key)
    noise = mx.random.normal(shape=parent.shape, scale=sigma, key=sub)
    child = parent + noise
    mx.eval(child)
    return child, rng_key


def _selection_step(
    state: EvolutionState,
    config: EvolutionConfig,
    rng: np.random.Generator,
    rng_key: mx.array,
) -> mx.array:
    """In-place: replace the bottom-k of the population with mutated tournament
    winners. Resets fitness/plays for replaced slots so they get fresh samples.
    """
    P = config.pop_size
    eligible = state.plays >= config.min_plays_for_selection
    if not np.any(eligible):
        return rng_key
    ranked = np.argsort(-state.fitness)  # best first
    keep = max(config.elites, 1)
    elites = ranked[:keep].tolist()
    targets = [int(i) for i in ranked[keep:] if int(i) in set(range(P)) - set(elites)]
    if not targets:
        return rng_key
    for slot in targets:
        parent_idx = _tournament_pick(rng, state.fitness, config.tournament_k)
        child, rng_key = _mutate(state.population[parent_idx], config.mutation_sigma, rng_key)
        state.population[slot] = child
        state.fitness[slot] = -np.inf
        state.plays[slot] = 0
    state.generation += 1
    return rng_key


def run_one_batch(
    state: EvolutionState,
    config: EvolutionConfig,
    rng: np.random.Generator,
    record_game_idx: Optional[int] = None,
    record: Optional[RecordConfig] = None,
    hand_coded_seats: Optional[dict[int, str]] = None,
) -> tuple[list[list[int | str]], list[GameState], list[np.ndarray], list[int]]:
    """Sample G rosters from the population, play them in parallel, fold all
    per-seat fitness back into running means.

    Returns (seat_ids_per_game, frames, final_cells_per_game, decision_ticks).
    `frames` is the recorded game's frame list (empty if record_game_idx is None).
    """
    P = config.pop_size
    G = config.games_per_batch
    S = config.game.num_players
    hand_coded_seats = hand_coded_seats or {}

    seat_ids_per_game: list[list[int | str]] = []
    rosters: list[list] = []
    for g in range(G):
        seat_ids: list[int | str] = []
        roster: list = []
        for seat in range(S):
            if seat in hand_coded_seats:
                name = hand_coded_seats[seat]
                seat_ids.append(name)
                roster.append(name)
            else:
                gid = int(rng.integers(0, P))
                seat_ids.append(gid)
                roster.append(state.population[gid])
        seat_ids_per_game.append(seat_ids)
        rosters.append(roster)

    results = play_batch_games(
        rosters, config.game,
        record_game_idx=record_game_idx, record=record,
    )
    state.games_played += G

    # Aggregate per-genome fitness across all G games.
    by_genome: dict[int, list[float]] = {}
    for g, ids in enumerate(seat_ids_per_game):
        for seat, sid in enumerate(ids):
            if isinstance(sid, str):
                continue
            by_genome.setdefault(sid, []).append(float(results[g].fitness[seat]))
    for gid, vals in by_genome.items():
        mean_fit = float(np.mean(vals))
        prev_plays = int(state.plays[gid])
        if prev_plays == 0 or state.fitness[gid] == -np.inf:
            state.fitness[gid] = mean_fit
        else:
            state.fitness[gid] = (state.fitness[gid] * prev_plays + mean_fit) / (prev_plays + 1)
        state.plays[gid] += 1

    # Update best.
    best_idx = int(np.argmax(state.fitness))
    state.best_fitness = float(state.fitness[best_idx])
    if state.best_fitness > state.all_time_best_fitness:
        state.all_time_best_fitness = state.best_fitness
        state.all_time_best_genome = state.population[best_idx]

    frames = results[record_game_idx].frames if record_game_idx is not None else []
    final_cells_per_game = [r.final_cells for r in results]
    decision_ticks = [r.decision_tick for r in results]
    return seat_ids_per_game, frames, final_cells_per_game, decision_ticks


def write_replay(
    out_path: Path,
    frames: list[GameState],
    config: EvolutionConfig,
    record: RecordConfig,
    metadata: dict,
) -> int:
    """Write frames as a replay. Returns the file size."""
    if not frames:
        return 0
    header = ReplayHeader(
        radius=config.game.radius,
        distance=config.game.distance,
        num_players=config.game.num_players,
        num_nodes=len(frames[0].nodes),
        tick_stride=record.tick_stride,
        dt_per_tick_ms=record.dt_per_tick_ms,
        metadata=metadata,
    )
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        w = ReplayWriter(f, header)
        for fr in frames:
            w.write_frame(fr)
        w.close()
    import os
    os.replace(tmp, out_path)
    return out_path.stat().st_size


def prepare_checkpoint_payload(state: EvolutionState) -> dict:
    """Snapshot the live state into plain numpy arrays. Cheap — main-thread
    safe. The returned dict is safe to hand to a background thread for I/O,
    because subsequent main-thread mutations to `state` won't touch these
    arrays."""
    weights = np.stack([np.array(g, copy=False) for g in state.population], axis=0)
    payload: dict = dict(
        weights=weights,
        fitness=state.fitness.copy(),
        plays=state.plays.copy(),
        generation=np.int64(state.generation),
        games_played=np.int64(state.games_played),
        best_fitness=np.float64(state.best_fitness),
        all_time_best_fitness=np.float64(state.all_time_best_fitness),
    )
    if state.all_time_best_genome is not None:
        payload["all_time_best_genome"] = np.array(
            state.all_time_best_genome, copy=False
        )
    return payload


def write_checkpoint_payload(payload: dict, path: Path) -> None:
    """I/O only. Safe to call from a background thread."""
    assert path.suffix == ".npz", "checkpoint path must end in .npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def save_checkpoint(state: EvolutionState, path: Path) -> None:
    """Synchronous save. Equivalent to prepare + write_checkpoint_payload."""
    write_checkpoint_payload(prepare_checkpoint_payload(state), path)


def load_checkpoint(path: Path) -> Optional[EvolutionState]:
    """Load evolution state from a previous save. Returns None if missing."""
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    weights = data["weights"]  # (P, W)
    pop: list[mx.array] = []
    for i in range(weights.shape[0]):
        g = mx.array(weights[i].astype(np.float32))
        mx.eval(g)
        pop.append(g)
    all_time = None
    if "all_time_best_genome" in data.files:
        all_time = mx.array(data["all_time_best_genome"].astype(np.float32))
        mx.eval(all_time)
    return EvolutionState(
        population=pop,
        fitness=data["fitness"].astype(np.float64),
        plays=data["plays"].astype(np.int64),
        generation=int(data["generation"]),
        games_played=int(data["games_played"]),
        best_fitness=float(data["best_fitness"]),
        all_time_best_fitness=float(data["all_time_best_fitness"]),
        all_time_best_genome=all_time,
    )


def save_champion_json(
    genome: mx.array,
    path: Path,
    generation: int,
    best_fitness: float,
    note: str = "",
) -> None:
    """Write a genome to the existing public/champions/<name>.json format that
    the browser already knows how to load via the 'load champion' button."""
    weights = np.array(genome, copy=False).astype(np.float64).tolist()
    payload = {
        "weights": weights,
        "generation": generation,
        "bestFitness": best_fitness,
        "savedAt": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def append_index(out_dir: Path, entry: dict, cap: int = 50) -> None:
    """Append entry to index.json. Trims to `cap` newest, and unlinks any
    .flxr files that fall off the index so the directory doesn't grow
    unboundedly during long training runs."""
    index_path = out_dir / "index.json"
    entries: list[dict] = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text())
            if not isinstance(entries, list):
                entries = []
        except json.JSONDecodeError:
            entries = []
    entries.append(entry)
    entries.sort(key=lambda e: e.get("saved_at", ""), reverse=True)
    kept = entries[:cap]
    dropped = entries[cap:]
    for e in dropped:
        f = e.get("file")
        if not f:
            continue
        try:
            (out_dir / f).unlink(missing_ok=True)
        except OSError:
            pass
    index_path.write_text(json.dumps(kept, indent=2) + "\n")
