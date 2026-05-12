"""MLX evolution training loop with batched parallel games.

Each batch plays G games in parallel on the GPU. After every batch, fitness
samples for every genome that appeared as a seat are folded into a running
mean, and the bottom-k of the population is replaced with mutated tournament
winners. Replays drop every `replay_every_sec` wall-clock seconds (from the
recorded game in the most recent batch). Champion JSON saved on each new
all-time best.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --pop 24 --games-per-batch 24 --ticks 10000
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from flux.evolve_mlx import (  # noqa: E402
    EvolutionConfig,
    append_index,
    load_checkpoint,
    make_initial_evolution_state,
    prepare_checkpoint_payload,
    run_one_batch,
    save_champion_json,
    save_checkpoint,
    write_checkpoint_payload,
    write_replay,
    _selection_step,
)
from flux.game_loop import GameConfig, RecordConfig  # noqa: E402
import mlx.core as mx  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "public" / "replays"


def default_checkpoint(model: str) -> Path:
    # v1 stays at the pre-existing path for backwards-compat with the running
    # job. v2 lives in its own subdir.
    if model == "v1":
        return REPO_ROOT / "python" / "checkpoints" / "latest.npz"
    return REPO_ROOT / "python" / "checkpoints" / model / "latest.npz"


def default_champion_dir(model: str) -> Path:
    if model == "v1":
        return REPO_ROOT / "public" / "champions"
    return REPO_ROOT / "public" / "champions" / model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v1", "v2", "v3"], default="v1",
                    help="v1: 91-feat 2-hop MLP (3571 weights). "
                         "v2: 181-feat 3-hop MLP (6451 weights). "
                         "v3: 2-layer GCN (2995 weights). Distinct artifact dirs.")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--games-per-batch", type=int, default=24,
                    help="G — parallel games per evolution step. Bigger = more "
                         "GPU saturation + cleaner fitness signal.")
    ap.add_argument("--radius", type=int, default=18)
    ap.add_argument("--num-players", type=int, default=12)
    ap.add_argument("--ticks", type=int, default=10000,
                    help="hard tick cap per game (game exits early on win)")
    ap.add_argument("--ai-period-ticks", type=int, default=5)
    ap.add_argument("--init-std", type=float, default=0.5)
    ap.add_argument("--mutation-sigma", type=float, default=0.05)
    ap.add_argument("--elites", type=int, default=6)
    ap.add_argument("--tournament-k", type=int, default=3)
    ap.add_argument("--selection-every-batches", type=int, default=1)
    ap.add_argument("--batches", type=int, default=0,
                    help="stop after N batches (0 = run forever)")
    ap.add_argument("--replay-every-sec", type=float, default=0.0,
                    help="min wall-clock seconds between replay writes. "
                         "0 = write every batch (writes are async).")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="checkpoint .npz path (default: per-model)")
    ap.add_argument("--checkpoint-every-batches", type=int, default=1)
    ap.add_argument("--champion-dir", type=Path, default=None,
                    help="champion JSON dir (default: per-model)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoint and start a new run")
    ap.add_argument("--aggressive-seat", type=int, default=0,
                    help="seat index that plays the hand-coded aggressive AI. "
                         "-1 disables.")
    ap.add_argument("--wandb", action="store_true",
                    help="log per-batch metrics to Weights & Biases. "
                         "Requires WANDB_API_KEY or `wandb login` beforehand.")
    ap.add_argument("--wandb-project", default="flux")
    ap.add_argument("--wandb-run-name", default=None,
                    help="run name shown in the W&B UI (default: model + timestamp)")
    ap.add_argument("--seed", type=int, default=int(time.time()) & 0xFFFFFFFF)
    args = ap.parse_args()
    if args.checkpoint is None:
        args.checkpoint = default_checkpoint(args.model)
    if args.champion_dir is None:
        args.champion_dir = default_champion_dir(args.model)

    out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    game_config = GameConfig(
        radius=args.radius,
        num_players=args.num_players,
        ticks=args.ticks,
        ai_period_ticks=args.ai_period_ticks,
        model_kind=args.model,
    )
    config = EvolutionConfig(
        pop_size=args.pop,
        games_per_batch=args.games_per_batch,
        init_std=args.init_std,
        mutation_sigma=args.mutation_sigma,
        elites=args.elites,
        tournament_k=args.tournament_k,
        selection_every_batches=args.selection_every_batches,
        game=game_config,
    )
    record_config = RecordConfig(tick_stride=10, dt_per_tick_ms=100)

    state = None
    if not args.fresh:
        state = load_checkpoint(args.checkpoint)
        if state is not None:
            print(f"resumed from {args.checkpoint.relative_to(REPO_ROOT)}: "
                  f"gen={state.generation} games={state.games_played} "
                  f"all_time_best={state.all_time_best_fitness:.2f}")
    if state is None:
        state = make_initial_evolution_state(config, args.seed)
        print("starting from a fresh population")
    rng = np.random.default_rng(args.seed ^ 0xDEADBEEF)
    rng_key = mx.random.key(args.seed ^ 0xC0FFEE)

    print(
        f"training [{args.model}]: pop={config.pop_size}, "
        f"G={config.games_per_batch}, seats={game_config.num_players}, "
        f"ticks_cap={game_config.ticks}, "
        f"ai_period={game_config.ai_period_ticks}, "
        f"replay every {args.replay_every_sec}s wall, "
        f"checkpoint every {args.checkpoint_every_batches} batch → "
        f"{args.checkpoint.relative_to(REPO_ROOT)}"
    )
    if args.aggressive_seat >= 0:
        print(f"  seat {args.aggressive_seat} = hand-coded aggressive AI")
    args.champion_dir.mkdir(parents=True, exist_ok=True)
    prev_all_time_best = state.all_time_best_fitness

    wandb_run = None
    if args.wandb:
        import wandb  # type: ignore
        run_name = args.wandb_run_name or (
            f"{args.model}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model,
                "pop": config.pop_size,
                "games_per_batch": config.games_per_batch,
                "ticks_cap": game_config.ticks,
                "ai_period_ticks": game_config.ai_period_ticks,
                "radius": game_config.radius,
                "num_players": game_config.num_players,
                "init_std": config.init_std,
                "mutation_sigma": config.mutation_sigma,
                "elites": config.elites,
                "tournament_k": config.tournament_k,
                "selection_every_batches": config.selection_every_batches,
                "early_weight": game_config.early_weight,
                "linger_penalty": game_config.linger_penalty,
                "aggressive_seat": args.aggressive_seat,
                "resumed_from_gen": state.generation,
            },
        )
        print(f"  wandb run: {wandb_run.url if wandb_run else '(init failed)'}")

    run_start_t = time.time()
    cumulative_batches = 0

    # Background writer thread for checkpoint + replay + champion JSON I/O.
    # Main loop snapshots state before submitting so subsequent mutations
    # (selection step, fitness updates) don't race with the write.
    write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flux-io")
    in_flight: list[Future] = []

    def submit_write(fn, /, *fn_args, **fn_kwargs) -> None:
        in_flight.append(write_executor.submit(fn, *fn_args, **fn_kwargs))
        # Reap completed futures to surface exceptions.
        for f in list(in_flight):
            if f.done():
                in_flight.remove(f)
                exc = f.exception()
                if exc is not None:
                    raise exc

    last_replay_t = 0.0
    batch_i = 0
    try:
        while True:
            if args.batches > 0 and batch_i >= args.batches:
                break
            now = time.time()
            should_record = (now - last_replay_t) >= args.replay_every_sec
            record_game_idx = 0 if should_record else None
            record = record_config if should_record else None

            hand_coded = {}
            if args.aggressive_seat >= 0:
                hand_coded[args.aggressive_seat] = "aggressive"

            t0 = time.time()
            seat_ids_per_game, frames, final_cells_per_game, decision_ticks = run_one_batch(
                state, config, rng,
                record_game_idx=record_game_idx, record=record,
                hand_coded_seats=hand_coded,
            )
            dt = time.time() - t0
            batch_i += 1
            G = config.games_per_batch

            min_dt = min(decision_ticks)
            med_dt = sorted(decision_ticks)[G // 2]
            max_dt = max(decision_ticks)
            # Best (highest fitness) seat across all games in this batch.
            batch_best_fit = -float("inf")
            batch_best = (0, 0)
            for g, ids in enumerate(seat_ids_per_game):
                final = final_cells_per_game[g]
                for s, sid in enumerate(ids):
                    if isinstance(sid, str):
                        continue
                    # Re-derive fitness for printing (we don't have per-game
                    # fitness handy here; use cells as a proxy or recompute).
                    pass
            # Use the per-game winner cell-count as a quick narrative metric.
            winners = []
            for g in range(G):
                w_seat = int(np.argmax(final_cells_per_game[g]))
                w_id = seat_ids_per_game[g][w_seat]
                winners.append((w_id, int(final_cells_per_game[g][w_seat])))
            aggressive_wins = sum(1 for w, _ in winners if isinstance(w, str))
            print(
                f"batch {batch_i:4d}  {dt:5.1f}s  gen={state.generation}  "
                f"games={G}  decision_tick min/med/max={min_dt}/{med_dt}/{max_dt}  "
                f"aggressive_wins={aggressive_wins}/{G}  "
                f"best_fit={state.best_fitness:8.2f}  "
                f"all_time={state.all_time_best_fitness:8.2f}"
            )

            cumulative_batches += 1
            elapsed_since_start = time.time() - run_start_t
            gens_per_min_rolling = (
                cumulative_batches * config.selection_every_batches * 60 / elapsed_since_start
                if elapsed_since_start > 0 else 0.0
            )
            if wandb_run is not None:
                log_payload = {
                    "batch": batch_i,
                    "generation": state.generation,
                    "games_played": state.games_played,
                    "best_fitness": state.best_fitness,
                    "all_time_best_fitness": state.all_time_best_fitness,
                    "decision_tick_min": min_dt,
                    "decision_tick_med": med_dt,
                    "decision_tick_max": max_dt,
                    "aggressive_wins": aggressive_wins,
                    "batch_wall_sec": dt,
                    "gens_per_min": gens_per_min_rolling,
                    "all_time_best_advanced": state.all_time_best_fitness > prev_all_time_best,
                }
                # Per-player metrics averaged across this batch's G games.
                # Seat 0 is aggressive (or whatever --aggressive-seat is);
                # others are random NN genomes sampled from the population.
                for p in range(game_config.num_players):
                    per_game = [int(fc[p]) for fc in final_cells_per_game]
                    log_payload[f"cells/seat_{p:02d}"] = float(np.mean(per_game))
                log_payload["cells/aggressive_mean"] = float(np.mean([
                    int(final_cells_per_game[g][args.aggressive_seat])
                    for g in range(G)
                ])) if args.aggressive_seat >= 0 else 0.0
                # Mean cell count across all NN seats (everyone except aggressive).
                nn_seats = [p for p in range(game_config.num_players)
                            if p != args.aggressive_seat]
                if nn_seats:
                    log_payload["cells/nn_mean"] = float(np.mean([
                        int(final_cells_per_game[g][p])
                        for g in range(G) for p in nn_seats
                    ]))
                wandb_run.log(log_payload, step=state.generation)

            if should_record and frames:
                last_replay_t = now
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                name = f"train_{args.model}_{stamp}_b{batch_i:05d}_gen{state.generation}.flxr"
                replay_path = out_dir / name
                metadata = {
                    "kind": "train_batch",
                    "model": args.model,
                    "generation": state.generation,
                    "batch_index": batch_i,
                    "games_per_batch": G,
                    "seat_ids": seat_ids_per_game[0],
                    "decision_tick": int(decision_ticks[0]),
                    "final_cells": [int(c) for c in final_cells_per_game[0]],
                    "best_fitness": state.best_fitness,
                    "all_time_best_fitness": state.all_time_best_fitness,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }
                index_entry = {
                    "file": name,
                    "saved_at": metadata["saved_at"],
                    "kind": "train_batch",
                    "model": args.model,
                    "generation": state.generation,
                    "batch_index": batch_i,
                    "ticks": game_config.ticks,
                    "radius": game_config.radius,
                    "num_players": game_config.num_players,
                    "best_fitness": state.best_fitness,
                }

                def _write_replay_and_index(p, fr, cfg, rec, meta, od, ent):
                    sz = write_replay(p, fr, cfg, rec, meta)
                    append_index(od, ent)
                    print(f"  replay {p.name}  ({sz} bytes)")

                # frames + metadata are immutable Python objects (GameState
                # tuples); safe to hand off directly to the writer thread.
                submit_write(
                    _write_replay_and_index,
                    replay_path, frames, config, record_config, metadata,
                    out_dir, index_entry,
                )

            # New all-time best → drop a champion JSON.
            if (state.all_time_best_fitness > prev_all_time_best
                    and state.all_time_best_genome is not None):
                prev_all_time_best = state.all_time_best_fitness
                cname = (
                    f"{args.model}_gen{state.generation:04d}_b{batch_i:05d}"
                    f"_fit{state.all_time_best_fitness:.0f}.json"
                )
                save_champion_json(
                    state.all_time_best_genome,
                    args.champion_dir / cname,
                    generation=state.generation,
                    best_fitness=state.all_time_best_fitness,
                    note=f"mlx-train {args.model}, batch {batch_i}",
                )
                print(f"  champion {cname}")

            if batch_i % config.selection_every_batches == 0:
                rng_key = _selection_step(state, config, rng, rng_key)

            if batch_i % args.checkpoint_every_batches == 0:
                # Snapshot synchronously (fast), write async.
                ckpt_payload = prepare_checkpoint_payload(state)
                submit_write(write_checkpoint_payload, ckpt_payload, args.checkpoint)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        # Drain in-flight writes, then do a final synchronous save.
        write_executor.shutdown(wait=True)
        save_checkpoint(state, args.checkpoint)
        print(f"checkpoint saved → {args.checkpoint.relative_to(REPO_ROOT)}")
        if wandb_run is not None:
            wandb_run.finish()

    print(f"done. batches={batch_i} generation={state.generation} "
          f"all_time_best={state.all_time_best_fitness:.2f}")


if __name__ == "__main__":
    main()
