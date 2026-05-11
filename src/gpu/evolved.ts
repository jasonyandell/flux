import type { Action, GameState, Player } from '../game/state';
import { buildNeighborTable, nnInferCell, randomGenome, type Genome, NN_OUT_DIM, NEIGHBOR_STRIDE } from './genome';
import { mulberry32 } from '../ai/rng';

// Champion genome lives in module state. Evolution loop writes here; aiThink reads.
// Roster is an optional per-seat override so each `evolved` seat can play a
// different genome (e.g. the full population) — without it, every seat clones
// the champion and the multi-seat game is just one policy mirrored N times.
let champion: Genome | null = null;
let roster: Genome[] | null = null;
let neighborCache: { state: GameState; nb: Int32Array } | null = null;

export function setChampion(g: Genome | null): void {
  champion = g;
}

export function getChampion(): Genome | null {
  return champion;
}

export function setRoster(genomes: Genome[] | null): void {
  roster = genomes && genomes.length > 0 ? genomes : null;
}

export function ensureChampion(): Genome {
  if (!champion) {
    const rng = mulberry32(0xC0FFEE);
    champion = randomGenome(rng);
  }
  return champion;
}

function neighborsFor(state: GameState): Int32Array {
  if (neighborCache && neighborCache.state === state) return neighborCache.nb;
  // Fall back to rebuild if state structure changed (e.g., respawn).
  if (!neighborCache || neighborCache.state.nodes.length !== state.nodes.length || neighborCache.state.edges.length !== state.edges.length) {
    neighborCache = { state, nb: buildNeighborTable(state) };
    return neighborCache.nb;
  }
  // Same topology: reuse but update reference (positions unchanged).
  neighborCache.state = state;
  return neighborCache.nb;
}

// aiThink for the evolved controller. Pure: uses champion genome (or random
// fallback) to choose per-cell actions, then emits toggleFlow actions that
// install the desired flow state — replacing any existing flows from this seat
// that point elsewhere.
export function aiThink(state: GameState, player: Player, _seed = 0): Action[] {
  const genome = roster ? roster[player % roster.length] : ensureChampion();
  const nb = neighborsFor(state);
  const actions: Action[] = [];

  // Current flows owned by this player, indexed by src.
  const desiredBySrc = new Map<number, number>(); // src -> dst (action choice)
  for (let cell = 0; cell < state.nodes.length; cell++) {
    if (state.nodes[cell].owner !== player) continue;
    const a = nnInferCell(genome, state, cell, player, nb);
    if (a < 0 || a >= NEIGHBOR_STRIDE) continue;
    const dst = nb[cell * NEIGHBOR_STRIDE + a];
    if (dst < 0) continue;
    desiredBySrc.set(cell, dst);
  }

  // Cancel any existing flow from this player whose dst doesn't match desired,
  // and cancel friendly self-target flows.
  for (const f of state.flows) {
    if (f.player !== player) continue;
    const desired = desiredBySrc.get(f.src);
    if (desired === undefined || desired !== f.dst) {
      // Toggle to remove
      actions.push({ kind: 'toggleFlow', src: f.src, dst: f.dst, player });
    } else {
      // Already in desired state — drop from map to avoid double-toggling.
      desiredBySrc.delete(f.src);
    }
  }

  // Add any remaining desired flows.
  for (const [src, dst] of desiredBySrc) {
    actions.push({ kind: 'toggleFlow', src, dst, player });
  }

  return actions;
}

// Unused param keeps TS strict happy without dead arg names.
void NN_OUT_DIM;
