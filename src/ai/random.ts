import type { Action, GameState, Player } from '../game/state';
import { mulberry32 } from './rng';
import { setup } from './utils';

export function aiThink(state: GameState, player: Player, seed = 0): Action[] {
  const ctx = setup(state, player);
  const actions: Action[] = [...ctx.cancelActions];
  const rng = mulberry32((seed ^ (state.tick * 7919) ^ (player * 31)) >>> 0);
  for (const src of ctx.openSrcs) {
    const candidates: number[] = [];
    for (const id of ctx.adj[src.id]) {
      if (state.nodes[id].owner !== player) candidates.push(id);
    }
    if (candidates.length === 0) continue;
    const dst = candidates[Math.floor(rng() * candidates.length)];
    actions.push({ kind: 'toggleFlow', src: src.id, dst, player });
  }
  return actions;
}
