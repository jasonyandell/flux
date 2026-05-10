import type { Action, GameState, Player } from '../game/state';
import { setup } from './utils';

export function aiThink(state: GameState, player: Player, _seed = 0): Action[] {
  const ctx = setup(state, player);
  const actions: Action[] = [...ctx.cancelActions];
  for (const src of ctx.openSrcs) {
    let bestId = -1, bestFriendlyCount = -1, bestStrength = Infinity;
    for (const id of ctx.adj[src.id]) {
      const n = state.nodes[id];
      if (n.owner === player) continue;
      let friendlyCount = 0;
      for (const nid of ctx.adj[id]) {
        if (state.nodes[nid].owner === player) friendlyCount++;
      }
      if (friendlyCount > bestFriendlyCount ||
          (friendlyCount === bestFriendlyCount && n.strength < bestStrength)) {
        bestFriendlyCount = friendlyCount;
        bestStrength = n.strength;
        bestId = id;
      }
    }
    if (bestId >= 0) actions.push({ kind: 'toggleFlow', src: src.id, dst: bestId, player });
  }
  return actions;
}
