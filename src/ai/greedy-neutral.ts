import type { Action, GameState, Player } from '../game/state';
import { setup } from './utils';

export function aiThink(state: GameState, player: Player, _seed = 0): Action[] {
  const ctx = setup(state, player);
  const actions: Action[] = [...ctx.cancelActions];
  for (const src of ctx.openSrcs) {
    let bestNeutralId = -1, bestNeutralStr = Infinity;
    let bestEnemyId = -1, bestEnemyStr = Infinity;
    for (const id of ctx.adj[src.id]) {
      const n = state.nodes[id];
      if (n.owner === player) continue;
      if (n.owner === null) {
        if (n.strength < bestNeutralStr) { bestNeutralStr = n.strength; bestNeutralId = id; }
      } else {
        if (n.strength < bestEnemyStr) { bestEnemyStr = n.strength; bestEnemyId = id; }
      }
    }
    const dst = bestNeutralId >= 0 ? bestNeutralId : bestEnemyId;
    if (dst >= 0) actions.push({ kind: 'toggleFlow', src: src.id, dst, player });
  }
  return actions;
}
