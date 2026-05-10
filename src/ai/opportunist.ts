import type { Action, GameState, Player } from '../game/state';
import { ATTACK_BONUS, REGEN_PER_SEC, TRANSFER_PER_SEC } from '../game/state';
import { setup } from './utils';

const HORIZON_TICKS = 10;
const TICK_DT = 0.1;

export function aiThink(state: GameState, player: Player, _seed = 0): Action[] {
  const ctx = setup(state, player);
  const actions: Action[] = [...ctx.cancelActions];
  const damagePerTick = TRANSFER_PER_SEC * TICK_DT * (1 + ATTACK_BONUS);
  const enemyRegenPerTick = REGEN_PER_SEC * TICK_DT;
  for (const src of ctx.openSrcs) {
    let bestId = -1, bestTime = Infinity;
    for (const id of ctx.adj[src.id]) {
      const n = state.nodes[id];
      if (n.owner === player) continue;
      const regen = n.owner === null ? 0 : enemyRegenPerTick;
      const netDrain = damagePerTick - regen;
      if (netDrain <= 0) continue;
      const ticksToCapture = n.strength / netDrain;
      if (ticksToCapture > HORIZON_TICKS) continue;
      if (ticksToCapture < bestTime) { bestTime = ticksToCapture; bestId = id; }
    }
    if (bestId >= 0) actions.push({ kind: 'toggleFlow', src: src.id, dst: bestId, player });
  }
  return actions;
}
