import type { Action, GameState, Player } from '../game/state';
import { pickWeakestNonFriendly, setup } from './utils';

const DEFENSIVE_THRESHOLD = 50;

export function aiThink(state: GameState, player: Player, _seed = 0): Action[] {
  const ctx = setup(state, player, DEFENSIVE_THRESHOLD);
  const actions: Action[] = [...ctx.cancelActions];
  for (const src of ctx.openSrcs) {
    const dst = pickWeakestNonFriendly(ctx, src.id);
    if (dst >= 0) actions.push({ kind: 'toggleFlow', src: src.id, dst, player });
  }
  return actions;
}
