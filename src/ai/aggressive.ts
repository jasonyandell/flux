import type { Action, GameState, Player } from '../game/state';
import { pickWeakestNonFriendly, setup } from './utils';

export function aiThink(state: GameState, player: Player, _seed = 0): Action[] {
  const ctx = setup(state, player);
  const actions: Action[] = [...ctx.cancelActions];
  for (const src of ctx.openSrcs) {
    const dst = pickWeakestNonFriendly(ctx, src.id);
    if (dst >= 0) actions.push({ kind: 'toggleFlow', src: src.id, dst, player });
  }
  return actions;
}
