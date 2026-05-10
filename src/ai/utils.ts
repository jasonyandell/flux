import type { Action, GameNode, GameState, NodeId, Player } from '../game/state';

export type Ctx = {
  state: GameState;
  player: Player;
  adj: NodeId[][];
  openSrcs: GameNode[];
  cancelActions: Action[];
};

export function buildAdjacency(state: GameState): NodeId[][] {
  const adj: NodeId[][] = state.nodes.map(() => []);
  for (const e of state.edges) {
    adj[e.a].push(e.b);
    adj[e.b].push(e.a);
  }
  return adj;
}

export function setup(state: GameState, player: Player, strengthThreshold = 5): Ctx {
  const cancelActions: Action[] = [];
  for (const f of state.flows) {
    if (f.player !== player) continue;
    if (state.nodes[f.dst].owner === player) {
      cancelActions.push({ kind: 'toggleFlow', src: f.src, dst: f.dst, player });
    }
  }
  const cancelKeys = new Set(cancelActions.map(a => `${a.src}-${a.dst}`));
  const srcsBlocked = new Set(
    state.flows
      .filter(f => f.player === player && !cancelKeys.has(`${f.src}-${f.dst}`))
      .map(f => f.src),
  );
  const openSrcs = state.nodes.filter(n =>
    n.owner === player && n.strength > strengthThreshold && !srcsBlocked.has(n.id),
  );
  return { state, player, adj: buildAdjacency(state), openSrcs, cancelActions };
}

export function pickWeakestNonFriendly(ctx: Ctx, srcId: NodeId): NodeId {
  let bestId = -1;
  let bestStrength = Infinity;
  let bestSecondary = 0;
  for (const id of ctx.adj[srcId]) {
    const n = ctx.state.nodes[id];
    if (n.owner === ctx.player) continue;
    const secondary = ctx.player === 0 ? id : -id;
    if (n.strength < bestStrength || (n.strength === bestStrength && secondary < bestSecondary)) {
      bestStrength = n.strength;
      bestSecondary = secondary;
      bestId = id;
    }
  }
  return bestId;
}
