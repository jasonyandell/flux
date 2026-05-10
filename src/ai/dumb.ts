import type { Action, GameState, NodeId, Player } from '../game/state';

export function aiThink(state: GameState, player: Player): Action[] {
  const actions: Action[] = [];

  for (const f of state.flows) {
    if (f.player !== player) continue;
    if (state.nodes[f.dst].owner === player) {
      actions.push({ kind: 'toggleFlow', src: f.src, dst: f.dst, player });
    }
  }

  const cancelKeys = new Set(actions.map(a => `${a.src}-${a.dst}`));
  const srcWithOutgoing = new Set(
    state.flows
      .filter(f => f.player === player && !cancelKeys.has(`${f.src}-${f.dst}`))
      .map(f => f.src),
  );

  const adj = buildAdjacency(state);
  const myStrong = state.nodes.filter(n => n.owner === player && n.strength > 5);
  for (const src of myStrong) {
    if (srcWithOutgoing.has(src.id)) continue;
    const ns = adj[src.id];
    let bestId: NodeId = -1;
    let bestStrength = Infinity;
    let bestSecondary = 0;
    for (const id of ns) {
      const n = state.nodes[id];
      if (n.owner === player) continue;
      const secondary = player === 0 ? id : -id;
      if (n.strength < bestStrength || (n.strength === bestStrength && secondary < bestSecondary)) {
        bestStrength = n.strength;
        bestSecondary = secondary;
        bestId = id;
      }
    }
    if (bestId >= 0) actions.push({ kind: 'toggleFlow', src: src.id, dst: bestId, player });
  }
  return actions;
}

function buildAdjacency(state: GameState): NodeId[][] {
  const adj: NodeId[][] = state.nodes.map(() => []);
  for (const e of state.edges) {
    adj[e.a].push(e.b);
    adj[e.b].push(e.a);
  }
  return adj;
}
