import type { Action, GameState, NodeId, Player } from '../game/state';

export function aiThink(state: GameState, player: Player): Action[] {
  const actions: Action[] = [];

  // Cancel my flows pointing at nodes I now own (target was captured).
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

  // For each owned, strong, non-sending node: attack weakest non-friendly neighbor.
  const myStrong = state.nodes.filter(n => n.owner === player && n.strength > 5);
  for (const src of myStrong) {
    if (srcWithOutgoing.has(src.id)) continue;
    const targets = neighborsOf(state, src.id)
      .map(id => state.nodes[id])
      .filter(n => n.owner !== player);
    if (targets.length === 0) continue;
    targets.sort((a, b) =>
      a.strength - b.strength || (player === 0 ? a.id - b.id : b.id - a.id),
    );
    actions.push({ kind: 'toggleFlow', src: src.id, dst: targets[0].id, player });
  }
  return actions;
}

function neighborsOf(state: GameState, n: NodeId): NodeId[] {
  const out: NodeId[] = [];
  for (const e of state.edges) {
    if (e.a === n) out.push(e.b);
    else if (e.b === n) out.push(e.a);
  }
  return out;
}
