import type { Edge, GameNode, GameState } from './state';

export function makeInitialState(): GameState {
  const layout: { pos: { x: number; y: number }; owner: number | null; strength: number }[] = [
    { pos: { x: -5, y:  0 }, owner: 0,    strength: 30 }, // 0: P0 base
    { pos: { x: -2, y:  2 }, owner: null, strength: 10 }, // 1
    { pos: { x: -2, y: -2 }, owner: null, strength: 10 }, // 2
    { pos: { x:  0, y:  0 }, owner: null, strength: 10 }, // 3: center
    { pos: { x:  2, y:  2 }, owner: null, strength: 10 }, // 4
    { pos: { x:  2, y: -2 }, owner: null, strength: 10 }, // 5
    { pos: { x:  5, y:  0 }, owner: 1,    strength: 30 }, // 6: P1 base
  ];

  const nodes: GameNode[] = layout.map((n, i) => ({ id: i, pos: n.pos, owner: n.owner, strength: n.strength }));

  const edges: Edge[] = [
    { a: 0, b: 1 }, { a: 0, b: 2 },
    { a: 1, b: 2 }, { a: 1, b: 3 }, { a: 1, b: 4 },
    { a: 2, b: 3 }, { a: 2, b: 5 },
    { a: 3, b: 4 }, { a: 3, b: 5 },
    { a: 4, b: 5 }, { a: 4, b: 6 },
    { a: 5, b: 6 },
  ];

  return { nodes, edges, flows: [], tick: 0, numPlayers: 2 };
}
