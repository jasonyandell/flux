export type Player = number;
export type Owner = Player | null;
export type NodeId = number;

export type Vec2 = { x: number; y: number };

export type GameNode = {
  id: NodeId;
  pos: Vec2;
  owner: Owner;
  strength: number;
};

export type Edge = { a: NodeId; b: NodeId };

export type Flow = { src: NodeId; dst: NodeId; player: Player };

export type GameState = {
  nodes: GameNode[];
  edges: Edge[];
  flows: Flow[];
  tick: number;
  numPlayers: number;
};

export type Action =
  | { kind: 'toggleFlow'; src: NodeId; dst: NodeId; player: Player };

export const REGEN_PER_SEC = 1.0;
export const TRANSFER_PER_SEC = 3.0;
export const MAX_STRENGTH = 100;
export const MIN_STRENGTH_TO_SEND = 0.1;
export const LOOP_BONUS = 0.5;
