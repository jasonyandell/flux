import type { Action, GameState, Player } from '../game/state';
import { aiThink as aggressive } from './aggressive';
import { aiThink as cluster } from './cluster';
import { aiThink as defensive } from './defensive';
import { aiThink as greedyNeutral } from './greedy-neutral';
import { aiThink as opportunist } from './opportunist';
import { aiThink as random } from './random';
import { aiThink as evolved } from '../gpu/evolved';

export type AIFn = (state: GameState, player: Player, seed?: number) => Action[];

export type AIName =
  | 'aggressive'
  | 'random'
  | 'defensive'
  | 'greedy-neutral'
  | 'opportunist'
  | 'cluster'
  | 'evolved';

export const AIs: Record<AIName, AIFn> = {
  aggressive,
  random,
  defensive,
  'greedy-neutral': greedyNeutral,
  opportunist,
  cluster,
  evolved,
};

export const AI_NAMES: AIName[] = [
  'aggressive', 'random', 'defensive', 'greedy-neutral', 'opportunist', 'cluster', 'evolved',
];
