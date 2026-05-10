import process from 'node:process';
import { aiThink } from '../ai/dumb';
import { makeInitialState } from '../game/graph';
import type { GameState, Player } from '../game/state';
import { applyAction, step } from '../game/step';

const TICK_DT = 0.1;
const MAX_TICKS = 5000;
const AI_PERIOD_TICKS = 5;

type Result = { winner: Player | 'draw'; ticks: number };

function runOne(): Result {
  let state: GameState = makeInitialState();

  for (let t = 0; t < MAX_TICKS; t++) {
    if (t % AI_PERIOD_TICKS === 0) {
      for (const p of [0, 1] as Player[]) {
        for (const a of aiThink(state, p)) state = applyAction(state, a);
      }
    }
    state = step(state, TICK_DT);

    let p0 = 0, p1 = 0;
    for (const n of state.nodes) {
      if (n.owner === 0) p0++;
      else if (n.owner === 1) p1++;
    }
    if (p0 > 0 && p1 === 0) return { winner: 0, ticks: t };
    if (p1 > 0 && p0 === 0) return { winner: 1, ticks: t };
    if (p0 === 0 && p1 === 0) return { winner: 'draw', ticks: t };
  }
  return { winner: 'draw', ticks: MAX_TICKS };
}

const N = parseInt(process.argv[2] ?? '10', 10);
const results: Result[] = [];
const t0 = performance.now();
for (let i = 0; i < N; i++) results.push(runOne());
const elapsed = performance.now() - t0;

const wins = [0, 0, 0]; // P0, P1, draw
let totalTicks = 0;
for (const r of results) {
  if (r.winner === 'draw') wins[2]++;
  else wins[r.winner]++;
  totalTicks += r.ticks;
}
console.log(`runs:        ${N}`);
console.log(`P0 wins:     ${wins[0]}`);
console.log(`P1 wins:     ${wins[1]}`);
console.log(`draws:       ${wins[2]}`);
console.log(`avg ticks:   ${(totalTicks / N).toFixed(1)}`);
console.log(`elapsed:     ${elapsed.toFixed(0)}ms (${(N / (elapsed / 1000)).toFixed(0)} runs/sec)`);
