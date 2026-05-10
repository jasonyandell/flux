import process from 'node:process';
import { AI_NAMES, AIs, type AIFn, type AIName } from '../ai';
import { makeInitialState } from '../game/graph';
import type { GameState, Player } from '../game/state';
import { applyAction, step } from '../game/step';

const TICK_DT = 0.1;
const MAX_TICKS = 5000;
const AI_PERIOD_TICKS = 5;

type Result = { winner: Player | 'draw'; ticks: number };

function runOne(p0Fn: AIFn, p1Fn: AIFn, seed = 0): Result {
  let state: GameState = makeInitialState();
  for (let t = 0; t < MAX_TICKS; t++) {
    if (t % AI_PERIOD_TICKS === 0) {
      for (const a of p0Fn(state, 0, seed * 10007)) state = applyAction(state, a);
      for (const a of p1Fn(state, 1, seed * 10007 + 1)) state = applyAction(state, a);
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

type PairResult = { p0: number; p1: number; draws: number; avgTicks: number; elapsedMs: number };

function runPair(p0Name: AIName, p1Name: AIName, n: number): PairResult {
  const t0 = performance.now();
  let p0 = 0, p1 = 0, draws = 0, totalTicks = 0;
  for (let i = 0; i < n; i++) {
    const r = runOne(AIs[p0Name], AIs[p1Name], i + 1);
    if (r.winner === 0) p0++;
    else if (r.winner === 1) p1++;
    else draws++;
    totalTicks += r.ticks;
  }
  return { p0, p1, draws, avgTicks: totalTicks / n, elapsedMs: performance.now() - t0 };
}

function printPair(p0Name: AIName, p1Name: AIName, n: number, r: PairResult): void {
  console.log(`pairing:     ${p0Name} (P0) vs ${p1Name} (P1)`);
  console.log(`runs:        ${n}`);
  console.log(`P0 wins:     ${r.p0}`);
  console.log(`P1 wins:     ${r.p1}`);
  console.log(`draws:       ${r.draws}`);
  console.log(`avg ticks:   ${r.avgTicks.toFixed(1)}`);
  console.log(`elapsed:     ${r.elapsedMs.toFixed(0)}ms`);
}

const args = process.argv.slice(2);

if (args[0] === 'tournament') {
  const n = parseInt(args[1] ?? '3', 10);
  const labelWidth = Math.max(...AI_NAMES.map(s => s.length), 5);
  const cellWidth = Math.max(...AI_NAMES.map(s => s.length), 7) + 2;
  console.log(`Tournament: ${n} runs/pairing, ${AI_NAMES.length} AIs (${AI_NAMES.length ** 2} pairings)`);
  console.log('cells: P0 wins / P1 wins / draws\n');

  let head = 'P0\\P1'.padEnd(labelWidth + 2);
  for (const name of AI_NAMES) head += name.padStart(cellWidth);
  console.log(head);

  const t0 = performance.now();
  for (const p0Name of AI_NAMES) {
    let row = p0Name.padEnd(labelWidth + 2);
    for (const p1Name of AI_NAMES) {
      const r = runPair(p0Name, p1Name, n);
      row += `${r.p0}/${r.p1}/${r.draws}`.padStart(cellWidth);
    }
    console.log(row);
  }
  console.log(`\nelapsed: ${(performance.now() - t0).toFixed(0)}ms`);
} else if (args.length >= 2 && Number.isNaN(parseInt(args[0], 10))) {
  const p0Name = args[0] as AIName;
  const p1Name = args[1] as AIName;
  const n = parseInt(args[2] ?? '10', 10);
  if (!AI_NAMES.includes(p0Name) || !AI_NAMES.includes(p1Name)) {
    console.error(`unknown AI. Available: ${AI_NAMES.join(', ')}`);
    process.exit(1);
  }
  printPair(p0Name, p1Name, n, runPair(p0Name, p1Name, n));
} else {
  const n = parseInt(args[0] ?? '10', 10);
  printPair('aggressive', 'aggressive', n, runPair('aggressive', 'aggressive', n));
}
