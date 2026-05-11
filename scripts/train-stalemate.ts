// CPU-only evolutionary trainer for the "watch them get smarter" scene.
//
// Goal: produce a genome that, when played as seats 1..11 on the default
// 12-seat board with seat 0 = `aggressive`, leaves the game NOT a runaway by
// tick 4000 — at least 2 players still alive AND no single player owning more
// than 60% of owned cells.
//
// The GPU evolution loop in src/gpu/evolution.ts is WebGPU-only and can't run
// from Node. This is a small parallel CPU-only mini-evolver that uses only
// the pure TS modules (step, applyAction, aiThink).
//
// Run with: npx tsx scripts/train-stalemate.ts

import { writeFileSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import type { GameState } from '../src/game/state.ts';
import { applyAction, step } from '../src/game/step.ts';
import { makeInitialState } from '../src/game/graph.ts';
import { AIs, type AIName } from '../src/ai/index.ts';
import { setChampion } from '../src/gpu/evolved.ts';
import { WEIGHTS_PER_GENOME, gaussian, mutate, type Genome } from '../src/gpu/genome.ts';
import { mulberry32 } from '../src/ai/rng.ts';

// --- Constants mirroring runner.ts presimGame ----------------------------
const TICK_HZ = 10;
const TICK_DT = 1 / TICK_HZ;
const AI_PERIOD_SEC = 0.5;
const AI_PERIOD_TICKS = Math.max(1, Math.round(AI_PERIOD_SEC / TICK_DT));
const NUM_PLAYERS = 12;
const TARGET_TICK = 4000;

// --- Evolver hyperparameters ---------------------------------------------
const POP = 16;
const ELITES = 4;
const MAX_GENS = 30;
const WARM_START_SIGMA = 0.4;
const MUTATE_SIGMA_START = 0.15;
const MUTATE_SIGMA_END = 0.04;
const HARD_TIME_BUDGET_MS = 10 * 60 * 1000; // 10 minutes wall-clock cap
const SEED = 0xC0FFEE_2000;

// --- Fitness targets ------------------------------------------------------
const MIN_ALIVE = 2;
const MAX_SHARE = 0.60;

// --- Paths ---------------------------------------------------------------
const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');
const strongPath = resolve(repoRoot, 'public/champions/strong.json');
const outPath = resolve(repoRoot, 'public/champions/gen2000.json');
const stillPath = resolve(repoRoot, 'public/champions/gen2000-still.json');

// --- Seats: 0 = aggressive, rest = evolved -------------------------------
function defaultSeats(n: number): AIName[] {
  const out: AIName[] = Array(n).fill('evolved' as AIName);
  out[0] = 'aggressive';
  return out;
}

// --- Run one game and report alive/share at TARGET_TICK ------------------
type GameResult = { aliveCount: number; maxShare: number; finalTick: number };

function runGame(seats: AIName[]): GameResult {
  return runGameRecording(seats).result;
}

function runGameRecording(seats: AIName[]): { result: GameResult; finalState: GameState } {
  let s: GameState = makeInitialState(undefined, undefined, NUM_PLAYERS);
  for (let t = 0; t < TARGET_TICK; t++) {
    s = step(s, TICK_DT);
    if (t % AI_PERIOD_TICKS === 0) {
      for (let p = 0; p < s.numPlayers; p++) {
        const fn = AIs[seats[p]];
        for (const a of fn(s, p, s.tick)) s = applyAction(s, a);
      }
    }
  }
  return { result: measure(s), finalState: s };
}

function measure(s: GameState): GameResult {
  const counts = new Array<number>(s.numPlayers).fill(0);
  let owned = 0;
  for (const n of s.nodes) {
    if (n.owner !== null) {
      counts[n.owner]++;
      owned++;
    }
  }
  let aliveCount = 0;
  let maxC = 0;
  for (const c of counts) {
    if (c > 0) aliveCount++;
    if (c > maxC) maxC = c;
  }
  const maxShare = owned > 0 ? maxC / owned : 0;
  return { aliveCount, maxShare, finalTick: s.tick };
}

// --- Fitness composite ----------------------------------------------------
function fitness(r: GameResult): number {
  let f = 0;
  if (r.aliveCount >= MIN_ALIVE) f += 10;
  if (r.maxShare <= MAX_SHARE) f += 20;
  f += r.aliveCount;
  f -= Math.max(0, r.maxShare - MAX_SHARE) * 50;
  return f;
}

function meetsTargets(r: GameResult): boolean {
  return r.aliveCount >= MIN_ALIVE && r.maxShare <= MAX_SHARE;
}

// --- Initial population: warm-start from strong.json ---------------------
function loadStrong(): Genome {
  const raw = JSON.parse(readFileSync(strongPath, 'utf-8')) as { weights: number[] };
  if (!Array.isArray(raw.weights) || raw.weights.length !== WEIGHTS_PER_GENOME) {
    throw new Error(`strong.json: expected ${WEIGHTS_PER_GENOME} weights, got ${raw.weights?.length}`);
  }
  return new Float32Array(raw.weights);
}

function perturb(parent: Genome, rand: () => number, sigma: number): Genome {
  const child = new Float32Array(parent.length);
  for (let i = 0; i < parent.length; i++) child[i] = parent[i] + gaussian(rand) * sigma;
  return child;
}

// --- Main loop ------------------------------------------------------------
type Candidate = { genome: Genome; result: GameResult | null; score: number };

async function main(): Promise<void> {
  const rand = mulberry32(SEED);
  const strong = loadStrong();
  const seats = defaultSeats(NUM_PLAYERS);

  console.log(`[init] warm-start from strong.json (sigma=${WARM_START_SIGMA}), pop=${POP}, max_gens=${MAX_GENS}`);

  // Build initial population: candidate 0 is the unmodified strong (gives a
  // baseline measurement), the rest are mutations of it.
  let pop: Candidate[] = [];
  pop.push({ genome: new Float32Array(strong), result: null, score: -Infinity });
  for (let i = 1; i < POP; i++) {
    pop.push({ genome: perturb(strong, rand, WARM_START_SIGMA), result: null, score: -Infinity });
  }

  const t0 = Date.now();
  let bestEver: Candidate | null = null;
  let bestEverGen = 0;
  let solvedAt: number | null = null;

  for (let gen = 0; gen < MAX_GENS; gen++) {
    const elapsed = Date.now() - t0;
    if (elapsed >= HARD_TIME_BUDGET_MS) {
      console.warn(`[stop] hard time budget hit at gen ${gen} (elapsed=${(elapsed / 1000).toFixed(1)}s) — saving best-so-far`);
      break;
    }

    // Evaluate every unscored candidate this generation.
    for (let i = 0; i < pop.length; i++) {
      const c = pop[i];
      if (c.result !== null) continue;
      setChampion(c.genome);
      const tStart = Date.now();
      const r = runGame(seats);
      const dt = Date.now() - tStart;
      c.result = r;
      c.score = fitness(r);
      console.log(
        `  gen ${gen} cand ${i.toString().padStart(2)}: ` +
        `alive=${r.aliveCount} share=${(r.maxShare * 100).toFixed(1)}% ` +
        `score=${c.score.toFixed(2)} (${dt}ms)`,
      );
      if (!bestEver || c.score > bestEver.score) {
        bestEver = c;
        bestEverGen = gen;
      }
      if (meetsTargets(r)) {
        solvedAt = gen;
        console.log(`[solved] gen ${gen} cand ${i} hit both targets — alive=${r.aliveCount} share=${(r.maxShare * 100).toFixed(1)}%`);
        break;
      }
    }
    if (solvedAt !== null) break;

    // Sort descending by score; keep elites; fill rest by mutating an elite.
    pop.sort((a, b) => b.score - a.score);
    const elites = pop.slice(0, ELITES);
    const t = gen / Math.max(1, MAX_GENS - 1);
    const sigma = MUTATE_SIGMA_START + (MUTATE_SIGMA_END - MUTATE_SIGMA_START) * t;
    const nextPop: Candidate[] = elites.map((c) => ({ ...c })); // carry results, score
    while (nextPop.length < POP) {
      const parent = elites[Math.floor(rand() * elites.length)].genome;
      nextPop.push({ genome: mutate(parent, rand, sigma), result: null, score: -Infinity });
    }
    pop = nextPop;
    console.log(`[gen ${gen} summary] top=${elites[0].score.toFixed(2)} sigma_next=${sigma.toFixed(3)}`);
  }

  const winner = solvedAt !== null ? pop.find((c) => c.result !== null && meetsTargets(c.result))! : bestEver!;
  const wr = winner.result!;
  const savedAt = new Date().toISOString();
  const payload = {
    weights: Array.from(winner.genome),
    generation: 2000,
    bestFitness: winner.score,
    savedAt,
    note: 'trained for stalemate at tick 4000 (warm-started from strong.json, CPU-evolved)',
  };
  writeFileSync(outPath, JSON.stringify(payload));

  // Re-run the winning genome to capture the final state. The browser will
  // load this still and freeze on it for the scene's duration — bypasses any
  // live-presim divergence (cache, async champion races, etc).
  setChampion(winner.genome);
  const replay = runGameRecording(seats);
  const fs = replay.finalState;
  const still = {
    tick: fs.tick,
    numPlayers: fs.numPlayers,
    boardConfig: { radius: 18, distance: 2, numPlayers: NUM_PLAYERS },
    owners: fs.nodes.map((n) => n.owner),
    strengths: fs.nodes.map((n) => n.strength),
    flows: fs.flows.map((f) => [f.src, f.dst, f.player]),
    expected: { atTick: TARGET_TICK, alive: replay.result.aliveCount, maxShare: replay.result.maxShare },
    savedAt,
  };
  writeFileSync(stillPath, JSON.stringify(still));

  const totalGens = solvedAt !== null ? solvedAt + 1 : MAX_GENS;
  const elapsedS = ((Date.now() - t0) / 1000).toFixed(1);
  console.log('');
  console.log(`=== DONE (${elapsedS}s, ${totalGens} gen) ===`);
  console.log(`solved: ${solvedAt !== null ? `yes at gen ${solvedAt}` : `no — best-so-far from gen ${bestEverGen}`}`);
  console.log(`alive=${wr.aliveCount}  max_share=${(wr.maxShare * 100).toFixed(1)}%  fitness=${winner.score.toFixed(2)}`);
  console.log(`replay: alive=${replay.result.aliveCount} share=${(replay.result.maxShare * 100).toFixed(1)}% (sanity check, should match winner)`);
  console.log(`saved: ${outPath}`);
  console.log(`saved: ${stillPath}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
