#!/usr/bin/env node
// Generate deterministic placeholder champion JSONs for the showcase demo's
// gen100 / gen200 / gen1000 scenes. No real saved snapshots exist for these
// generations, so we seed `mulberry32` and emit increasingly-spread random
// genomes that read as "less weak" without claiming any real training.
//
// `mulberry32` and `gaussian` are inlined from src/ai/rng.ts and src/gpu/genome.ts
// so this can be run with plain `node scripts/gen-champions.mjs` (no TS toolchain).

import { writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const WEIGHTS_PER_GENOME = 3571; // 91*32 + 32 + 32*19 + 19

function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand) {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function randomGenome(rand, std) {
  const g = new Array(WEIGHTS_PER_GENOME);
  for (let i = 0; i < g.length; i++) g[i] = gaussian(rand) * std;
  return g;
}

const __dirname = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(__dirname, '..', 'public', 'champions');
mkdirSync(outDir, { recursive: true });

const specs = [
  { name: 'gen100',  generation: 100,  seed: 100,  std: 0.05 },
  { name: 'gen200',  generation: 200,  seed: 200,  std: 0.15 },
  { name: 'gen1000', generation: 1000, seed: 1000, std: 0.30 },
];

const savedAt = new Date().toISOString();
for (const s of specs) {
  const rng = mulberry32(s.seed);
  const weights = randomGenome(rng, s.std);
  const payload = {
    weights,
    generation: s.generation,
    bestFitness: null,
    savedAt,
    note: 'placeholder, random-seeded',
  };
  const path = resolve(outDir, `${s.name}.json`);
  writeFileSync(path, JSON.stringify(payload));
  console.log(`wrote ${path} (seed=${s.seed} std=${s.std} weights=${weights.length})`);
}
