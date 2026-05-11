/// <reference types="@webgpu/types" />

import { makeInitialState } from '../game/graph';
import { ATTACK_BONUS, MAX_STRENGTH, MIN_STRENGTH_TO_SEND, REGEN_PER_SEC, TRANSFER_PER_SEC } from '../game/state';
import { mulberry32 } from '../ai/rng';
import {
  buildNeighborTable,
  mutate,
  randomGenome,
  WEIGHTS_PER_GENOME,
  NN_IN_DIM,
  NN_HIDDEN,
  NN_OUT_DIM,
  NEIGHBOR_STRIDE,
  type Genome,
} from './genome';
import type { GPUCtx } from './runtime';
import nnShader from './shaders/nn.wgsl?raw';
import stepShader from './shaders/step.wgsl?raw';
import { setChampion } from './evolved';
import { FLOW_STRIDE } from './step';

export type EvolutionConfig = {
  popSize: number;
  numSeatsPerGame: number;
  gamesPerGenome: number;
  ticksPerGame: number;
  aiPeriodTicks: number;
  mutationSigma: number;
  elites: number;
  tournamentK: number;
  boardRadius: number;
  initStd: number;
};

export const DEFAULT_EVOLUTION_CONFIG: EvolutionConfig = {
  popSize: 12,
  numSeatsPerGame: 4,
  gamesPerGenome: 2,
  ticksPerGame: 500,
  aiPeriodTicks: 5,
  mutationSigma: 0.05,
  elites: 3,
  tournamentK: 3,
  boardRadius: 9, // ~271 cells — fast eval board (per user spec ~250)
  initStd: 0.5,
};

export type EvolutionState = {
  generation: number;
  bestFitness: number;        // best in current generation
  allTimeBest: number;        // best ever observed
  champion: Genome;           // all-time best genome
  population: Genome[];
  fitnessHistory: number[];
};

const STATE_KEY = 'flux-evolution-state';
const ENABLED_KEY = 'flux-evolution-enabled';

export function saveEvolutionState(s: EvolutionState): void {
  try {
    const payload = {
      generation: s.generation,
      bestFitness: s.bestFitness,
      allTimeBest: s.allTimeBest,
      champion: Array.from(s.champion),
      population: s.population.map(g => Array.from(g)),
      fitnessHistory: s.fitnessHistory,
    };
    localStorage.setItem(STATE_KEY, JSON.stringify(payload));
  } catch (err) {
    console.warn('saveEvolutionState failed:', err);
  }
}

export function loadEvolutionState(): EvolutionState | null {
  try {
    const raw = localStorage.getItem(STATE_KEY);
    if (!raw) return null;
    const d = JSON.parse(raw);
    return {
      generation: d.generation ?? 0,
      bestFitness: d.bestFitness ?? 0,
      allTimeBest: d.allTimeBest ?? 0,
      champion: new Float32Array(d.champion),
      population: (d.population as number[][]).map(g => new Float32Array(g)),
      fitnessHistory: d.fitnessHistory ?? [],
    };
  } catch (err) {
    console.warn('loadEvolutionState failed:', err);
    return null;
  }
}

export function clearEvolutionState(): void {
  localStorage.removeItem(STATE_KEY);
  localStorage.removeItem(ENABLED_KEY);
}

export function saveEvolveEnabled(enabled: boolean): void {
  try { localStorage.setItem(ENABLED_KEY, enabled ? '1' : '0'); } catch { /* ignore */ }
}

export function loadEvolveEnabled(): boolean {
  return localStorage.getItem(ENABLED_KEY) === '1';
}

export function createInitialPopulation(cfg: EvolutionConfig, rand: () => number): Genome[] {
  const pop: Genome[] = [];
  for (let i = 0; i < cfg.popSize; i++) pop.push(randomGenome(rand, cfg.initStd));
  return pop;
}

// Compute number of games per generation. Each genome plays in `gamesPerGenome`
// games. Total seat-slots = popSize * gamesPerGenome. Each game has numSeatsPerGame
// seats. So numGames = ceil(popSize * gamesPerGenome / numSeatsPerGame).
function plannedGames(cfg: EvolutionConfig): number {
  return Math.ceil((cfg.popSize * cfg.gamesPerGenome) / cfg.numSeatsPerGame);
}

// Random assignment: which genome plays in which (game, seat) slot.
// Returns seatGenome[game * numSeats + seat] = genome index, and inverse map
// genomeGames[genome] = array of (game, seat) for fitness aggregation.
function assignSeats(cfg: EvolutionConfig, numGames: number, rand: () => number): {
  seatGenome: Uint32Array;
  genomeGames: Array<Array<{ game: number; seat: number }>>;
} {
  const slots: number[] = [];
  for (let g = 0; g < cfg.popSize; g++) {
    for (let k = 0; k < cfg.gamesPerGenome; k++) slots.push(g);
  }
  // Shuffle.
  for (let i = slots.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [slots[i], slots[j]] = [slots[j], slots[i]];
  }
  // Pad with random genomes if slot count < numGames * numSeats.
  while (slots.length < numGames * cfg.numSeatsPerGame) {
    slots.push(Math.floor(rand() * cfg.popSize));
  }

  const seatGenome = new Uint32Array(numGames * cfg.numSeatsPerGame);
  const genomeGames: Array<Array<{ game: number; seat: number }>> = [];
  for (let i = 0; i < cfg.popSize; i++) genomeGames.push([]);
  for (let game = 0; game < numGames; game++) {
    for (let seat = 0; seat < cfg.numSeatsPerGame; seat++) {
      const idx = game * cfg.numSeatsPerGame + seat;
      const genome = slots[idx];
      seatGenome[idx] = genome;
      genomeGames[genome].push({ game, seat });
    }
  }
  return { seatGenome, genomeGames };
}

// One generation: evaluate population on GPU, compute fitness, select+mutate.
export async function runGeneration(
  ctx: GPUCtx,
  cfg: EvolutionConfig,
  evo: EvolutionState,
  rand: () => number,
): Promise<EvolutionState> {
  const state = makeInitialState(cfg.boardRadius, 2, cfg.numSeatsPerGame);
  const cells = state.nodes.length;
  const numGames = plannedGames(cfg);
  const maxFlows = Math.max(64, cells); // each owned cell can produce one flow
  const { seatGenome, genomeGames } = assignSeats(cfg, numGames, rand);

  // Per-game initial state (cloned).
  const stateStrength = new Float32Array(cells);
  const stateOwner = new Int32Array(cells);
  for (let i = 0; i < cells; i++) {
    stateStrength[i] = state.nodes[i].strength;
    stateOwner[i] = state.nodes[i].owner === null ? -1 : state.nodes[i].owner as number;
  }
  const strengthAll = new Float32Array(cells * numGames);
  const ownerAll = new Int32Array(cells * numGames);
  for (let g = 0; g < numGames; g++) {
    strengthAll.set(stateStrength, g * cells);
    ownerAll.set(stateOwner, g * cells);
  }

  const neighbors = buildNeighborTable(state);

  // Pack genomes into one flat buffer.
  const genomesFlat = new Float32Array(cfg.popSize * WEIGHTS_PER_GENOME);
  for (let i = 0; i < cfg.popSize; i++) {
    genomesFlat.set(evo.population[i], i * WEIGHTS_PER_GENOME);
  }

  // GPU buffers.
  const { device, queue } = ctx;
  const mkStorage = (bytes: number, init?: ArrayBufferView | ArrayBuffer) => {
    const buf = device.createBuffer({
      size: Math.max(16, bytes),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    if (init) queue.writeBuffer(buf, 0, init as BufferSource);
    return buf;
  };
  const mkUniform = (bytes: number, init: ArrayBufferView | ArrayBuffer) => {
    const buf = device.createBuffer({
      size: Math.max(16, bytes),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    queue.writeBuffer(buf, 0, init as BufferSource);
    return buf;
  };

  // Game state buffers — ping-pong like step.ts but here we manage manually.
  const strengthA = mkStorage(cells * numGames * 4, strengthAll);
  const strengthB = mkStorage(cells * numGames * 4);
  const ownerA = mkStorage(cells * numGames * 4, ownerAll);
  const ownerB = mkStorage(cells * numGames * 4);
  const flowsA = mkStorage(maxFlows * numGames * FLOW_STRIDE);
  const flowsB = mkStorage(maxFlows * numGames * FLOW_STRIDE);
  const numFlowsA = mkStorage(numGames * 4);
  const numFlowsB = mkStorage(numGames * 4);
  // Initialize flows to empty (player = -1).
  {
    const empty = new ArrayBuffer(maxFlows * numGames * FLOW_STRIDE);
    const dv = new DataView(empty);
    for (let i = 0; i < maxFlows * numGames; i++) dv.setInt32(i * FLOW_STRIDE + 8, -1, true);
    queue.writeBuffer(flowsA, 0, empty);
    queue.writeBuffer(flowsB, 0, empty);
    queue.writeBuffer(numFlowsA, 0, new Uint32Array(numGames));
    queue.writeBuffer(numFlowsB, 0, new Uint32Array(numGames));
  }
  const neighborsBuf = mkStorage(neighbors.byteLength, neighbors);
  const seatGenomeBuf = mkStorage(seatGenome.byteLength, seatGenome);
  const genomesBuf = mkStorage(genomesFlat.byteLength, genomesFlat);
  const actionBuf = mkStorage(cells * numGames * 4);

  // NN params uniform.
  const nnParams = new ArrayBuffer(48);
  {
    const dv = new DataView(nnParams);
    dv.setUint32(0, cells, true);
    dv.setUint32(4, numGames, true);
    dv.setUint32(8, cfg.numSeatsPerGame, true);
    dv.setUint32(12, NN_HIDDEN, true);
    dv.setUint32(16, NN_IN_DIM, true);
    dv.setUint32(20, NN_OUT_DIM, true);
    dv.setUint32(24, NEIGHBOR_STRIDE, true);
    dv.setUint32(28, maxFlows, true);
    dv.setFloat32(32, MAX_STRENGTH, true);
  }
  const nnParamsBuf = mkUniform(48, nnParams);

  // Step params uniform.
  const TICK_DT = 0.1;
  const stepParams = new ArrayBuffer(48);
  {
    const dv = new DataView(stepParams);
    dv.setUint32(0, cells, true);
    dv.setUint32(4, maxFlows, true);
    dv.setUint32(8, numGames, true);
    dv.setUint32(12, cfg.numSeatsPerGame, true);
    dv.setFloat32(16, REGEN_PER_SEC, true);
    dv.setFloat32(20, TRANSFER_PER_SEC, true);
    dv.setFloat32(24, ATTACK_BONUS, true);
    dv.setFloat32(28, MAX_STRENGTH, true);
    dv.setFloat32(32, MIN_STRENGTH_TO_SEND, true);
    dv.setFloat32(36, TICK_DT, true);
  }
  const stepParamsBuf = mkUniform(48, stepParams);

  // NN pipeline.
  const nnModule = device.createShaderModule({ code: nnShader });
  const nnLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const nnPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [nnLayout] });
  const nnForward = device.createComputePipeline({
    layout: nnPipelineLayout,
    compute: { module: nnModule, entryPoint: 'nn_forward' },
  });
  const nnBuildFlows = device.createComputePipeline({
    layout: nnPipelineLayout,
    compute: { module: nnModule, entryPoint: 'build_flows' },
  });

  // Step pipeline.
  const stepModule = device.createShaderModule({ code: stepShader });
  const stepLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const stepPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [stepLayout] });
  const stepCells = device.createComputePipeline({
    layout: stepPipelineLayout,
    compute: { module: stepModule, entryPoint: 'step_cells' },
  });
  const stepPrune = device.createComputePipeline({
    layout: stepPipelineLayout,
    compute: { module: stepModule, entryPoint: 'prune_flows' },
  });

  // Two step bind groups: A->B and B->A.
  const stepBgAB = device.createBindGroup({
    layout: stepLayout,
    entries: [
      { binding: 0, resource: { buffer: stepParamsBuf } },
      { binding: 1, resource: { buffer: strengthA } },
      { binding: 2, resource: { buffer: ownerA } },
      { binding: 3, resource: { buffer: flowsA } },
      { binding: 4, resource: { buffer: numFlowsA } },
      { binding: 5, resource: { buffer: strengthB } },
      { binding: 6, resource: { buffer: ownerB } },
      { binding: 7, resource: { buffer: flowsB } },
      { binding: 8, resource: { buffer: numFlowsB } },
    ],
  });
  const stepBgBA = device.createBindGroup({
    layout: stepLayout,
    entries: [
      { binding: 0, resource: { buffer: stepParamsBuf } },
      { binding: 1, resource: { buffer: strengthB } },
      { binding: 2, resource: { buffer: ownerB } },
      { binding: 3, resource: { buffer: flowsB } },
      { binding: 4, resource: { buffer: numFlowsB } },
      { binding: 5, resource: { buffer: strengthA } },
      { binding: 6, resource: { buffer: ownerA } },
      { binding: 7, resource: { buffer: flowsA } },
      { binding: 8, resource: { buffer: numFlowsA } },
    ],
  });

  // NN bind groups: reads "current" state and writes flows into "current" buffers
  // (since AI period happens AFTER previous step). We need two too: NN-on-A,
  // NN-on-B. They write to the SAME side flows (overwriting).
  const nnBgA = device.createBindGroup({
    layout: nnLayout,
    entries: [
      { binding: 0, resource: { buffer: nnParamsBuf } },
      { binding: 1, resource: { buffer: strengthA } },
      { binding: 2, resource: { buffer: ownerA } },
      { binding: 3, resource: { buffer: neighborsBuf } },
      { binding: 4, resource: { buffer: seatGenomeBuf } },
      { binding: 5, resource: { buffer: genomesBuf } },
      { binding: 6, resource: { buffer: actionBuf } },
      { binding: 7, resource: { buffer: flowsA } },
      { binding: 8, resource: { buffer: numFlowsA } },
    ],
  });
  const nnBgB = device.createBindGroup({
    layout: nnLayout,
    entries: [
      { binding: 0, resource: { buffer: nnParamsBuf } },
      { binding: 1, resource: { buffer: strengthB } },
      { binding: 2, resource: { buffer: ownerB } },
      { binding: 3, resource: { buffer: neighborsBuf } },
      { binding: 4, resource: { buffer: seatGenomeBuf } },
      { binding: 5, resource: { buffer: genomesBuf } },
      { binding: 6, resource: { buffer: actionBuf } },
      { binding: 7, resource: { buffer: flowsB } },
      { binding: 8, resource: { buffer: numFlowsB } },
    ],
  });

  // Run ticks. Current state = side `cur`. After each step, cur flips.
  let cur: 'A' | 'B' = 'A';
  const cellsX = Math.ceil(cells / 64);
  const gamesX = Math.ceil(numGames / 64);
  for (let t = 0; t < cfg.ticksPerGame; t++) {
    const enc = device.createCommandEncoder();
    if (t % cfg.aiPeriodTicks === 0) {
      // NN forward + build flows on current side.
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, cur === 'A' ? nnBgA : nnBgB);
      pass.setPipeline(nnForward);
      pass.dispatchWorkgroups(cellsX, numGames, 1);
      pass.setPipeline(nnBuildFlows);
      pass.dispatchWorkgroups(numGames, 1, 1);
      pass.end();
    }
    // Step: cur -> other.
    const pass2 = enc.beginComputePass();
    pass2.setBindGroup(0, cur === 'A' ? stepBgAB : stepBgBA);
    pass2.setPipeline(stepCells);
    pass2.dispatchWorkgroups(cellsX, numGames, 1);
    pass2.setPipeline(stepPrune);
    pass2.dispatchWorkgroups(gamesX, 1, 1);
    pass2.end();
    queue.submit([enc.finish()]);
    cur = cur === 'A' ? 'B' : 'A';
  }

  // Read back owner buffer to compute per-seat cell counts.
  const ownerSrc = cur === 'A' ? ownerA : ownerB;
  const staging = device.createBuffer({
    size: cells * numGames * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(ownerSrc, 0, staging, 0, cells * numGames * 4);
    queue.submit([enc.finish()]);
  }
  await staging.mapAsync(GPUMapMode.READ);
  const ownerFinal = new Int32Array(staging.getMappedRange().slice(0));
  staging.unmap(); staging.destroy();

  // Compute per-(game, seat) cell counts and aggregate fitness per genome.
  const fitness = new Float32Array(cfg.popSize);
  const games = new Float32Array(cfg.popSize);
  for (let game = 0; game < numGames; game++) {
    const counts = new Array(cfg.numSeatsPerGame).fill(0);
    for (let i = 0; i < cells; i++) {
      const o = ownerFinal[game * cells + i];
      if (o >= 0 && o < cfg.numSeatsPerGame) counts[o]++;
    }
    for (let seat = 0; seat < cfg.numSeatsPerGame; seat++) {
      const genome = seatGenome[game * cfg.numSeatsPerGame + seat];
      fitness[genome] += counts[seat];
      games[genome] += 1;
    }
  }
  for (let i = 0; i < cfg.popSize; i++) {
    if (games[i] > 0) fitness[i] /= games[i];
  }
  void genomeGames;

  // Destroy GPU buffers.
  for (const b of [
    strengthA, strengthB, ownerA, ownerB, flowsA, flowsB,
    numFlowsA, numFlowsB, neighborsBuf, seatGenomeBuf, genomesBuf, actionBuf,
    nnParamsBuf, stepParamsBuf,
  ]) b.destroy();

  // Selection: sort by fitness desc, take elites, fill rest by tournament.
  const order = Array.from({ length: cfg.popSize }, (_, i) => i)
    .sort((a, b) => fitness[b] - fitness[a]);
  const newPop: Genome[] = [];
  for (let e = 0; e < cfg.elites; e++) newPop.push(evo.population[order[e]]);
  while (newPop.length < cfg.popSize) {
    // Tournament of K
    let best = -1;
    for (let k = 0; k < cfg.tournamentK; k++) {
      const c = Math.floor(rand() * cfg.popSize);
      if (best < 0 || fitness[c] > fitness[best]) best = c;
    }
    newPop.push(mutate(evo.population[best], rand, cfg.mutationSigma));
  }

  const bestIdx = order[0];
  const bestFit = fitness[bestIdx];
  const genChampion = evo.population[bestIdx];
  const allTimeBest = bestFit > evo.allTimeBest ? bestFit : evo.allTimeBest;
  const champion = bestFit > evo.allTimeBest ? genChampion : evo.champion;
  setChampion(champion);

  return {
    generation: evo.generation + 1,
    bestFitness: bestFit,
    allTimeBest,
    champion,
    population: newPop,
    fitnessHistory: [...evo.fitnessHistory, bestFit].slice(-200),
  };
}

export function makeInitialEvolutionState(cfg: EvolutionConfig): EvolutionState {
  const rng = mulberry32(0xBABA);
  const population = createInitialPopulation(cfg, rng);
  setChampion(population[0]);
  return {
    generation: 0,
    bestFitness: 0,
    allTimeBest: 0,
    champion: population[0],
    population,
    fitnessHistory: [],
  };
}
