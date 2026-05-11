/// <reference types="@webgpu/types" />

import type { GameState } from '../game/state';
import { applyAction, step } from '../game/step';
import { makeInitialState } from '../game/graph';
import type { GPUCtx } from './runtime';
import { createGPUStep, recordSteps, uploadInitial, downloadState, destroyGPUStep } from './step';

const TICK_DT = 0.1;

function gameStateToBuffers(s: GameState, maxFlows: number) {
  const strength = new Float32Array(s.nodes.length);
  const owner = new Int32Array(s.nodes.length);
  for (let i = 0; i < s.nodes.length; i++) {
    strength[i] = s.nodes[i].strength;
    owner[i] = s.nodes[i].owner === null ? -1 : s.nodes[i].owner as number;
  }
  const flows = new Int32Array(maxFlows * 3);
  for (let f = 0; f < s.flows.length && f < maxFlows; f++) {
    flows[f * 3] = s.flows[f].src;
    flows[f * 3 + 1] = s.flows[f].dst;
    flows[f * 3 + 2] = s.flows[f].player;
  }
  return { strength, owner, flows, numFlows: Math.min(s.flows.length, maxFlows) };
}

export type ParityResult = {
  ok: boolean;
  maxStrengthDiff: number;
  ownerMismatches: number;
  flowMismatches: number;
  ticksTested: number;
};

export async function runParityTest(ctx: GPUCtx, opts?: {
  ticks?: number;
  radius?: number;
  numPlayers?: number;
  tolerance?: number;
  seed?: number;
}): Promise<ParityResult> {
  const ticks = opts?.ticks ?? 50;
  const radius = opts?.radius ?? 6; // smaller board for parity (still ~127 cells)
  const numPlayers = opts?.numPlayers ?? 4;
  const tol = opts?.tolerance ?? 1e-3;
  const seed = opts?.seed ?? 1;

  let jsState = makeInitialState(radius, 2, numPlayers);

  // Seed some flows deterministically using the seed.
  function pseudoRand(s: number): number {
    let x = (s | 0) >>> 0;
    x = Math.imul(x ^ (x >>> 16), 0x7feb352d) >>> 0;
    x = Math.imul(x ^ (x >>> 15), 0x846ca68b) >>> 0;
    return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
  }

  // Pre-seed: give each player some scattered cells so flows can fire.
  // Easiest: just inject some owned cells around each seat.
  const startCells = jsState.nodes.length;
  void startCells;
  // Manually add an action sequence: every 5 ticks, try toggling some flows.
  const actionSchedule = new Map<number, { src: number; dst: number; player: number }[]>();
  for (let t = 0; t < ticks; t += 5) {
    const acts: { src: number; dst: number; player: number }[] = [];
    for (let p = 0; p < numPlayers; p++) {
      for (let k = 0; k < 3; k++) {
        const src = Math.floor(pseudoRand(seed + t * 31 + p * 7 + k * 11) * jsState.nodes.length);
        const dst = Math.floor(pseudoRand(seed + t * 31 + p * 7 + k * 11 + 1) * jsState.nodes.length);
        acts.push({ src, dst, player: p });
      }
    }
    actionSchedule.set(t, acts);
  }

  const cells = jsState.nodes.length;
  const maxFlows = Math.min(512, cells * 2);

  const gpu = createGPUStep(ctx, {
    cells,
    maxFlows,
    numGames: 1,
    numPlayers,
    dt: TICK_DT,
  });

  // We will sync state JS -> GPU at each step rather than letting GPU run
  // multiple ticks alone (so action toggles match exactly). Actually, since
  // we test step parity, we just sync the START state once, then run K ticks
  // in lockstep, applying actions BEFORE each step on both sides.
  let maxDiff = 0;
  let ownerMismatches = 0;
  let flowMismatches = 0;

  // Initial upload
  const init = gameStateToBuffers(jsState, maxFlows);
  uploadInitial(gpu, ctx, [init]);

  for (let t = 0; t < ticks; t++) {
    // Apply scheduled actions on JS
    const sched = actionSchedule.get(t);
    if (sched) {
      for (const a of sched) {
        jsState = applyAction(jsState, {
          kind: 'toggleFlow', src: a.src, dst: a.dst, player: a.player,
        });
      }
      // Re-upload JS state to GPU to mirror action effect (we treat actions as
      // an external mutation outside of step()).
      const snap = gameStateToBuffers(jsState, maxFlows);
      uploadInitial(gpu, ctx, [snap]);
    }

    // Step both
    jsState = step(jsState, TICK_DT);
    const enc = ctx.device.createCommandEncoder();
    recordSteps(gpu, enc, 1);
    ctx.queue.submit([enc.finish()]);

    // Compare every few ticks (or just at the end) to limit readback cost.
    if (t === ticks - 1 || t % 10 === 9) {
      const dl = await downloadState(gpu, ctx);
      for (let i = 0; i < cells; i++) {
        const jsStrength = jsState.nodes[i].strength;
        const gpuStrength = dl.strength[i];
        const d = Math.abs(jsStrength - gpuStrength);
        if (d > maxDiff) maxDiff = d;
        const jsOwner = jsState.nodes[i].owner === null ? -1 : jsState.nodes[i].owner as number;
        if (jsOwner !== dl.owner[i]) ownerMismatches++;
      }
      const jsFlows = jsState.flows;
      const gpuFlowCount = dl.numFlows[0];
      if (jsFlows.length !== gpuFlowCount) flowMismatches++;
      // Compare flow contents as sets.
      const jsSet = new Set(jsFlows.map(f => `${f.src}-${f.dst}-${f.player}`));
      for (let f = 0; f < gpuFlowCount; f++) {
        const k = `${dl.flows[f * 3]}-${dl.flows[f * 3 + 1]}-${dl.flows[f * 3 + 2]}`;
        if (!jsSet.has(k)) flowMismatches++;
      }
    }
  }

  destroyGPUStep(gpu);

  return {
    ok: maxDiff <= tol && ownerMismatches === 0 && flowMismatches === 0,
    maxStrengthDiff: maxDiff,
    ownerMismatches,
    flowMismatches,
    ticksTested: ticks,
  };
}

