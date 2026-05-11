/// <reference types="@webgpu/types" />

import {
  ATTACK_BONUS,
  MAX_STRENGTH,
  MIN_STRENGTH_TO_SEND,
  REGEN_PER_SEC,
  TRANSFER_PER_SEC,
} from '../game/state';
import stepShader from './shaders/step.wgsl?raw';
import type { GPUCtx } from './runtime';

// Layout per game:
//   strength: f32[cells]
//   owner:    i32[cells]   (-1 = neutral)
//   flows:    FlowEnc[maxFlows] (player < 0 = empty slot)
//   numFlows: u32 (active prefix length)
//
// FlowEnc: { src: u32, dst: u32, player: i32, _pad: u32 } = 16 bytes
export const FLOW_STRIDE = 16; // bytes

export type StepConfig = {
  cells: number;
  maxFlows: number;
  numGames: number;
  numPlayers: number;
  dt: number;
};

export type GPUStep = {
  cfg: StepConfig;
  paramsBuf: GPUBuffer;
  strengthA: GPUBuffer;
  strengthB: GPUBuffer;
  ownerA: GPUBuffer;
  ownerB: GPUBuffer;
  flowsA: GPUBuffer;
  flowsB: GPUBuffer;
  numFlowsA: GPUBuffer;
  numFlowsB: GPUBuffer;
  bindGroupA: GPUBindGroup; // reads A, writes B
  bindGroupB: GPUBindGroup;
  pipelineStep: GPUComputePipeline;
  pipelinePrune: GPUComputePipeline;
  layout: GPUBindGroupLayout;
  swap: boolean;
};

export function createGPUStep(ctx: GPUCtx, cfg: StepConfig): GPUStep {
  const { device } = ctx;
  const cellsPerGame = cfg.cells;
  const totalCells = cfg.cells * cfg.numGames;
  const totalFlows = cfg.maxFlows * cfg.numGames;

  const paramsBuf = device.createBuffer({
    size: 48, // 10 fields, padded to 16-byte align
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const mkF32 = (n: number) => device.createBuffer({
    size: Math.max(4, n * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const mkI32 = mkF32;
  const mkU32 = mkF32;
  const mkFlows = (n: number) => device.createBuffer({
    size: Math.max(16, n * FLOW_STRIDE),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const strengthA = mkF32(totalCells);
  const strengthB = mkF32(totalCells);
  const ownerA = mkI32(totalCells);
  const ownerB = mkI32(totalCells);
  const flowsA = mkFlows(totalFlows);
  const flowsB = mkFlows(totalFlows);
  const numFlowsA = mkU32(cfg.numGames);
  const numFlowsB = mkU32(cfg.numGames);

  // Pack params: cells, maxFlows, numGames, numPlayers (u32), then 6 f32.
  const params = new ArrayBuffer(48);
  const dv = new DataView(params);
  dv.setUint32(0, cellsPerGame, true);
  dv.setUint32(4, cfg.maxFlows, true);
  dv.setUint32(8, cfg.numGames, true);
  dv.setUint32(12, cfg.numPlayers, true);
  dv.setFloat32(16, REGEN_PER_SEC, true);
  dv.setFloat32(20, TRANSFER_PER_SEC, true);
  dv.setFloat32(24, ATTACK_BONUS, true);
  dv.setFloat32(28, MAX_STRENGTH, true);
  dv.setFloat32(32, MIN_STRENGTH_TO_SEND, true);
  dv.setFloat32(36, cfg.dt, true);
  device.queue.writeBuffer(paramsBuf, 0, params);

  const layout = device.createBindGroupLayout({
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

  const module = device.createShaderModule({ code: stepShader });
  const pipelineStep = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: 'step_cells' },
  });
  const pipelinePrune = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: 'prune_flows' },
  });

  const bindGroupA = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
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
  const bindGroupB = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
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

  return {
    cfg,
    paramsBuf,
    strengthA, strengthB,
    ownerA, ownerB,
    flowsA, flowsB,
    numFlowsA, numFlowsB,
    bindGroupA, bindGroupB,
    pipelineStep, pipelinePrune,
    layout,
    swap: false,
  };
}

// Encode N step kernels back to back (no readback between steps).
export function recordSteps(gpu: GPUStep, encoder: GPUCommandEncoder, n: number) {
  const cellsX = Math.ceil(gpu.cfg.cells / 64);
  const cellsY = gpu.cfg.numGames;
  const gamesX = Math.ceil(gpu.cfg.numGames / 64);

  for (let i = 0; i < n; i++) {
    const bg = gpu.swap ? gpu.bindGroupB : gpu.bindGroupA;
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, bg);
    pass.setPipeline(gpu.pipelineStep);
    pass.dispatchWorkgroups(cellsX, cellsY, 1);
    pass.setPipeline(gpu.pipelinePrune);
    pass.dispatchWorkgroups(gamesX, 1, 1);
    pass.end();
    gpu.swap = !gpu.swap;
  }
}

// "Current" output buffers (the ones that the next step would READ).
export function currentBuffers(gpu: GPUStep) {
  // After recordSteps, the "current" state is the side we just wrote.
  // If swap is true after toggling, we wrote into A this round and would next read A.
  // Logic: bindGroupA reads from A and writes to B. After running, swap flips so
  // next round we use bindGroupB which reads from B. So "current" = the read-side
  // of the bind group we'd use NEXT, which is B if swap is true, else A.
  return gpu.swap
    ? { strength: gpu.strengthB, owner: gpu.ownerB, flows: gpu.flowsB, numFlows: gpu.numFlowsB }
    : { strength: gpu.strengthA, owner: gpu.ownerA, flows: gpu.flowsA, numFlows: gpu.numFlowsA };
}

export function uploadInitial(
  gpu: GPUStep,
  ctx: GPUCtx,
  perGame: { strength: Float32Array; owner: Int32Array; flows: Int32Array; numFlows: number }[],
) {
  const cells = gpu.cfg.cells;
  const maxFlows = gpu.cfg.maxFlows;
  const numGames = gpu.cfg.numGames;
  if (perGame.length !== numGames) throw new Error('perGame length mismatch');

  const strength = new Float32Array(cells * numGames);
  const owner = new Int32Array(cells * numGames);
  const flows = new ArrayBuffer(maxFlows * numGames * FLOW_STRIDE);
  const numFlowsArr = new Uint32Array(numGames);
  const dv = new DataView(flows);

  for (let g = 0; g < numGames; g++) {
    strength.set(perGame[g].strength, g * cells);
    owner.set(perGame[g].owner, g * cells);
    const src = perGame[g].flows;
    const baseB = g * maxFlows * FLOW_STRIDE;
    for (let f = 0; f < maxFlows; f++) {
      const o = baseB + f * FLOW_STRIDE;
      if (f < perGame[g].numFlows) {
        dv.setUint32(o, src[f * 3], true);
        dv.setUint32(o + 4, src[f * 3 + 1], true);
        dv.setInt32(o + 8, src[f * 3 + 2], true);
      } else {
        dv.setUint32(o, 0, true);
        dv.setUint32(o + 4, 0, true);
        dv.setInt32(o + 8, -1, true);
      }
      dv.setUint32(o + 12, 0, true);
    }
    numFlowsArr[g] = perGame[g].numFlows;
  }

  const cur = currentBuffers(gpu);
  ctx.queue.writeBuffer(cur.strength, 0, strength.buffer, strength.byteOffset, strength.byteLength);
  ctx.queue.writeBuffer(cur.owner, 0, owner.buffer, owner.byteOffset, owner.byteLength);
  ctx.queue.writeBuffer(cur.flows, 0, flows);
  ctx.queue.writeBuffer(cur.numFlows, 0, numFlowsArr.buffer, numFlowsArr.byteOffset, numFlowsArr.byteLength);
}

export async function downloadState(
  gpu: GPUStep,
  ctx: GPUCtx,
): Promise<{ strength: Float32Array; owner: Int32Array; flows: Int32Array; numFlows: Uint32Array }> {
  const cells = gpu.cfg.cells;
  const maxFlows = gpu.cfg.maxFlows;
  const numGames = gpu.cfg.numGames;
  const cur = currentBuffers(gpu);

  const sBytes = cells * numGames * 4;
  const oBytes = cells * numGames * 4;
  const fBytes = maxFlows * numGames * FLOW_STRIDE;
  const nBytes = numGames * 4;

  const stagingS = ctx.device.createBuffer({ size: sBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const stagingO = ctx.device.createBuffer({ size: oBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const stagingF = ctx.device.createBuffer({ size: fBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const stagingN = ctx.device.createBuffer({ size: nBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const enc = ctx.device.createCommandEncoder();
  enc.copyBufferToBuffer(cur.strength, 0, stagingS, 0, sBytes);
  enc.copyBufferToBuffer(cur.owner, 0, stagingO, 0, oBytes);
  enc.copyBufferToBuffer(cur.flows, 0, stagingF, 0, fBytes);
  enc.copyBufferToBuffer(cur.numFlows, 0, stagingN, 0, nBytes);
  ctx.queue.submit([enc.finish()]);

  await Promise.all([
    stagingS.mapAsync(GPUMapMode.READ),
    stagingO.mapAsync(GPUMapMode.READ),
    stagingF.mapAsync(GPUMapMode.READ),
    stagingN.mapAsync(GPUMapMode.READ),
  ]);

  const strength = new Float32Array(stagingS.getMappedRange().slice(0));
  const owner = new Int32Array(stagingO.getMappedRange().slice(0));
  const flowsRaw = new Int32Array(stagingF.getMappedRange().slice(0));
  const numFlows = new Uint32Array(stagingN.getMappedRange().slice(0));

  stagingS.unmap(); stagingS.destroy();
  stagingO.unmap(); stagingO.destroy();
  stagingF.unmap(); stagingF.destroy();
  stagingN.unmap(); stagingN.destroy();

  // Convert flows: stride 16 bytes = 4 i32. Pack as [src, dst, player] triples ignoring pad.
  const flows = new Int32Array(maxFlows * numGames * 3);
  for (let i = 0; i < maxFlows * numGames; i++) {
    flows[i * 3] = flowsRaw[i * 4];
    flows[i * 3 + 1] = flowsRaw[i * 4 + 1];
    flows[i * 3 + 2] = flowsRaw[i * 4 + 2];
  }

  return { strength, owner, flows, numFlows };
}

export function destroyGPUStep(gpu: GPUStep) {
  gpu.paramsBuf.destroy();
  gpu.strengthA.destroy(); gpu.strengthB.destroy();
  gpu.ownerA.destroy(); gpu.ownerB.destroy();
  gpu.flowsA.destroy(); gpu.flowsB.destroy();
  gpu.numFlowsA.destroy(); gpu.numFlowsB.destroy();
}
