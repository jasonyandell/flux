import { makeInitialState } from '../game/graph';
import { MAX_STRENGTH, type GameState, type Owner } from '../game/state';

const MAGIC = 0x52584c46; // "FLXR" little-endian as u32

export type ReplayHeader = {
  version: number;
  radius: number;
  distance: number;
  numPlayers: number;
  numNodes: number;
  tickStride: number;
  dtPerTickMs: number;
  numFrames: number;
  metadata: Record<string, unknown>;
};

export type Replay = {
  header: ReplayHeader;
  base: GameState;   // initial deterministic state (positions, edges)
  frames: ReplayFrame[];
};

export type ReplayFrame = {
  owners: Int8Array;    // length numNodes, -1 = neutral
  strengths: Uint8Array; // length numNodes, quantized 0..255 from 0..MAX_STRENGTH
  flows: { src: number; dst: number; player: number }[];
};

export function parseReplay(buffer: ArrayBuffer): Replay {
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC) throw new Error('replay: bad magic');
  const version = view.getUint8(4);
  if (version !== 1) throw new Error(`replay: unsupported version ${version}`);
  const radius = view.getInt16(6, true);
  const distance = view.getInt16(8, true);
  const numPlayers = view.getUint8(10);
  const numNodes = view.getUint32(12, true);
  const tickStride = view.getUint16(16, true);
  const dtPerTickMs = view.getUint16(18, true);
  const numFrames = view.getUint32(20, true);
  const metadataLen = view.getUint32(24, true);
  const metadataBytes = new Uint8Array(buffer, 28, metadataLen);
  const metadataText = new TextDecoder('utf-8').decode(metadataBytes);
  let metadata: Record<string, unknown> = {};
  try {
    metadata = JSON.parse(metadataText);
  } catch (_e) {
    // tolerate empty / malformed metadata
  }

  const base = makeInitialState(radius, distance, numPlayers);
  if (base.nodes.length !== numNodes) {
    throw new Error(`replay: node count mismatch (header=${numNodes}, derived=${base.nodes.length})`);
  }

  const frames: ReplayFrame[] = [];
  let off = 28 + metadataLen;
  for (let i = 0; i < numFrames; i++) {
    const owners = new Int8Array(buffer.slice(off, off + numNodes));
    off += numNodes;
    const strengths = new Uint8Array(buffer.slice(off, off + numNodes));
    off += numNodes;
    const flowCount = view.getUint16(off, true);
    off += 2;
    const flows = new Array<{ src: number; dst: number; player: number }>(flowCount);
    for (let f = 0; f < flowCount; f++) {
      const src = view.getUint16(off, true); off += 2;
      const dst = view.getUint16(off, true); off += 2;
      const player = view.getUint8(off); off += 1;
      flows[f] = { src, dst, player };
    }
    frames.push({ owners, strengths, flows });
  }

  return {
    header: {
      version, radius, distance, numPlayers, numNodes,
      tickStride, dtPerTickMs, numFrames, metadata,
    },
    base,
    frames,
  };
}

/** Mutate the base state's per-node owner/strength + flows from a recorded frame.
 *  Returns the same base state object (the renderer reads it immediately). */
export function applyFrame(base: GameState, frame: ReplayFrame): GameState {
  const scale = MAX_STRENGTH / 255;
  for (let i = 0; i < base.nodes.length; i++) {
    const o = frame.owners[i];
    const owner: Owner = o === -1 ? null : o;
    base.nodes[i].owner = owner;
    base.nodes[i].strength = frame.strengths[i] * scale;
  }
  base.flows.length = 0;
  for (const f of frame.flows) base.flows.push(f);
  return base;
}
