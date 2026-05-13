/**
 * FLXR v2 replay format reader.
 *
 * Layout (little-endian) — same fixed header as v1 with version=2.
 * Per-frame flow record is 6 bytes (src u16, dst u16, player u8, pressure_q u8).
 */
import { buildBoard, type Board } from '../board';

const MAGIC = 0x52584c46;   // "FLXR" LE u32

export const V2_VERSION = 2;
export const MAX_STRENGTH = 100.0;
export const MAX_EDGE = 100.0;

export type ReplayHeader = {
  version: number;
  radius: number;
  numPlayers: number;
  numNodes: number;
  tickStride: number;
  dtPerTickMs: number;
  numFrames: number;
  metadata: Record<string, unknown>;
};

export type Frame = {
  owners: Int8Array;          // length N, -1 = neutral, -2 = dead
  strengths: Uint8Array;      // length N, quantized 0..255 from 0..MAX_STRENGTH
  flows: {
    src: number;
    dst: number;
    player: number;
    pressure: number;          // de-quantized to 0..MAX_EDGE
  }[];
};

export type Replay = {
  header: ReplayHeader;
  board: Board;
  frames: Frame[];
};

export function parseReplay(buffer: ArrayBuffer): Replay {
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC) throw new Error('replay: bad magic');
  const version = view.getUint8(4);
  if (version !== V2_VERSION) {
    throw new Error(`replay: expected v2 (version=${V2_VERSION}), got ${version}`);
  }
  const radius = view.getInt16(6, true);
  // distance slot is unused in v2 (K=6); skip.
  const numPlayers = view.getUint8(10);
  const numNodes = view.getUint32(12, true);
  const tickStride = view.getUint16(16, true);
  const dtPerTickMs = view.getUint16(18, true);
  const numFrames = view.getUint32(20, true);
  const metadataLen = view.getUint32(24, true);
  const metadataBytes = new Uint8Array(buffer, 28, metadataLen);
  const metadataText = new TextDecoder('utf-8').decode(metadataBytes);
  let metadata: Record<string, unknown> = {};
  try { metadata = JSON.parse(metadataText); } catch (_e) {}

  const board = buildBoard(radius, numPlayers);
  if (board.N !== numNodes) {
    throw new Error(`replay: node count mismatch (header=${numNodes}, derived=${board.N})`);
  }

  const frames: Frame[] = [];
  let off = 28 + metadataLen;
  const edgeScale = MAX_EDGE / 255;
  for (let i = 0; i < numFrames; i++) {
    const owners = new Int8Array(buffer.slice(off, off + numNodes));
    off += numNodes;
    const strengths = new Uint8Array(buffer.slice(off, off + numNodes));
    off += numNodes;
    const flowCount = view.getUint16(off, true);
    off += 2;
    const flows = new Array<{ src: number; dst: number; player: number; pressure: number }>(flowCount);
    for (let f = 0; f < flowCount; f++) {
      const src = view.getUint16(off, true); off += 2;
      const dst = view.getUint16(off, true); off += 2;
      const player = view.getUint8(off); off += 1;
      const pressureQ = view.getUint8(off); off += 1;
      flows[f] = { src, dst, player, pressure: pressureQ * edgeScale };
    }
    frames.push({ owners, strengths, flows });
  }

  return {
    header: { version, radius, numPlayers, numNodes, tickStride, dtPerTickMs, numFrames, metadata },
    board,
    frames,
  };
}
