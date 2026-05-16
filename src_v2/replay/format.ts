/**
 * FLXR v3 binary replay format reader.
 *
 * Layout (little-endian):
 *   magic        4 bytes  "FLXR"
 *   version      u8       = 3
 *   reserved     u8
 *   header_len   u32      length of JSON header blob
 *   header_json  bytes    UTF-8 JSON: radius, num_players, num_nodes,
 *                                     tick_stride, dt_per_tick_ms, num_frames,
 *                                     max_strength, max_edge, metadata
 *   frames_gz    bytes    gzip-compressed frame stream (to EOF)
 *
 * Frame stream (concatenated, fixed-size per game):
 *   owners            N bytes int8     -2 dead, -1 neutral, 0..P-1 seat
 *   strengths         N bytes uint8    quantized 0..255 / [0, max_strength]
 *   outflow_bits      ceil(N*K/8)      bit i: cell (i // K), slot (i % K)
 *   pressure_bytes    popcount bytes   uint8 quantized / [0, max_edge],
 *                                      same iteration order as outflow_bits
 *
 * v1/v2 readers don't speak v3. Historical replays are disposable
 * (regenerable in seconds) — see wiki/topics/v2-overnight-research.md.
 */
import { buildBoard, K, type Board } from '../board';

const MAGIC = 0x52584c46;   // "FLXR" LE u32

export const V3_VERSION = 3;

export type ReplayHeader = {
  version: number;
  radius: number;
  numPlayers: number;
  numNodes: number;
  tickStride: number;
  dtPerTickMs: number;
  numFrames: number;
  maxStrength: number;
  maxEdge: number;
  metadata: Record<string, unknown>;
};

export type Flow = {
  src: number;
  dst: number;
  player: number;
  pressure: number;
};

export type Frame = {
  owners: Int8Array;
  strengths: Uint8Array;       // de-quantize via header.maxStrength / 255
  flows: Flow[];
};

export type Replay = {
  header: ReplayHeader;
  board: Board;
  frames: Frame[];
};

async function decompressGzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([new Uint8Array(bytes)]).stream().pipeThrough(
    new DecompressionStream('gzip'),
  );
  const out = await new Response(stream).arrayBuffer();
  return new Uint8Array(out);
}

export async function parseReplay(buffer: ArrayBuffer): Promise<Replay> {
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC) throw new Error('replay: bad magic');
  const version = view.getUint8(4);
  if (version !== V3_VERSION) {
    throw new Error(`replay: expected v3 (version=${V3_VERSION}), got ${version}`);
  }
  const headerLen = view.getUint32(6, true);
  const headerBytes = new Uint8Array(buffer, 10, headerLen);
  const headerJson = JSON.parse(new TextDecoder('utf-8').decode(headerBytes));
  const hdr: ReplayHeader = {
    version,
    radius: headerJson.radius,
    numPlayers: headerJson.num_players,
    numNodes: headerJson.num_nodes,
    tickStride: headerJson.tick_stride,
    dtPerTickMs: headerJson.dt_per_tick_ms,
    numFrames: headerJson.num_frames,
    maxStrength: headerJson.max_strength,
    maxEdge: headerJson.max_edge,
    metadata: headerJson.metadata ?? {},
  };

  const board = buildBoard(hdr.radius, hdr.numPlayers);
  if (board.N !== hdr.numNodes) {
    throw new Error(
      `replay: node count mismatch (header=${hdr.numNodes}, derived=${board.N})`,
    );
  }

  const compressed = new Uint8Array(buffer, 10 + headerLen);
  const raw = await decompressGzip(compressed);

  const N = hdr.numNodes;
  const outflowBytes = (N * K + 7) >> 3;
  const edgeScale = hdr.maxEdge / 255;

  const frames: Frame[] = new Array(hdr.numFrames);
  let off = 0;
  for (let i = 0; i < hdr.numFrames; i++) {
    const owners = new Int8Array(raw.buffer, raw.byteOffset + off, N).slice();
    off += N;
    const strengths = new Uint8Array(raw.buffer, raw.byteOffset + off, N).slice();
    off += N;
    const outflowBits = new Uint8Array(raw.buffer, raw.byteOffset + off, outflowBytes);
    off += outflowBytes;

    // Count active outflow bits to know how many pressure bytes follow.
    let active = 0;
    for (let b = 0; b < outflowBytes; b++) {
      let v = outflowBits[b];
      v = v - ((v >> 1) & 0x55);
      v = (v & 0x33) + ((v >> 2) & 0x33);
      active += (((v + (v >> 4)) & 0x0f));
    }
    // Trim trailing pad bits (set to 0 by the writer but defensive).
    const padBits = outflowBytes * 8 - N * K;
    if (padBits > 0) {
      // Subtract any spurious bits set in the pad region. Writer should never
      // emit them, but be defensive.
      const lastByte = outflowBits[outflowBytes - 1];
      const padMask = (1 << padBits) - 1;
      active -= popcount8(lastByte & padMask);
    }

    const pressureBytes = new Uint8Array(raw.buffer, raw.byteOffset + off, active);
    off += active;

    // Reconstruct flows by iterating bits in order.
    const flows: Flow[] = new Array(active);
    let pIdx = 0;
    for (let c = 0; c < N; c++) {
      const ownerC = owners[c];
      for (let k = 0; k < K; k++) {
        const bitIdx = c * K + k;
        const byteIdx = bitIdx >> 3;
        const inByte = 7 - (bitIdx & 7);    // packbits big-bit-order
        if (((outflowBits[byteIdx] >> inByte) & 1) === 0) continue;
        const d = board.neighbors[c * K + k];
        const pressure = pressureBytes[pIdx] * edgeScale;
        pIdx++;
        if (d < 0) continue;                 // off-grid; skip rendering
        flows[pIdx - 1] = { src: c, dst: d, player: ownerC, pressure };
      }
    }
    // Compact out any holes from off-grid skips.
    const compact: Flow[] = [];
    for (let f = 0; f < flows.length; f++) {
      if (flows[f]) compact.push(flows[f]);
    }
    frames[i] = { owners, strengths, flows: compact };
  }

  return { header: hdr, board, frames };
}

function popcount8(x: number): number {
  x = x - ((x >> 1) & 0x55);
  x = (x & 0x33) + ((x >> 2) & 0x33);
  return (x + (x >> 4)) & 0x0f;
}
