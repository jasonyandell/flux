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

export type ReplayStreamCallbacks = {
  onReplayReady?(replay: Replay): void;
  onProgress?(loadedFrames: number, totalFrames: number, replay: Replay): void;
};

async function decompressGzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([new Uint8Array(bytes)]).stream().pipeThrough(
    new DecompressionStream('gzip'),
  );
  const out = await new Response(stream).arrayBuffer();
  return new Uint8Array(out);
}

function appendBytes(a: Uint8Array, b: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function headerFromBuffer(buffer: ArrayBuffer, version: number, headerLen: number): ReplayHeader {
  const headerBytes = new Uint8Array(buffer, 10, headerLen);
  const headerJson = JSON.parse(new TextDecoder('utf-8').decode(headerBytes));
  return {
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
}

export async function parseReplay(buffer: ArrayBuffer): Promise<Replay> {
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC) throw new Error('replay: bad magic');
  const version = view.getUint8(4);
  if (version !== V3_VERSION) {
    throw new Error(`replay: expected v3 (version=${V3_VERSION}), got ${version}`);
  }
  const headerLen = view.getUint32(6, true);
  const hdr = headerFromBuffer(buffer, version, headerLen);

  const board = buildBoard(hdr.radius, hdr.numPlayers);
  if (board.N !== hdr.numNodes) {
    throw new Error(
      `replay: node count mismatch (header=${hdr.numNodes}, derived=${board.N})`,
    );
  }

  const compressed = new Uint8Array(buffer, 10 + headerLen);
  const raw = await decompressGzip(compressed);

  return parseRawFrames(hdr, board, raw);
}

export async function parseReplayResponse(
  res: Response,
  callbacks: ReplayStreamCallbacks = {},
): Promise<Replay> {
  if (!res.body || typeof DecompressionStream === 'undefined') {
    return parseReplay(await res.arrayBuffer());
  }

  const reader = res.body.getReader();
  let head: Uint8Array<ArrayBuffer> = new Uint8Array(0);
  while (head.length < 10) {
    const { value, done } = await reader.read();
    if (done || !value) throw new Error('replay: truncated before header');
    head = appendBytes(head, value);
  }

  const prefix = new DataView(head.buffer, head.byteOffset, head.byteLength);
  if (prefix.getUint32(0, true) !== MAGIC) throw new Error('replay: bad magic');
  const version = prefix.getUint8(4);
  if (version !== V3_VERSION) {
    throw new Error(`replay: expected v3 (version=${V3_VERSION}), got ${version}`);
  }
  const headerLen = prefix.getUint32(6, true);
  const headerEnd = 10 + headerLen;
  while (head.length < headerEnd) {
    const { value, done } = await reader.read();
    if (done || !value) throw new Error('replay: truncated header');
    head = appendBytes(head, value);
  }

  const hdr = headerFromBuffer(
    head.buffer.slice(head.byteOffset, head.byteOffset + headerEnd),
    version,
    headerLen,
  );
  const board = buildBoard(hdr.radius, hdr.numPlayers);
  if (board.N !== hdr.numNodes) {
    throw new Error(
      `replay: node count mismatch (header=${hdr.numNodes}, derived=${board.N})`,
    );
  }

  const replay: Replay = { header: hdr, board, frames: [] };
  let ready = false;
  const emitProgress = () => {
    if (!ready && replay.frames.length >= 2) {
      ready = true;
      callbacks.onReplayReady?.(replay);
    }
    callbacks.onProgress?.(replay.frames.length, hdr.numFrames, replay);
  };

  const firstCompressed = head.subarray(headerEnd);
  const compressedStream = new ReadableStream<Uint8Array>({
    async start(controller) {
      if (firstCompressed.length > 0) controller.enqueue(firstCompressed);
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (value) controller.enqueue(value);
        }
        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
  });
  const gzipStream = new DecompressionStream('gzip') as unknown as {
    readable: ReadableStream<Uint8Array>;
    writable: WritableStream<Uint8Array>;
  };
  const rawReader = compressedStream.pipeThrough(gzipStream).getReader();

  const N = hdr.numNodes;
  const outflowBytes = (N * K + 7) >> 3;
  const edgeScale = hdr.maxEdge / 255;
  let raw: Uint8Array<ArrayBuffer> = new Uint8Array(0);

  async function ensureAvailable(n: number): Promise<boolean> {
    while (raw.length < n) {
      const { value, done } = await rawReader.read();
      if (done) return false;
      if (value) raw = appendBytes(raw, value);
    }
    return true;
  }

  for (let i = 0; i < hdr.numFrames; i++) {
    const baseBytes = N + N + outflowBytes;
    if (!await ensureAvailable(baseBytes)) break;

    const outflowBits = raw.subarray(N + N, baseBytes);
    const active = countActiveOutflows(outflowBits, N);
    if (!await ensureAvailable(baseBytes + active)) break;

    const frameRaw = raw.subarray(0, baseBytes + active);
    replay.frames.push(decodeFrame(frameRaw, hdr, board, edgeScale));
    raw = raw.subarray(baseBytes + active);
    emitProgress();
  }

  if (!ready) callbacks.onReplayReady?.(replay);
  callbacks.onProgress?.(replay.frames.length, hdr.numFrames, replay);
  return replay;
}

function parseRawFrames(hdr: ReplayHeader, board: Board, raw: Uint8Array): Replay {
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

    const active = countActiveOutflows(outflowBits, N);

    const pressureBytes = new Uint8Array(raw.buffer, raw.byteOffset + off, active);
    off += active;

    frames[i] = decodeFrameParts(owners, strengths, outflowBits, pressureBytes, board, edgeScale);
  }

  return { header: hdr, board, frames };
}

function decodeFrame(raw: Uint8Array, hdr: ReplayHeader, board: Board, edgeScale: number): Frame {
  const N = hdr.numNodes;
  const outflowBytes = (N * K + 7) >> 3;
  const owners = new Int8Array(raw.slice(0, N).buffer);
  const strengths = raw.slice(N, N + N);
  const outflowBits = raw.subarray(N + N, N + N + outflowBytes);
  const pressureBytes = raw.subarray(N + N + outflowBytes);
  return decodeFrameParts(owners, strengths, outflowBits, pressureBytes, board, edgeScale);
}

function decodeFrameParts(
  owners: Int8Array,
  strengths: Uint8Array,
  outflowBits: Uint8Array,
  pressureBytes: Uint8Array,
  board: Board,
  edgeScale: number,
): Frame {
    // Reconstruct flows by iterating bits in order.
    const flows: Flow[] = new Array(pressureBytes.length);
    let pIdx = 0;
    for (let c = 0; c < owners.length; c++) {
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
  return { owners, strengths, flows: compact };
}

function countActiveOutflows(outflowBits: Uint8Array, numNodes: number): number {
  let active = 0;
  for (let b = 0; b < outflowBits.length; b++) {
    active += popcount8(outflowBits[b]);
  }
  const padBits = outflowBits.length * 8 - numNodes * K;
  if (padBits > 0) {
    const lastByte = outflowBits[outflowBits.length - 1];
    const padMask = (1 << padBits) - 1;
    active -= popcount8(lastByte & padMask);
  }
  return active;
}

function popcount8(x: number): number {
  x = x - ((x >> 1) & 0x55);
  x = (x & 0x33) + ((x >> 2) & 0x33);
  return (x + (x >> 4)) & 0x0f;
}
