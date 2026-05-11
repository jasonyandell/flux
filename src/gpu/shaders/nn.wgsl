// Per-cell NN forward pass + flows builder.
//
// nn_forward: one thread per (cell, game). For owned cells, fetch the genome of
// the cell's owner-seat, compute 91-feature → 32 (tanh) → 19 forward pass,
// argmax. Write actionOut[game*cells + cell].
//
// build_flows: one thread per game. Walk actionOut, emit one flow per cell whose
// action picks a valid neighbor. Replaces flow state for the next step.

struct NNParams {
  cells: u32,
  numGames: u32,
  numSeats: u32,
  hidden: u32,
  inDim: u32,
  outDim: u32,
  neighborStride: u32,
  maxFlows: u32,
  maxStrength: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

struct FlowEnc {
  src: u32,
  dst: u32,
  player: i32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> P: NNParams;
@group(0) @binding(1) var<storage, read> strengthIn: array<f32>;
@group(0) @binding(2) var<storage, read> ownerIn: array<i32>;
@group(0) @binding(3) var<storage, read> neighbors: array<i32>;
@group(0) @binding(4) var<storage, read> seatGenome: array<u32>;
@group(0) @binding(5) var<storage, read> genomes: array<f32>;
@group(0) @binding(6) var<storage, read_write> actionBuf: array<i32>;
@group(0) @binding(7) var<storage, read_write> flowsOut: array<FlowEnc>;
@group(0) @binding(8) var<storage, read_write> numFlowsOut: array<u32>;

const MAX_HIDDEN: u32 = 32u;
const MAX_OUT: u32 = 19u;

@compute @workgroup_size(64)
fn nn_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell = gid.x;
  let game = gid.y;
  if (cell >= P.cells || game >= P.numGames) { return; }

  let base = game * P.cells;
  let myOwner = ownerIn[base + cell];

  if (myOwner < 0) {
    actionBuf[base + cell] = i32(P.outDim) - 1;
    return;
  }

  let seat = u32(myOwner);
  let genomeIdx = seatGenome[game * P.numSeats + seat];
  let weightsPerGenome = P.inDim * P.hidden + P.hidden + P.hidden * P.outDim + P.outDim;
  let g0 = genomeIdx * weightsPerGenome;
  let W1_off = g0;
  let b1_off = g0 + P.inDim * P.hidden;
  let W2_off = b1_off + P.hidden;
  let b2_off = W2_off + P.hidden * P.outDim;

  var h: array<f32, MAX_HIDDEN>;
  for (var j: u32 = 0u; j < P.hidden; j = j + 1u) {
    h[j] = genomes[b1_off + j];
  }

  let f0 = strengthIn[base + cell] / P.maxStrength;
  for (var j: u32 = 0u; j < P.hidden; j = j + 1u) {
    h[j] = h[j] + f0 * genomes[W1_off + j];
  }

  for (var n: u32 = 0u; n < P.neighborStride; n = n + 1u) {
    let nbId = neighbors[cell * P.neighborStride + n];
    var sFeat: f32 = 0.0;
    var isFriendA: f32 = 0.0;
    var isFriendB: f32 = 0.0;
    var isEnemy: f32 = 0.0;
    var isNeutral: f32 = 0.0;
    if (nbId >= 0) {
      sFeat = strengthIn[base + u32(nbId)] / P.maxStrength;
      let nbOwner = ownerIn[base + u32(nbId)];
      if (nbOwner == i32(seat)) {
        isFriendA = 1.0;
        isFriendB = 1.0;
      } else if (nbOwner < 0) {
        isNeutral = 1.0;
      } else {
        isEnemy = 1.0;
      }
    } else {
      isNeutral = 1.0;
    }
    let featBase = 1u + n * 5u;
    for (var j: u32 = 0u; j < P.hidden; j = j + 1u) {
      h[j] = h[j]
        + sFeat * genomes[W1_off + (featBase + 0u) * P.hidden + j]
        + isFriendA * genomes[W1_off + (featBase + 1u) * P.hidden + j]
        + isFriendB * genomes[W1_off + (featBase + 2u) * P.hidden + j]
        + isEnemy * genomes[W1_off + (featBase + 3u) * P.hidden + j]
        + isNeutral * genomes[W1_off + (featBase + 4u) * P.hidden + j];
    }
  }

  for (var j: u32 = 0u; j < P.hidden; j = j + 1u) {
    let x = h[j];
    let e1 = exp(x);
    let e2 = exp(-x);
    h[j] = (e1 - e2) / (e1 + e2);
  }

  var out: array<f32, MAX_OUT>;
  for (var k: u32 = 0u; k < P.outDim; k = k + 1u) {
    out[k] = genomes[b2_off + k];
  }
  for (var j: u32 = 0u; j < P.hidden; j = j + 1u) {
    let hj = h[j];
    for (var k: u32 = 0u; k < P.outDim; k = k + 1u) {
      out[k] = out[k] + hj * genomes[W2_off + j * P.outDim + k];
    }
  }

  var bestK: u32 = 0u;
  var bestV: f32 = out[0];
  for (var k: u32 = 1u; k < P.outDim; k = k + 1u) {
    if (out[k] > bestV) { bestV = out[k]; bestK = k; }
  }
  actionBuf[base + cell] = i32(bestK);
}

@compute @workgroup_size(1)
fn build_flows(@builtin(global_invocation_id) gid: vec3<u32>) {
  let game = gid.x;
  if (game >= P.numGames) { return; }
  let base = game * P.cells;
  let outBase = game * P.maxFlows;
  var write: u32 = 0u;
  for (var cell: u32 = 0u; cell < P.cells; cell = cell + 1u) {
    let owner = ownerIn[base + cell];
    if (owner < 0) { continue; }
    let a = actionBuf[base + cell];
    if (a < 0 || u32(a) >= P.neighborStride) { continue; }
    let nbId = neighbors[cell * P.neighborStride + u32(a)];
    if (nbId < 0) { continue; }
    if (write >= P.maxFlows) { break; }
    flowsOut[outBase + write] = FlowEnc(cell, u32(nbId), owner, 0u);
    write = write + 1u;
  }
  for (var f: u32 = write; f < P.maxFlows; f = f + 1u) {
    flowsOut[outBase + f] = FlowEnc(0u, 0u, -1, 0u);
  }
  numFlowsOut[game] = write;
}
