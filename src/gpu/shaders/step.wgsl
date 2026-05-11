// Port of src/game/step.ts. Parameters laid out across N parallel games.
// Per-game data is contiguous: strength[game*cells + i], owner[game*cells + i],
// flows[game*maxFlows + f] (with player == -1 meaning "slot empty").
// numFlows[game] gives active prefix length.

struct Params {
  cells: u32,
  maxFlows: u32,
  numGames: u32,
  numPlayers: u32,
  regen: f32,
  transfer: f32,
  attackBonus: f32,
  maxStrength: f32,
  minStrengthToSend: f32,
  dt: f32,
  inboundBonus: f32,
};

struct FlowEnc {
  src: u32,
  dst: u32,
  player: i32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> strengthIn: array<f32>;
@group(0) @binding(2) var<storage, read> ownerIn: array<i32>;
@group(0) @binding(3) var<storage, read> flowsIn: array<FlowEnc>;
@group(0) @binding(4) var<storage, read> numFlowsIn: array<u32>;
@group(0) @binding(5) var<storage, read_write> strengthOut: array<f32>;
@group(0) @binding(6) var<storage, read_write> ownerOut: array<i32>;
@group(0) @binding(7) var<storage, read_write> flowsOut: array<FlowEnc>;
@group(0) @binding(8) var<storage, read_write> numFlowsOut: array<atomic<u32>>;

const MAX_PLAYERS: u32 = 12u;

@compute @workgroup_size(64)
fn step_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell = gid.x;
  let game = gid.y;
  if (cell >= params.cells || game >= params.numGames) { return; }

  let base = game * params.cells;
  let myOwner = ownerIn[base + cell];
  var myStrength = strengthIn[base + cell];

  var force: array<f32, MAX_PLAYERS>;
  for (var p: u32 = 0u; p < MAX_PLAYERS; p = p + 1u) { force[p] = 0.0; }

  let flowsBase = game * params.maxFlows;
  let nF = numFlowsIn[game];
  let k = params.transfer * params.dt;

  var inboundFriendly: u32 = 0u;

  for (var f: u32 = 0u; f < nF; f = f + 1u) {
    let flow = flowsIn[flowsBase + f];
    if (flow.player < 0) { continue; }
    let srcOwner = ownerIn[base + flow.src];
    if (srcOwner != flow.player) { continue; }
    let srcStrength = strengthIn[base + flow.src];
    if (srcStrength < params.minStrengthToSend) { continue; }

    if (flow.src == cell) {
      force[u32(flow.player)] = force[u32(flow.player)] - k;
    }
    if (flow.dst == cell) {
      let dstOwner = ownerIn[base + flow.dst];
      let enemy = dstOwner != flow.player;
      let amt = select(k, k * (1.0 + params.attackBonus), enemy);
      force[u32(flow.player)] = force[u32(flow.player)] + amt;
      if (!enemy && i32(flow.player) == myOwner) {
        inboundFriendly = inboundFriendly + 1u;
      }
    }
  }

  if (myOwner >= 0) {
    let bonus = params.inboundBonus * f32(inboundFriendly);
    force[u32(myOwner)] = force[u32(myOwner)] + (params.regen + bonus) * params.dt;
  }

  var newOwner = myOwner;
  var newStrength = myStrength;

  if (myOwner >= 0) {
    newStrength = newStrength + force[u32(myOwner)];
    for (var p: u32 = 0u; p < params.numPlayers; p = p + 1u) {
      if (i32(p) != myOwner) { newStrength = newStrength - force[p]; }
    }
  } else {
    for (var p: u32 = 0u; p < params.numPlayers; p = p + 1u) {
      if (force[p] > 0.0) { newStrength = newStrength - force[p]; }
    }
  }

  if (newStrength < 0.0) {
    var bestP: i32 = -1;
    var best: f32 = 0.0;
    for (var p: u32 = 0u; p < params.numPlayers; p = p + 1u) {
      if (i32(p) != myOwner && force[p] > best) {
        best = force[p];
        bestP = i32(p);
      }
    }
    if (bestP >= 0) {
      newOwner = bestP;
      newStrength = -newStrength;
    } else {
      newStrength = 0.0;
    }
  }

  if (newStrength > params.maxStrength) { newStrength = params.maxStrength; }
  if (newStrength < 0.0) { newStrength = 0.0; }

  strengthOut[base + cell] = newStrength;
  ownerOut[base + cell] = newOwner;
}

// Second kernel: prune flows whose src.owner no longer matches flow.player.
// Mirrors JS: aliveFlows is kept in order, no-op flows just continue.
@compute @workgroup_size(64)
fn prune_flows(@builtin(global_invocation_id) gid: vec3<u32>) {
  let game = gid.x;
  if (game >= params.numGames) { return; }
  let base = game * params.cells;
  let inBase = game * params.maxFlows;
  let outBase = game * params.maxFlows;
  let nF = numFlowsIn[game];
  var write: u32 = 0u;
  for (var f: u32 = 0u; f < nF; f = f + 1u) {
    let flow = flowsIn[inBase + f];
    if (flow.player < 0) { continue; }
    let srcOwner = ownerIn[base + flow.src];
    if (srcOwner != flow.player) { continue; }
    flowsOut[outBase + write] = flow;
    write = write + 1u;
  }
  // Zero the rest for cleanliness.
  for (var f: u32 = write; f < params.maxFlows; f = f + 1u) {
    flowsOut[outBase + f] = FlowEnc(0u, 0u, -1, 0u);
  }
  atomicStore(&numFlowsOut[game], write);
}
