/**
 * v2 sphere scene renderer — three.js perspective camera, orbit controls,
 * instanced node spheres on a sphere shell, instanced "barbell" tubes for
 * every undirected edge.
 *
 * Edges are full 3D capsules joining adjacent cells. With zero pressure they
 * sit as thin uniform tubes — the static graph wireframe. With pressure, a
 * vertex shader applies a directional bulge + traveling wave; a fragment
 * shader pushes brightness with `vPressure` and clips the final color at 95%
 * (hue-preserving) so loaded edges glow without blowing out post-bloom.
 *
 * Bloom is restricted to the edge-tube layer; nodes/halo/globe stay matte.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js';
import type { Board } from '../board';
import { SPHERE_RADIUS } from '../board';
import { MAX_STRENGTH } from '../replay/format';

export const PLAYER_COLORS_HEX = [
  0x4a90e2, 0xe24a4a, 0x4ae28a, 0xe2c44a,
  0xa44ae2, 0xe2884a, 0x4ae2e2, 0xe24a88,
  0x88e24a, 0xe2e24a, 0x4a88e2, 0xa4e24a,
];
const NEUTRAL = new THREE.Color(0x666666);
const DEAD = new THREE.Color(0x18141a);
// Idle-edge color for the barbell tube — matches the prior LineSegments hue
// so the inactive graph still reads as "infrastructure" not "off."
const IDLE_EDGE_COLOR = new THREE.Color(0x18243a);

const tmpMatrix = new THREE.Matrix4();
const tmpPos = new THREE.Vector3();
const tmpScale = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion();
const tmpColor = new THREE.Color();
const ownerColors = PLAYER_COLORS_HEX.map(c => new THREE.Color(c));

// Owned-cell brightness band. Bloom is layer-restricted to the edge tubes,
// so cell brightness here just controls how readable the cells are. The
// values below are the *ceiling* — cells decay multiplicatively from this
// based on how long they've sat unchanged (see NODE_FADE_*).
const NODE_BLOOM_MIN = 0.85;
const NODE_BLOOM_RANGE = 0.55;   // → 0.85..1.40 across strength 0..1

// Activity-fade for nodes. A cell whose owner and quantized strength haven't
// changed for a while loses brightness exponentially toward NODE_FADE_FLOOR;
// the moment something about it changes, it snaps back to the ceiling. Lets
// the eye find the action — quiet regions desaturate while active fronts
// stay bright. Tau is in *displayed frames*, which at stride-1 ai-period-1
// matches game ticks; at higher strides each tick covers more game time.
const NODE_FADE_TAU = 50;        // 1/e fall-off in ~50 frames
const NODE_FADE_FLOOR = 0.30;    // long-quiet cells dim to 30% of their ceiling

// Globally-shared time uniform; barbell shader binds to this so all
// pulse/wave effects share a heartbeat.
const timeUniform: { value: number } = { value: 0 };

// Layer that bloom is restricted to. Only the edge-tube mesh enables this
// layer; nodes/globe/halo stay on layer 0. The camera's layer mask is
// flipped per-pass in render() rather than swapping materials, which avoids
// the BackSide halo turning into a solid black sphere during bloom.
const BLOOM_LAYER = 1;

export type Scene = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  bloomComposer: EffectComposer;
  finalComposer: EffectComposer;
  globe: THREE.Mesh;
  nodeInstanced: THREE.InstancedMesh;
  edgeInstanced: THREE.InstancedMesh;
  edgeMaterial: THREE.ShaderMaterial;
  edgePressureAttr: THREE.InstancedBufferAttribute;
  edgeDirAttr: THREE.InstancedBufferAttribute;
  flowToInstance: Map<number, { id: number; dir: number }>;
  // Per-cell fade tracking. Lazily allocated to nodeCount on first
  // updateScene call; reset when the board geometry is rebuilt.
  framesSinceChange: Int32Array | null;
  prevOwner: Int8Array | null;
  prevStrength: Uint8Array | null;
  lastFrameIndex: number;
  domElement: HTMLCanvasElement;
  nodeCount: number;
  edgeCount: number;
};

// Mean great-circle spacing between adjacent cells, used as the size unit
// for nodes and tube widths so a 2562-cell sphere doesn't render as a wall
// of touching dots and a 91-cell board doesn't get lost.
function cellSpacing(board: Board): number {
  let sum = 0;
  let count = 0;
  for (let c = 0; c < board.N; c++) {
    const cx = board.pos3d[c * 3];
    const cy = board.pos3d[c * 3 + 1];
    const cz = board.pos3d[c * 3 + 2];
    for (let k = 0; k < 6; k++) {
      const d = board.neighbors[c * 6 + k];
      if (d < 0 || d < c) continue;
      const dx = board.pos3d[d * 3] - cx;
      const dy = board.pos3d[d * 3 + 1] - cy;
      const dz = board.pos3d[d * 3 + 2] - cz;
      sum += Math.hypot(dx, dy, dz);
      count++;
    }
  }
  return count > 0 ? sum / count : SPHERE_RADIUS * 0.3;
}

// ---------------------------------------------------------------------------
// Barbell-tube shaders
// ---------------------------------------------------------------------------
// Vertex: distort the cylinder radially based on per-instance pressure and
// a traveling sine wave. Idle (aPressure=0): perfect uniform tube. Active:
// middle bulges, source-end gets a slight extra fattening, and a wave moves
// along the bar in the flow direction.
//
// Cylinder geometry is unit (radius=1, height=1), pre-rotated so position.y
// runs along the local axis, mapped to local t ∈ [0,1] via t = position.y + 0.5.
// instanceMatrix scales/positions/rotates each cylinder to span its edge.
//
// Three.js auto-injects `instanceMatrix` and `instanceColor` for any
// InstancedMesh; we declare them via the standard `<common>` chunks. Custom
// per-instance scalars (`aPressure`, `aDirection`) are bound via
// InstancedBufferAttribute.
const EDGE_VERTEX_SHADER = `
  attribute float aPressure;
  attribute float aDirection;
  varying float vT;
  varying float vPressure;
  varying float vDirection;
  varying vec3 vColor;
  uniform float uTime;
  void main() {
    vT = position.y + 0.5;            // 0..1 along the bar's local axis
    vPressure = aPressure;
    vDirection = aDirection;
    vColor = instanceColor;

    float t = vT;

    // Visibility floor: gate on aDirection (set ⇒ ±1), not aPressure, so
    // an outflow that's been set but hasn't yet built up overflow still
    // shows as an active connection at FLOW_FLOOR thickness/brightness.
    bool flowing = abs(aDirection) > 0.5;
    float effP = flowing ? max(aPressure, 0.30) : 0.0;

    // Bell envelope: bulge peaks at the midpoint and tapers naturally back
    // to idle thickness at the endpoints. This keeps the cylinder *inside*
    // each node sphere at the cylinder's tips — even at full active width
    // (radial ~1.75 at the midpoint), the cylinder endpoint is at idle
    // radius which fits inside the smallest sphere. Result: connectors
    // never visually overlay the node spheres they meet.
    float envelope = sin(t * 3.14159);              // 0..1..0 across length

    // Traveling wave; sign of aDirection sends it src→dst or dst→src.
    float waveT = aDirection * t;
    float wave = sin(uTime * 2.0 - waveT * 5.5) * 0.5 + 0.5;  // 0..1

    // Idle: 1.0×. Active midpoint peak: ~1.75×. Endpoints: 1.0× (matches idle).
    float radial = 1.0 + effP * (0.35 + 0.40 * wave) * envelope;

    // Source-end asymmetry: slight extra fattening at the originating end
    // so flow direction reads even when the wave is at a trough.
    float srcWeight = aDirection > 0.0 ? (1.0 - t)
                    : aDirection < 0.0 ? t
                    : 0.0;
    radial += effP * 0.20 * srcWeight;

    vec3 displaced = position;
    displaced.x *= radial;
    displaced.z *= radial;
    gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(displaced, 1.0);
  }
`;

// Fragment: brightness scales hard with pressure (idle 0.55× → loaded 4.5×),
// plus a directional spark that travels along the bar so the eye can read
// flow direction even on a static frame. Final color is clipped at 95% with
// a hue-preserving normalization so saturated edges don't blow out into
// pure white post-bloom.
const EDGE_FRAGMENT_SHADER = `
  uniform float uTime;
  varying float vT;
  varying float vPressure;
  varying float vDirection;
  varying vec3 vColor;
  void main() {
    // Visibility floor: any *set* outflow (vDirection != 0) snaps the
    // effective pressure up to FLOW_FLOOR so a connection is always clearly
    // readable — including the moment the seat sets an outflow but no
    // overflow has yet built up (pressure==0 but the edge is claimed).
    // Truly idle edges (no flow record at all → vDirection==0) stay dim.
    const float FLOW_FLOOR = 0.30;
    bool flowing = abs(vDirection) > 0.5;
    float effP = flowing ? max(vPressure, FLOW_FLOOR) : 0.0;

    // Spark traveling along the bar in the flow direction.
    float speed = 0.7 + effP * 1.6;
    float dirSign = vDirection >= 0.0 ? 1.0 : -1.0;
    float t = fract(uTime * speed - dirSign * vT);
    float spark = exp(-t * t * 80.0);
    float trail = exp(-t * 7.0) * 0.30 * effP;
    float energy = spark * (0.9 + 1.4 * effP) + trail;

    // Base glow scales with pressure. Slope dropped from 3.95 to 2.2 so
    // even a fully-loaded edge sits closer to the bloom threshold instead
    // of multiple stops past it; previously two crossing loaded edges blew
    // out to a featureless white blob.
    float baseBoost = 0.55 + 2.20 * effP;
    vec3 color = vColor * baseBoost + vColor * energy;

    // Hue-preserving 75% brightness cap (was 95%). Aggressive clip means
    // bloom never has anything close to pure-white to work from.
    float maxComp = max(max(color.r, color.g), color.b);
    color = (maxComp > 0.75) ? color * (0.75 / maxComp) : color;

    gl_FragColor = vec4(color, 1.0);
  }
`;

export function createScene(canvas: HTMLCanvasElement, board: Board): Scene {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x02030a);
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  // ACES tone mapping: smooth roll-off near white instead of hard clipping.
  // Combined with the 95% cap in the edge fragment, post-bloom highlights
  // stay punchy but never crater into a featureless blob.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(
    45,
    window.innerWidth / Math.max(window.innerHeight, 1),
    0.1, 1000,
  );
  camera.position.set(0, 0, SPHERE_RADIUS * 3.0);
  camera.lookAt(0, 0, 0);
  // Camera sees both layer 0 (everything else) and layer 1 (edge tubes).
  // render() narrows to BLOOM_LAYER for the bloom pass and restores both
  // for the final pass.
  camera.layers.enable(BLOOM_LAYER);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.8;
  controls.zoomSpeed = 0.7;
  controls.minDistance = SPHERE_RADIUS * 1.2;
  controls.maxDistance = SPHERE_RADIUS * 8;
  controls.enablePan = false;

  // Inner globe — kept dark so emissive edges pop.
  const globeGeom = new THREE.SphereGeometry(SPHERE_RADIUS * 0.985, 64, 48);
  const globeMat = new THREE.MeshBasicMaterial({
    color: 0x070a14, transparent: true, opacity: 0.92,
  });
  const globe = new THREE.Mesh(globeGeom, globeMat);
  scene.add(globe);

  // Atmospheric halo: backside fresnel shell.
  const haloGeom = new THREE.SphereGeometry(SPHERE_RADIUS * 1.06, 64, 48);
  const haloMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.BackSide,
    uniforms: { uColor: { value: new THREE.Color(0x4070ff) } },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vView;
      void main() {
        vNormal = normalize(normalMatrix * normal);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vView = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      varying vec3 vNormal;
      varying vec3 vView;
      void main() {
        float fres = pow(1.0 - abs(dot(vNormal, vView)), 2.6);
        gl_FragColor = vec4(uColor * fres * 1.4, fres * 0.85);
      }
    `,
  });
  const halo = new THREE.Mesh(haloGeom, haloMat);
  scene.add(halo);

  const built = buildSceneInstances(board);
  scene.add(built.nodeInstanced);
  scene.add(built.edgeInstanced);
  built.edgeInstanced.layers.enable(BLOOM_LAYER);

  // Selective bloom — bloomComposer sees only BLOOM_LAYER (edge tubes),
  // finalComposer renders everything and the ShaderPass adds bloom on top.
  const bloomComposer = new EffectComposer(renderer);
  bloomComposer.renderToScreen = false;
  bloomComposer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  bloomComposer.setSize(window.innerWidth, window.innerHeight);
  bloomComposer.addPass(new RenderPass(scene, camera));
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.50,   // strength  — was 1.00; halves the post-bloom intensity
    0.42,   // radius    — was 0.65; tighter halo so glow doesn't bleed across
    0.55,   // threshold — was 0.35; only the brightest edge shafts contribute
  );
  bloomComposer.addPass(bloomPass);

  const combinePass = new ShaderPass(new THREE.ShaderMaterial({
    uniforms: {
      baseTexture: { value: null },
      bloomTexture: { value: bloomComposer.renderTarget2.texture },
    },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform sampler2D baseTexture;
      uniform sampler2D bloomTexture;
      varying vec2 vUv;
      void main() {
        // 0.65 attenuates the bloom contribution so the halo never out-shouts
        // the bar itself. Combined with the bloom-pass strength + threshold,
        // this is the final brightness governor — drop further (e.g. 0.45)
        // for an even softer look.
        gl_FragColor = texture2D(baseTexture, vUv)
                     + vec4(0.65) * texture2D(bloomTexture, vUv);
      }
    `,
    defines: {},
  }), 'baseTexture');
  combinePass.needsSwap = true;

  const finalComposer = new EffectComposer(renderer);
  finalComposer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  finalComposer.setSize(window.innerWidth, window.innerHeight);
  finalComposer.addPass(new RenderPass(scene, camera));
  finalComposer.addPass(combinePass);
  finalComposer.addPass(new OutputPass());

  return {
    renderer, scene, camera, controls,
    bloomComposer, finalComposer,
    globe,
    nodeInstanced: built.nodeInstanced,
    edgeInstanced: built.edgeInstanced,
    edgeMaterial: built.edgeMaterial,
    edgePressureAttr: built.edgePressureAttr,
    edgeDirAttr: built.edgeDirAttr,
    flowToInstance: built.flowToInstance,
    framesSinceChange: null,
    prevOwner: null,
    prevStrength: null,
    lastFrameIndex: -1,
    domElement: canvas,
    nodeCount: board.N,
    edgeCount: built.edgeCount,
  };
}

// Base node radius — half of the prior 0.42 default so arrows/edges read as
// the dominant signal, not the cells.
const NODE_RADIUS_FRAC = 0.21;
const NODE_SCALE_MIN = 0.45;
const NODE_SCALE_RANGE = 1.0;

// Idle barbell-tube radius (unscaled). Bulge can push it up to ~1.55× this
// at full pressure middle — well below the node radius so the bar stays
// readable as "infrastructure" while a strong cell stays the visual anchor.
const EDGE_TUBE_FRAC = 0.07;

type Built = {
  nodeInstanced: THREE.InstancedMesh;
  edgeInstanced: THREE.InstancedMesh;
  edgeMaterial: THREE.ShaderMaterial;
  edgePressureAttr: THREE.InstancedBufferAttribute;
  edgeDirAttr: THREE.InstancedBufferAttribute;
  flowToInstance: Map<number, { id: number; dir: number }>;
  edgeCount: number;
};

function buildSceneInstances(board: Board): Built {
  // ---- Nodes ----
  const r = cellSpacing(board) * NODE_RADIUS_FRAC;
  const nodeGeom = new THREE.SphereGeometry(r, 14, 10);
  const nodeMat = new THREE.MeshBasicMaterial({ vertexColors: false });
  const nodeInstanced = new THREE.InstancedMesh(nodeGeom, nodeMat, board.N);
  nodeInstanced.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(board.N * 3), 3);
  for (let i = 0; i < board.N; i++) {
    tmpPos.set(board.pos3d[i * 3], board.pos3d[i * 3 + 1], board.pos3d[i * 3 + 2]);
    tmpScale.setScalar(1);
    tmpMatrix.compose(tmpPos, tmpQuat, tmpScale);
    nodeInstanced.setMatrixAt(i, tmpMatrix);
    nodeInstanced.setColorAt(i, NEUTRAL);
  }
  nodeInstanced.instanceMatrix.needsUpdate = true;
  if (nodeInstanced.instanceColor) nodeInstanced.instanceColor.needsUpdate = true;

  // ---- Barbell edges: one cylinder instance per undirected adjacency ----
  const tubeR = cellSpacing(board) * EDGE_TUBE_FRAC;
  // Cylinder of unit height (-0.5..+0.5 along y) and unit radius. instanceMatrix
  // scales (tubeR, length, tubeR), rotates local-y to span src→dst, translates
  // to the midpoint.
  const cylGeom = new THREE.CylinderGeometry(1, 1, 1, 14, 8, true);
  // Lift slightly so the tube sits just above the globe shell, matching the
  // node radius so endpoints visually meet the spheres.
  const lift = 1.005;

  // Walk every undirected adjacency once; build edge list + flow lookup.
  const edges: Array<{ a: number; b: number }> = [];
  const flowToInstance = new Map<number, { id: number; dir: number }>();
  for (let c = 0; c < board.N; c++) {
    for (let k = 0; k < 6; k++) {
      const d = board.neighbors[c * 6 + k];
      if (d < 0 || d < c) continue;        // dedupe undirected
      const id = edges.length;
      edges.push({ a: c, b: d });
      // c→d on this edge counts as direction +1; d→c as -1.
      flowToInstance.set(c * board.N + d, { id, dir: +1 });
      flowToInstance.set(d * board.N + c, { id, dir: -1 });
    }
  }
  const edgeCount = edges.length;

  const upY = new THREE.Vector3(0, 1, 0);
  const A = new THREE.Vector3();
  const B = new THREE.Vector3();
  const dir = new THREE.Vector3();
  const mid = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  const m = new THREE.Matrix4();

  // ShaderMaterial: instanceColor + the custom per-instance scalars below.
  const edgeMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: timeUniform },
    vertexShader: EDGE_VERTEX_SHADER,
    fragmentShader: EDGE_FRAGMENT_SHADER,
    transparent: false, depthWrite: true,
    blending: THREE.NormalBlending,
  });

  const edgeInstanced = new THREE.InstancedMesh(cylGeom, edgeMaterial, edgeCount);
  edgeInstanced.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(edgeCount * 3), 3);
  // Per-instance scalars — created once, mutated each frame.
  const pressures = new Float32Array(edgeCount);
  const dirs = new Float32Array(edgeCount);
  const edgePressureAttr = new THREE.InstancedBufferAttribute(pressures, 1);
  const edgeDirAttr = new THREE.InstancedBufferAttribute(dirs, 1);
  edgePressureAttr.setUsage(THREE.DynamicDrawUsage);
  edgeDirAttr.setUsage(THREE.DynamicDrawUsage);
  cylGeom.setAttribute('aPressure', edgePressureAttr);
  cylGeom.setAttribute('aDirection', edgeDirAttr);

  for (let i = 0; i < edgeCount; i++) {
    const { a, b } = edges[i];
    A.set(
      board.pos3d[a * 3] * lift,
      board.pos3d[a * 3 + 1] * lift,
      board.pos3d[a * 3 + 2] * lift,
    );
    B.set(
      board.pos3d[b * 3] * lift,
      board.pos3d[b * 3 + 1] * lift,
      board.pos3d[b * 3 + 2] * lift,
    );
    dir.subVectors(B, A);
    const length = dir.length() || 1;
    dir.multiplyScalar(1 / length);
    mid.addVectors(A, B).multiplyScalar(0.5);
    quat.setFromUnitVectors(upY, dir);
    scale.set(tubeR, length, tubeR);
    m.compose(mid, quat, scale);
    edgeInstanced.setMatrixAt(i, m);
    edgeInstanced.setColorAt(i, IDLE_EDGE_COLOR);
  }
  edgeInstanced.instanceMatrix.needsUpdate = true;
  if (edgeInstanced.instanceColor) edgeInstanced.instanceColor.needsUpdate = true;

  return {
    nodeInstanced, edgeInstanced, edgeMaterial,
    edgePressureAttr, edgeDirAttr,
    flowToInstance, edgeCount,
  };
}

export type FrameRender = {
  owners: Int8Array;
  strengths: Uint8Array;
  flows: { src: number; dst: number; player: number; pressure: number }[];
};

export function updateScene(
  s: Scene, board: Board, frame: FrameRender, frameIndex: number,
): void {
  const strengthScale = MAX_STRENGTH / 255;

  // Lazy-allocate fade buffers. They get re-allocated to the right size when
  // the board changes (rebuildSceneGeometry nulls them out).
  if (
    !s.framesSinceChange || s.framesSinceChange.length !== board.N ||
    !s.prevOwner || !s.prevStrength
  ) {
    s.framesSinceChange = new Int32Array(board.N);   // 0 = "just changed"
    s.prevOwner = new Int8Array(board.N);
    s.prevStrength = new Uint8Array(board.N);
    // Seed prev with the current frame so the first render doesn't flag every
    // cell as "just changed" (which would have them all flash at full bright).
    s.prevOwner.set(frame.owners);
    s.prevStrength.set(frame.strengths);
    s.lastFrameIndex = frameIndex;
  }

  // Only tick the fade counters when the *actual playback frame* advances —
  // otherwise paused/HMR re-renders would tick the counter at 60fps and fade
  // everything away in two seconds of just sitting on a frame.
  const frameAdvanced = frameIndex !== s.lastFrameIndex;
  if (frameAdvanced) {
    const fsc = s.framesSinceChange;
    const po = s.prevOwner;
    const ps = s.prevStrength;
    for (let i = 0; i < board.N; i++) {
      if (frame.owners[i] !== po[i] || frame.strengths[i] !== ps[i]) {
        fsc[i] = 0;
        po[i] = frame.owners[i];
        ps[i] = frame.strengths[i];
      } else {
        fsc[i] += 1;
      }
    }
    s.lastFrameIndex = frameIndex;
  }

  const fsc = s.framesSinceChange;
  for (let i = 0; i < board.N; i++) {
    const owner = frame.owners[i];
    const strength = frame.strengths[i] * strengthScale;
    const scale = NODE_SCALE_MIN + (strength / MAX_STRENGTH) * NODE_SCALE_RANGE;
    tmpPos.set(board.pos3d[i * 3], board.pos3d[i * 3 + 1], board.pos3d[i * 3 + 2]);
    tmpScale.setScalar(scale);
    tmpMatrix.compose(tmpPos, tmpQuat, tmpScale);
    s.nodeInstanced.setMatrixAt(i, tmpMatrix);

    // Activity fade: 1.0 immediately after a change → NODE_FADE_FLOOR
    // asymptotically. Applied multiplicatively to all cell types so a
    // stagnant region (neutral or owned) recedes while active fronts pop.
    const fade = NODE_FADE_FLOOR + (1 - NODE_FADE_FLOOR) * Math.exp(-fsc[i] / NODE_FADE_TAU);

    if (owner === -1) {
      tmpColor.copy(NEUTRAL).multiplyScalar(fade);
      s.nodeInstanced.setColorAt(i, tmpColor);
    } else if (owner === -2) {
      // Dead cells are inert wall-tile; no fade — keep them readable as
      // permanent obstacles, not "stuff that hasn't changed in a while."
      s.nodeInstanced.setColorAt(i, DEAD);
    } else {
      tmpColor.copy(ownerColors[owner % ownerColors.length]);
      const dim = NODE_BLOOM_MIN + NODE_BLOOM_RANGE * (strength / MAX_STRENGTH);
      tmpColor.multiplyScalar(dim * fade);
      s.nodeInstanced.setColorAt(i, tmpColor);
    }
  }
  s.nodeInstanced.instanceMatrix.needsUpdate = true;
  if (s.nodeInstanced.instanceColor) s.nodeInstanced.instanceColor.needsUpdate = true;

  // ---- Per-edge pressure / direction / color update ----
  // Reset every instance to "idle." This is O(edgeCount) per frame (~7680
  // entries at subdiv 4); cheaper than tracking which edges changed.
  const press = s.edgePressureAttr.array as Float32Array;
  const dirs = s.edgeDirAttr.array as Float32Array;
  const colorAttr = s.edgeInstanced.instanceColor;
  press.fill(0);
  dirs.fill(0);
  if (colorAttr) {
    const cArr = colorAttr.array as Float32Array;
    for (let i = 0; i < s.edgeCount; i++) {
      cArr[i * 3]     = IDLE_EDGE_COLOR.r;
      cArr[i * 3 + 1] = IDLE_EDGE_COLOR.g;
      cArr[i * 3 + 2] = IDLE_EDGE_COLOR.b;
    }
  }

  // Frame-relative pressure normalization, same trick as before.
  let frameMaxPressure = 0;
  for (let i = 0; i < frame.flows.length; i++) {
    const p = frame.flows[i].pressure;
    if (p > frameMaxPressure) frameMaxPressure = p;
  }
  const pressureNorm = Math.max(frameMaxPressure, 1.0);

  for (let i = 0; i < frame.flows.length; i++) {
    const f = frame.flows[i];
    const lookup = s.flowToInstance.get(f.src * board.N + f.dst);
    if (!lookup) continue;
    const id = lookup.id;
    const p = Math.pow(Math.max(0, Math.min(1, f.pressure / pressureNorm)), 0.7);
    // Take the first flow record on this edge (so a set-but-zero-pressure
    // outflow still claims the edge for its owner), and let any later flow
    // with higher pressure override. dirs[id] == 0 ⇔ "no flow yet this
    // frame"; once set, lookup.dir is always ±1 so the gate is reliable.
    if (dirs[id] === 0 || p > press[id]) {
      press[id] = p;
      dirs[id] = lookup.dir;
      const color = ownerColors[f.player % ownerColors.length];
      if (colorAttr) {
        const cArr = colorAttr.array as Float32Array;
        cArr[id * 3]     = color.r;
        cArr[id * 3 + 1] = color.g;
        cArr[id * 3 + 2] = color.b;
      }
    }
  }
  s.edgePressureAttr.needsUpdate = true;
  s.edgeDirAttr.needsUpdate = true;
  if (colorAttr) colorAttr.needsUpdate = true;
}

export function rebuildSceneGeometry(s: Scene, board: Board): void {
  s.scene.remove(s.nodeInstanced);
  s.nodeInstanced.dispose();
  s.nodeInstanced.geometry.dispose();
  (s.nodeInstanced.material as THREE.Material).dispose();
  s.scene.remove(s.edgeInstanced);
  s.edgeInstanced.dispose();
  s.edgeInstanced.geometry.dispose();
  // Keep the shader material — it's reusable across boards.

  const built = buildSceneInstances(board);
  s.scene.add(built.nodeInstanced);
  s.scene.add(built.edgeInstanced);
  built.edgeInstanced.layers.enable(BLOOM_LAYER);
  // Replace the per-edge material with the same shader material so the
  // updateScene attribute references stay valid through the rebuild.
  built.edgeInstanced.material = s.edgeMaterial;

  s.nodeInstanced = built.nodeInstanced;
  s.edgeInstanced = built.edgeInstanced;
  s.edgePressureAttr = built.edgePressureAttr;
  s.edgeDirAttr = built.edgeDirAttr;
  s.flowToInstance = built.flowToInstance;
  s.nodeCount = board.N;
  s.edgeCount = built.edgeCount;
  // Force fade buffers to be re-allocated on the next updateScene — old
  // arrays are sized to the previous board's N and seeded from its frames.
  s.framesSinceChange = null;
  s.prevOwner = null;
  s.prevStrength = null;
  s.lastFrameIndex = -1;
}

export function resizeRenderer(s: Scene): void {
  const w = window.innerWidth, h = window.innerHeight;
  s.renderer.setSize(w, h, false);
  s.bloomComposer.setSize(w, h);
  s.finalComposer.setSize(w, h);
  s.camera.aspect = w / Math.max(h, 1);
  s.camera.updateProjectionMatrix();
}

export function render(s: Scene): void {
  s.controls.update();
  timeUniform.value = performance.now() * 0.001;

  // Pass 1: bloom-only. Restrict the camera to BLOOM_LAYER so only the
  // edge-tube mesh renders into the bloom composer's target.
  s.camera.layers.set(BLOOM_LAYER);
  s.bloomComposer.render();

  // Pass 2: full scene render + additive bloom composite + tonemap output.
  s.camera.layers.enable(0);
  s.finalComposer.render();
}
