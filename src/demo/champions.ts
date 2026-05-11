// Fetch helper for scene champions. Pure module: no three/lil-gui/DOM.
// The runner calls `loadSceneChampion(label)` between scenes and feeds the
// returned Float32Array into `setChampion` (or `null` for the untrained
// gen0 scene, which lets `ensureChampion` mint a fresh random genome).

export type SceneLabel = 'gen0' | 'gen50' | 'gen100' | 'gen2000' | 'gen20k';

type Index = Record<SceneLabel, string | null>;

let indexPromise: Promise<Index> | null = null;

function base(): string {
  // Vite injects BASE_URL (string ending in '/'). Falls back to '/' in non-Vite
  // contexts (tests) so absolute fetches still resolve when served from root.
  const env = (import.meta as unknown as { env?: { BASE_URL?: string } }).env;
  return env?.BASE_URL ?? '/';
}

function loadIndex(): Promise<Index> {
  if (!indexPromise) {
    indexPromise = fetch(`${base()}champions/index.json`).then((r) => {
      if (!r.ok) throw new Error(`champions/index.json: ${r.status}`);
      return r.json() as Promise<Index>;
    });
  }
  return indexPromise;
}

export async function loadSceneChampion(label: SceneLabel): Promise<Float32Array | null> {
  const index = await loadIndex();
  const filename = index[label];
  if (filename === null || filename === undefined) return null;
  const res = await fetch(`${base()}champions/${filename}`);
  if (!res.ok) throw new Error(`champions/${filename}: ${res.status}`);
  const data = (await res.json()) as { weights: number[] };
  if (!Array.isArray(data.weights)) throw new Error(`champions/${filename}: missing weights array`);
  return new Float32Array(data.weights);
}
