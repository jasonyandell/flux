/**
 * Event log reader for `public/v2/replays/events.jsonl`.
 *
 * The writer (python/flux_v2/replay.py::append_index) appends one JSON
 * line per replay. The viewer tails the file, filters by the user's
 * last-closed timestamp (persisted in localStorage), and surfaces a
 * new-arrivals count to the hamburger badge.
 *
 * Composition style: each step is a Kleisli arrow `A -> Promise<B>`
 * (i.e. an async function), wired together with `andThen` so the
 * pipeline reads top-to-bottom in one place.
 */

export type ReplayEvent = {
  ts: string;
  event: 'replay_added';
  file: string;
  iteration?: number;
  generation?: number;
  radius?: number;
  num_players?: number;
  kind?: string;
};

// Kleisli composition over Promise: (A -> Promise<B>) >=> (B -> Promise<C>)
//                                   ============================
//                                   = A -> Promise<C>
export function andThen<A, B, C>(
  f: (a: A) => Promise<B>,
  g: (b: B) => Promise<C>,
): (a: A) => Promise<C> {
  return async (a) => g(await f(a));
}

export const fetchEventsText = async (url: string): Promise<string> => {
  const res = await fetch(`${url}?t=${Date.now()}`, { cache: 'no-cache' });
  if (!res.ok) {
    // 404 is normal before any replay has been written — treat as empty.
    if (res.status === 404) return '';
    throw new Error(`events http ${res.status}`);
  }
  return res.text();
};

export const parseEvents = async (text: string): Promise<ReplayEvent[]> => {
  const out: ReplayEvent[] = [];
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (!t) continue;
    try {
      const obj = JSON.parse(t);
      if (obj && obj.event === 'replay_added' && typeof obj.ts === 'string') {
        out.push(obj as ReplayEvent);
      }
    } catch {
      // Truncated/corrupt line — skip; next poll picks up the rest.
    }
  }
  return out;
};

export const sinceTs = (sinceMs: number) =>
  async (events: ReplayEvent[]): Promise<ReplayEvent[]> => {
    if (!Number.isFinite(sinceMs) || sinceMs <= 0) return events;
    return events.filter((e) => {
      const ms = Date.parse(e.ts);
      return Number.isFinite(ms) && ms > sinceMs;
    });
  };

export type EventsTailer = {
  // Returns events newer than `sinceMs` (ms-since-epoch). The pipeline
  // itself is allocation-light; callers cache the result and recompute
  // the badge count whenever they re-render.
  fetchNewer: (sinceMs: number) => Promise<ReplayEvent[]>;
};

export function createEventsTailer(url: string): EventsTailer {
  return {
    fetchNewer(sinceMs) {
      const pipeline = andThen(
        () => fetchEventsText(url),
        andThen(parseEvents, sinceTs(sinceMs)),
      );
      return pipeline(undefined as void);
    },
  };
}
