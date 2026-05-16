/**
 * Event log reader for public/v2/replays/events.jsonl.
 *
 * The replay writer appends one JSON line per replay. The viewer tails the
 * file and uses a local "last closed" cursor to badge new arrivals without
 * diffing index snapshots.
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

export const fetchEventsText = async (url: string): Promise<string> => {
  const res = await fetch(`${url}?t=${Date.now()}`, { cache: 'no-cache' });
  if (!res.ok) {
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
      // Ignore partial/corrupt lines; the next poll will see a complete file.
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
  fetchNewer: (sinceMs: number) => Promise<ReplayEvent[]>;
};

export function createEventsTailer(url: string): EventsTailer {
  return {
    async fetchNewer(sinceMs) {
      const text = await fetchEventsText(url);
      return sinceTs(sinceMs)(await parseEvents(text));
    },
  };
}
