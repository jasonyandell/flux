import type { Replay } from '../replay/format';

export type RunHeader = {
  setReplay: (file: string | null, replay: Replay | null) => void;
  setOnCopyLink: (handler: () => void) => void;
};

function str(meta: Record<string, unknown>, key: string): string | null {
  const v = meta[key];
  return typeof v === 'string' && v.length > 0 ? v : null;
}

function num(meta: Record<string, unknown>, key: string): number | null {
  const v = meta[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function partsFor(file: string | null, replay: Replay | null): string[] {
  if (!file || !replay) return ['No replay loaded'];
  const meta = replay.header.metadata as Record<string, unknown>;
  const out = [file];
  const ruleset = str(meta, 'ruleset');
  if (ruleset) out.push(ruleset);
  const edgeAlpha = num(meta, 'edge_alpha');
  if (edgeAlpha !== null) out.push(`ea ${edgeAlpha}`);
  const seed = num(meta, 'seed');
  if (seed !== null) out.push(`seed ${seed}`);
  const winner = num(meta, 'winner');
  if (winner !== null && winner >= 0) out.push(`winner ${winner}`);
  const ticks = num(meta, 'ticks');
  if (ticks !== null) out.push(`${ticks} ticks`);
  return out;
}

export function createRunHeader(): RunHeader {
  const root = document.createElement('div');
  root.id = 'flux-v2-run-header';
  root.style.cssText =
    'position:fixed;top:max(52px,calc(env(safe-area-inset-top) + 52px));right:max(12px,env(safe-area-inset-right));' +
    'max-width:min(520px,calc(100vw - 32px));z-index:9;' +
    'display:flex;align-items:center;gap:8px;color:#ddd;' +
    'font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;' +
    'background:rgba(12,12,18,0.62);border:1px solid rgba(70,70,92,0.55);' +
    'border-radius:7px;padding:7px 8px;backdrop-filter:blur(8px);' +
    '-webkit-backdrop-filter:blur(8px);box-shadow:0 2px 14px rgba(0,0,0,0.32);';

  const text = document.createElement('div');
  text.style.cssText =
    'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:0.72;min-width:0;';
  text.textContent = 'No replay loaded';
  root.appendChild(text);

  const copyBtn = document.createElement('button');
  copyBtn.type = 'button';
  copyBtn.textContent = 'link';
  copyBtn.setAttribute('aria-label', 'Copy current replay link');
  copyBtn.style.cssText =
    'flex:0 0 auto;background:rgba(74,144,226,0.12);border:1px solid #2a2a3a;' +
    'border-radius:5px;color:#b9d6ff;font:inherit;font-size:10px;padding:4px 7px;' +
    'cursor:pointer;pointer-events:auto;';
  root.appendChild(copyBtn);

  let onCopy = () => {};
  copyBtn.addEventListener('click', () => onCopy());
  document.body.appendChild(root);

  return {
    setReplay(file, replay) {
      text.textContent = partsFor(file, replay).join(' · ');
      copyBtn.disabled = !file;
      copyBtn.style.opacity = file ? '1' : '0.45';
    },
    setOnCopyLink(handler) { onCopy = handler; },
  };
}
