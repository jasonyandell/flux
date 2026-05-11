export type Overlay = {
  root: HTMLDivElement;
  title: HTMLDivElement;
  caption: HTMLDivElement;
  showTitle: (text: string) => void;
  hideTitle: () => void;
  showCaption: (text: string) => void;
  hideCaption: () => void;
  destroy: () => void;
};

const MONO = 'ui-monospace,SFMono-Regular,Menlo,monospace';

export function createOverlay(): Overlay {
  const root = document.createElement('div');
  root.id = 'flux-demo-overlay';
  root.style.cssText =
    'position:fixed;inset:0;pointer-events:none;z-index:11;font-family:' + MONO + ';color:#fff;';

  const title = document.createElement('div');
  title.style.cssText =
    'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);' +
    'font-size:clamp(40px,8vw,96px);letter-spacing:8px;font-weight:bold;' +
    'text-shadow:0 2px 16px rgba(0,0,0,0.85);opacity:0;transition:opacity 0.6s ease;' +
    'white-space:nowrap;';
  root.appendChild(title);

  const caption = document.createElement('div');
  caption.style.cssText =
    'position:absolute;left:50%;bottom:18%;transform:translateX(-50%);' +
    'font-size:clamp(20px,3.2vw,40px);letter-spacing:3px;font-weight:bold;' +
    'text-shadow:0 2px 12px rgba(0,0,0,0.85);opacity:0;transition:opacity 0.5s ease;' +
    'white-space:nowrap;text-align:center;';
  root.appendChild(caption);

  document.body.appendChild(root);

  return {
    root,
    title,
    caption,
    showTitle: (text) => { title.textContent = text; title.style.opacity = '1'; },
    hideTitle: () => { title.style.opacity = '0'; },
    showCaption: (text) => { caption.textContent = text; caption.style.opacity = '1'; },
    hideCaption: () => { caption.style.opacity = '0'; },
    destroy: () => { root.remove(); },
  };
}
