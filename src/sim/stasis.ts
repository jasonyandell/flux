export function detectStasis(samples: readonly number[][], epsilon: number, windowSize: number): boolean {
  if (samples.length < windowSize || samples.length === 0) return false;

  // 1v1 endgame: let it play out — that's where the tricky finishing moves live.
  const latest = samples[samples.length - 1];
  let alive = 0;
  for (const c of latest) if (c > 0) alive++;
  if (alive <= 2) return false;

  // Cleanup phase: one player dominates and the runner-up has scraps. The
  // game just hasn't finished mopping them up yet. Absolute threshold (not a
  // percentage) so this only fires when the leader has genuinely earned it —
  // an early-game 80%/10%/10% split keeps stasis enabled because the runner-up
  // still has real territory.
  let max = 0, secondMax = 0;
  for (const c of latest) {
    if (c > max) { secondMax = max; max = c; }
    else if (c > secondMax) secondMax = c;
  }
  if (secondMax < 5) return false;

  const numPlayers = samples[0].length;
  for (let p = 0; p < numPlayers; p++) {
    let sum = 0;
    for (const s of samples) sum += s[p];
    const mean = sum / samples.length;
    let sq = 0;
    for (const s of samples) {
      const d = s[p] - mean;
      sq += d * d;
    }
    if (sq / samples.length >= epsilon) return false;
  }
  return true;
}
