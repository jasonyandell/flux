export function detectStasis(samples: readonly number[][], epsilon: number, windowSize: number): boolean {
  if (samples.length < windowSize || samples.length === 0) return false;

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
