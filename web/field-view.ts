// Extend exterior samples under the vector particle mask so image interpolation
// never blends the dark missing-data colour into the exterior field. This copy
// is for drawing only: the solver's masked values and scores stay untouched.
export function padField(field: Float64Array, n: number): Float64Array {
  const padded = field.slice();
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const offset = (y * n + x) * 6;
      if (Number.isFinite(field[offset])) continue;
      let weight = 0;
      const sum = new Float64Array(6);
      for (let dy = -2; dy <= 2; dy++)
        for (let dx = -2; dx <= 2; dx++) {
          const xx = x + dx,
            yy = y + dy;
          if (xx < 0 || xx >= n || yy < 0 || yy >= n) continue;
          const source = (yy * n + xx) * 6;
          if (!Number.isFinite(field[source])) continue;
          const w = 1 / (dx * dx + dy * dy);
          weight += w;
          for (let j = 0; j < 6; j++) sum[j] = sum[j]! + w * field[source + j]!;
        }
      if (weight)
        for (let j = 0; j < 6; j++) padded[offset + j] = sum[j]! / weight;
    }
  return padded;
}

// A labelled display gain, independent of animation phase. Never rescale data.
export function patternGain(field: Float64Array): number {
  let power = 0,
    count = 0;
  for (let i = 0; i < field.length; i += 6) {
    if (!Number.isFinite(field[i])) continue;
    for (let j = 0; j < 6; j++) power += field[i + j]! ** 2;
    count++;
  }
  return power > 0 ? Math.max(1, Math.sqrt(count / power)) : 1;
}
