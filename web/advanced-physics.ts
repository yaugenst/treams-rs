import init, {
  metasurface,
  crystal_spectrum,
  crystal_field,
} from "./wasm/treams_wasm.js";
export { init };
export type ArrayState = {
  kind: "array";
  wavelength: number;
  period: number;
  radius: number;
  angle: number;
};
export type CrystalState = {
  kind: "crystal";
  frequency: number;
  index: number;
  fill: number;
  periods: number;
};
export type AdvancedState = ArrayState | CrystalState;
export type Point = {
  values: Float64Array;
  field?: Float64Array;
  positions?: Float64Array;
};
export type Curve = {
  axis: Float64Array;
  values: Float64Array;
  stride: number;
  gaps: number;
};
export const initialArray = (): ArrayState => ({
  kind: "array",
  wavelength: 1.55,
  period: 0.8,
  radius: 0.23,
  angle: 0,
});
export const initialCrystal = (): CrystalState => ({
  kind: "crystal",
  frequency: 0.27,
  index: 2.6,
  fill: 0.64,
  periods: 8,
});
export const limits = (s: AdvancedState): [number, number] =>
  s.kind === "array" ? [0.65, 1.65] : [0.08, 0.65];
export const selected = (s: AdvancedState) =>
  s.kind === "array" ? s.wavelength : s.frequency;
const linspace = (a: number, b: number, n: number) =>
  Float64Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
const arrayPoint = (s: ArrayState, wavelength: number) =>
  metasurface(4, s.radius, s.period, wavelength, s.angle, 12.25, 0, 0);
export function point(s: AdvancedState): Point {
  if (s.kind === "array") return { values: arrayPoint(s, s.wavelength) };
  const positions = linspace(-1, s.periods + 1, 257);
  return {
    values: crystal_spectrum(
      1.45,
      s.index,
      s.fill,
      s.periods,
      new Float64Array([s.frequency]),
    ),
    positions,
    field: crystal_field(
      1.45,
      s.index,
      s.fill,
      s.periods,
      s.frequency,
      positions,
    ),
  };
}
let cached: { key: string; curve: Curve } | undefined;
export async function spectrum(
  s: AdvancedState,
  cancelled: () => boolean,
): Promise<Curve | undefined> {
  const key = JSON.stringify(
    s.kind === "array"
      ? [s.kind, s.radius, s.period, s.angle]
      : [s.kind, s.index, s.fill, s.periods],
  );
  if (cached?.key === key) return cached.curve;
  const [lo, hi] = limits(s),
    axis = linspace(lo, hi, s.kind === "array" ? 65 : 161);
  if (s.kind === "crystal") {
    const curve = {
      axis,
      values: crystal_spectrum(1.45, s.index, s.fill, s.periods, axis),
      stride: 5,
      gaps: 0,
    };
    cached = { key, curve };
    return curve;
  }
  const values = new Float64Array(axis.length * 3);
  let gaps = 0;
  for (let i = 0; i < axis.length; i++) {
    if (cancelled()) return;
    try {
      values.set(arrayPoint(s, axis[i]!).subarray(0, 3), i * 3);
    } catch (error) {
      // Exact diffraction thresholds are unsupported; show a break, not invented data.
      if (!String(error).toLowerCase().includes("graz")) throw error;
      values.fill(NaN, i * 3, (i + 1) * 3);
      gaps++;
    }
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  }
  const curve = { axis, values, stride: 3, gaps };
  cached = { key, curve };
  return curve;
}
