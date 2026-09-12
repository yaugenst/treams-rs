import init, {
  ScatteringSystem,
  direct_plane_field,
  cluster_target_gradient,
} from "./wasm/treams_wasm.js";
import {
  geometryValid,
  type State,
  type Result,
  type Particle,
} from "./model.js";
export { init };
const array = (values: number[]) => new Float64Array(values);
const power = (values: Float64Array) =>
  values.reduce((sum, v) => sum + v * v, 0);
const material = (epsilon: number, kappa = 0) => [
  epsilon,
  0.02,
  1,
  0,
  kappa,
  0,
];
const vacuum = [1, 0, 1, 0, 0, 0];
const clustered = (s: State) =>
  s.experiment === "particles" || s.experiment === "design";
const direction = (s: State) =>
  array([
    Math.cos((s.angle * Math.PI) / 180),
    Math.sin((s.angle * Math.PI) / 180),
    0,
  ]);
const radii = (s: State) => array(s.particles.map((p) => p.radius));
const positions = (s: State) =>
  array(s.particles.flatMap((p) => [p.x, p.y, 0]));
const epsilon = (s: State) =>
  array(s.particles.flatMap(() => [s.epsilon, 0.02]));
function sphere(s: State, k0: number, bare = false) {
  const shell = s.experiment === "shell" && !bare;
  return ScatteringSystem.sphere(
    5,
    k0,
    array(shell ? [s.radius, s.radius + s.shell] : [s.radius]),
    array([
      ...material(s.epsilon, s.experiment === "chirality" ? s.chirality : 0),
      ...(shell ? material(s.shellIndex ** 2) : []),
      ...vacuum,
    ]),
  );
}
function create(s: State, k0: number) {
  if (!clustered(s)) return sphere(s, k0);
  const solve =
    s.experiment === "particles" && s.compare
      ? ScatteringSystem.independent_cluster
      : ScatteringSystem.cluster;
  return solve(6, k0, radii(s), epsilon(s), positions(s));
}
// Only the current geometry is retained: illumination changes reuse its solved T matrix.
let cached: { key: string; system: ScatteringSystem } | undefined;
function systemFor(s: State, k0: number) {
  const key = JSON.stringify([
    s.experiment,
    s.wavelength,
    s.radius,
    s.epsilon,
    s.shell,
    s.shellIndex,
    s.chirality,
    s.compare,
    s.particles,
  ]);
  if (cached?.key !== key) {
    const system = create(s, k0);
    cached?.system.free();
    cached = { key, system };
  }
  return cached!.system;
}
function incidentFor(s: State, system: ScatteringSystem) {
  if (s.experiment !== "mixer")
    return system.plane_wave(direction(s), s.helicity);
  const a = new Float64Array(system.modes * 2);
  // Electric N multipoles are equal-weight helicities: l=1,m=0 and l=2,m=1.
  const first = (4 * Math.sqrt(1 - s.mixture)) / Math.SQRT2,
    second = (4 * Math.sqrt(s.mixture)) / Math.SQRT2,
    phase = (s.phase * Math.PI) / 180;
  for (const index of [2, 3]) a[index * 2] = first;
  for (const index of [12, 13]) {
    a[index * 2] = second * Math.cos(phase);
    a[index * 2 + 1] = second * Math.sin(phase);
  }
  return a;
}
function orderPowers(b: Float64Array) {
  const out: number[] = [];
  let index = 0;
  for (let l = 1; l <= 5; l++) {
    let sum = 0;
    for (let m = -l; m <= l; m++)
      for (let p = 0; p < 2; p++) {
        sum += b[index]! ** 2 + b[index + 1]! ** 2;
        index += 2;
      }
    out.push(sum);
  }
  return out;
}
function filterOrder(b: Float64Array, order: number) {
  let i = 0;
  for (let l = 1; l <= 5; l++)
    for (let m = -l; m <= l; m++)
      for (let p = 0; p < 2; p++) {
        if (l !== order) {
          b[i] = 0;
          b[i + 1] = 0;
        }
        i += 2;
      }
}
let spectrumCache: { key: string; values: Float64Array } | undefined;
function spectrumFor(s: State) {
  if (s.experiment !== "resonance") return new Float64Array();
  const key = JSON.stringify([s.radius, s.epsilon, s.helicity, s.order]);
  if (spectrumCache?.key === key) return spectrumCache.values;
  const values = new Float64Array(65 * 2);
  for (let i = 0; i < 65; i++) {
    const wavelength = 1.1 + (i / 64) * 1.4,
      k0 = (2 * Math.PI) / wavelength,
      system = sphere(s, k0);
    try {
      const b = system.scatter(system.plane_wave(array([1, 0, 0]), s.helicity));
      values[i * 2] = wavelength;
      const selectedPower = s.order ? orderPowers(b)[s.order - 1]! : power(b);
      values[i * 2 + 1] =
        selectedPower / (k0 * k0 * Math.PI * s.radius * s.radius);
    } finally {
      system.free();
    }
  }
  spectrumCache = { key, values };
  return values;
}
function targetScore(s: State) {
  const k0 = (2 * Math.PI) / s.wavelength,
    system = create(s, k0);
  try {
    const points = array([s.target[0], s.target[1], 0]);
    const field = system.electric_field(
      system.scatter(system.plane_wave(direction(s), s.helicity)),
      points,
      true,
    );
    const direct = direct_plane_field(k0, direction(s), s.helicity, points);
    return field.reduce((sum, v, i) => sum + (v + direct[i]!) ** 2, 0);
  } finally {
    system.free();
  }
}
export function improve(s: State): {
  state: State;
  message: string;
  accepted: boolean;
} {
  if (s.experiment !== "design")
    throw new Error("Improve is available in the target experiment");
  const gradient = cluster_target_gradient(
    6,
    (2 * Math.PI) / s.wavelength,
    radii(s),
    epsilon(s),
    positions(s),
    direction(s),
    s.helicity,
    array([s.target[0], s.target[1], 0]),
  );
  const before = gradient[0]!,
    offset = 1 + s.particles.length;
  const norm = Math.hypot(
    ...s.particles.flatMap((_, i) => [
      gradient[offset + 3 * i]!,
      gradient[offset + 3 * i + 1]!,
    ]),
  );
  if (!Number.isFinite(norm) || norm < 1e-12)
    return {
      state: s,
      accepted: false,
      message: "The position gradient is flat here. Try moving a particle.",
    };
  for (let attempt = 0; attempt < 9; attempt++) {
    const next = structuredClone(s),
      step = 0.12 / 2 ** attempt;
    next.particles.forEach((p, i) => {
      p.x += (step * gradient[offset + 3 * i]!) / norm;
      p.y += (step * gradient[offset + 3 * i + 1]!) / norm;
    });
    if (!geometryValid(next)) continue;
    const after = targetScore(next);
    if (after > before * (1 + 1e-7))
      return {
        state: next,
        accepted: true,
        message: `${before.toFixed(2)}× → ${after.toFixed(2)}× · +${((after / before - 1) * 100).toFixed(1)}%`,
      };
  }
  return {
    state: s,
    accepted: false,
    message:
      "No improving step fits here. Move a particle or target and try again.",
  };
}
export function simulate(id: number, s: State, n: number): Result {
  const started = performance.now(),
    k0 = (2 * Math.PI) / s.wavelength,
    system = systemFor(s, k0);
  const a = incidentFor(s, system),
    b = system.scatter(a);
  const orders = clustered(s)
    ? []
    : orderPowers(b).map((v) => v / (k0 * k0 * Math.PI * s.radius * s.radius));
  let score = power(b) / (k0 * k0),
    reference = 1;
  if (s.experiment === "resonance") score /= Math.PI * s.radius * s.radius;
  if (s.experiment === "mixer") score = power(b) / power(a);
  if (s.experiment === "shell") {
    const bare = sphere(s, k0, true);
    try {
      reference =
        power(bare.scatter(bare.plane_wave(direction(s), s.helicity))) /
        (k0 * k0);
    } finally {
      bare.free();
    }
  }
  if (s.experiment === "chirality")
    reference =
      power(system.scatter(system.plane_wave(direction(s), 1 - s.helicity))) /
      (k0 * k0);
  if (s.experiment === "resonance" && s.order) filterOrder(b, s.order);
  const particles: Particle[] = clustered(s)
    ? s.particles
    : [
        {
          x: 0,
          y: 0,
          radius: s.radius + (s.experiment === "shell" ? s.shell : 0),
        },
      ];
  const span = 3.2,
    points: number[] = [],
    indices: number[] = [];
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const px = ((x + 0.5) / n - 0.5) * span,
        py = (0.5 - (y + 0.5) / n) * span;
      if (
        particles.some(
          (p) => Math.hypot(px - p.x, py - p.y) <= p.radius * 1.015,
        )
      )
        continue;
      indices.push(y * n + x);
      points.push(px, py, 0);
    }
  if (s.experiment === "design") points.push(s.target[0], s.target[1], 0);
  const coords = array(points),
    values = system.electric_field(b, coords, true);
  // A multipole-only resonance view intentionally shows that scattered order.
  if (!(s.experiment === "resonance" && s.order)) {
    const incoming =
      s.experiment === "mixer"
        ? system.electric_field(a, coords, false)
        : direct_plane_field(k0, direction(s), s.helicity, coords);
    for (let i = 0; i < values.length; i++)
      values[i] = values[i]! + incoming[i]!;
  }
  const field = new Float64Array(n * n * 6);
  field.fill(NaN);
  let peak = 0;
  for (let i = 0; i < indices.length; i++) {
    const offset = indices[i]! * 6;
    let intensity = 0;
    for (let j = 0; j < 6; j++) {
      const value = values[i * 6 + j]!;
      field[offset + j] = value;
      intensity += value * value;
    }
    peak = Math.max(peak, intensity);
  }
  if (s.experiment === "design") score = power(values.slice(-6));
  if (s.experiment === "particles") score = peak;
  const gradient =
    s.experiment === "design"
      ? cluster_target_gradient(
          6,
          k0,
          radii(s),
          epsilon(s),
          positions(s),
          direction(s),
          s.helicity,
          array([s.target[0], s.target[1], 0]),
        )
      : new Float64Array();
  return {
    id,
    state: s,
    field,
    n,
    span,
    peak,
    score,
    reference,
    orders,
    spectrum: spectrumFor(s),
    gradient,
    milliseconds: performance.now() - started,
  };
}
