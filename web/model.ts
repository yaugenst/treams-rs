export type Experiment =
  "particles" | "resonance" | "mixer" | "shell" | "chirality" | "design";
export interface Particle {
  x: number;
  y: number;
  radius: number;
}
export interface State {
  experiment: Experiment;
  wavelength: number;
  angle: number;
  radius: number;
  epsilon: number;
  shell: number;
  shellIndex: number;
  chirality: number;
  helicity: number;
  phase: number;
  mixture: number;
  order: number;
  compare: boolean;
  particles: Particle[];
  target: [number, number];
}
export interface Result {
  id: number;
  state: State;
  field: Float64Array;
  n: number;
  span: number;
  peak: number;
  score: number;
  reference: number;
  orders: number[];
  spectrum: Float64Array;
  gradient: Float64Array;
  milliseconds: number;
  improvement?: { message: string; accepted: boolean };
}
export const presets: Record<
  Experiment,
  { title: string; short: string; description: string; explanation: string }
> = {
  particles: {
    title: "Move the particles",
    short: "Shape the light",
    description: "Bring two particles together. Watch the space between them.",
    explanation:
      "Each particle scatters the incoming light. Its scattered wave then illuminates its neighbour, which scatters again. Moving a particle changes the phase of this exchange and the pattern around both particles.",
  },
  resonance: {
    title: "Find a resonance",
    short: "Find a resonance",
    description: "Small changes in wavelength. A very different response.",
    explanation:
      "A sphere can support several electromagnetic resonances. Scrub through the spectrum and compare the field at a peak with the field between peaks. Higher multipole orders describe finer spatial structure.",
  },
  mixer: {
    title: "Mix the waves",
    short: "Mix the waves",
    description: "Choose the waves going in. See what comes out.",
    explanation:
      "Incoming waves can be described by a list of complex coefficients, a. The T-matrix maps them to outgoing coefficients, b = T a. Their amplitudes and relative phase decide whether fields reinforce or cancel.",
  },
  shell: {
    title: "A little less visible",
    short: "Add a shell",
    description: "Can a coating make a particle scatter less light?",
    explanation:
      "Light scattered by a coating can interfere with the response of its core. Adjust the shell and try to reduce the scattering score. This is a comparison at one wavelength, not broadband invisibility.",
  },
  chirality: {
    title: "A different handedness",
    short: "Flip the light",
    description: "Same sphere. Opposite circular polarization.",
    explanation:
      "In a materially chiral medium, the two circular polarizations experience different optical responses. The chirality control changes the material, not the spherical shape. Compare both handednesses at the same wavelength.",
  },
  design: {
    title: "Give light a destination",
    short: "Let it improve",
    description: "Place a target. Let the particles work towards it.",
    explanation:
      "The objective is the total electric-field intensity at the target. A native analytic adjoint finds how moving each particle changes that objective. Run follows accepted steps uphill until no further step fits; pause or try one step at a time. Radii stay fixed and particles stay apart. This is a local search, so different starting points lead to different outcomes.",
  },
};
export function initial(experiment: Experiment = "particles"): State {
  return {
    experiment,
    wavelength: experiment === "shell" ? 1.4 : 1.65,
    angle: 0,
    radius: 0.23,
    epsilon: experiment === "shell" ? 16 : 9,
    shell: 0.06,
    shellIndex: 2.8,
    chirality: 0.16,
    helicity: 0,
    phase: 0,
    mixture: 0.5,
    order: 0,
    compare: false,
    particles: [
      { x: -0.43, y: 0, radius: 0.23 },
      { x: 0.43, y: 0, radius: 0.23 },
    ],
    target: [0, 0.65],
  };
}
export function geometryValid(s: State): boolean {
  return s.particles.every(
    (p, i) =>
      p &&
      [p.x, p.y, p.radius].every(Number.isFinite) &&
      Math.abs(p.x) <= 1.1 &&
      Math.abs(p.y) <= 1.1 &&
      p.radius >= 0.12 &&
      p.radius <= 0.35 &&
      s.particles.every(
        (q, j) =>
          i === j ||
          Math.hypot(p.x - q.x, p.y - q.y) >= p.radius + q.radius + 0.15,
      ) &&
      (s.experiment !== "design" ||
        Math.hypot(p.x - s.target[0], p.y - s.target[1]) >= p.radius + 0.1),
  );
}
// Links are the only external input boundary. Reject invalid states as a whole.
export function fromHash(hash: string): State {
  if (!hash) return initial();
  const raw = JSON.parse(decodeURIComponent(hash.replace(/^#/, ""))) as State;
  if (!raw || !Object.hasOwn(presets, raw.experiment))
    throw new Error("Unknown experiment");
  const ranges: Record<string, [number, number]> = {
    wavelength: [1.1, 2.5],
    angle: [-180, 180],
    radius: [0.12, 0.35],
    epsilon: [1.96, 16],
    shell: [0.01, 0.25],
    shellIndex: [1, 2.8],
    chirality: [-0.5, 0.5],
    helicity: [0, 1],
    phase: [-180, 180],
    mixture: [0, 1],
    order: [0, 2],
  };
  for (const [key, [min, max]] of Object.entries(ranges)) {
    const v = raw[key as keyof State];
    if (typeof v !== "number" || !Number.isFinite(v) || v < min || v > max)
      throw new Error("Parameter outside experiment range");
  }
  if (
    !Number.isInteger(raw.helicity) ||
    !Number.isInteger(raw.order) ||
    typeof raw.compare !== "boolean" ||
    !Array.isArray(raw.particles) ||
    raw.particles.length !== 2 ||
    !Array.isArray(raw.target) ||
    raw.target.length !== 2 ||
    !raw.target.every((v) => Number.isFinite(v) && Math.abs(v) <= 1.3) ||
    !geometryValid(raw)
  )
    throw new Error("Invalid experiment geometry");
  return raw;
}
