import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { init, simulate, improve } from "./dist/physics.js";
import { initial, fromHash, geometryValid } from "./dist/model.js";
import { padField, patternGain } from "./dist/field-view.js";
await init({
  module_or_path: await readFile(
    new URL("./dist/wasm/treams_wasm_bg.wasm", import.meta.url),
  ),
});
for (const experiment of [
  "particles",
  "resonance",
  "mixer",
  "shell",
  "chirality",
  "design",
]) {
  const state = initial(experiment),
    result = simulate(1, state, 20);
  assert(Number.isFinite(result.score) && result.score > 0, experiment);
  assert(result.field.some(Number.isFinite));
  assert.deepEqual(
    fromHash("#" + encodeURIComponent(JSON.stringify(state))),
    state,
  );
  console.log(
    `${experiment}: score=${result.score.toFixed(6)} reference=${result.reference.toFixed(6)} time=${result.milliseconds.toFixed(0)}ms`,
  );
}
const base = initial();
// A constant exterior must remain constant right up to a subpixel circular mask.
// Padding is display-only and must preserve every scientific sample, including NaNs.
for (const n of [8, 24, 56]) {
  const field = new Float64Array(n * n * 6);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const masked = Math.hypot(x - n * 0.43, y - n * 0.51) < n * 0.2;
      for (let j = 0; j < 6; j++)
        field[(y * n + x) * 6 + j] = masked ? NaN : j - 2;
    }
  const original = field.slice(),
    padded = padField(field, n);
  assert.deepEqual(field, original);
  for (let i = 0; i < field.length; i++) {
    if (Number.isFinite(field[i])) assert.equal(padded[i], field[i]);
    if (Number.isFinite(padded[i]))
      assert(Math.abs(padded[i] - ((i % 6) - 2)) < 1e-12);
  }
  for (let y = 1; y < n - 1; y++)
    for (let x = 1; x < n - 1; x++)
      if (
        [-1, 1, -n, n].some((d) => Number.isFinite(field[(y * n + x + d) * 6]))
      )
        assert(
          Number.isFinite(padded[(y * n + x) * 6]),
          "no dark missing-data seam at the boundary",
        );
}
const weakQuad = simulate(
  1,
  { ...initial("resonance"), wavelength: 1.45, order: 2 },
  24,
);
assert(
  Math.abs(weakQuad.orders.reduce((a, b) => a + b, 0) - weakQuad.score) < 1e-12,
);
assert(
  weakQuad.orders[1] / weakQuad.score < 0.001,
  "screenshot is a dipole resonance",
);
assert(
  Math.abs(weakQuad.spectrum[16 * 2 + 1] - weakQuad.orders[1]) < 1e-12,
  "spectrum follows selected mode",
);
assert(
  patternGain(weakQuad.field) > 10,
  "weak-mode contrast boost is explicit",
);
const strongQuad = simulate(
  1,
  { ...initial("resonance"), radius: 0.3, wavelength: 1.31, order: 2 },
  20,
);
assert(
  strongQuad.orders[1] / strongQuad.score > 0.6,
  "quadrupole can dominate at a different resonance",
);
const coupled = simulate(1, base, 20),
  independent = simulate(2, { ...base, compare: true }, 20);
assert(
  Math.abs(coupled.score - independent.score) > 0.001,
  "multiple scattering changes field",
);
const chiral = initial("chirality");
chiral.chirality = 0;
const neutral = simulate(1, chiral, 20);
assert(
  Math.abs(neutral.score - neutral.reference) < 1e-10,
  "achiral sphere has equal helicity response",
);
const mixer = initial("mixer");
const zero = simulate(1, mixer, 20),
  shifted = simulate(2, { ...mixer, phase: 90 }, 20);
assert(
  Math.abs(zero.score - shifted.score) < 1e-12,
  "orthogonal input mode power is phase independent",
);
assert(
  zero.field.some(
    (v, i) => Number.isFinite(v) && Math.abs(v - shifted.field[i]) > 1e-4,
  ),
  "local interference depends on phase",
);
const design = initial("design"),
  before = simulate(1, design, 20),
  step = improve(design),
  after = simulate(2, step.state, 20);
assert(geometryValid(step.state));
assert(after.score > before.score, "accepted adjoint step improves objective");
console.log(`Adjoint step: ${step.message}`);
let runState = design,
  runScore = before.score,
  runSteps = 0,
  stopped = false;
for (let attempt = 0; attempt < 40; attempt++) {
  const next = improve(runState);
  if (!next.accepted) {
    assert.deepEqual(
      next.state,
      runState,
      "stopping preserves the last accepted geometry",
    );
    stopped = true;
    break;
  }
  assert(
    geometryValid(next.state),
    "every accepted move respects separation and target clearance",
  );
  const nextScore = simulate(attempt, next.state, 6).score;
  assert(
    nextScore > runScore,
    "every automatic step improves the actual target objective",
  );
  runState = next.state;
  runScore = nextScore;
  runSteps++;
}
assert(
  stopped && runSteps > 3,
  "default run advances and eventually stops on its own",
);
assert(runScore > 2 * before.score, "default run reaches the stronger focus");
console.log(
  `Automatic optimization: ${runSteps} accepted steps, ${before.score.toFixed(6)} → ${runScore.toFixed(6)}, then stopped.`,
);
assert.throws(() =>
  fromHash(
    "#" + encodeURIComponent(JSON.stringify({ ...base, wavelength: 0 })),
  ),
);
assert.throws(() =>
  fromHash(
    "#" +
      encodeURIComponent(
        JSON.stringify({
          ...base,
          particles: [base.particles[0], base.particles[0]],
        }),
      ),
  ),
);
console.log("Browser experiment numerical checks passed.");
