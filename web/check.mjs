import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { init, simulate, improve } from "./dist/physics.js";
import { initial, fromHash, geometryValid } from "./dist/model.js";
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
