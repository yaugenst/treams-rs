import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  init,
  initialArray,
  initialCrystal,
  point,
  spectrum,
} from "./dist/advanced-physics.js";
await init({
  module_or_path: await readFile(
    new URL("./dist/wasm/treams_wasm_bg.wasm", import.meta.url),
  ),
});
for (const state of [
  initialArray(),
  { ...initialArray(), wavelength: 0.73, angle: 20 },
]) {
  const p = point(state).values;
  assert(
    Math.abs(p[0] + p[1] - 1) < 2e-8,
    "lossless infinite array conserves incident power",
  );
  let reflected = 0,
    transmitted = 0;
  for (let i = 0; i < p[3]; i++) {
    const at = 4 + 7 * i;
    assert(
      Math.abs(p[at + 2] ** 2 + p[at + 3] ** 2 + p[at + 4] ** 2 - 1) < 1e-12,
      "every displayed beam lies on the light cone",
    );
    reflected += p[at + 5];
    transmitted += p[at + 6];
  }
  assert(
    Math.abs(reflected - p[0]) < 1e-12 && Math.abs(transmitted - p[1]) < 1e-12,
    "beam powers add to displayed R and T",
  );
  if (state.angle)
    assert(p[3] > 1, "short oblique illumination opens diffraction orders");
  else
    assert(
      p[0] > 0.9,
      "default surface demonstrates strong collective reflection",
    );
}
const crystal = initialCrystal(),
  gap = point(crystal),
  pass = point({ ...crystal, frequency: 0.12 });
assert(
  gap.values[3] > 0.5 && gap.values[0] < 0.001,
  "default frequency lies in a strongly reflecting bandgap",
);
assert(
  pass.values[3] < 1e-8 && pass.values[0] > 0.7,
  "pass-band frequency transmits",
);
assert(
  point({ ...crystal, periods: 4 }).values[0] > gap.values[0],
  "more periods suppress gap transmission",
);
assert(
  Math.abs(gap.values[0] + gap.values[1] - 1) < 1e-12,
  "lossless finite stack conserves power",
);
const last = gap.field.length - 2;
assert(
  Math.abs(gap.field[last] ** 2 + gap.field[last + 1] ** 2 - gap.values[0]) <
    1e-12,
  "displayed transmitted field matches measured power",
);
const uniform = await spectrum({ ...crystal, index: 1.45 }, () => false);
for (let i = 0; i < uniform.axis.length; i++)
  assert(
    uniform.values[5 * i + 3] < 1e-7,
    "equal indices close every infinite-crystal gap",
  );
const curves = await spectrum(crystal, () => false);
for (let i = 0; i < curves.axis.length; i++)
  assert(
    Math.abs(curves.values[5 * i] + curves.values[5 * i + 1] - 1) < 1e-11,
    "full displayed spectrum conserves power",
  );
assert.equal(
  await spectrum({ ...initialArray(), period: 0.81 }, () => true),
  undefined,
  "superseded array sweeps stop before computing samples",
);
console.log(
  "Advanced WASM: array diffraction power/directions, collective resonance, finite-stack fields, Bloch gap closure and cancellation passed.",
);
