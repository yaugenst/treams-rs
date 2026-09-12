// This same check runs in Node and a browser against independently generated treams data.
function check(condition, message) {
  if (!condition) throw new Error(message);
}

function close(actual, expected, label) {
  check(actual.length === expected.length, `${label}: shape mismatch`);
  let maximum = 0;
  for (let i = 0; i < actual.length; i += 2) {
    const error = Math.hypot(actual[i] - expected[i], actual[i + 1] - expected[i + 1]);
    const magnitude = Math.hypot(expected[i], expected[i + 1]);
    check(error <= 2e-12 + 2e-11 * magnitude, `${label}[${i / 2}]: error ${error}`);
    maximum = Math.max(maximum, error);
  }
  return maximum;
}

function rejects(call, label) {
  let rejected = false;
  try { call(); } catch (error) { rejected = error instanceof Error; }
  check(rejected, `${label}: expected a JavaScript Error`);
}

export function verify(ScatteringSystem, reference) {
  const results = [];
  for (const c of reference.cases) {
    const material = m => [...m.epsilon, ...m.mu, ...m.kappa];
    const system = c.spheres.length > 1
      ? ScatteringSystem.cluster(c.lmax, c.k0, c.spheres.map(s => s.radii[0]), c.spheres.flatMap(s => s.materials[0].epsilon), c.positions.flat())
      : ScatteringSystem.sphere(c.lmax, c.k0, c.spheres[0].radii, [...c.spheres[0].materials, c.embedding].flatMap(material));
    try {
      check(system.modes === c.mode_count, `${c.name}: mode count`);
      const matrix = system.tmatrix();
      const incident = system.plane_wave(c.direction, c.polarization);
      const scattered = system.scatter(incident);
      const field = system.electric_field(scattered, c.points.flat(), true);
      const errors = {
        tmatrix: close(matrix, c.tmatrix, `${c.name}: T matrix`),
        incident: close(incident, c.incident, `${c.name}: incident`),
        scattered: close(scattered, c.scattered, `${c.name}: scattered`),
        electric_field: close(field, c.electric_field, `${c.name}: field`),
      };
      close(system.plane_wave(c.direction.map(x => 3 * x), c.polarization), incident, "direction scale invariance");
      close(system.electric_field(scattered.map(x => 2 * x), c.points.flat(), true), field.map(x => 2 * x), "field linearity");
      // Returned arrays must own their storage: mutating one does not mutate the solver.
      matrix.fill(0);
      close(system.scatter(incident), scattered, "owned matrix output");
      rejects(() => system.scatter(new Float64Array(1)), "malformed complex array");
      rejects(() => system.scatter(new Float64Array(2)), "wrong incident dimension");
      rejects(() => system.plane_wave([0, 0, 0], 0), "zero direction");
      rejects(() => system.plane_wave([0, 0, 1], 2), "invalid helicity");
      rejects(() => system.electric_field(scattered, [0, Number.NaN, 1], true), "nonfinite point");
      results.push({name: c.name, modes: system.modes, maximum_absolute_error: errors});
      if (c.lossless) {
        const t = system.tmatrix();
        let scattering = 0, extinction = 0;
        for (let i = 0; i < t.length; i++) scattering += t[i] ** 2;
        for (let i = 0; i < system.modes; i++) extinction -= t[2 * i * (system.modes + 1)];
        check(Math.abs(scattering - extinction) <= 2e-12 * scattering, "lossless optical theorem");
      }
    } finally {
      system.free();
    }
  }
  rejects(() => ScatteringSystem.cluster(2, 1, [0.2, 0.2], [2, 0, 2, 0], [0, 0, 0, 0.1, 0, 0]), "overlapping spheres");
  return {passed: true, reference: reference.reference, cases: results};
}
