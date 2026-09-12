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

export function verify(ScatteringSystem, reference, helpers) {
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
      const sparse = new Float64Array(scattered.length);
      // For clusters, the first origin is entirely inactive. The remaining mode
      // labels must still reference the original positions and both helicities.
      const active = c.spheres.length > 1
        ? [system.modes / 2, system.modes / 2 + 3, system.modes - 2, system.modes - 1]
        : [0, 3, Math.floor(system.modes / 2), system.modes - 1];
      for (const i of active) {
        sparse[2 * i] = scattered[2 * i];
        sparse[2 * i + 1] = scattered[2 * i + 1];
      }
      const zero = new Float64Array(scattered.length);
      for (const outgoing of [false, true]) {
        const sparseField = system.electric_field(sparse, c.points.flat(), outgoing);
        const fullField = system.electric_field(scattered, c.points.flat(), outgoing);
        const sum = scattered.map((x, i) => x + sparse[i]);
        check(Array.from({length: system.modes}, (_, i) => Math.hypot(sum[2 * i], sum[2 * i + 1])).every(x => x > 0), "full reference retains every mode");
        const fullSumField = system.electric_field(sum, c.points.flat(), outgoing);
        close(sparseField, fullSumField.map((x, i) => x - fullField[i]), "sparse equals full-basis difference");
        close(system.electric_field(phase(sparse, 0.73), c.points.flat(), outgoing), phase(sparseField, 0.73), "sparse complex linearity");
        close(system.electric_field(sparse.map(x => x * 1e-30), c.points.flat(), outgoing).map(x => x * 1e30), sparseField, "nonzero coefficients have no amplitude cutoff");
        const zeroField = system.electric_field(zero, c.points.flat(), outgoing);
        check(zeroField.length === 6 * c.points.length && zeroField.every(x => x === 0), "all-zero field preserves shape");
      }
      close(system.electric_field(zero, c.positions.flat(), false), new Float64Array(6 * c.positions.length), "zero regular field at expansion origins");
      rejects(() => system.electric_field(zero, c.positions[0], true), "zero outgoing field retains singular-origin restriction");
      rejects(() => system.electric_field(sparse, c.positions[0], true), "inactive outgoing origin remains unsupported");
      rejects(() => system.electric_field(zero.slice(2), c.points.flat(), true), "zero field validates coefficient count");
      rejects(() => system.electric_field(zero, [0, Number.NaN, 1], true), "zero field validates geometry");
      // Returned arrays must own their storage: mutating one does not mutate the solver.
      matrix.fill(0);
      close(system.scatter(incident), scattered, "owned matrix output");
      rejects(() => system.scatter(new Float64Array(1)), "malformed complex array");
      rejects(() => system.scatter(new Float64Array(2)), "wrong incident dimension");
      rejects(() => system.plane_wave([0, 0, 0], 0), "zero direction");
      rejects(() => system.plane_wave([0, 0, 1], 2), "invalid helicity");
      rejects(() => system.electric_field(scattered, [0, Number.NaN, 1], true), "nonfinite point");
      results.push({name: c.name, modes: system.modes, maximum_absolute_error: errors, sparse_field_invariants: true});
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
  return {passed: true, reference: reference.reference, cases: results, helpers: verifyHelpers(ScatteringSystem, helpers, reference)};
}

function phase(values, angle) {
  const result = new Float64Array(values.length), c = Math.cos(angle), s = Math.sin(angle);
  for (let i = 0; i < values.length; i += 2) {
    result[i] = c * values[i] - s * values[i + 1];
    result[i + 1] = s * values[i] + c * values[i + 1];
  }
  return result;
}

function verifyHelpers(ScatteringSystem, {direct_plane_field, cluster_target_gradient}, reference) {
  const c = reference.cases.find(c => c.spheres.length === 2);
  const radii = c.spheres.map(s => s.radii[0]);
  const epsilon = c.spheres.flatMap(s => s.materials[0].epsilon);
  const positions = c.positions.flat();
  const points = c.points.flat();
  const direct = direct_plane_field(c.k0, c.direction, c.polarization, points);
  const origin = direct_plane_field(c.k0, c.direction, c.polarization, [0, 0, 0]);
  for (let i = 0; i < c.points.length; i++) {
    const angle = c.k0 * c.points[i].reduce((sum, x, axis) => sum + x * c.direction[axis], 0);
    const e = direct.slice(6 * i, 6 * i + 6);
    close(e, phase(origin, angle), "direct plane-wave phase");
    const intensity = e.reduce((sum, x) => sum + x * x, 0);
    check(Math.abs(intensity - 1) < 2e-14, "unit incident intensity");
    for (const component of [0, 1]) {
      const longitudinal = c.direction.reduce((sum, d, axis) => sum + d * e[2 * axis + component], 0);
      check(Math.abs(longitudinal) < 2e-14, "incident transversality");
    }
  }

  const independent = ScatteringSystem.independent_cluster(c.lmax, c.k0, radii, epsilon, positions);
  try {
    const a = independent.plane_wave(c.direction, c.polarization);
    close(a, c.incident, "independent incident origin phases");
    const b = independent.scatter(a);
    const field = independent.electric_field(b, points, true);
    const expected = new Float64Array(field.length);
    for (let particle = 0; particle < radii.length; particle++) {
      const local = ScatteringSystem.sphere(c.lmax, c.k0, [radii[particle]], [...epsilon.slice(2 * particle, 2 * particle + 2), 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]);
      try {
        const localA = local.plane_wave(c.direction, c.polarization);
        const localB = local.scatter(localA);
        const angle = c.k0 * c.positions[particle].reduce((sum, x, axis) => sum + x * c.direction[axis], 0);
        close(b.slice(2 * particle * local.modes, 2 * (particle + 1) * local.modes), phase(localB, angle), "independent local scattering");
        const shifted = c.points.flatMap(p => p.map((x, axis) => x - c.positions[particle][axis]));
        const contribution = phase(local.electric_field(localB, shifted, true), angle);
        for (let i = 0; i < expected.length; i++) expected[i] += contribution[i];
      } finally { local.free(); }
    }
    close(field, expected, "coherent independent scattered field");
    check(field.some((x, i) => Math.abs(x - c.electric_field[i]) > 1e-8), "mutual scattering changes the field");
  } finally { independent.free(); }

  function objective(rs, centers, helicity, eps = epsilon, target = [0.16, 0.64, 0.41]) {
    const system = ScatteringSystem.cluster(c.lmax, c.k0, rs, eps, centers);
    try {
      const a = system.plane_wave(c.direction, helicity);
      const scattered = system.electric_field(system.scatter(a), target, true);
      const incident = direct_plane_field(c.k0, c.direction, helicity, target);
      return incident.reduce((sum, x, i) => sum + (x + scattered[i]) ** 2, 0);
    } finally { system.free(); }
  }
  const gradientCases = [];
  for (const helicity of [0, 1]) {
    const target = [0.16, 0.64, 0.41];
    const actual = cluster_target_gradient(c.lmax, c.k0, radii, epsilon, positions, c.direction, helicity, target);
    check(actual.length === 1 + 4 * radii.length, "gradient result layout");
    check(Math.abs(actual[0] - objective(radii, positions, helicity)) < 2e-13, "objective matches public forward path");
    let maximum = 0;
    const parameters = [...radii, ...positions];
    for (let i = 0; i < parameters.length; i++) {
      const h = 1e-5 * Math.max(1, Math.abs(parameters[i]));
      const plus = parameters.slice(), minus = parameters.slice();
      plus[i] += h; minus[i] -= h;
      // Finite differences are only this test oracle; the exported pullback is analytic.
      const expected = (objective(plus.slice(0, radii.length), plus.slice(radii.length), helicity)
        - objective(minus.slice(0, radii.length), minus.slice(radii.length), helicity)) / (2 * h);
      const error = Math.abs(actual[i + 1] - expected);
      maximum = Math.max(maximum, error);
      check(error < 2e-7 + 2e-5 * Math.abs(expected), `analytic shape gradient ${helicity}/${i}: ${actual[i + 1]} != ${expected}`);
    }
    const shift = [0.21, -0.17, 0.13];
    const translated = cluster_target_gradient(c.lmax, c.k0, radii, epsilon, positions.map((x, i) => x + shift[i % 3]), c.direction, helicity, target.map((x, i) => x + shift[i]));
    check(actual.every((x, i) => Math.abs(x - translated[i]) < 2e-11), "joint scene/target translation preserves intensity and gradient");
    gradientCases.push({helicity, parameter_count: parameters.length, maximum_absolute_error: maximum});
  }

  // Index-matched particles produce no scattering. Incident intensity stays one,
  // independently of how many local expansion origins the cluster contains.
  for (const count of [2, 3]) {
    const centers = [-1, 0, 0, 1, 0, 0, 0, 1, 0].slice(0, 3 * count);
    const rs = Array(count).fill(0.15), eps = Array(count).fill([1, 0]).flat();
    const result = cluster_target_gradient(3, c.k0, rs, eps, centers, c.direction, 1, [0.1, 0.2, 0.7]);
    check(Math.abs(result[0] - 1) < 2e-12, "incident field is added only once");
    check(result.slice(1).every(x => Math.abs(x) < 2e-12), "index-matched shape gradient vanishes");
  }
  rejects(() => direct_plane_field(0, c.direction, 1, points), "invalid vacuum wavenumber");
  rejects(() => direct_plane_field(c.k0, [0, 0, 0], 1, points), "invalid direct direction");
  rejects(() => ScatteringSystem.independent_cluster(2, 1, [0.2, 0.2], [2, 0, 2, 0], [0, 0, 0, 0.1, 0, 0]), "independent overlapping spheres");
  rejects(() => cluster_target_gradient(c.lmax, c.k0, radii, epsilon, positions, c.direction, 1, c.positions[0]), "interior optimization target");
  return {direct_plane_field: true, independent_cluster: true, analytic_gradient: gradientCases};
}
