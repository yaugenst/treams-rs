// Independent oracle: the installed tfp-photonics/treams 0.4.5 package.
// For each case:
// tm = treams.TMatrix.sphere(lmax, 2*pi/wavelength, [radius], [epsilon, 1])
// q = [k0*sin(angle), 0]; cell = eye(2)*period
// sm = treams.SMatrices.from_array(tm.latticeinteraction.solve(cell, q), ports)
// ports contain both helicities of every open reciprocal-lattice order.
// Excite helicity 0 of order (0, 0). R/T are sm.tr; individual powers are
// the corresponding outgoing helicity norms times kz/kz_incident.
const cases = [
  {
    args: [4, 0.23, 0.8, 1.25, 15, 12.25, 0, 0],
    powers: [0.6185651345109535, 0.38143486548905337],
    orders: [
      [
        0, 0, 0.25881904510252074, 0, 0.9659258262890683, 0.6185651345109534,
        0.3814348654890535,
      ],
    ],
    lmax6: [0.6185712961031988, 0.381428703896809],
  },
  {
    args: [4, 0.23, 0.8, 0.73, 20, 12.25, 0, 0],
    powers: [0.15165569721246472, 0.8483443027875369],
    orders: [
      [
        -1, 0, -0.5704798566743312, 0, 0.8213115932025035, 0.023299996948144813,
        0.21274587551171828,
      ],
      [
        0, -1, 0.3420201433256687, -0.9125, 0.22442364304923196,
        0.057006745381917856, 0.03142516895442831,
      ],
      [
        0, 0, 0.3420201433256687, 0, 0.9396926207859084, 0.028587217712334547,
        0.4726697319400726,
      ],
      [
        0, 1, 0.3420201433256687, 0.9125, 0.22442364304923196,
        0.042761737170067406, 0.13150352638131746,
      ],
    ],
    lmax6: [0.1516027446796177, 0.8483972553203838],
  },
  {
    args: [4, 0.23, 0.8, 1.25, 15, 12.25, 0.15, 0],
    powers: [0.5148069785286966, 0.25542735723994053],
    orders: [
      [
        0, 0, 0.25881904510252074, 0, 0.9659258262890683, 0.5148069785286965,
        0.2554273572399406,
      ],
    ],
    lmax6: [0.5148108133371763, 0.2554196043684774],
  },
];

function check(condition, message) {
  if (!condition) throw new Error(`Metasurface: ${message}`);
}

/** Run on the actual initialized WASM module, in Node or a browser. */
export function verifyMetasurface(wasm) {
  let maxReferenceError = 0;
  let maxBalanceError = 0;
  let maxCutoffDifference = 0;
  const elapsedMs = [];
  for (const fixture of cases) {
    const start = performance.now();
    const value = wasm.metasurface(...fixture.args);
    elapsedMs.push(performance.now() - start);
    const expected = [
      ...fixture.powers,
      1 - fixture.powers[0] - fixture.powers[1],
      fixture.orders.length,
      ...fixture.orders.flat(),
    ];
    check(value.length === expected.length, "open diffraction order count");
    for (let i = 0; i < value.length; i++) {
      const error = Math.abs(value[i] - expected[i]);
      maxReferenceError = Math.max(maxReferenceError, error);
      check(error < 2e-9, `upstream reference mismatch at ${i}: ${error}`);
    }
    check(value[2] > -2e-9, "passive array creates energy");
    if (fixture.args[6] === 0) {
      maxBalanceError = Math.max(
        maxBalanceError,
        Math.abs(value[0] + value[1] - 1),
      );
      check(Math.abs(value[0] + value[1] - 1) < 2e-9, "lossless power balance");
    }
    let r = 0;
    let t = 0;
    for (let i = 4; i < value.length; i += 7) {
      check(
        Math.abs(
          value[i + 2] ** 2 + value[i + 3] ** 2 + value[i + 4] ** 2 - 1,
        ) < 1e-12,
        "outgoing wavevector lies on vacuum light cone",
      );
      r += value[i + 5];
      t += value[i + 6];
    }
    check(
      Math.abs(r - value[0]) < 1e-12 && Math.abs(t - value[1]) < 1e-12,
      "resolved orders sum to total power",
    );
    for (let i = 0; i < 2; i++) {
      maxCutoffDifference = Math.max(
        maxCutoffDifference,
        Math.abs(value[i] - fixture.lmax6[i]),
      );
    }
  }
  // A scale transformation leaves Maxwell scattering unchanged, and reversing
  // helicity mirrors the achiral square cell without changing total power.
  const original = wasm.metasurface(...cases[1].args);
  const scaled = wasm.metasurface(4, 0.69, 2.4, 2.19, 20, 12.25, 0, 0);
  const reversed = wasm.metasurface(4, 0.23, 0.8, 0.73, 20, 12.25, 0, 1);
  for (let i = 0; i < original.length; i++) {
    check(Math.abs(original[i] - scaled[i]) < 2e-9, "length-scale invariance");
  }
  for (let i = 0; i < 3; i++) {
    check(
      Math.abs(original[i] - reversed[i]) < 2e-9,
      "achiral helicity symmetry",
    );
  }
  const transparent = wasm.metasurface(4, 0.23, 0.8, 0.73, 20, 1, 0, 0);
  check(
    Math.abs(transparent[0]) < 1e-20 && Math.abs(transparent[1] - 1) < 1e-13,
    "vacuum cell transmits all power",
  );
  let thresholdRejected = false;
  try {
    wasm.metasurface(4, 0.23, 0.8, 0.8, 0, 12.25, 0, 0);
  } catch (error) {
    thresholdRejected = /grazing diffraction threshold/.test(String(error));
  }
  check(thresholdRejected, "exact diffraction threshold is explicit");
  check(
    maxCutoffDifference < 6e-5,
    "selected fixtures agree with lmax6 oracle",
  );
  return {
    cases: cases.length,
    maxReferenceError,
    maxBalanceError,
    maxCutoffDifference,
    elapsedMs,
  };
}

/** Optional full qualification using metasurface-convergence.json. */
export function verifyMetasurfaceConvergence(wasm, reference) {
  const start = performance.now();
  let maxReferenceError = 0;
  let maxBalanceError = 0;
  let maxCutoffDifference = 0;
  for (const [
    r,
    p,
    wavelength,
    angle,
    expected,
    higherCutoff,
  ] of reference.rows) {
    const [reflection, transmission] = wasm.metasurface(
      4,
      r,
      p,
      wavelength,
      angle,
      12.25,
      0,
      0,
    );
    const error = Math.abs(reflection - expected);
    const balance = Math.abs(reflection + transmission - 1);
    check(
      error < 2e-9,
      `grid oracle mismatch at ${[r, p, wavelength, angle]}: ${error}`,
    );
    check(
      balance < 2e-9,
      `grid power balance at ${[r, p, wavelength, angle]}: ${balance}`,
    );
    maxReferenceError = Math.max(maxReferenceError, error);
    maxBalanceError = Math.max(maxBalanceError, balance);
    maxCutoffDifference = Math.max(
      maxCutoffDifference,
      Math.abs(reflection - higherCutoff),
    );
  }
  return {
    cases: reference.rows.length,
    thresholdExclusions: reference.threshold_exclusions.length,
    maxReferenceError,
    maxBalanceError,
    maxCutoffDifference,
    elapsedMs: performance.now() - start,
  };
}
