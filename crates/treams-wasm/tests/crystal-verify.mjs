// Shared Node/browser qualification. Reference values use tfp-photonics/treams
// 1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39, SMatrices.slab(...).tr([1, 0])
// and bands_kz(1): thicknesses=[fill,1-fill]*N, materials=[1,nA²,nB²,...,1].
const reference = [
  [0.12, 0.7804810783846864, 0.2195189216153141, 1.4774648060137494, 0],
  [0.20, 0.3469815068482828, 0.6530184931517231, 2.575279543886322, 0],
  [0.27, 0.00035114916055480706, 0.9996488508394435, Math.PI, 0.5837006670070595],
  [0.35, 0.9807376614975117, 0.019262338502488995, 2.3719024309024137, 0],
  [0.45, 0.7520445185714069, 0.2479554814285914, 1.0675864738082035, 0],
];

function check(condition, label) {
  if (!condition) throw new Error(`crystal: ${label}`);
}

function close(actual, expected, label, tolerance = 2e-11) {
  const error = Math.abs(actual - expected);
  check(error <= tolerance, `${label}: ${actual} vs ${expected}, error ${error}`);
  return error;
}

function rejects(call, label) {
  let rejected = false;
  try { call(); } catch (error) { rejected = error instanceof Error; }
  check(rejected, `${label} must throw a JavaScript Error`);
}

export function verifyCrystal({crystal_spectrum, crystal_field}) {
  const nA = 1.45, nB = 2.6, fill = nB / (nA + nB);
  let maxReferenceError = 0, maxEnergyError = 0, maxTraceError = 0;
  const result = crystal_spectrum(nA, nB, fill, 8, reference.map(row => row[0]));
  check(result.length === reference.length * 5, "spectrum shape");
  for (const [i, [, t, r, phase, decay]] of reference.entries()) {
    for (const [j, expected] of [t, r, phase / Math.PI, decay].entries()) {
      maxReferenceError = Math.max(maxReferenceError, close(result[5 * i + j], expected, `upstream row ${i} column ${j}`));
    }
  }
  // Independent transfer-trace identity for any positive lossless indices.
  const frequencies = Array.from({length: 37}, (_, i) => 0.02 + i * 0.017);
  for (const [a, b] of [[1, 1], [1.7, 1.7], [1.45, 2.6], [4, 1]]) {
    for (const filling of [0.1, 0.5, 0.9]) {
      for (const count of [1, 8, 24]) {
        const rows = crystal_spectrum(a, b, filling, count, frequencies);
        for (const [i, frequency] of frequencies.entries()) {
          const [t, r, phase, decay, trace] = rows.slice(5 * i, 5 * i + 5);
          maxEnergyError = Math.max(maxEnergyError, close(t + r, 1, "lossless power"));
          const da = 2 * Math.PI * frequency * a * filling;
          const db = 2 * Math.PI * frequency * b * (1 - filling);
          const expectedTrace = Math.cos(da) * Math.cos(db) - 0.5 * (a / b + b / a) * Math.sin(da) * Math.sin(db);
          maxTraceError = Math.max(maxTraceError, close(trace, expectedTrace, "independent Bloch trace"));
          check(phase >= 0 && phase <= 1 + 1e-12, "principal Bloch phase");
          if (a === b) close(decay, 0, "zero contrast closes the infinite gap");
          if (a === 1 && b === 1) close(r, 0, "vacuum reflection");
        }
      }
    }
  }
  const center = 1 / (4 * nA * fill);
  let previous = 0;
  for (const count of [1, 2, 4, 8, 12, 24]) {
    const [, r] = crystal_spectrum(nA, nB, fill, count, [center]);
    const contrast = (nB / nA) ** (2 * count);
    close(r, ((contrast - 1) / (contrast + 1)) ** 2, "quarter-wave Bragg reflectance");
    check(r > previous, "gap reflection strengthens with period count");
    previous = r;
  }
  // Exact complex vacuum field and layer-boundary continuity, including the
  // final vacuum interface, qualify the animated quantity rather than a picture.
  const positions = [-0.5, 0, 0.4, 1.7, 4.9, 8, 8.5];
  const vacuum = crystal_field(1, 1, 0.37, 8, 0.27, positions);
  for (const [i, z] of positions.entries()) {
    close(vacuum[2 * i], Math.cos(2 * Math.PI * 0.27 * z), "vacuum field real");
    close(vacuum[2 * i + 1], Math.sin(2 * Math.PI * 0.27 * z), "vacuum field imaginary");
  }
  const boundaries = Array.from({length: 8}, (_, i) => [i, i + fill]).flat().concat([8]);
  const around = boundaries.flatMap(z => [z - 1e-9, z + 1e-9]);
  const field = crystal_field(nA, nB, fill, 8, 0.27, around);
  let maxContinuityError = 0;
  for (let i = 0; i < field.length; i += 4) {
    const error = Math.hypot(field[i] - field[i + 2], field[i + 1] - field[i + 3]);
    check(error < 2e-8, "tangential field continuity");
    maxContinuityError = Math.max(maxContinuityError, error);
  }
  const transmitted = crystal_field(nA, nB, fill, 8, 0.27, [8.5]);
  close(transmitted[0] ** 2 + transmitted[1] ** 2, reference[2][1], "field agrees with power transmission");
  // Independent upstream SMatrices.illuminate on a stack split at each point,
  // followed by treams.efield; equal helicities [-1,-1]/sqrt(2) give unit Ex.
  const fieldReference = [
    [1.521164879208683, 0.04166783376370231],
    [1.998316942306833, 0.05488290756309504],
    [1.7792005484614646, 0.04892221596631696],
    [0.021108087803159276, 0.0007009081185875843],
    [-0.6993015513460372, -0.01911345806267234],
    [-0.9924873118291986, -0.02736226222957497],
    [-0.19751314402014858, -0.006010035985042184],
    [0.012094461998260496, -0.005147835888777194],
    [0.018727665582738962, 0.0006509242474371136],
    [0.01417020212554557, 0.012261913891232106],
  ].flat();
  const sampled = crystal_field(nA, nB, fill, 8, 0.27, [-0.4, 0, 0.2, fill, 0.8, 1.2, 3.4, 7.8, 8, 8.4]);
  let maxFieldReferenceError = 0;
  for (const [i, expected] of fieldReference.entries()) {
    maxFieldReferenceError = Math.max(maxFieldReferenceError, close(sampled[i], expected, "upstream complex Ex"));
  }
  rejects(() => crystal_spectrum(nA, nB, 0, 8, [0.27]), "zero layer width");
  rejects(() => crystal_spectrum(nA, nB, fill, 0, [0.27]), "empty stack");
  rejects(() => crystal_spectrum(nA, nB, fill, 8, [NaN]), "nonfinite frequency");
  rejects(() => crystal_field(nA, nB, fill, 8, 0.27, [NaN]), "nonfinite position");
  return {
    passed: true,
    model: "lossless normal-incidence 1D dielectric crystal",
    upstream_reference_cases: reference.length,
    invariant_spectral_samples: frequencies.length * 4 * 3 * 3,
    maximum_reference_error: maxReferenceError,
    maximum_energy_error: maxEnergyError,
    maximum_trace_error: maxTraceError,
    maximum_complex_field_reference_error: maxFieldReferenceError,
    maximum_boundary_field_difference: maxContinuityError,
  };
}
