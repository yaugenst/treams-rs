// Read-only truncation qualification for Light Lab. Node >=24, no dependencies.
// node scripts/qualify_browser.mjs . > benchmarks/browser/convergence.json
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve, join} from 'node:path';
import {pathToFileURL} from 'node:url';

const root = resolve(process.argv[2] ?? '.');
const pkg = join(root, 'target/wasm-pkg');
const binary = await readFile(join(pkg, 'treams_wasm_bg.wasm'));
const {default: init, ScatteringSystem, direct_plane_field} = await import(pathToFileURL(join(pkg, 'treams_wasm.js')));
await init({module_or_path: binary});
const f = a => new Float64Array(a), vacuum = [1, 0, 1, 0, 0, 0];
const material = (index, chirality = 0) => [index * index, .02, 1, 0, chirality, 0];

function sphere(lmax, {radius = .23, index = 3, wavelength = 1.65, thickness = 0, shellIndex = 1.6, chirality = 0, helicity = 0}) {
  const k0 = 2 * Math.PI / wavelength;
  const system = ScatteringSystem.sphere(lmax, k0, f(thickness ? [radius, radius + thickness] : [radius]),
    f([...material(index, chirality), ...(thickness ? material(shellIndex) : []), ...vacuum]));
  try {
    const a = system.plane_wave(f([1, 0, 0]), helicity), b = system.scatter(a);
    return {lmax, scattering: b.reduce((sum, x) => sum + x * x, 0) / k0 ** 2,
      extinction: -b.reduce((sum, x, i) => sum + x * a[i], 0) / k0 ** 2};
  } finally { system.free(); }
}

function cluster(lmax, {radius = .23, index = 3, wavelength = 1.65, gap = .4, targets = [[0, .65, 0]], angles = [0]}) {
  const k0 = 2 * Math.PI / wavelength, center = radius + gap / 2;
  const system = ScatteringSystem.cluster(lmax, k0, f([radius, radius]), f([index * index, .02, index * index, .02]),
    f([-center, 0, 0, center, 0, 0]));
  try {
    const values = [];
    for (const angle of angles) {
      const theta = angle * Math.PI / 180, direction = f([Math.cos(theta), Math.sin(theta), 0]);
      const b = system.scatter(system.plane_wave(direction, 0)), points = f(targets.flat());
      const scattered = system.electric_field(b, points, true), incident = direct_plane_field(k0, direction, 0, points);
      for (let point = 0; point < targets.length; point++) {
        let intensity = 0;
        for (let i = 6 * point; i < 6 * point + 6; i++) intensity += (scattered[i] + incident[i]) ** 2;
        values.push({angle, target: targets[point], intensity});
      }
    }
    return {lmax, values};
  } finally { system.free(); }
}

const spherical = Object.entries({
  default: {}, shellOriginal: {thickness: .09}, chiral0: {chirality: .16}, chiral1: {chirality: .16, helicity: 1},
  small: {radius: .12, index: 1.4, wavelength: 2.5},
  large: {radius: .35, index: 4, wavelength: 1.1},
  chiralBoundary: {radius: .35, index: 4, wavelength: 1.1, chirality: .5, helicity: 1},
  shellBoundary: {radius: .35, index: 4, wavelength: 1.1, thickness: .25, shellIndex: 2.8},
}).map(([name, parameters]) => ({name, parameters, results: [5, 6, 8].map(l => sphere(l, parameters))}));

const shellParameters = {radius: .23, index: 4, wavelength: 1.4, thickness: .06, shellIndex: 2.8};
const shell = [{}, {wavelength: 1.37}, {wavelength: 1.43}, {thickness: .05}, {thickness: .07}, {shellIndex: 2.75}].map(change => {
  const parameters = {...shellParameters, ...change};
  return {parameters, results: [5, 6, 8].map(lmax => {
    const coated = sphere(lmax, parameters), bare = sphere(lmax, {...parameters, thickness: 0});
    return {lmax, coated: coated.scattering, bareCore: bare.scattering, ratio: coated.scattering / bare.scattering};
  })};
});

const practical = [];
for (const [radius, index] of [[.3, 3.5], [.3, 3], [.35, 4]]) {
  for (const gap of [.1, .15]) for (const wavelength of [1.1, 1.3, 1.65, 2.5]) {
    const center = radius + gap / 2, clearance = .100001;
    const targets = [[center + radius + clearance, 0, 0],
      [0, Math.sqrt((radius + clearance) ** 2 - center ** 2), 0], [center, radius + clearance, 0]];
    const parameters = {radius, index, gap, wavelength, targets, angles: [0, 45, 90]};
    practical.push({parameters, results: [4, 6, 8].map(l => cluster(l, parameters))});
  }
}

const extended = [
  {name: 'default', parameters: {}, cutoffs: [4, 6, 8]},
  {name: 'originalExtreme', parameters: {radius: .35, index: 4, wavelength: 1.1, gap: .025, targets: [[0, .45, 0], [.7825, 0, 0]]}, cutoffs: [4, 6, 8, 10, 12]},
  {name: 'practicalLarge', parameters: {radius: .35, index: 4, wavelength: 1.1, gap: .15, targets: [[0, .1479050371048938, 0]], angles: [90]}, cutoffs: [6, 8, 10, 12]},
  {name: 'practicalCancellation', parameters: {radius: .23, index: 4, wavelength: 1.4, gap: .15, targets: [[.635001, 0, 0]]}, cutoffs: [6, 8, 10, 12]},
].map(({name, parameters, cutoffs}) => ({name, parameters, results: cutoffs.map(l => cluster(l, parameters))}));

console.log(JSON.stringify({node: process.version, wasmSha256: createHash('sha256').update(binary).digest('hex'),
  conventions: 'Vacuum; epsilon=index^2+0.02i, mu=1. Lengths and cross-section areas share the scene units. Cluster score is total |E|^2 with incident intensity one, not a norm of local scattering coefficients.',
  scope: 'Sampled truncation sensitivity, not a uniform error bound. Higher cutoffs are comparators; lmax8 is not assumed converged for close resonant clusters. Near-dark targets amplify relative errors.',
  spherical, shell, practical, extended}, null, 2));
