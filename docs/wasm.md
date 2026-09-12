# WebAssembly

The complete Rust core compiles for `wasm32-unknown-unknown`. The separate
`treams-wasm` crate exposes double-precision layered/chiral spheres, interacting
sphere clusters, plane-wave illumination, Cartesian electric fields, and a
fixed-target intensity gradient to JavaScript. Bounded exports also cover square
sphere-array diffraction and normal-incidence one-dimensional dielectric
crystals. These paths share the CPU numerical
implementations; Python, NumPy, CUDA, and server computation are absent from this
build.

The browser build runs serially. Native builds use the
`faer` `std`, `linalg`, and `rayon` features. On WASM only, `faer/rayon` is disabled
because its `spindle` dependency requires unsupported atomic waits. The core's
Rayon iterators use Rayon's [supported single-thread fallback](https://docs.rs/rayon-core/latest/rayon_core/#global-fallback-when-threading-is-unsupported).
Browser and native builds share the same numerical kernels.

## Build and qualify

Use the repository's Rust 1.94.0 toolchain and Node.js 22 or newer:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.128 --locked
node scripts/check_wasm.mjs
```

The script builds an optimized WASM module, emits browser ES modules and
TypeScript declarations with `wasm-bindgen --target web`, and executes actual
WASM numerics in Node against committed upstream treams 0.4.5 fixtures. It writes
the package and numerical/size report under `target/wasm-pkg/`.

To additionally run the same checks in a real Chrome/Chromium browser over a
temporary loopback HTTP server:

```sh
node scripts/check_wasm.mjs --browser /path/to/chrome
cargo clippy --locked -p treams-wasm --target wasm32-unknown-unknown -- -D warnings
```

Three particle fixtures cover a 30-mode multilayer chiral sphere, a 60-mode interacting
two-sphere cluster (including the dense LU solve), and a 48-mode lossless sphere.
The checks compare every T-matrix entry, incident and scattered amplitude, and
sampled electric-field component. They also verify the lossless optical theorem,
field linearity, direction normalization, output ownership, and invalid-input
errors. Further checks compare analytic radius/position gradients for both
helicities against central differences (maximum absolute error `6.0e-11`),
joint translation invariance, direct incident illumination, and independently
assembled coherent uncoupled-sphere fields. Regenerate the independent fixtures with
`uv run --script scripts/generate_wasm_reference.py`.

The [qualification report](../crates/treams-wasm/tests/qualification.json) records
Node 26 and Chrome 153 results. This build is **1,001,440 bytes (978 KiB)**, or
**371,942 bytes (363 KiB) gzip**, excluding the generated JavaScript glue and
TypeScript declarations. Maximum absolute differences from upstream were
`1.8e-17` for T matrices, `6.5e-15` for incident coefficients, and `4.9e-17` for
sampled electric fields. Three square-array fixtures agree with upstream within
`8.83e-15` absolute and conserve lossless power within `2.0e-15`. Crystal checks
cover five upstream fixtures and 1,332 spectral invariant samples: maximum
reference error `6.78e-15`, energy-balance error `4.01e-13`, and complex-field
reference error `8.89e-16`. These are finite checks, not an accuracy claim
for every core operation or parameter range on WASM.

## Browser use

Serve the generated `.js` and `.wasm` together over HTTP. This uses ordinary
[browser ES modules without a bundler](https://wasm-bindgen.github.io/wasm-bindgen/examples/without-a-bundler.html):

```javascript
import init, {ScatteringSystem} from "./treams_wasm.js";
await init();

// Every material is [epsilon.re, epsilon.im, mu.re, mu.im, kappa.re, kappa.im].
const sphere = ScatteringSystem.sphere(
  4, 1.3, new Float64Array([0.35]),
  new Float64Array([4, 0, 1, 0, 0, 0,  // sphere
                   1, 0, 1, 0, 0, 0]) // vacuum outside
);
try {
  const incident = sphere.plane_wave(new Float64Array([0.36, -0.48, 0.8]), 1);
  const scattered = sphere.scatter(incident);
  const field = sphere.electric_field(
    scattered, new Float64Array([1.1, 0.8, 0.7, -0.9, -0.8, 0.6]), true
  );
  // field: Ex.re, Ex.im, Ey.re, Ey.im, Ez.re, Ez.im for each sample point.
  console.log(sphere.modes, field);
} finally {
  sphere.free();
}
```

`ScatteringSystem.cluster(lmax, k0, radii, epsilon, positions)` solves homogeneous
nonmagnetic spheres in vacuum, including their mutual scattering. Permittivities
are interleaved real/imaginary pairs and positions are consecutive xyz triples.
The solved object can be reused for many requested illuminations through
`plane_wave` and `scatter`. `tmatrix()` copies its column-major complex matrix;
the other complex arrays also use real/imaginary pairs. Returned typed arrays own
their storage and remain valid after the solved object is freed.

Run larger computations in a module Web Worker to keep sliders and rendering
responsive. The same imports and calculations work there; return the field with
`postMessage(field, [field.buffer])`. This serial package needs no
`SharedArrayBuffer` or cross-origin isolation headers.

The JavaScript interface exposes selected workflows, not the complete Python API.
Its cluster constructor forms the full dense T matrix. General periodic-system
objects, particle-interior fields, generic pullbacks, CUDA/WebGPU, and a
multithreaded browser runtime are not exposed. The array and crystal helpers below
have narrower contracts. WASM32 and browser memory limits still apply;
sampled outgoing fields are valid outside the particles. GPU execution in the
native package does not imply GPU execution in a browser.

## Direct fields and analytic target gradients

`direct_plane_field(k0, direction, helicity, points)` evaluates a unit-amplitude
vacuum plane wave directly. Add it **once** to the outgoing field for total fields.
For a cluster, each local regular expansion returned by `plane_wave` represents
the same incoming wave; summing their regular fields would multiply that wave
by the number of origins.

`ScatteringSystem.independent_cluster(...)` accepts the same arguments as
`cluster` and builds the independent local response. It disables repeated
scattering while retaining coherent interference when the outgoing fields are
summed. This is useful for controlled comparisons.

`cluster_target_gradient(lmax, k0, radii, epsilon, positions, direction, helicity,
target)` returns `[J, ...N_radius_gradients, ...3N_xyz_position_gradients]`, where
`J = |E_incident(target) + E_scattered(target)|²`. It holds frequency, material,
illumination and the world-space target fixed. The native analytic pullback
includes coupling, incident expansion phases, and outgoing expansion origins.
There is no finite differencing in production and no Python/autodiff runtime.

## Periodic showcases

`metasurface(lmax, radius, period, wavelength, angle_deg, epsilon_re, epsilon_im,
helicity)` returns diffraction powers for an infinite square array of identical,
achiral, nonmagnetic passive spheres in vacuum. It solves one requested
illumination using shared Ewald coupling and spherical-to-plane channels.
The output is `[R, T, A, N]`, followed by N records
`[order_x, order_y, kx/k0, ky/k0, abs(kz)/k0, R_order, T_order]`. All propagating
orders are included; evanescent coupling remains in the Ewald calculation.
Absorptance is the unmodified balance `1-R-T`, including tiny negative roundoff.

The export requires `lmax` 1–4, disjoint spheres, positive real permittivity,
nonnegative imaginary permittivity, incidence within ±60°, and
`period/wavelength <= 2`. Exact grazing diffraction thresholds are unsupported;
a spectral curve must show a gap there. This is a power/direction API, not a
periodic near-field export. The educational UI uses order 4 and limits radius to
0.12–0.23 μm. The [cutoff audit](../crates/treams-wasm/tests/metasurface-convergence.json)
compares orders 4 and 6 at 558 sampled radius/spacing/wavelength/angle combinations,
excluding nine exact thresholds. Actual WASM in Node matches all 558 order-4
reference values within `5.71e-14`, with power-balance error at most `3.34e-14`.
The maximum order-4/order-6 reflectance difference is 0.00320049 (0.320 percentage
points), not a uniform accuracy bound between samples. Chrome qualification uses
the three separate array fixtures described above.

`crystal_spectrum(n_a, n_b, fill_a, periods, frequencies)` evaluates a lossless
normal-incidence alternating dielectric stack with vacuum ports. The unit period
is `a=1`; A occupies `fill_a`, followed by B. Indices are in [1,4], filling in
(0,1), periods in 1–64, and up to 1,024 frequencies `a/lambda` in (0,2] are accepted.
Each row is `[T, R, abs(Re(K*a))/pi, abs(Im(K*a)), Re(cos(K*a))]`: T/R describe the
finite stack and K the infinite crystal. `abs(Im(K*a))` is amplitude attenuation
per period, not intensity attenuation. This is the exact one-dimensional layered
model; it does not model oblique incidence, absorption, or a two-dimensional
photonic crystal.

`crystal_field(n_a, n_b, fill_a, periods, frequency, positions)` returns interleaved
complex Ex at up to 4,096 positions `z/a`, inside or outside the stack. Unit
x-polarized illumination enters from below; the field includes coherent internal
reflections. Animate with `Re(Ex * exp(-i*phase))`. These periodic helpers are
forward-only JavaScript exports.

The [Light Lab](../web/README.md) combines six introductory experiments with the
[advanced array/crystal page](../web/advanced.html). A browser worker evaluates
the selected point first, then fills the spectrum; array sweeps yield between
samples so new inputs can interrupt them. No numerical work requires a server.
