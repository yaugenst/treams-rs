# WebAssembly

The complete Rust core compiles for `wasm32-unknown-unknown`. The separate
`treams-wasm` crate exposes double-precision layered/chiral spheres, interacting
sphere clusters, plane-wave illumination, Cartesian electric fields, and a
fixed-target intensity gradient to JavaScript. It shares the CPU numerical implementations; Python, NumPy, CUDA,
and server computation are absent from this build.

The browser build runs serially. Native builds retain exactly the existing
`faer` `std`, `linalg`, and `rayon` features. On WASM only, `faer/rayon` is disabled
because its `spindle` dependency requires unsupported atomic waits. The core's
Rayon iterators use Rayon's [supported single-thread fallback](https://docs.rs/rayon-core/latest/rayon_core/#global-fallback-when-threading-is-unsupported).
No parallel kernels were replaced or duplicated.

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

The three fixtures cover a 30-mode multilayer chiral sphere, a 60-mode interacting
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
Node 26 and Chrome 153 results. This build is **623,014 bytes (608 KiB)**, or
**228,390 bytes (223 KiB) gzip**, excluding the generated JavaScript glue and
TypeScript declarations. Maximum absolute differences from upstream were
`1.8e-17` for T matrices, `6.5e-15` for incident coefficients, and `4.9e-17` for
sampled electric fields. These are finite fixture checks, not an accuracy claim
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

The JavaScript interface currently covers the workflows above, not every Python
method. Its cluster constructor forms the full dense T matrix. It does not yet
expose generic native pullbacks, periodic systems, internal fields, CUDA/WebGPU, or a
multithreaded browser runtime. WASM32 and browser memory limits still apply;
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

The [Light Lab](../web/README.md) uses these exports in six mobile-friendly
experiments, with a gradient-arrow overlay and accepted adjoint ascent steps.
It is a static website: the browser worker does all numerical computation.
