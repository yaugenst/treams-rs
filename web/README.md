# Light Lab

Six static, mobile-friendly experiments using the real `treams-wasm` solver:
interacting spheres, resonances and multipoles, coherent mode mixing, core-shell
scattering suppression, material chirality, and target-intensity optimization.
The [advanced page](advanced.html) adds square-array diffraction and a
one-dimensional photonic crystal.
The gradient overlay shows the analytic position derivative of the target's
**total** electric intensity. Arrow lengths share a relative scale; they are not
forces. Run repeats the native analytic adjoint/backtracking step and accepts only
increases. Particle motion and the solved field transition together between steps;
the intermediate animation is presentation, not another solved state. Step once
remains available. Pause, manual edits, or hiding the tab interrupt the run and
discard an in-flight proposal. Automatic runs use a 32² field preview and refine
the last accepted geometry when stopped; the point objective does not depend on
the display grid. A run stops when the existing search finds no improving step,
including a flat gradient or a separation/target-clearance constraint. This is
not a guarantee of an unconstrained optimum. The default scene takes 12 accepted
steps from intensity 0.955903 to 2.191525, then stalls near target clearance.

## Run

From the repository root, with the [WASM toolchain](../docs/wasm.md) installed:

```sh
node scripts/check_wasm.mjs
cd web
npm ci
npm test
python3 -m http.server 8780 --bind 127.0.0.1 --directory dist
```

Open `http://127.0.0.1:8780/`. Python here only serves static files. All simulation,
analytic differentiation and optimization execute as Rust/WASM in a browser
module worker. There is no Python, remote solve, CUDA or WebGPU in the page.
`just web-check` also runs WASM Clippy and formatting. CI runs the browser
numerical and touch-interaction checks after building and qualifying WASM.

The build copies the generated binding package from `target/wasm-pkg`; generated
bindings, dependencies and `dist` are ignored. It also creates `NOTICES.txt` from
the project licenses and selected WASM dependency graph, plus
`RUST-STDLIB-NOTICES.html` from the pinned Rust toolchain. Registry packages that
omit license files use the commit-pinned texts in `dependency-licenses.json`;
license texts are read locally. Cargo may fetch locked registry packages missing
from a cold cache; the build does not fetch licenses from upstream websites.
Native sliders and buttons provide
keyboard operation; precise position inputs are an alternative to dragging.
Share copies the complete experiment into the URL fragment. The fragment is
validated when loaded and is not sent to the static server.

## Numerical scope

- Coordinates and wavelength use μm. Materials use `epsilon = n² + 0.02i`, `mu=1`;
  embedding vacuum, with constant material parameters across the spectrum.
- Fields show the central plane through three-dimensional spheres. Outgoing
  expansions are exterior-only; numerical samples mask particles and a 1.5% margin.
  A display-only copy pads missing texels with neighbouring exterior values beneath
  smooth vector circles, preventing the coarse mask from bleeding into the field.
  This does not compute interior fields or change scientific samples or scores.
- Single spheres use multipoles through order 5; clusters use order 6. Cluster
  surfaces stay at least 0.15 μm apart and targets at least 0.10 μm outside them.
  These are finite-truncation educational experiments, not uniformly converged
  research calculations. Near cancellation minima and surfaces need particular care.
- The total field includes exactly one incident plane wave. Independent-particle
  mode disables rescattering but retains coherent interference between particles.
- Single-sphere scattering uses `sum(abs(b)²)/k0²`; efficiency divides by projected
  outer area. The shell score compares absolute cross-sections to the same bare
  core, so changing the normalization area cannot manufacture suppression.
- The mode mixer uses electric `l=1,m=0` and `l=2,m=1` regular incident multipoles,
  with fixed incident coefficient norm 4. Its score is a coefficient-norm ratio,
  not a plane-wave cross-section. Isolating a resonance order also selects its
  scattering score and spectrum; its fraction of total scattering stays visible.
- Intensity uses a fixed logarithmic colour scale, saturating at 7; wave motion
  uses a fixed `tanh` scale for `Re(Ez)`. Grid peak intensity depends on sampling.
  Isolated dipole/quadrupole views default to an explicitly labelled field boost,
  scaling their sampled RMS field amplitude to one when weaker. Tap the badge
  for true strength. The intensity legend reflects the gain squared; numerical
  values are unchanged. All-wave and other experiments retain the fixed scale.
- Only the current solved T matrix and single spectrum are cached. Slider/drag
  updates are coarser than settled views; stale calculations never overwrite
  newer requested settings. Particle outlines and live controls track input
  immediately. The previous field is dimmed while a solve is pending, and actual
  new fields crossfade in; intermediate frames are presentation, not extra solves.
  Animation reuses the complex field and pauses when the tab is hidden.
  Reduced-motion preference disables autoplay.

Mobile keeps the field and two primary controls together in one viewport. More
opens the remaining settings in a separately scrolling panel. The field itself
accepts wavelength, mode, coating and chirality gestures.

`npm run test:ui` uses Playwright with actual Chrome on macOS (or an installed
Playwright Chromium elsewhere). It verifies touch dragging updates the rendered
particle in two animation frames while the field is pending, all six interactions,
visible primary controls at phone sizes, narrow layouts, sharing and the adjoint
step. Screenshots are written to the ignored `web/output` folder.

`npm test` exercises real WASM for all six paths, coherent coupling, helicity
symmetry, phase-sensitive local interference with invariant mode power, link
validation, and an accepted adjoint improvement. Binding qualification separately
checks analytic radius/position gradients against central differences, translation
invariance, independent sphere fields and upstream numerical fixtures in Node and
actual Chrome. Finite differences appear only in tests.

The supported agent browser API is feature-detected as `document.modelContext`.
It exposes read/configure/improve tools sharing the visible state and solver.
Live WebMCP registration and execution are not part of the qualified browser
baseline; ordinary browser operation does not require that API.

## Advanced experiments

Open `advanced.html`, or use **Advanced** beside the Light Lab title. Both views
use the same Rust/WASM module in a worker. The selected point is calculated first;
the spectrum follows. Square-array sweeps yield between samples and discard stale
work after a new input, keeping controls responsive.

- **Metasurface:** a square array of lossless achiral spheres in vacuum, index 3.5,
  order 4, radius 0.12–0.23 μm, spacing 0.65–1.10 μm, wavelength 0.65–1.65 μm, and
  incidence 0–45°. The display shows diffraction directions and normalized order
  powers, not a computed near-field image. Every propagating order contributes to
  R/T; exact grazing thresholds appear as gaps. Actual WASM matches 558 order-4
  reference samples within `5.71e-14`. The order-4/order-6 cutoff audit found a
  maximum reflectance difference of 0.00320049; finite sampling does not establish
  that bound throughout the continuous slider domain.
- **Photonic crystal:** the exact normal-incidence, lossless one-dimensional
  layered model. The finite stack spectrum and complex Ex field use all coherent
  internal reflections; the band diagram describes the infinite crystal. A/B
  indices are 1.45 and 1.45–3.5; filling is 0.1–0.9, the finite stack has 1–20
  periods, and frequency is `a/lambda` in 0.08–0.65. The evanescent Bloch quantity
  describes amplitude attenuation per period, not intensity decay.

[Export contracts and numerical qualification](../docs/wasm.md#periodic-showcases)
record the broader native input bounds, fixture errors, and sampled cutoff audit.
`npm test` includes both advanced numerical paths; `npm run test:ui` also exercises
the advanced page. The general Python periodic API and periodic gradients are not
exposed by these browser helpers.

## Deploy

Build and test the site as above, then upload the **contents of `web/dist/`** to a
static HTTPS host. Keep both HTML pages, their JavaScript/CSS files, and `wasm/`
in their existing relative layout. TypeScript declarations (`*.d.ts`) are
not needed at runtime. Keep both generated notice files with the deployment.
No build service, Python runtime, backend API, database,
or repository access is required on the host.

Configure the host to serve JavaScript with a JavaScript MIME type and `.wasm`
with `application/wasm`. The worker and WASM files use the same origin as the
page; this serial build needs no cross-origin isolation headers. Open the deployed
HTTPS URL on a phone and check a particle drag, a wavelength change, and an
optimization step before sharing it.

The Share button preserves the selected experiment and its parameters in the
URL fragment. Access control belongs to the static host; the page itself does
not authenticate visitors.
