# Light Lab

Six static, mobile-friendly experiments using the real `treams-wasm` solver:
interacting spheres, resonances and multipoles, coherent mode mixing, core-shell
scattering suppression, material chirality, and target-intensity optimization.
The gradient overlay shows the analytic position derivative of the target's
**total** electric intensity. Arrow lengths share a relative scale; they are not
forces. Improve uses a constrained backtracking step and accepts only increases.

## Run

From the repository root, with the toolchain in `docs/wasm.md` installed:

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
`just web-check` also runs WASM Clippy and formatting. CI runs the same browser
numerical checks after its WASM qualification job.

The build copies the generated binding package from `target/wasm-pkg`; generated
bindings, dependencies and `dist` are ignored. Native sliders and buttons provide
keyboard operation; precise position inputs are an alternative to dragging.
Share copies the complete experiment into the URL fragment. The fragment is
validated when loaded and is not sent to the static server.

## Numerical scope

- Coordinates and wavelength use μm. Materials use `epsilon = n² + 0.02i`, `mu=1`;
  embedding vacuum, with constant material parameters across the spectrum.
- Fields show the central plane through three-dimensional spheres. Outgoing
  expansions are exterior-only; the page masks the particles and a 1.5% margin.
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
  not a plane-wave cross-section. The resonance score includes all orders even
  when the field view isolates one order.
- Intensity uses a fixed logarithmic colour scale, saturating at 7; wave motion
  uses a fixed `tanh` scale for `Re(Ez)`. Grid peak intensity depends on sampling.
- Only the current solved T matrix and single spectrum are cached. Slider/drag
  updates are coarser than settled views; stale calculations never overwrite
  newer requested settings. Particle outlines and live controls track input
  immediately. The previous field is dimmed while a solve is pending, and actual
  new fields crossfade in; intermediate frames are presentation, not extra solves. Animation reuses the complex field, and pauses its
  work when the tab is hidden. Reduced-motion preference disables autoplay.

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
The available Chrome preview lacks that API, so live WebMCP registration and
execution remain unverified; ordinary browser operation is unaffected.

## Private preview

The current preview is served from
`~/personal/previews/treams-light-lab` on [redacted-host]:

- loopback-only static HTTP: `127.0.0.1:8781`;
- persistent user service: `treams-light-lab.service`;
- Tailscale HTTPS: `https://localhost:8446/`.

Tailscale Serve is used, not Funnel. Existing Serve ports are preserved.
Only built static assets are deployed. To update after a successful build:

```sh
rsync -az --exclude='*.d.ts' web/dist/ \
  localhost:~/personal/previews/treams-light-lab/
```

The service is a task-owned user unit outside chezmoi management. Disable this
preview without changing other hosted services:

```sh
ssh localhost \
  'tailscale serve --https=8446 off; systemctl --user disable --now treams-light-lab.service'
```
