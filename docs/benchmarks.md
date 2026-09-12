# CPU scattering and field benchmarks

For optional GPU measurements and their precision/setup limits, see
[GPU opportunities](gpu-opportunities.md) and [cached GPU sampling](gpu-sampling.md).
[WASM qualification](wasm.md) covers the browser exports separately.

## Reference qualification

The reference grid passed all 527 runtime comparisons and 525 peak-RSS comparisons
against treams 0.4.5. It covers scalar and batched functions, geometry, local
waves, finite and periodic scattering, fields, planar stacks, power observables
and recorded native pullbacks. The [qualification manifest](../benchmarks/complete-qualification.json)
records the commands, result paths and build fingerprints; [raw results](../benchmarks/results/final/)
retain all timing samples and process measurements. The lowest measured speedup
is 1.02x and the largest gated RSS ratio is 0.912x. The two recorded-adjoint RSS
exceptions are described below; there is no universal all-input speed guarantee.

Run `just bench-all` after installing the locked development dependencies. Each
case first checks numerical agreement, then benchmarks both implementations in
separate processes with four threads each. Sub-millisecond calls use fourteen
alternating paired timings with batches calibrated to at least 20 ms. The strict
threshold is speedup >= 1 and Rust peak RSS <= upstream peak RSS. The reference run
used CPU affinity 8-11 on the Ryzen 9 9950X host; this differs from some historical
runs below. Native binary, Python source and benchmark hashes identify each result.

Two recorded internal-illumination cases at 1,024 channels retain owned inputs
needed for arbitrary amplitude pullbacks and exceed upstream's forward-only RSS.
Their runtime gates remain strict. With one/eight incident columns, recorded
forward speedups are 1.40x/1.28x and peak RSS is 1.56x/1.75x upstream. The matching
forward-only calls achieve 1.83x/1.71x speedups with 0.91x/0.91x upstream RSS. Upstream has
no equivalent reverse pass; reverse timings are reported separately, never treated
as an upstream speed comparison.

Shared native dispatch preserves NumPy broadcasting, output-buffer and subclass
paths. Cylindrical rotation uses a real sine/cosine pair for its unit phase
instead of a general complex exponential. Selected Linux results:

| Operation | Input size | Speedup | Rust / upstream peak RSS |
| --- | ---: | ---: | ---: |
| Cartesian to spherical coordinates | 1 | 1.83x | 0.65x |
| Cartesian to spherical vector components | 1 | 1.58x | 0.65x |
| Plane-wave M field | 1 | 3.84x | 0.66x |
| Cylindrical rotation | 128 | 1.13x | 0.65x |
| Two-dimensional cell volume | 1 | 2.56x | 0.65x |
| Two-dimensional reciprocal cell | 1 | 1.98x | 0.65x |
| EBCM, degree 3 / 96 quadrature nodes | 30 modes | 83.89x | 0.53x |
| EBCM, degree 4 / 96 quadrature nodes | 48 modes | 127.85x | 0.53x |

The EBCM rows use the explicit legacy integral for like-for-like timing; the
corrected surface element and its physical checks are described below.

A separate [macOS qualification](../benchmarks/mac-qualification.json) passed all
30 scalar/batched regression cases for these boundaries. Its lowest measured
speedup is 1.30x and largest RSS ratio is 0.70x. The complete Linux grid is the
broader qualification; the macOS subset is not a full-platform parity claim.

Peak RSS includes imports and allocator retention. These measurements qualify the
listed inputs on this CPU, not every size, host or conditioning regime. Numerical
accuracy gates remain independent of timing gates.

## Evidence provenance

Historical artifact paths have been neutralized to remove user and machine
identifiers. Manifest digests of redacted raw JSON files were refreshed to match
the redistributed bytes. Numerical results, timing samples, and recorded source,
native-library, and benchmark-executable fingerprints remain unchanged. This
metadata cleanup did not rerun or requalify the historical measurements.

## Historical measurements

The following measurements document individual kernels and the effects of specific
optimizations. They use different historical builds and benchmark protocols,
identified by the cited artifacts. Their qualification counts and test totals
describe those builds, not the current source tree. Use the reference manifest
above for the broader CPU comparison.

Unless stated otherwise, the host was an AMD Ryzen 9 9950X (16 physical cores),
Linux x86-64, Python 3.13.1 and treams 0.4.5. Rust used an optimized build, faer
and Rayon. Unqualified result filenames below refer to
[`benchmarks/results/`](../benchmarks/results/). Several exploratory timings were
not archived; those sections say so explicitly. Their printed summaries are
historical observations, not independently inspectable raw evidence.

### Axisymmetric EBCM

For r(theta)=0.3(1+0.23 cos²(theta)), k0=1.3, inner eps=3.1+0.2i,
mu=1.2+0.1i, kappa=0.07 and vacuum outside, the complete outgoing Q integral
was compared before timing at rtol=2e-9, atol=1e-12. These measurements explicitly
use `legacy=True` to compare the same integral as upstream, which omits a radial
surface-area factor; the corrected default is physically checked separately.
Rust uses 96 Gauss-Legendre nodes, while upstream uses its adaptive SciPy
quadrature. Four matched threads, seven samples after warmup, isolated processes.

| Degree | Modes | treams ms | Rust ms | Speedup | Rust reverse ms | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 30 | 284.26 | 3.54 | 80.3x | 1.99 | 83.87 / 42.18 |
| 4 | 48 | 727.40 | 5.90 | 123.4x | 3.00 | 84.93 / 42.45 |

Reverse covers every sampled radius/slope and all complex wavenumbers and
impedances. It uses Rayon over nodes and cached waves within each node; the
residual retains inputs only. Peak RSS through reverse stayed at the forward
peak in these cases. Results include the native/Python boundary and residual
creation, but exclude shape/basis setup.

The degree-6 strict comparison fails on six of 9,216 nearly zero entries,
with differences up to 4.98e-10. The discrepancy comes from cancellation in both
implementations: all affected entries have m=0 and odd l_out+l_in, so their
integrands are odd under theta -> pi-theta for this equatorially symmetric surface.
Their exact integrals vanish. An independent m=0 spherical-wave calculation at
80 decimal digits gives magnitudes below 8e-82 and confirms the angular reflection
identity. Its integrand also agrees with upstream at an ordinary sample point.

The largest integral of the absolute integrand is about 7.31e5; one double-precision
epsilon times that magnitude is 1.62e-10. The Rust residuals at 48, 96 and 192 nodes
range from 0.04 to 2.43 times their corresponding epsilon-scaled absolute integrals.
Upstream leaves up to 6.62e-10, and tighter adaptive tolerances return the same
values with a roundoff warning. Increasing quadrature order alone does not remove
this floating-point floor. Neither these data nor the symmetry diagnostic justify
relaxing the generic accuracy gate or claiming a degree-6 speedup.

The corrected integral's independent physical convergence tests remain separate.
Reproduce this diagnostic with
`uv run --no-sync --with mpmath python scripts/qualify_ebcm_cancellation.py`.
All eight m=0 degree-1/3 to degree-6 entries, their high-precision results, quadrature
samples and condition estimates are recorded in
[the cancellation diagnostic](../benchmarks/results/ebcm-l6-cancellation.json).
The raw degree-3/4 timings and original strict-failure log were temporary
measurements and are not archived. The independent cancellation diagnostic is
archived; the timing table remains a historical summary.

```sh
uv run --no-sync python scripts/benchmark_cluster.py --workload ebcm --particles 1 --lmax 4 --samples 96 --threads 4
```

### Finite sphere clusters

| Spheres | lmax | Matrix dimension | Threads | treams ms | Rust ms | Speedup | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 3 | 240 | 1 | 88.23 | 3.01 | 29.3× | 71.8 / 44.2 |
| 16 | 3 | 480 | 1 | 389.64 | 13.91 | 28.0× | 90.3 / 57.2 |
| 8 | 5 | 560 | 1 | 686.49 | 22.88 | 30.0× | 98.4 / 65.1 |
| 32 | 3 | 960 | 1 | 1677.50 | 88.36 | 19.0× | 154.9 / 113.2 |
| 8 | 3 | 240 | 4 | 90.63 | 1.96 | 46.2× | 73.6 / 42.7 |
| 16 | 3 | 480 | 4 | 391.17 | 9.35 | 41.9× | 91.6 / 57.8 |
| 8 | 5 | 560 | 4 | 691.96 | 12.69 | 54.5× | 100.2 / 64.5 |
| 32 | 3 | 960 | 4 | 1636.14 | 43.41 | 37.7× | 156.4 / 113.6 |

With packed in-place LU, matched-four-thread rechecks were faster and smaller
than upstream: dimension 240 took 2.19 ms versus 89.95 ms
(41.0x; 46.6 versus 75.0 MiB), and dimension 960 took 35.63 ms
versus 1641.15 ms (46.1x; 100.9 versus 159.0 MiB). These are separate comparisons,
not controlled speed ratios against the older Rust measurements above. Full raw
results are in `benchmarks/results/packed-lu-cluster-d{240,960}.json`.

Raw samples and environment details are in `benchmarks/results/release-*.json`.
Each row uses seven samples after a warmup. A separate correctness process first
compares the entire complex T-matrix with treams at `rtol=2e-9, atol=1e-12`.
Each backend then runs in its own process. BLAS and Rayon get equal thread budgets.
The benchmark refuses a Rust debug build and records the native binary hash.

Both timings include particle coefficient construction, translations and the full
dense interacting T-matrix. Rust also retains its pullback residual. They exclude
imports. Peak process RSS includes imports, allocator retention and benchmark
bookkeeping; import baselines are recorded separately. It is not a pure measure
of the numerical kernel's working memory. Tests used chains of spheres with radii
0.15–0.25, spacing 0.8, relative permittivity 4+0.1i and vacuum wavenumber 1.3.
No CPU affinity, frequency or thermal isolation was imposed.

The major improvements were reusing Wigner couplings across pairs, computing each
radial/angular term once per displacement, replacing generic LU/matmul with faer,
storing local particle matrices as blocks, and using disjoint Rayon workers to
assemble coupling columns. Dense interacting output and LU still cost O(D²)
space; factorization remains O(D³). No claim is made here about GPU performance,
cylinders, or arbitrary user geometries.

Reproduce after `uv sync --locked --group dev` and `just build-ext-release`:

```sh
uv run --no-sync python scripts/benchmark_cluster.py --particles 16 --lmax 3 --threads 4
```

### Batched spherical fields

The same harness accepts `--workload field --samples 2048`. It compares weighted
outgoing electric fields, including a retained native pullback context, against
`treams.efield(...) @ coefficients`. Basis setup and input generation are outside
the timed region for both implementations; upstream's field operator allocation
is part of its evaluation. Correctness is checked before isolated timing.

For four origins, order 3 (120 modes), 2,048 samples and four threads, the measured
median was 40.82 ms in Rust versus 822.27 ms in treams (20.14x). Peak process RSS
was 40.98 MiB versus 96.94 MiB; post-import baselines were 40.22 and 64.00 MiB.
Raw samples and binary identity: `benchmarks/results/fields-n4-l3-p2048-t4.json`.
The native path contracts amplitudes as it evaluates each sample, so it does not
allocate upstream's sample-by-component-by-mode matrix. Reverse mode recomputes
local wave derivatives and reduces cotangents with Rayon; no dense Jacobian is
retained. This is one qualified field workload, not a claim about all field sizes.

The shared spherical/cylindrical field path was rechecked on the same spherical
workload: 40.64 ms forward, 71.11 ms reverse, 41.28 MiB peak RSS. Forward remained
20.24x faster than treams (`shared-fields-n4-l3-p2048-t4.json`).

With the full operator API, the same 120-mode/2,048-sample/four-thread workload
returns the entire (2048, 3, 120) array in 45.47 ms versus 856.08 ms for treams
(18.83x). Its reverse takes 75.11 ms. Forward peak RSS is 64.20 versus 97.68 MiB;
the returned operator and transient array transfer account for more memory than
weighted evaluation. The reverse context itself retains only geometry. The
weighted path after this shared implementation remains 41.86 ms forward,
70.65 ms reverse, and 41.63 MiB peak RSS (19.60x faster than treams).
Raw results: `field-operator-n4-l3-p2048-t4.json` and
`field-geometry-n4-l3-p2048-t4.json`; use `--workload field-operator` to reproduce.

The cylindrical workload uses four origins, orders -3 through 3, axial labels
0.2 and -0.3 and both helicities (112 modes), with the same 2,048 points and four
threads. It measures 47.57 ms forward versus treams' 502.92 ms (10.57x), 64.24 ms
reverse, and 41.64 versus 94.11 MiB forward peak RSS. Without radial reuse, the
implementation took 138.96 ms forward; sharing the radial value and its first two
derivatives across adjacent orders removed repeated Bessel calls. The forward residual remains
linear in modes plus samples. Both versions and raw samples are recorded in
`cylindrical-fields*-n4-l3-p2048-t4.json`. Reproduce with `--workload cylindrical-field`.

### Periodic sphere arrays and adjoints

The periodic workload includes local sphere coefficients, the 2D Ewald coupling,
and the full interacting response matrix, retaining every native pullback context.
The unit cell contains a square grid with spacing 0.8, the same particle parameters
as above, and Bloch wavevector (0.1, 0.15). Basis setup is outside both timings.

| Spheres | lmax | Threads | treams ms | Rust ms | Speedup | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3 | 1 | 103.57 | 49.24 | 2.10x | 89.2 / 41.6 |
| 4 | 3 | 4 | 118.94 | 14.06 | 8.46x | 89.5 / 40.7 |
| 9 | 3 | 4 | 571.90 | 70.05 | 8.16x | 184.5 / 49.8 |

Before shared-polarization Ewald derivatives, complete native reverse passes took
146.59 ms and 749.60 ms for the four-thread
four- and nine-sphere cases, respectively. These include the dense adjoint solve,
all particle pullbacks, and both origin sets, medium wavenumbers, Bloch vector and
lattice geometry. Peak process RSS through the reverse pass was 43.4 and 60.0 MiB.
Reverse timings exclude preparing a fresh forward context and use a fixed complex
output cotangent. That benchmark spent about ten forward evaluations'
time on one complete periodic reverse pass. Recomputation kept residual memory
small; the S-matrix measurements below include shared Ewald derivatives. No
derivative speedup over Dreams or another autodiff implementation has been measured.

Raw results: `periodic-n4-l3-t1.json`, `periodic-adjoint-n4-l3-t4.json`, and
`periodic-adjoint-n9-l3-t4.json` in `benchmarks/results`. The harness records
native reverse timings separately from the forward comparison:

```sh
uv run --no-sync python scripts/benchmark_cluster.py --workload periodic --particles 4 --lmax 3 --threads 4
```

### Complete periodic S matrices

The `array` workload adds plane-wave incidence and radiation into ten ports
(zero and four first diffraction orders, both polarizations). Correctness checks
the full S matrix against treams before timing; reverse mode includes the channel
and radiation pullbacks as well as the periodic solve and all particle contexts.

| Spheres | lmax | Threads | treams ms | Rust ms | Speedup | Rust reverse ms | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3 | 1 | 105.41 | 50.35 | 2.09x | 254.38 | 90.6 / 43.3 |
| 4 | 3 | 4 | 117.41 | 15.27 | 7.69x | 71.30 | 90.8 / 42.8 |
| 9 | 3 | 4 | 570.15 | 73.00 | 7.81x | 351.90 | 187.9 / 52.4 |

For four spheres/four threads the complete reverse fell from 141.24 to 71.30 ms
after sharing Ewald derivatives between equal-wavenumber polarizations and using
only the active lattice parameters in local chain rules. This is about 4.7 forward
evaluations per reverse, with no retained mode-by-parameter Jacobian. The larger
case peaks at 60.5 MiB through reverse. Raw samples: `array-before-n4-l3-t4.json`
and `array-n*-l3-t*.json`. Derivative speed relative to Dreams remains unmeasured.

```sh
uv run --no-sync python scripts/benchmark_cluster.py --workload array --particles 9 --lmax 3 --threads 4
```

### Multipole rotations

For two spherical origins and all degrees through 8 (320 modes), the complete
rotation takes 0.510 ms versus treams' 5.597 ms (10.98x) with four matched threads.
Native reverse takes 0.753 ms; forward peak RSS is 42.27 versus 67.03 MiB. Both
forward timings include constructing angular blocks. Rust retains the small-d
blocks for an analytic Euler-angle reverse pass. Raw samples and binary identity
are in `rotation-n2-l8-t4.json`; reproduce with
`--workload rotation --particles 2 --lmax 8 --threads 4`.

### Cylindrical-to-spherical conversion

Release, four matched threads, one common origin, spherical lmax=12 (336 modes),
32 axial wavenumbers between -0.7 and 0.7 and cylindrical mmax=12 (1,600 modes).
The full 336-by-1,600 matrix is checked against treams before timing.
treams takes 23.13 ms and Rust 3.34 ms: **6.92x** on this case. The native pullback
for arbitrary origin and complex-wavenumber cotangents takes 5.80 ms; forward peak
RSS is 73.9 MiB versus 55.9 MiB. Seven samples after warmup, isolated processes.

Without azimuthal selection, the same workload took 20.48 ms forward and
23.63 ms reverse. Skipping analytically zero azimuthal orders at coincident
transverse origins reduced those
times without suppressing adjacent-order position derivatives. This measurement
covers common-origin conversion; it does not establish displaced-origin speedup.
The raw conversion timings, including the baseline, were temporary measurements
and are not archived. These values are retained as a historical summary.

### Plane fields and direct NumPy buffer transfer

Four matched threads, 128 plane modes (64 transverse vectors, both polarizations),
4,096 Cartesian samples, including propagating and evanescent waves at k0=1.3.
Every benchmark first compares the full result against upstream.

| Output | treams forward | Rust forward | Speedup | Rust reverse | Forward peak RSS, treams / Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weighted plane field | 139.37 ms | 1.95 ms | 71.5x | 3.58 ms | 88.3 / 42.1 MiB |
| Full plane field operator | 137.13 ms | 2.70 ms | 50.7x | 9.76 ms | 88.1 / 65.7 MiB |

Without the shared contraction, the weighted reverse pass took 6.10 ms.
Contracting polarization cotangents over all samples before differentiating each
mode reduced it to 3.58 ms. The copying implementation of the full operator took
13.32 ms forward and 89.4 MiB peak RSS. Transferring the owned Rust buffer directly
to NumPy removes that extra copy. The reverse
pass packs the returned stride layout with a bulk copy and still accepts arbitrary
cotangent strides. These are complete Python/native boundary timings, including
input conversion, output transfer and residual creation. No output Jacobian is
retained. Seven samples after warmup; process and scheduling variability remain.

The same buffer transfer applies to multipole field operators. Rechecking four
spherical origins, lmax=3 and 2,048 points gives 823.02 / 42.83 ms (19.2x), with
75.81 ms reverse and 97.1 / 53.5 MiB forward peak RSS. The earlier copying path used
64.2 MiB for Rust. Weighted multipole fields already avoid the full operator.

The raw timings for these plane fields, packed operators, buffer transfers and
copying baselines were temporary measurements and are not archived. The values
here are historical summaries.

### Plane-to-spherical illumination

Four matched threads, two spherical origins, lmax=8 (320 modes), and 64 transverse
vectors with both polarizations (128 plane modes). At k0=1.3 the directions include
propagating and evanescent waves. Complete output agreement is checked first.
treams takes 14.07 ms and Rust 0.598 ms: **23.6x** for this case. The native reverse
pass, including all origin and full complex-wavevector cotangents, takes 1.23 ms.
Forward peak RSS is 67.5 / 40.2 MiB. Rust shares the normalized direction across
multipoles, uses Rayon over incident modes, and transfers the output buffer to
NumPy without retaining it in the residual. Seven samples after warmup.
The raw timing was a temporary measurement and is not archived; these values
are a historical summary.

### Complete cylindrical arrays

Four matched threads, one-dimensional arrays along x, kz=0.2, k0=1.3,
radii 0.15–0.25, permittivity 4+0.1j, spacing 0.8 and period 0.8N.
Each evaluation includes particle T matrices, Ewald coupling, the periodic solve
and all four plane-wave scattering blocks for ten ports. Seven samples after
warmup; arbitrary full complex cotangents for the native reverse.

| Cylinders | mmax | Modes | treams ms | Rust ms | Speedup | Rust reverse ms | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 5 | 88 | 2101.29 | 8.26 | 254.4x | 21.13 | 69.1 / 41.6 |
| 9 | 3 | 126 | 2982.63 | 16.64 | 179.2x | 45.15 | 70.9 / 43.8 |

Both solvers use eta=0.7. The nine-cylinder reference at its automatic split
differs from its converged answer by about 8.4e-8 in the final S matrix. Rust's
coupling stays stable as eta changes; a regression checks it against the converged
reference. Benchmark correctness tolerances remain rtol=2e-9, atol=1e-12.

Sharing Ewald evaluations between equal-wavenumber polarizations roughly halved
Rust forward and reverse times in the four-cylinder automatic-split baseline
(18.42 to 9.38 ms forward, 45.84 to 22.99 ms reverse). Polarization wavenumber
gradients remain separate and are checked by independent perturbations. No dense
parameter Jacobian is retained. Results do not imply this speedup for every array.

The raw converged-array timings were temporary measurements and are not archived.
The historical workload can be rerun with
`--workload cylindrical-array --particles 9 --lmax 3 --threads 4 --repeats 7`.

### Cylindrical plane illumination and phase reuse

Four origins, mmax=12 (200 cylindrical modes), kz=0.2, and 128 transverse
vectors with both polarizations (256 plane modes). Four matched threads, seven
samples after warmup, complete matrix checked against treams. Rust takes 0.220 ms
versus 6.376 ms (**28.9x**); the native origin/transverse-wavevector reverse takes
0.446 ms. Forward peak RSS is 39.9 MiB versus 67.7 MiB.

Without phase reuse, the implementation took 0.377 ms forward and 0.537 ms
reverse. Reusing each origin phase across its multipoles reduced both. The same change improves
spherical plane illumination: the previously described two-origin, lmax=8 case
measured 0.494 ms versus 14.143 ms (**28.6x**), with 1.163 ms reverse.
Cylindrical axial labels remain fixed in these gradients.

The raw phase-reuse timings were temporary measurements and are not archived.
The cylindrical workload uses `--workload cylindrical-plane-expansion --particles 4
--lmax 12 --samples 128 --threads 4 --repeats 7`.

### Compact planar multilayers

Four interior layers, alternating permittivities 2.3+0.1j and 1.7+0.05j in
vacuum, thicknesses 0.1–0.4, k0=1.3, transverse qx=0.1–0.8 and qy=0.2.
Four matched threads, seven samples after warmup, isolated backend processes.
Both forward timings use the public `SMatrices.slab` and return the complete
four-block dense array; it is compared with upstream before timing.

| Channels | Plane modes | treams ms | Rust ms | Speedup | Compact Rust reverse ms | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 256 | 84.37 | 2.81 | 30.0x | 1.16 | 88.9 / 52.0 |
| 512 | 1024 | 2152.16 | 23.58 | 91.3x | 3.28 | 378.2 / 187.9 |

Reverse timings use the compact `diff.layer_stack` output cotangent and cover
all medium wavenumbers, impedances, transverse components and layer thicknesses.
They exclude packing a dense cotangent into its independent channel blocks.
The native solve and retained residual scale linearly with channel count;
materializing the legacy dense output still costs quadratic time and memory.
For objectives on selected transmission/reflection channels, the compact Advect
API avoids that output expansion. Partial-polarization slab bases use
the general projected composition path instead of this optimization.

Profiling the earlier dense implementation put most slab time in generic
S-matrix composition. A preliminary single-layer, 256-mode case improved from
about 23 ms to 1.5 ms after solving channels independently. The table above is the
subsequent isolated four-layer measurement, not that preliminary timing.
The raw slab timings were temporary measurements and are not archived.
Rerun the historical workload with `--workload slab --layers 4 --channels 512
--threads 4 --repeats 7`.

### Periodic spherical-to-cylindrical conversion

One common origin, lmax=mmax=12, 336 spherical inputs and 1,600 cylindrical
outputs across 32 axial diffraction orders. Wavenumber 1.3, period 200 and Bloch
component 0.2. Four matched threads, seven samples after warmup; the complete
matrix is checked against treams before timing.

Rust takes **1.158 ms** versus **24.266 ms** for treams (**21.0x**). Its native
reverse takes **6.114 ms**, covering both origins, complex medium wavenumbers,
per-output axial wavenumbers and period. Forward peak RSS is **48.5 MiB** versus
**73.6 MiB**; forward plus reverse peaks at 65.1 MiB for Rust. The residual holds
geometry only. Axis selection eliminates zero coefficients while retaining
adjacent azimuthal orders for their nonzero position derivatives.

The raw conversion timing was a temporary measurement and is not archived.
Rerun the historical workload with `--workload periodic-conversion --particles 1 --lmax 12
--samples 32 --threads 4 --repeats 7`. This is conversion timing, excluding
particle construction and the periodic interaction solve.

### Dense internal illumination

Two general dense S matrices, four matched threads, seven samples after warmup.
Random complex reflections scale as 0.1/sqrt(N), with identity transmission plus
similarly sized perturbations; seed 81. All four fields are checked against treams
before timing. Construction of the supplied S matrices is excluded.

These measurements use batches lasting at least 20 ms per sample and
include destruction of returned fields and residuals. Reverse samples aggregate
the same number of fresh contexts, with forward preparation outside the timer.
Earlier tables in this document used individual calls and excluded forward result
cleanup. Raw results record the batch size and methodology explicitly.

The former 1024-mode regression (68 ms and 355.7 MiB versus treams' 60.7 ms and
258.0 MiB) is corrected on the ordinary forward path. `SMatrices.illuminate`
borrows its input blocks and avoids recording unused adjoint inputs. The shared
native LU overwrites the operator in one packed buffer instead of copying it and
allocating separate dense L and U matrices.

| Modes | RHS columns | treams ms | Rust forward ms | Speedup | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 3.54 | 0.87 | 4.07x | 82.6 / 56.6 |
| 256 | 8 | 4.19 | 1.51 | 2.79x | 83.0 / 57.5 |
| 1024 | 1 | 62.09 | 34.04 | 1.82x | 259.2 / 234.0 |
| 1024 | 8 | 63.52 | 30.49 | 2.08x | 259.4 / 235.1 |

Differentiable illumination includes owned input snapshots plus the retained LU.
Snapshots preserve the pullback after Python input mutation. Large blocks copy in
parallel; contiguous row-major inputs retain their layout without a transpose.
In reverse, their buffers are overwritten with the block gradients after the
incident-amplitude cotangents have been computed. Rank-P contractions avoid dense
operator cotangents and conjugate-transpose copies.

| Modes | RHS columns | treams forward ms | Rust recorded forward ms | Speedup | Rust reverse ms | Rust forward / through-reverse peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 3.94 | 0.93 | 4.23x | 0.30 | 65.6 / 85.6 |
| 256 | 8 | 3.82 | 1.11 | 3.43x | 0.50 | 65.2 / 85.2 |
| 1024 | 1 | 61.40 | 40.97 | 1.50x | 29.96 | 332.1 / 420.6 |
| 1024 | 8 | 66.26 | 41.26 | 1.61x | 33.34 | 332.4 / 423.8 |

Recording the 1024-mode adjoint still uses more memory than upstream's forward-only
operation: about 332 versus 259 MiB. This cost is explicit, not hidden in the
forward-only result. RSS includes imports, setup allocations and allocator
retention, not just live arrays. These cases establish no universal speed
guarantee for arbitrary matrices or hosts.

Raw samples, binary hashes and environments are stored in
`benchmarks/results/internal-{forward,adjoint}-l{128,512}-p{1,8}.json`.
`just bench-performance` reruns these eight accuracy-checked cases and the two
compact permutation cases below on an idle host, failing if Rust is slower;
forward-only illumination and permutation also gate peak RSS against upstream.
The harness accepts `--require-speedup` and `--require-rss-ratio` for other
workloads. Timing gates are separate from shared hosted correctness CI.
Here lmax is only a size argument: the dense matrix has 2*lmax modes.

### Compact plane permutations

Cyclic Cartesian-axis transforms return both output polarizations for each input
mode, shape (2, N). The reference is `treams.pw.permute_xyz` broadcast to the same
shape; these are not timings for a dense N-by-N public operator. Four matched
threads, release, accuracy checked first, seven batched samples including output
and residual destruction as described above.

| Modes | treams us | Rust us | Speedup | Rust reverse us | treams / Rust forward peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 7.52 | 5.41 | 1.39x | 16.78 | 65.0 / 42.2 |
| 1024 | 170.23 | 34.68 | 4.91x | 77.78 | 65.2 / 43.4 |

Geometry is shared between adjacent equal directions, while their input
cotangents remain independent. The residual stores only the vectors and discrete
labels. Raw results: `benchmarks/results/plane-permutation-l{64,512}.json`.

The shared scaled polarization and complex quotient arithmetic was also rechecked
on the full plane-field operator (128 modes, 4096 samples). It takes 1.81 ms versus
138.43 ms upstream (76.55x), with 9.77 ms reverse and forward peak RSS of 67.5 versus
89.5 MiB. The result is in `plane-transform-check-field-n128.json`.

### Plane translation phases

Compact exp(i k.r) tables at 4096 displacements, including propagating and
evanescent wavevectors. Both backends receive precomputed vectors; the reference
is the treams 0.4.5 `pw.translate` ufunc. Release extension, matched four-thread
limits, separate processes, seven timed repetitions and the harness's unchanged
accuracy gate (rtol 2e-9, atol 1e-12).

| Plane modes | treams ms | Rust ms | Speedup | Rust reverse ms | treams / Rust forward peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 7.81 | 1.81 | 4.31x | 7.95 | 90.5 / 50.2 |
| 1024 | 72.69 | 22.53 | 3.23x | 64.42 | 258.9 / 108.0 |

The context retains only positions and wavevectors. Rust transfers the phase
table directly into NumPy. Borrowing contiguous cotangents reduced the large
forward-plus-reverse peak from 235.1 to 172.3 MiB and reverse time from 79.33 to
64.42 ms; strided inputs still require packing. Peak RSS includes interpreter,
imports, output tables and cotangents. These measurements cover the compact phase
kernel, not an entire scattering solve or the dense masked translation operator.

The raw phase-table timings were temporary measurements and are not archived.
Rerun the historical workload with
`--workload plane-phases --particles 16 --lmax 32 --samples 4096 --threads 4`;
the number of plane modes is 2*particles*lmax. The smaller case uses 8 and 8.

### Oriented chirality forms

Compact signed-helicity up/down/cross coefficients, averaged from -0.2 to 0.7
along x, including propagating and evanescent waves. The independent reference
constructs Cartesian polarizations with `treams.special.vpw_A` and contracts their
inner products and analytic exponential averages. This measures the equivalent
Cartesian calculation, not upstream's xy-only high-level chirality implementation.
Both outputs have shape (3, N); accuracy is checked before timing.

| Modes | Reference us | Rust us | Speedup | Rust reverse us | Reference / Rust forward peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 54.72 | 24.45 | 2.24x | 114.67 | 65.3 / 42.8 |
| 1024 | 619.42 | 65.88 | 9.40x | 280.72 | 65.3 / 43.4 |

Release, four matched threads, seven batched samples including result destruction.
Reverse covers both real transverse components, complex normal components and
interval endpoints; the residual stores only input geometry. Large mode sets use
Rayon. Raw results: `benchmarks/results/oriented-chirality-l{64,512}.json`.
Reproduce with `--workload oriented-chirality --particles 1 --lmax 512 --threads 4`.

### Heterogeneous particle clusters and block adjoints

Spherical particles with alternating cutoffs 3 and 4, radii 0.15–0.25,
permittivity 4+0.1j, vacuum wavenumber 1.3, and spacing 0.8. Local particle
construction is outside both timings. Both solvers return the complete interacting
matrix; full agreement is checked before timing. Four matched threads, release,
seven batched samples including result destruction.

| Particles | Modes | Path | treams ms | Rust ms | Speedup | Rust reverse ms | treams / Rust forward peak MiB |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 156 | Native block boundary | 35.43 | 3.24 | 10.95x | 0.91 | 71.3 / 44.0 |
| 16 | 624 | Native block boundary | 734.82 | 17.51 | 41.96x | 22.74 | 109.2 / 68.5 |
| 4 | 156 | Public cluster + solve | 35.67 | 3.97 | 8.99x | — | 71.7 / 45.1 |
| 16 | 624 | Public cluster + solve | 737.35 | 23.76 | 31.03x | — | 107.9 / 86.3 |

The native boundary accepts separate local arrays, preserving their block structure
without a dense local matrix. The public `TMatrix.cluster(...).interaction.solve()`
retains its explicit dense array semantics but routes the solve through this block
boundary. Reverse returns only local diagonal-block gradients, plus every position
and both embedding-wavenumber cotangents. It avoids a dense local gradient whose
off-diagonal entries would be discarded. The 624-mode native process peaks at
107.1 MiB through reverse.

The same adjoint change was rechecked on homogeneous sphere clusters with lmax=3.
At 240 modes the full forward takes 2.48 ms versus 90.32 ms upstream (36.42x),
reverse 2.34 ms and forward/through-reverse RSS 46.1/50.9 MiB. At 960 modes it
takes 37.50 ms versus 1678.25 ms (44.75x), reverse 63.31 ms and RSS 102.6/177.5 MiB.
Earlier single-call measurements are not method-identical to these batched results.

Raw results: `particle-cluster{-public,}-n{4,16}-l3.json` and
`block-adjoint-cluster-n{8,32}-l3.json` under `benchmarks/results`.
Use `--workload particle-cluster` or `--workload particle-cluster-public` with
`--particles 16 --lmax 3 --threads 4` to reproduce the heterogeneous cases.

### Broadcast Bessel functions

Outgoing cylindrical Hankel H1 of order 3 at 128 or 4096 complex arguments,
uniform real part 0.6–8.0 and imaginary part 0.2. Results and first derivatives
are compared with the corresponding `treams.special` functions before timing.
Release, four matched threads, seven batched samples including result destruction.

| Values | Operation | Path | treams us | Rust us | Speedup | Rust reverse us |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 128 | H1 | Public forward | 60.86 | 32.13 | 1.89x | — |
| 4096 | H1 | Public forward | 1905.60 | 550.44 | 3.46x | — |
| 128 | H1 derivative | Public forward | 93.62 | 39.13 | 2.39x | — |
| 4096 | H1 derivative | Public forward | 3035.05 | 936.35 | 3.24x | — |
| 128 | H1 | Recorded | 61.54 | 28.52 | 2.16x | 36.16 |
| 4096 | H1 | Recorded | 1911.90 | 554.59 | 3.45x | 935.10 |
| 128 | H1 derivative | Recorded | 93.36 | 48.76 | 1.91x | 62.43 |
| 4096 | H1 derivative | Recorded | 3034.26 | 882.62 | 3.44x | 1389.10 |

A 128-value baseline failed the runtime gate (65.59 us versus 61.08 us).
Parallel evaluation starts at 64 values, and ordinary forward calls borrow
inputs without creating an unused residual. Both forward-only and recorded cases
passed. Separate runs have scheduling variability, so the small differences between
recorded and ordinary forward numbers are not evidence that recording is free.
Rust peak RSS was 39.8–41.0 MiB versus 64.9–65.4 MiB upstream for these cases.
Scalar call overhead and other special-function families are not qualified by
this table. No derivative-order or higher-order autodiff support is implied.

Raw results: `benchmarks/results/bessel{,-derivative}{,-forward}-n{128,4096}.json`.
Use `--workload bessel-forward --samples 4096 --particles 1 --lmax 3 --threads 4`;
the other workload names match the result filenames. All use the same accuracy gate.

### Heterogeneous cylindrical clusters

Alternating azimuthal cutoffs 3/4, two fixed axial channels (0.2, 0.4), release
extension and matched four threads. Local cylinder construction is outside both
timers. These include forward result destruction and use independent processes.
The public path includes cluster construction and the interaction solve.

| Particles / modes | Path | Upstream ms | Rust ms | Speedup | Reverse ms | Upstream / Rust forward RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 4 / 128 | Native blocks | 4.161 | 1.373 | 3.03x | 1.298 | 70.5 / 42.5 |
| 16 / 512 | Native blocks | 67.254 | 20.640 | 3.26x | 25.596 | 96.1 / 59.3 |
| 4 / 128 | Public API | 4.255 | 1.336 | 3.18x | — | 70.2 / 43.9 |
| 16 / 512 | Public API | 67.142 | 23.677 | 2.84x | — | 96.8 / 70.2 |

The native 512-mode run peaks at 83.0 MiB through reverse. Parallel cylindrical
translation contractions reduced its reverse from 59.38 to 25.60 ms in the same
benchmark; the 128-mode reverse fell from 3.01 to 1.30 ms. The serial measurements
were exploratory; committed JSON files contain the qualified measurements.
Raw data: `benchmarks/results/cylindrical-particle-cluster{,-public}-n{4,16}-l3.json`.
All four correctness, runtime and forward-RSS gates passed and are included in
`just bench-performance`.

### Scalar overhead and NumPy ufunc qualification

A scalar outgoing Hankel call exposed a Python-dispatch regression: an exploratory
measurement gave 1.94 us versus upstream's 0.82 us. Native scalar entry points
skip temporary arrays and broadcasting. Array calls use actual NumPy ufunc loops,
retaining NumPy's output allocation, masks, broadcasting and overlap handling.
The Rust complex-Hankel derivative obtains its adjacent orders in one library
sequence evaluation, using the [Bessel derivative recurrence](https://dlmf.nist.gov/10.6.ii).
No numerical tolerance was relaxed.

The following release results use the same independent-process, four-thread,
seven-batch timing protocol, including output/context destruction. Scalar arguments
are Python values at z=1.3+0.2i; arrays cover Re(z)=0.6..8 with Im(z)=0.2.
Order is 3. All twelve correctness/runtime/forward-RSS gates passed. This supersedes
the earlier Bessel implementation's table above; its historical JSON is retained.

| Values | Operation | Path | Upstream us | Rust us | Speedup | Reverse us |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Value | Ordinary | 0.957 | 0.596 | 1.61x | — |
| 128 | Value | Ordinary | 61.576 | 35.080 | 1.76x | — |
| 4096 | Value | Ordinary | 1901.426 | 553.620 | 3.43x | — |
| 1 | Derivative | Ordinary | 1.067 | 1.004 | 1.06x | — |
| 128 | Derivative | Ordinary | 95.156 | 24.925 | 3.82x | — |
| 4096 | Derivative | Ordinary | 3091.247 | 554.712 | 5.57x | — |
| 1 | Value | Recorded | 0.910 | 0.859 | 1.06x | 0.796 |
| 128 | Value | Recorded | 61.387 | 29.670 | 2.07x | 26.041 |
| 4096 | Value | Recorded | 2009.643 | 564.970 | 3.56x | 575.693 |
| 1 | Derivative | Recorded | 1.081 | 0.876 | 1.23x | 0.812 |
| 128 | Derivative | Recorded | 94.985 | 33.888 | 2.80x | 31.131 |
| 4096 | Derivative | Recorded | 3216.664 | 567.114 | 5.67x | 582.907 |

Rust forward peak RSS was 39.2–41.3 MiB, versus upstream's 64.2–65.2 MiB.
The recorded scalar value has only a small runtime margin; these microsecond
results should be rechecked on the target host. The broad performance claim remains
limited to measured workloads, not every input, machine or thread count.
Raw files are `benchmarks/results/ufunc-bessel{,-derivative}{,-forward}-n{1,128,4096}.json`.
`just bench-performance` runs all twelve cases with the same required ratios.

### Public angular functions

The angular-function qualification passed all 50 runtime gates in its
`just bench-performance` snapshot, with four matched threads.
Ordinary forward paths and the special-function/particle adjoint recordings also
passed the peak-RSS gates. Recorded internal illumination retains additional
owned inputs and is intentionally measured separately from its ordinary forward
RSS gate, as described above. These results qualify the listed inputs and host;
they do not establish a universal speed guarantee for every possible argument.

Integer-degree Legendre, pi and tau use native NumPy loops, a direct scalar path,
and the same factored recurrences for forward and native reverse. Fixed-label
recording avoids Python broadcast expansion. Tau shares one sine-power factor
between adjacent Legendre orders. This removed the measured 128-element recording
regression without changing the comparison tolerances.

Degree is 6, order is 2. The scalar argument is 0.3+0.1j; arrays span
[-0.8,0.8]+0.1j. Units below are microseconds. Recording includes owned residual
creation and result destruction; reverse consumes a fresh residual each time.

| Operation | Count | Upstream forward | Rust forward | Speedup | Rust reverse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legendre ordinary | 1 | 0.802 | 0.290 | 2.76x | — |
| Legendre ordinary | 128 | 6.192 | 3.545 | 1.75x | — |
| Legendre ordinary | 4096 | 362.485 | 46.219 | 7.84x | — |
| Legendre recorded | 1 | 0.780 | 0.430 | 1.82x | 0.402 |
| Legendre recorded | 128 | 6.267 | 4.643 | 1.35x | 6.061 |
| Legendre recorded | 4096 | 363.316 | 71.860 | 5.06x | 98.387 |
| Pi ordinary | 1 | 0.837 | 0.335 | 2.50x | — |
| Pi ordinary | 128 | 9.155 | 6.732 | 1.36x | — |
| Pi ordinary | 4096 | 652.243 | 69.679 | 9.36x | — |
| Pi recorded | 1 | 0.815 | 0.487 | 1.67x | 0.465 |
| Pi recorded | 128 | 9.161 | 7.882 | 1.16x | 11.552 |
| Pi recorded | 4096 | 651.226 | 71.144 | 9.15x | 111.898 |
| Tau ordinary | 1 | 0.879 | 0.362 | 2.43x | — |
| Tau ordinary | 128 | 12.946 | 8.792 | 1.47x | — |
| Tau ordinary | 4096 | 856.314 | 86.509 | 9.90x | — |
| Tau recorded | 1 | 0.850 | 0.492 | 1.73x | 0.455 |
| Tau recorded | 128 | 12.955 | 9.976 | 1.30x | 15.271 |
| Tau recorded | 4096 | 855.324 | 88.387 | 9.68x | 142.593 |

Peak RSS is 40.3–41.7 MiB for Rust versus 64.1–65.2 MiB for upstream.
The 18 `angular-*.json` files in [raw results](../benchmarks/results/) record every
sample, native-extension hash, Python/NumPy version, thread configuration and
reverse peak RSS. The other 32 recipe results were refreshed in the same run.

### Wigner elements, Euler pullbacks and public Ewald integrals

The Wigner/Ewald qualification passed all 68 runtime gates in its
`just bench-performance` snapshot, including 50 angular/baseline workloads and
18 Wigner/Ewald cases. The preceding angular table records the run saved in
commit 4f41708; the raw JSON files were refreshed by the 68-case run. The same RSS
qualification applies: ordinary forward and special-function/particle recording
paths must use no more peak RSS
than upstream; retained internal-illumination adjoint inputs are measured separately.

Individual Wigner elements use the [Jacobi recurrence](https://dlmf.nist.gov/18.9.E2),
without allocating a full angular matrix. Public ufuncs parallelize large strided
arrays while preserving masked/in-place behavior. Owned Wigner residuals retain
only labels and Euler arguments, and compute local angle derivatives in reverse.
The scalar Wigner-3j path bypasses NumPy dispatch; this removed its measured scalar
overhead. No accuracy tolerance was relaxed.

Wigner degree is 6, row 1, column -2, with outer Euler angles 0.2 and -0.1.
The polar argument is 0.7+0.1j for scalars and spans [0.3,1.3]+0.1j for arrays.
The 3j case uses degrees (6,6,6) and orders (1,-2,1). Gamma degree is 1.5;
Kambe order is -2 and eta=0.7+0.1j. Units below are microseconds.

| Operation | Count | Upstream forward | Rust forward | Speedup | Rust reverse |
| --- | ---: | ---: | ---: | ---: | ---: |
| wigner-forward | 1 | 1.510 | 1.236 | 1.22x | — |
| wigner-forward | 128 | 40.153 | 14.328 | 2.80x | — |
| wigner-forward | 4096 | 2256.407 | 135.234 | 16.69x | — |
| wigner-small-forward | 1 | 1.124 | 0.970 | 1.16x | — |
| wigner-small-forward | 128 | 39.366 | 12.470 | 3.16x | — |
| wigner-small-forward | 4096 | 2127.653 | 119.326 | 17.83x | — |
| wigner3j-forward | 1 | 1.302 | 0.292 | 4.47x | — |
| wigner3j-forward | 128 | 28.319 | 16.130 | 1.76x | — |
| wigner3j-forward | 4096 | 1003.028 | 141.485 | 7.09x | — |
| incgamma-forward | 1 | 0.672 | 0.314 | 2.14x | — |
| incgamma-forward | 128 | 20.111 | 15.654 | 1.28x | — |
| incgamma-forward | 4096 | 1085.018 | 164.622 | 6.59x | — |
| intkambe-forward | 1 | 0.835 | 0.374 | 2.23x | — |
| intkambe-forward | 128 | 24.098 | 22.111 | 1.09x | — |
| intkambe-forward | 4096 | 1358.198 | 204.507 | 6.64x | — |
| wigner | 1 | 1.428 | 0.718 | 1.99x | 0.742 |
| wigner | 128 | 39.724 | 24.048 | 1.65x | 22.380 |
| wigner | 4096 | 2357.991 | 144.681 | 16.30x | 274.425 |

Peak RSS spans 39.9–41.5 MiB for Rust versus 64.2–65.8 MiB for upstream. Native
hashes, timing samples, reverse measurements and environment details are in the
corresponding [raw results](../benchmarks/results/).

### Cylindrical axial field derivatives

For four particles, order 3 and four threads on the measured release build,
both the ordinary reverse and optional per-mode axial reverse passed accuracy,
forward-runtime and forward-RSS gates. The forward is identical for both choices;
recording adds no axial derivative table. Timings in milliseconds:

| Reverse requested | Samples | Upstream forward | Rust forward | Speedup | Rust reverse | Rust / upstream RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing | 128 | 32.071 | 3.120 | 10.28x | 4.045 | 44.8 / 67.7 |
| Existing | 2048 | 503.039 | 49.549 | 10.15x | 66.882 | 44.9 / 96.4 |
| All + axial | 128 | 33.642 | 3.094 | 10.87x | 4.180 | 43.8 / 67.7 |
| All + axial | 2048 | 504.578 | 48.038 | 10.50x | 65.684 | 44.3 / 96.3 |

The pre-change baseline at `ce0bde4` measured 3.172/47.935 ms forward and
4.197/62.601 ms reverse for 128/2048 samples. Subsequent fixed-path runs measured
3.093–3.120/48.038–49.549 ms forward; these short successive runs vary by a few
percent and do not establish a change at that scale. Axial reverse remains about
4.2/65.7 ms while computing one additional derivative per mode. The four field
cases brought the historical `just bench-performance` snapshot to 72 gates. Its 68-gate combined run and the
four field cases are recorded separately, not claimed as one run.

### Shared axial derivatives of finite and periodic cylindrical expansions

Four or sixteen particles, two axial groups, order 3 and four threads. Each
accuracy-checked case passed forward-runtime and peak-RSS gates against treams.
Periodic comparisons use the same explicit Ewald split eta=0.7 in both solvers.
At period 12.8, upstream's automatic split differs from the converged result by
up to 0.007466 in this matrix; explicit eta=0.5/0.8/1.0 agrees with Rust to about
1.5e-11 or better. The native automatic split also agrees with those values.
The scalar split regression covers both periods 7.2 and 12.8.

| Workload | Particles | Upstream forward ms | Rust forward ms | Speedup | Rust reverse ms | Rust / upstream RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| expansion | 4 | 2.319 | 0.583 | 3.98x | 0.606 | 41.9 / 67.1 |
| expansion | 16 | 38.088 | 10.762 | 3.54x | 10.599 | 48.2 / 74.8 |
| expansion-axial | 4 | 2.313 | 0.592 | 3.91x | 0.640 | 42.4 / 66.8 |
| expansion-axial | 16 | 37.895 | 10.523 | 3.60x | 10.716 | 48.0 / 74.8 |
| periodic | 4 | 1522.719 | 8.786 | 173.32x | 19.316 | 43.5 / 67.6 |
| periodic | 16 | 16303.387 | 88.316 | 184.60x | 238.705 | 50.4 / 86.0 |
| periodic-axial | 4 | 1510.388 | 9.131 | 165.41x | 20.026 | 43.4 / 66.8 |
| periodic-axial | 16 | 16334.604 | 88.576 | 184.41x | 238.965 | 50.3 / 85.8 |

The `-axial` cases evaluate the same forward and request the additional grouped
axial gradient in reverse. Periodic reverse reuses the same Ewald derivatives,
without retaining an axial Jacobian. The eight cases brought the historical
`just bench-performance` snapshot to 80 gates. They were run separately from the
68-gate combined run and the four field cases above.

The complete four-cylinder array path also passed its runtime/RSS gate:
4.729 to 4.511 ms forward and 12.138 to 11.760 ms reverse without/with the grouped
axial derivative, with measured peak RSS 42.5/43.9 MiB. This is a short same-host check rather than
a claim that the optional derivative improves the unchanged forward algorithm.
Raw baseline and grouped-derivative runs are saved as
`cylindrical-array-{before,after}-axial.json`.

### Coordinate transformations

All eight point and eight vector-frame conversions use native NumPy gufuncs
with analytic pullbacks. Component strides, NumPy broadcasting and output buffers
are supported. Direct factored rotations avoid temporary matrices; a constant
broadcast vector is retained once in an adjoint context.

The coordinate qualification passed 128 combined performance gates, including 48
coordinate comparisons at 1, 128 and 65,536 points. Every case first checks values
against upstream; the gates require forward speedup >= 1 and peak forward RSS no
higher than upstream. Representative forward speedups:

| Transform | 1 point | 128 points | 65,536 points |
| --- | ---: | ---: | ---: |
| car2sph | 1.08x | 1.21x | 2.31x |
| sph2car | 1.04x | 1.33x | 2.87x |
| car2pol | 1.08x | 1.51x | 4.23x |
| vsph2car | 1.01x | 1.31x | 2.96x |
| vpol2car | 1.07x | 1.43x | 5.04x |

Raw samples, RSS and binary identities are in `coordinate-*-n*.json`. These
submicrosecond scalar differences remain sensitive to machine noise; the larger
arrays expose the numerical throughput advantage more clearly.

The benchmark's forward-only helpers return the actual result directly.
Previously only the Rust helper added and destroyed a dummy `(value, None)`
residual tuple, which biased small operations. Recorded operations still return
and retain their real contexts, and metadata explicitly records
`forward_records_adjoint`. Both backends include result destruction. The complete
128-case run was repeated after this correction; no accuracy or performance
threshold was relaxed.

### Local wave functions

The local-wave qualification passed all 191 runtime gates and all 187 applicable
forward-RSS gates. The four recorded internal-illumination cases retained an owned
adjoint tape and were deliberately not compared with upstream forward-only RSS. No accuracy,
runtime or RSS threshold was relaxed. That build also passed 61 native tests and
1,512 Python tests through the then-current `just ci` recipe; its clean Linux wheel
passed SciPy-free Advect workflows and the optional HDF5 round trip.

The combined run used `taskset -c 8-11 just bench-performance`: both backends had
the same four physical cores and four BLAS/Rayon threads. Background activity on
the original first-four-core placement materially affected native solve timings;
those diagnostic runs are retained alongside the final results. Use otherwise
idle cores for reproducible qualification, rather than interpreting shared-host
contention as a kernel regression.

Every workload with isolated median times below one millisecond also runs both
backends in one process, alternating their order over 14 paired samples. Each
sample batches at least 20 milliseconds of work for each backend. The reported
speedup is the median paired ratio; isolated processes remain authoritative for
peak RSS and reverse cost. JSON retains both timing methods and all samples.
The smallest measured speedup in the complete run is 1.013x (scalar plane M wave),
so tiny scalar comparisons remain sensitive to CPU noise.

| Forward function | 1 sample | 128 samples | 4,096 samples |
| --- | ---: | ---: | ---: |
| sph_harm | 1.32x | 3.94x | 15.05x |
| vsw_rA | 1.73x | 4.59x | 20.36x |
| vcw_rA | 1.30x | 2.33x | 9.90x |
| vpw_A | 1.06x | 3.35x | 21.75x |

| Recorded function | 128 samples | 4,096 samples | Reverse at 4,096 samples |
| --- | ---: | ---: | ---: |
| vsw_rA | 4.40x | 26.88x | 0.644 ms |
| vcw_rA | 2.20x | 10.58x | 0.677 ms |
| vpw_A | 2.08x | 18.83x | 0.191 ms |

The wave kernels share normalized angular recurrences, fetch adjacent cylindrical
Bessel values together, and reuse plane polarizations for constant directions.
An algebraic complex square root removes trigonometric work from normalization;
scaled fallbacks preserve extreme magnitudes and signed branch limits. Coordinate
norms use an FMA path at ordinary magnitudes and scaled hypot at extremes.

Recorded illumination preserves contiguous input order, uses two independent
copy tasks for large stacks, and completes the snapshots before solving. This
keeps later Python mutation safe without competing with the solve for workers.
The contiguous finite-input scan vectorizes. Forward-only illumination continues
to borrow inputs and retains no tape. Superseded overlap and serial-copy timings
remain in `snapshot-*`, `serial-*` and `parallel-copy-*` JSON files.

Final wave results are `wave-<function>-n<size>.json` and
`wave-adjoint-final-<function>-n<size>.json`; each includes the native binary hash.
These measurements qualify the listed workloads, not every possible problem size,
conditioning regime or machine.

### Coefficient namespaces and integral adjoints

The coefficient qualification passed all 257 runtime gates and all 253 applicable
forward-RSS gates on the same four-core Linux configuration. The four recorded
internal-illumination cases retain the existing RSS exemption described above.
The smallest measured speedup is 1.006x for scalar cw.rotate; this remains a
noise-sensitive comparison. Accuracy, speed and RSS thresholds are unchanged.
The result manifest and native SHA256 are in `benchmarks/coefficient-qualification.json`.

| Forward function | 1 sample | 128 samples | 4,096 samples |
| --- | ---: | ---: | ---: |
| tl_vcw | 1.822x | 1.299x | 2.569x |
| tl_vcw_r | 2.427x | 1.368x | 4.606x |
| cw.translate | 2.195x | 1.510x | 4.272x |
| pw.permute_xyz | 3.592x | 1.367x | 10.028x |
| pw.translate | 2.223x | 1.226x | 1.236x |

Cylindrical coefficient arrays use the radial kernel's 64-element Rayon
threshold; scalar Python numbers bypass NumPy dispatch. Plane-coordinate
permutations share normalization factors and retain the scaled axis/extreme
branches. Plane phases evaluate one exponential and one sine/cosine pair. Kambe
recording avoids redundant broadcasts and recurrence setup. Raw baseline failures
and targeted rechecks remain alongside the passing combined results.

Paired timing calibrates each backend's batch independently to at least
20 milliseconds. This avoids oversampling the slower backend when the speedup
is large. Both backends still receive 14 alternating paired samples, with
per-call timings and independent-process RSS measurements. The first 91 cases
were retained from the preceding calibration method at the identical native
SHA256; the other 166 used independent batch sizes, recorded explicitly in JSON.
The manifest records which cases used each calibration method.

The measured build passed 65 native tests, 1,611 Python tests, strict
lint/type/rustdoc checks and isolated Linux wheel checks including Advect and
optional HDF5.

### Lattice geometry and material branches

The geometry qualification passed 42 geometry cases and 257 regression cases:
299 runtime gates and 295 applicable forward-RSS gates, with no threshold changes. Both backends used CPU cores 8-11 and four BLAS/Rayon threads.
The smallest measured runtime margin is 1.005x for scalar cw.rotate; scalar
comparisons remain sensitive to CPU noise. The largest applicable RSS ratio
is 0.912. The unchanged four recorded illumination cases remain RSS-exempt.

| Forward operation | 1 sample | 128 samples | 4,096 samples |
| --- | ---: | ---: | ---: |
| volume2 | 1.033x | 2.978x | 44.687x |
| volume3 | 1.047x | 2.647x | 25.899x |
| reciprocal2 | 1.026x | 6.004x | 8.677x |
| reciprocal3 | 1.034x | 3.121x | 3.577x |
| refractive_index | 3.213x | 3.466x | 1.955x |
| wave_vec_z | 1.904x | 3.197x | 1.607x |

Circular diffraction enumeration measures 7.915x, 64.034x and 256.860x at radius
1, 4 and 16; tests separately establish complete enumeration in skew cells.
The material and outgoing-normal-wavevector loops use serial SIMD-friendly
arithmetic for these cheap elements. The shared square root uses FMA at ordinary
scales and retains the scaled branch for extreme arguments. Cell determinants
inline fixed 1D/2D/3D arithmetic; cube boundaries enumerate output points directly.

The 42 geometry results are `geometry-<operation>-n<size>.json`; the 257
regression rechecks have the `geometry-recheck-` prefix, preserving the previous
coefficient qualification files. `benchmarks/geometry-qualification.json` records
all 299 paths and their shared native binary hash. Mac diagnostic results retain
their original filenames and platform metadata; they are not the Linux proof.
The measured build passed the Linux suite and isolated wheel checks
(68 Rust, 1,647 Python).

## Dense CPU scheduling

`Lu` selects faer parallelism from the matrix size and number of requested
right-hand sides. It uses the existing Rayon pool and never exceeds either that
pool or faer's configured worker budget. Configurations of four or fewer workers
retain their previous scheduling. WASM stays serial.

Scheduling fine-grained work in faer's recursive LU and triangular solves across
too many workers caused the measured slowdown. This path has no BLAS thread
pool. The worker limit changes scheduling without changing factorization or
pullback mathematics.

For larger pools the worker limit is
`max(1, min(budget, columns / 16, max(min(rows / 512, 4), rows / 2048)))`, with
integer division. Factorization supplies `columns = rows`; forward and adjoint
solves supply the actual RHS count. This keeps narrow illuminations serial and
allows larger matrices to use progressively more workers. It does not impose a
permanent four-worker ceiling.

On Ryzen 9950X with one 16-worker pool pinned to physical cores 0–15, the
[scheduling probe](../benchmarks/results/cpu-parallelism.json) measured these medians:

| Complex128 matrix / RHS | Previous factor + solve | Bounded factor + solve | Stage speedup |
|---|---:|---:|---:|
| 256 / 4 | 47.56 ms | 0.57 ms | 83.9× |
| 1024 / 64 | 527.15 ms | 15.85 ms | 33.3× |
| 2048 / 512 | 974.03 ms | 152.22 ms | 6.4× |
| 4096 / 64 | 2725.67 ms | 483.54 ms | 5.64× |
| 8192 / 64 | 6528.76 ms | 3279.13 ms | 1.99× |

These isolate scheduling overhead; they are not end-to-end API or GPU speedup
claims. The probe used Cargo's default release profile, a warmup and three
samples per case. Minor unrelated background CPU activity existed on the shared
host. Four-worker performance was already much better than the old 16-worker
path, so GPU comparisons must include the best CPU configuration. At 8192 rows,
factorization took 3.14 s with four workers, 3.28 s with eight and 5.03 s with
sixteen. Sizes above 8192 were not timed; their increasing worker budget is a
work-granularity heuristic, not a measured optimum.

The checked-in probe uses the workspace release profile and verifies complete
normal and Hermitian-adjoint equation residuals outside timing:

```sh
RAYON_NUM_THREADS=16 cargo run --release -p treams-core --example benchmark_lu -- 1024 64 2 5
```

Arguments are matrix rows, RHS columns, requested faer workers and measured
repetitions. Compare worker counts 1, 2, 4, 8 and 16 on the same CPU affinity.
