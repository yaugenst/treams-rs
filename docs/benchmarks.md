# Scattering and field benchmarks

Measured on [redacted-host], AMD Ryzen 9 9950X (16 physical cores), Linux x86-64,
Python 3.13.1, treams 0.4.5. Rust uses the optimized build, faer and Rayon.
These are measured local results for the implemented sphere-cluster path,
not a performance claim about all of treams.

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

## Batched spherical fields

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
reverse, and 41.64 versus 94.11 MiB forward peak RSS. The initial implementation
took 138.96 ms forward; reusing the radial value and its first two derivatives
across adjacent orders removed repeated Bessel calls. The forward residual remains
linear in modes plus samples. Both versions and raw samples are recorded in
`cylindrical-fields*-n4-l3-p2048-t4.json`. Reproduce with `--workload cylindrical-field`.

## Periodic sphere arrays and adjoints

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
output cotangent. That earlier benchmark spent about ten forward evaluations'
time on one complete periodic reverse pass; recomputation keeps residual memory
small, but reverse runtime still needs optimization. No derivative speedup over
Dreams or another autodiff implementation has been measured.

Raw results: `periodic-n4-l3-t1.json`, `periodic-adjoint-n4-l3-t4.json`, and
`periodic-adjoint-n9-l3-t4.json` in `benchmarks/results`. The harness now records
native reverse timings separately from the forward comparison:

```sh
uv run --no-sync python scripts/benchmark_cluster.py --workload periodic --particles 4 --lmax 3 --threads 4
```

## Complete periodic S matrices

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

## Multipole rotations

For two spherical origins and all degrees through 8 (320 modes), the complete
rotation takes 0.510 ms versus treams' 5.597 ms (10.98x) with four matched threads.
Native reverse takes 0.753 ms; forward peak RSS is 42.27 versus 67.03 MiB. Both
forward timings include constructing angular blocks. Rust retains the small-d
blocks for an analytic Euler-angle reverse pass. Raw samples and binary identity
are in `rotation-n2-l8-t4.json`; reproduce with
`--workload rotation --particles 2 --lmax 8 --threads 4`.

## Cylindrical-to-spherical conversion

Release, four matched threads, one common origin, spherical lmax=12 (336 modes),
32 axial wavenumbers between -0.7 and 0.7 and cylindrical mmax=12 (1,600 modes).
The full 336-by-1,600 matrix is checked against treams before timing.
treams takes 23.13 ms and Rust 3.34 ms: **6.92x** on this case. The native pullback
for arbitrary origin and complex-wavenumber cotangents takes 5.80 ms; forward peak
RSS is 73.9 MiB versus 55.9 MiB. Seven samples after warmup, isolated processes.

The first implementation took 20.48 ms forward and 23.63 ms reverse. Skipping
analytically zero azimuthal orders at coincident transverse origins reduced those
times without suppressing adjacent-order position derivatives. This measurement
covers common-origin conversion; it does not establish displaced-origin speedup.
Raw results: `/tmp/conversion-before-l12-k32-t4.json` and
`/tmp/conversion-l12-k32-t4.json` on [redacted-host].

## Plane fields and direct NumPy buffer transfer

Four matched threads, 128 plane modes (64 transverse vectors, both polarizations),
4,096 Cartesian samples, including propagating and evanescent waves at k0=1.3.
Every benchmark first compares the full result against upstream.

| Output | treams forward | Rust forward | Speedup | Rust reverse | Forward peak RSS, treams / Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weighted plane field | 139.37 ms | 1.95 ms | 71.5x | 3.58 ms | 88.3 / 42.1 MiB |
| Full plane field operator | 137.13 ms | 2.70 ms | 50.7x | 9.76 ms | 88.1 / 65.7 MiB |

The initial weighted reverse pass took 6.10 ms. Contracting polarization
cotangents over all samples before differentiating each mode reduced it to 3.58 ms.
The initial full operator took 13.32 ms forward and 89.4 MiB peak RSS: transferring
the owned Rust buffer directly to NumPy removes that extra copy. Its final reverse
pass packs the returned stride layout with a bulk copy and still accepts arbitrary
cotangent strides. These are complete Python/native boundary timings, including
input conversion, output transfer and residual creation. No output Jacobian is
retained. Seven samples after warmup; process and scheduling variability remain.

The same buffer transfer applies to multipole field operators. Rechecking four
spherical origins, lmax=3 and 2,048 points gives 823.02 / 42.83 ms (19.2x), with
75.81 ms reverse and 97.1 / 53.5 MiB forward peak RSS. The earlier copying path used
64.2 MiB for Rust. Weighted multipole fields already avoid the full operator.

Raw results on [redacted-host]: `/tmp/plane-field-d128-p4096-t4.json`,
`/tmp/plane-operator-packed-d128-p4096-t4.json` and
`/tmp/field-operator-zero-copy-n4-l3-p2048-t4.json`. Development baselines include
`/tmp/plane-field-before-d128-p4096-t4.json` and
`/tmp/plane-operator-d128-p4096-t4.json`.

## Plane-to-spherical illumination

Four matched threads, two spherical origins, lmax=8 (320 modes), and 64 transverse
vectors with both polarizations (128 plane modes). At k0=1.3 the directions include
propagating and evanescent waves. Complete output agreement is checked first.
treams takes 14.07 ms and Rust 0.598 ms: **23.6x** for this case. The native reverse
pass, including all origin and full complex-wavevector cotangents, takes 1.23 ms.
Forward peak RSS is 67.5 / 40.2 MiB. Rust shares the normalized direction across
multipoles, uses Rayon over incident modes, and transfers the output buffer to
NumPy without retaining it in the residual. Seven samples after warmup.
Raw result: `/tmp/plane-expansion-before-n2-l8-k64-t4.json` on [redacted-host].

## Complete cylindrical arrays

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

Raw results on [redacted-host]: `/tmp/cylindrical-array-converged-n4-m5-t4.json`
and `/tmp/cylindrical-array-converged-n9-m3-t4.json`. Reproduce with
`--workload cylindrical-array --particles 9 --lmax 3 --threads 4 --repeats 7`.

## Cylindrical plane illumination and phase reuse

Four origins, mmax=12 (200 cylindrical modes), kz=0.2, and 128 transverse
vectors with both polarizations (256 plane modes). Four matched threads, seven
samples after warmup, complete matrix checked against treams. Rust takes 0.220 ms
versus 6.376 ms (**28.9x**); the native origin/transverse-wavevector reverse takes
0.446 ms. Forward peak RSS is 39.9 MiB versus 67.7 MiB.

The initial implementation took 0.377 ms forward and 0.537 ms reverse. Reusing
each origin phase across its multipoles reduced both. The same change improves
spherical plane illumination: the previously described two-origin, lmax=8 case
now takes 0.494 ms versus 14.143 ms (**28.6x**), with 1.163 ms reverse.
Cylindrical axial labels remain fixed in these gradients.

Raw results: `/tmp/cylindrical-plane-expansion-n4-m12-k128-t4.json` and
`/tmp/plane-expansion-phases-n2-l8-k64-t4.json` on [redacted-host]. The cylindrical
case uses `--workload cylindrical-plane-expansion --particles 4 --lmax 12
--samples 128 --threads 4 --repeats 7`.

## Compact planar multilayers

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
API avoids that output expansion. Partial-polarization slab bases currently use
the general projected composition path instead of this optimization.

Profiling the earlier dense implementation put most slab time in generic
S-matrix composition. A preliminary single-layer, 256-mode case improved from
about 23 ms to 1.5 ms after solving channels independently. The table above is the
subsequent isolated four-layer measurement, not that preliminary timing.
Raw results: `/tmp/slab-l4-q128-t4.json` and `/tmp/slab-l4-q512-t4.json` on
[redacted-host]. Reproduce with `--workload slab --layers 4 --channels 512
--threads 4 --repeats 7`.

## Periodic spherical-to-cylindrical conversion

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

Raw result: `/tmp/periodic-conversion-n1-l12-k32-t4.json` on [redacted-host].
Reproduce with `--workload periodic-conversion --particles 1 --lmax 12
--samples 32 --threads 4 --repeats 7`. This is conversion timing, excluding
particle construction and the periodic interaction solve.

## Dense internal illumination

Two general dense S matrices, one incident right-hand side in each direction,
four matched threads and seven samples after warmup. Random complex reflections
scale as 0.1/sqrt(N), with identity transmission plus similarly sized perturbations;
seed 81 makes the well-conditioned algebra benchmark reproducible. This measures
illumination of supplied S matrices, excluding their physical construction. All
four outgoing/internal fields are checked against treams before timing.

| Modes | treams ms | Rust ms | Speedup | Rust reverse ms | treams / Rust forward peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 3.60 | 2.88 | 1.25x | 1.17 | 82.4 / 64.1 |
| 1024 | 60.65 | 68.00 | 0.89x | 32.15 | 258.0 / 355.7 |

The larger forward case remains slower and uses more memory. Rust includes owned
input copies and a retained LU for its native pullback; treams returns fields only.
Copying protects reverse correctness if Python subsequently mutates the inputs.
Tiled parallel input copies reduced the 1024-mode forward from an initial 92 ms
to 68 ms. Adjoint matrix views, rank-one contractions and releasing primal S
matrices before allocating their gradients reduced reverse from 92 ms to 32 ms.
The initial forward-plus-reverse peak was about 595 MiB. These improvements do not
establish universal outperformance of treams.

Raw results: `/tmp/internal-field-parallel-copy-n256-p1-t4.json` and
`/tmp/internal-field-parallel-copy-n1024-p1-t4.json` on [redacted-host]. Reproduce
with `--workload internal-field --particles 1 --lmax 512 --samples 1 --threads 4`.
Here lmax is only a size argument: the dense matrix has 2*lmax modes.
