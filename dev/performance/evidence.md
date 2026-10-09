# Benchmark evidence

## Evidence provenance

Each row is one set of measurements. Dates, commits, hosts and case counts come
from the linked summary files; "not recorded" means the value is missing.
Commits are the revisions of the original runs;
[`history-provenance.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/history-provenance.json) maps them
to this repository's history. The linked files identify each build with SHA-256
hashes of the native library, Python sources and benchmark scripts. Files under
`benchmarks/results/` keep
their recorded measurements; identifying data and archive metadata were redacted
([redacted paths](#redacted-paths)).

| Measurement set | Date | Source commit | Host | Cases | Outcome | Summary / manifest |
| --- | --- | --- | --- | ---: | --- | --- |
| CPU improvements after 0.1.0 | 2026-10-04 | `8bbcb87` plus archived candidate patch | AMD Ryzen 9 9950X, Linux, Python 3.13.1 | 36 targeted and 92 broad four-thread comparisons, plus thread and memory screens | Target timings pass; original broad outliers, rechecks and four periodic memory failures retained; predates dense-product scheduling change | [`cpu-speedups-20261004.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/cpu-speedups-20261004.json), [CPU report](cpu-speedups.md) |
| Dense-product scheduling and final CPU controls | 2026-10-04 | `8bbcb87` plus archived reconstructed candidate | AMD Ryzen 9 9950X, Linux, Python 3.13.1 | 12 paired 32-thread cases, four-thread controls and six bitwise workflow checks | Final comparisons with 0.1.0 pass; intermediate regressions and discarded policies retained | [`cpu-scheduling-20261004.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/cpu-scheduling-20261004.json), [scheduling report](cpu-speedups.md#small-product-scheduling) |
| Linux core correctness | 2026-09-19 | `2843a70` | AMD Ryzen 9 9950X, Linux, Python 3.13.1 | 615 | All pass | [`linux-core-qualification.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/linux-core-qualification.json), [summary](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/linux-core-qualification.md) |
| Linux core performance | 2026-09-19 to 2026-09-20 | `2470fbe` | AMD Ryzen 9 9950X, Linux, Python 3.13.1 | 1,300 planned, 769 timed | All measured; broad-grid median 5.12×; 4 slower cases (2 recorded internal illuminations at a solver switch point, 1 scalar recorded `tl_vcw_r` call, 1 matrix-free illumination case at 32 columns) | [`linux-core-performance.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/linux-core-performance.json), [summary](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/linux-core-qualification.md) |
| Reference replay grid | 2026-09-12 | not recorded | AMD Ryzen 9 9950X, Linux, Python 3.13.1, CPUs 8–11 | 527 | 527 runtime and 525 RSS checks pass; minimum 1.02× | [`complete-qualification.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/complete-qualification.json) |
| macOS dispatch grid | 2026-09-12 | not recorded | Apple M3, macOS 26.6.2, Python 3.13.12 | 30 | All pass; minimum 1.30×, largest RSS ratio 0.695 | [`mac-qualification.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/mac-qualification.json) |
| Mac and Linux comparison | not recorded (by 2026-09-12) | `7421ca9` (Linux) | Apple M3 and AMD Ryzen 9 9950X | 736 per platform | Every outcome kept; median 5.17× (M3) and 5.04× (Ryzen); 11 slower Mac cases | [`comparison-plan.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/comparison-plan.json), [platform comparison](platform-comparison.md) |
| Requested illumination | 2026-09-12 | not recorded | AMD Ryzen 9 9950X (CPUs 8–11) and Apple M3 | 9 | All complete; matrix-free gradient memory 30× below the full T-matrix at 512 spheres | [`illumination-linux-n512-l1-p1.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/results/illumination-linux-n512-l1-p1.json) and the other `illumination-*.json`, [large problems](large-problems.md) |
| Published applications | sources retrieved 2026-09-12 | not recorded | macOS 26.6.2 arm64, Python 3.13.12 | 5 spectra, 300 thermal frequencies | Within the stated tolerances; thermal absorption within 0.577% of the author data | [`papers/qualification.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/papers/qualification.json), [`papers/thermal-result.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/papers/thermal-result.json), [published applications](../validation/published-applications.md) |
| Public cluster recheck (macOS) | not recorded | `43bb7fe` | macOS 26.6.2 arm64, Python 3.13.12 | 2 | Both pass; spherical cluster 7.20×, cylindrical cluster 1.56×, lower peak memory (RSS ratios 0.729 and 0.701) | [`agent-usability/qualification.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/agent-usability/qualification.json) |
| Fractional Legendre check | not recorded | not recorded | not recorded | 532 | 530 finite cases within 2.17e-13 relative of 70-digit values; 2 expected overflows | [`fractional-legendre-physical.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/results/fractional-legendre-physical.json) |
| LU scheduling probe | not recorded | not recorded | AMD Ryzen 9950X, 16 Rayon workers on CPUs 0–15 | 35 configurations | Bounded worker counts 1.99× to 83.9× faster per stage than the full pool, in 7 selected sizes | [`cpu-parallelism.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/results/cpu-parallelism.json), [numerics](../design/numerics.md) |
| Pre-release speed comparison (not a validation run) | 2026-09-26 to 2026-09-30 | baseline `5dadf1d` | shared 4-CPU cloud VM, one thread | 556 cases of 203 workloads | Geometric-mean time ratio against `5dadf1d`: 0.77 forward and 0.68 reverse without lattice sums, 0.48 and 0.35 for lattice sums; `recsumcw1d` about 4% slower | not archived; workloads in [`benchmark_cluster.py`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/scripts/benchmark_cluster.py) |

The retained report does not record run dates for both platforms in the Mac and
Linux comparison.
[`correction-plan.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/correction-plan.json), created on
2026-09-12, uses it as its baseline, so the comparison ran on or before that day.

The pre-release comparison alternated optimized builds of a later revision and
`5dadf1d`, then rechecked suspected slowdowns by CPU time and instruction counts.
Among the largest gains (forward / reverse) are field evaluation 17× / 8.6×,
EBCM 8.4× / 5.0×, cylindrical
expansion 6.3× / 5.4×, cylindrical periodic arrays 4.4× / 3.2×, Wigner 3j 2.3×
and `intkambe` pullbacks 1.9×. The single-point 1D cylindrical reciprocal sum
(`recsumcw1d`) is the only slowdown beyond noise: the lattice accuracy checks
require extra Ewald shell, sheet and method tracking. Explicit lattice
splits below the automatic one can take longer because they sum more shells and
check their cancellation ([last section](#lattice-sums-at-explicit-small-splits-pre-release-timings)).

## Automatic-dispatch overhead

A 2026-10-03 check compared eight NumPy calls before and after automatic
autodiff selection: sphere scattering and cross sections at degrees 1 and 10,
one-point fields at degree 3, a two-sphere solve, and two-channel interface and
slab construction. Both versions used the same native library.

| Frameworks imported before timing | Added time in the seven smallest calls | Calls exceeding the 10% timing limit |
| --- | ---: | ---: |
| None | 0.3–2.8 µs | 3 of 8 |
| Advect, JAX, PyTorch and Autograd | 2.0–9.7 µs | 7 of 8 |

For example, slab construction changed from 19.3 to 22.0 µs without framework
imports, and from 19.1 to 28.8 µs with all four imported. The two-sphere solve
stayed within the timing limit in both conditions. All recorded values agreed
exactly. These timings cover small forward calls, not gradient computation or
large workloads; memory was not measured.

The check used [`compare_builds.py`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/scripts/compare_builds.py) on Linux,
an AMD Ryzen 9 9950X and Python 3.13.1: five paired rounds, three samples of at
least 20 ms per call, alternating measurement order, one native/BLAS thread
and CPU affinity 4–5. Imports and input setup were outside the timed calls.
Reported times are medians of each round's fastest sample; the limit uses the
median paired time ratio.
The raw rounds and source snapshots are retained privately. Earlier benchmark
results above remain unchanged.

### Measured source and library identities

The baseline was `e3e759f1bc46d1df35bb1dbb6d9c99a91fb691be`; the candidate
was the automatic-dispatch source snapshot, including its traversal
optimization and reviewed fixes. SHA-256 fingerprints identify these measured
builds, independently of later source changes:

```text
Baseline Python: 5fbd9d6ef41fbc04633607ab473ed5f1753d1759e1013f42a611c684fb9e0c08
Candidate Python: 839a44d639e7c3aa01004915066df57c9cc6a84e34938c766197e30c6781d596
Shared native: 9e8a39cb2fc8459b5069a65ffc349c9eee71024ced6464524e55f52f9b8ed3c5
```

## Redacted paths

The distributed evidence has privacy-only redactions documented in
[`benchmarks/privacy-provenance.json`](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/privacy-provenance.json).
Numerical values, timings, outcomes and numerical array bytes are unchanged;
archive owner metadata is normalized and distributed-file checksums are updated.
Original commit identifiers and measured source/library fingerprints describe
the historical runs. Sanitized source snapshots report separate content hashes.
No measurement was rerun for this cleanup.

## Archived per-kernel measurements

These measurements time individual numerical operations. Each belongs to the build
named in its result file, which may differ from the current source. Unless a
section says otherwise, the host is an AMD Ryzen 9 9950X (16 physical cores)
with Linux x86-64, Python 3.13.1, treams 0.4.5 and an optimized treams-rs
build with faer and Rayon. Both packages get the same thread budget, each runs
in its own process, and the result is compared with treams before timing at
`rtol=2e-9, atol=1e-12`. Result file names refer to
[`benchmarks/results/`](https://github.com/yaugenst/treams-rs/tree/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/results). Sections without a result
file summarize timings that were not archived.

Reproduce a workload with
`uv run --no-sync python scripts/benchmark_cluster.py --workload <name> ...`
after `just build-ext-release`.

### Axisymmetric EBCM

For r(theta)=0.3(1+0.23 cos²(theta)), k0=1.3, inner eps=3.1+0.2i,
mu=1.2+0.1i, kappa=0.07 and vacuum outside, the complete singular Q integral is
compared with treams at rtol=2e-9, atol=1e-12, using the treams surface
integral, which omits a radial area factor (`radial_area_factor=False`). treams-rs uses 96
Gauss-Legendre nodes; treams uses adaptive SciPy quadrature. Four threads, seven
samples after warmup. The reverse pass covers every sampled radius and slope,
and all complex wavenumbers and impedances; its peak RSS stays at the forward peak.
Timings include the Python call and saving the residual (the data kept for the
reverse pass), not shape and basis setup. The timings were not archived. On a
shared four-CPU host, parallel evaluation at quadrature nodes with shared radial
functions and solid harmonics brings the degree-4 forward from 5.4 to 0.74 ms and forward plus
reverse from 8.5 to 1.8 ms; the table shows the build without them.

| Degree | Modes | treams ms | Rust ms | Speedup | Rust reverse ms | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 30 | 284.26 | 3.54 | 80.3x | 1.99 | 83.87 / 42.18 |
| 4 | 48 | 727.40 | 5.90 | 123.4x | 3.00 | 84.93 / 42.45 |

At degree 6 the strict comparison fails on six of 9,216 entries, by up to
4.98e-10. All six have m=0 and odd l_out+l_in, so their integrands are odd
under theta -> pi-theta on this equatorially symmetric surface and their exact
integrals vanish (below 8e-82 at 80 digits). The largest integral of the
absolute integrand is about 7.31e5, and double-precision epsilon times it is
1.62e-10: treams-rs residuals at 48, 96 and 192 nodes lie between 0.04 and
2.43 times this floor, and treams leaves up to 6.62e-10. This is a rounding
floor in both packages, so the tolerance stays and no degree-6 speedup is
claimed. Reproduce it with
`uv run --no-sync --with mpmath python scripts/qualify_ebcm_cancellation.py`;
[the cancellation diagnostic](https://github.com/yaugenst/treams-rs/blob/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/results/ebcm-l6-cancellation.json)
holds all eight m=0 entries, their high-precision values, quadrature samples
and condition estimates. Workload: `--workload ebcm --particles 1 --lmax 4
--samples 96 --threads 4`.

### Finite sphere clusters

Chains of spheres with radii 0.15–0.25, spacing 0.8, relative permittivity
4+0.1i and vacuum wavenumber 1.3. Timings cover particle coefficients,
translations and the full dense interacting T-matrix, and treams-rs keeps
data for the pullback; they exclude imports. Peak RSS includes imports, memory
retained for reuse and the benchmark's own memory. Seven samples after warmup,
without fixed CPU placement or frequency. Raw samples: `benchmarks/results/release-*.json`.
With packed in-place LU and four threads, dimension 240 takes 2.19 ms against
89.95 ms (41.0x; 46.6 against 75.0 MiB) and dimension 960 takes 35.63 ms
against 1641.15 ms (46.1x; 100.9 against 159.0 MiB), in
`benchmarks/results/packed-lu-cluster-d{240,960}.json`. Dense output and LU
need O(D²) memory and the factorization O(D³) time. Workload:
`--particles 16 --lmax 3 --threads 4`.

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

### Batched spherical fields

Weighted singular electric fields at four positions, order 3 (120 modes), 2,048
samples and four threads, against `treams.efield(...) @ coefficients`. treams-rs
contracts the amplitudes at each sample, so it never builds the
sample-by-component-by-mode matrix; its reverse pass recomputes local wave
derivatives and keeps no dense Jacobian. Basis setup is outside the timings.

| Workload | treams ms | Rust ms | Speedup | Rust reverse ms | treams / Rust peak MiB | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Weighted field | 822.27 | 40.82 | 20.14x | — | 96.94 / 40.98 | `benchmarks/results/fields-n4-l3-p2048-t4.json` |
| Weighted field, shared spherical/cylindrical path | — | 40.64 | 20.24x | 71.11 | — / 41.28 | `shared-fields-n4-l3-p2048-t4.json` |
| Full operator, (2048, 3, 120) array | 856.08 | 45.47 | 18.83x | 75.11 | 97.68 / 64.20 | `field-operator-n4-l3-p2048-t4.json` |
| Weighted field, shared operator code | — | 41.86 | 19.60x | 70.65 | — / 41.63 | `field-geometry-n4-l3-p2048-t4.json` |
| Cylindrical, 112 modes | 502.92 | 47.57 | 10.57x | 64.24 | 94.11 / 41.64 | `cylindrical-fields*-n4-l3-p2048-t4.json` |

Post-import baselines of the first row are 64.00 MiB for treams and 40.22 MiB
for treams-rs. The cylindrical case uses orders -3 to 3, axial labels 0.2 and
-0.3 and both helicities; without sharing the radial value and its first two
derivatives between adjacent orders it takes 138.96 ms. On a shared four-core
machine (16 spheres at order 4, 4,000 samples, one thread), evaluating each
scaled radial function once per sample, position, wavenumber and order brings
weighted fields from 4.61 s to 0.26 s forward and from 12.3 s to 1.08 s forward
plus reverse, with values equal bit for bit. Workloads: `--workload field --samples 2048`,
`--workload field-operator`, `--workload cylindrical-field`.

### Periodic sphere arrays and adjoints

A square grid of spheres with spacing 0.8, the particle parameters above and
Bloch vector (0.1, 0.15). Timings cover local coefficients, the 2D Ewald
coupling and the full interacting response matrix, keeping the data for every
pullback; basis setup is excluded. The complete reverse pass (dense adjoint
solve, particle pullbacks, positions, wavenumbers, Bloch vector and lattice)
takes 146.59 and 749.60 ms for four and nine spheres at four threads, peaking at
43.4 and 60.0 MiB; this build computes the Ewald derivatives per polarization.
Raw results: `periodic-n4-l3-t1.json`, `periodic-adjoint-n4-l3-t4.json` and
`periodic-adjoint-n9-l3-t4.json`. Workload: `--workload periodic --particles 4
--lmax 3 --threads 4`.

| Spheres | lmax | Threads | treams ms | Rust ms | Speedup | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3 | 1 | 103.57 | 49.24 | 2.10x | 89.2 / 41.6 |
| 4 | 3 | 4 | 118.94 | 14.06 | 8.46x | 89.5 / 40.7 |
| 9 | 3 | 4 | 571.90 | 70.05 | 8.16x | 184.5 / 49.8 |

### Complete periodic S matrices

The `array` workload adds plane-wave incidence and radiation into ten ports
(the zeroth and four first diffraction orders, both polarizations); the full S
matrix is compared with treams. Reverse covers the channel and radiation
pullbacks, the periodic solve and all particle pullbacks. Sharing Ewald
derivatives between polarizations of equal wavenumber brings the
four-sphere, four-thread reverse from 141.24 to 71.30 ms, about 4.7 forward
evaluations, with no mode-by-parameter Jacobian. The nine-sphere case peaks at
60.5 MiB through reverse. Raw samples: `array-before-n4-l3-t4.json` and
`array-n*-l3-t*.json`. Workload: `--workload array --particles 9 --lmax 3
--threads 4`.

| Spheres | lmax | Threads | treams ms | Rust ms | Speedup | Rust reverse ms | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3 | 1 | 105.41 | 50.35 | 2.09x | 254.38 | 90.6 / 43.3 |
| 4 | 3 | 4 | 117.41 | 15.27 | 7.69x | 71.30 | 90.8 / 42.8 |
| 9 | 3 | 4 | 570.15 | 73.00 | 7.81x | 351.90 | 187.9 / 52.4 |

### Multipole rotations

Two spherical positions and all degrees through 8 (320 modes), four threads: the
complete rotation, including the angular blocks, takes 0.510 ms against
5.597 ms in treams (10.98x). The reverse pass takes 0.753 ms using the kept
small-d blocks; forward peak RSS is 42.27 against 67.03 MiB. Raw samples:
`rotation-n2-l8-t4.json`. Workload: `--workload rotation --particles 2 --lmax 8
--threads 4`.

### Cylindrical-to-spherical conversion

One common position, spherical lmax=12 (336 modes), 32 axial wavenumbers between
-0.7 and 0.7 and cylindrical mmax=12 (1,600 modes), four threads, seven samples.
The full 336-by-1,600 matrix takes 3.34 ms against 23.13 ms in treams
(**6.92x**); the pullback for positions and complex wavenumbers takes 5.80 ms.
Forward peak RSS is 73.9 MiB for treams and 55.9 MiB for treams-rs. treams-rs
skips the azimuthal orders that vanish at coincident transverse positions and
keeps adjacent orders for the position derivatives; without the skip, the
forward takes 20.48 ms and the reverse 23.63 ms. This covers coincident positions
only. The timings were not archived.

### Plane fields and direct NumPy buffer transfer

128 plane modes (64 transverse vectors, both polarizations), 4,096 Cartesian
samples with propagating and evanescent waves at k0=1.3, four threads, seven
samples. Timings include input conversion, output transfer and saving data for
the pullback; no output Jacobian is kept. Contracting polarization cotangents (the
gradients with respect to the outputs) over all samples before differentiating
each mode brings the weighted reverse from 6.10 to 3.58 ms. Handing the Rust
buffer to NumPy without a copy brings the full operator from 13.32 ms and
89.4 MiB to the row below. The reverse pass accepts any cotangent memory layout,
using one copy to make the data contiguous. For multipole field operators (four
positions, lmax=3, 2,048 points) the same transfer gives 823.02 / 42.83 ms (19.2x), 75.81 ms
reverse and 97.1 / 53.5 MiB forward peak RSS, against 64.2 MiB for treams-rs
with a copy. The timings were not archived.

| Output | treams forward | Rust forward | Speedup | Rust reverse | Forward peak RSS, treams / Rust |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weighted plane field | 139.37 ms | 1.95 ms | 71.5x | 3.58 ms | 88.3 / 42.1 MiB |
| Full plane field operator | 137.13 ms | 2.70 ms | 50.7x | 9.76 ms | 88.1 / 65.7 MiB |

### Plane-to-spherical illumination

Two spherical positions, lmax=8 (320 modes), and 64 transverse vectors with both
polarizations (128 plane modes) at k0=1.3, including evanescent directions; four
threads, seven samples. treams takes 14.07 ms and treams-rs 0.598 ms
(**23.6x**); the reverse pass for all positions and complex wavevectors takes
1.23 ms. Forward peak RSS is 67.5 / 40.2 MiB. treams-rs shares the normalized
direction across multipoles, runs incident modes in parallel and hands the
output to NumPy without keeping it in the residual. Reusing the phase of each
position across its multipoles gives 0.494 ms against 14.143 ms (**28.6x**) with
1.163 ms reverse. The timings were not archived.

### Complete cylindrical arrays

One-dimensional arrays along x with kz=0.2, k0=1.3, radii 0.15–0.25,
permittivity 4+0.1j, spacing 0.8 and period 0.8N; four threads, seven samples.
Each evaluation covers particle T-matrices, Ewald coupling, the periodic solve
and all four plane-wave blocks for ten ports; reverse takes arbitrary complex
cotangents. Both packages use the Ewald split eta=0.7: at its automatic split,
treams differs from the converged nine-cylinder answer by about 8.4e-8 in the
S matrix, while treams-rs stays stable as eta changes, and a regression test
checks it against the converged reference. Sharing Ewald evaluations between
polarizations of equal wavenumber brings the four-cylinder case at the
automatic split from 18.42 to 9.38 ms forward and from 45.84 to 22.99 ms
reverse. The timings were not archived. Workload: `--workload cylindrical-array
--particles 9 --lmax 3 --threads 4 --repeats 7`.

| Cylinders | mmax | Modes | treams ms | Rust ms | Speedup | Rust reverse ms | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 5 | 88 | 2101.29 | 8.26 | 254.4x | 21.13 | 69.1 / 41.6 |
| 9 | 3 | 126 | 2982.63 | 16.64 | 179.2x | 45.15 | 70.9 / 43.8 |

### Cylindrical plane illumination and phase reuse

Four positions, mmax=12 (200 cylindrical modes), kz=0.2, and 128 transverse
vectors with both polarizations (256 plane modes); four threads, seven samples.
treams-rs takes 0.220 ms against 6.376 ms (**28.9x**); the reverse pass for
positions and transverse wavevectors takes 0.446 ms, with axial labels fixed.
Forward peak RSS is 39.9 against 67.7 MiB. Without reusing the phase of each position
across its multipoles, the forward takes 0.377 ms and the reverse 0.537 ms. The
timings were not archived. Workload: `--workload cylindrical-plane-expansion
--particles 4 --lmax 12 --samples 128 --threads 4 --repeats 7`.

### Compact planar multilayers

Four interior layers with alternating permittivities 2.3+0.1j and 1.7+0.05j in
vacuum, thicknesses 0.1–0.4, k0=1.3, transverse qx=0.1–0.8 and qy=0.2; four
threads, seven samples. Both forward timings call the public slab constructor
and return the complete four-block dense array. Reverse timings start from the
compact `diff.layer_stack` cotangent and cover all medium wavenumbers,
impedances, transverse components and thicknesses. treams-rs solves each
channel on its own and composes each layer's reflectionless propagation in
closed form, as a diagonal phase scaling of the stack below; the forward
blocks remain equal bit for bit to the general composition. Its solve and residual
grow linearly with the channel count, while the dense output still costs
quadratic time and memory. Slab bases with partial polarization use the
general projected composition. The timings were not archived. Workload:
`--workload slab --layers 4 --channels 512 --threads 4 --repeats 7`.

| Channels | Plane modes | treams ms | Rust ms | Speedup | Compact Rust reverse ms | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 256 | 84.37 | 2.81 | 30.0x | 1.16 | 88.9 / 52.0 |
| 512 | 1024 | 2152.16 | 23.58 | 91.3x | 3.28 | 378.2 / 187.9 |

### Periodic spherical-to-cylindrical conversion

One common position, lmax=mmax=12, 336 spherical inputs and 1,600 cylindrical
outputs across 32 axial diffraction orders, wavenumber 1.3, period 200 and Bloch
component 0.2; four threads, seven samples. treams-rs takes **1.158 ms** against
**24.266 ms** (**21.0x**). Its reverse pass takes **6.114 ms** and covers both
positions, complex medium wavenumbers, per-output axial wavenumbers and the
period. Forward peak RSS is **48.5 MiB** against **73.6 MiB**, and forward plus
reverse peaks at 65.1 MiB; the residual holds geometry only. The timing covers
conversion only and was not archived. Workload: `--workload periodic-conversion
--particles 1 --lmax 12 --samples 32 --threads 4 --repeats 7`.

### Dense internal illumination

Two general dense S matrices, four threads, seven samples. Random complex
reflections scale as 0.1/sqrt(N), with identity transmission plus similar
perturbations; seed 81. All four fields are compared with treams; building the
S matrices is outside the timings. Each sample is a batch of at least 20 ms
that includes freeing the returned fields and residuals. Ordinary illumination
uses its input blocks without copies and factors the operator in one LU buffer.
Recorded illumination also copies the inputs, so the pullback survives later
changes to the Python arrays; at 1,024 modes this costs
about 332 MiB against 259 MiB for treams' forward-only call. Raw results:
`benchmarks/results/internal-{forward,adjoint}-l{128,512}-p{1,8}.json`.
`just bench-performance` reruns these eight cases and the two plane permutation
cases below into `benchmarks/results/local/geometry-recheck-*.json`. Here lmax
is only a size argument: the dense matrix has 2*lmax modes.

| Modes | RHS columns | treams ms | Rust forward ms | Speedup | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 3.54 | 0.87 | 4.07x | 82.6 / 56.6 |
| 256 | 8 | 4.19 | 1.51 | 2.79x | 83.0 / 57.5 |
| 1024 | 1 | 62.09 | 34.04 | 1.82x | 259.2 / 234.0 |
| 1024 | 8 | 63.52 | 30.49 | 2.08x | 259.4 / 235.1 |

| Modes | RHS columns | treams forward ms | Rust recorded forward ms | Speedup | Rust reverse ms | Rust forward / through-reverse peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 3.94 | 0.93 | 4.23x | 0.30 | 65.6 / 85.6 |
| 256 | 8 | 3.82 | 1.11 | 3.43x | 0.50 | 65.2 / 85.2 |
| 1024 | 1 | 61.40 | 40.97 | 1.50x | 29.96 | 332.1 / 420.6 |
| 1024 | 8 | 66.26 | 41.26 | 1.61x | 33.34 | 332.4 / 423.8 |

### Compact plane permutations

Cyclic Cartesian-axis transforms return both output polarizations for each
input mode, shape (2, N), against `treams.pw.permute_xyz` broadcast to the same
shape; this is not a dense N-by-N operator. Four threads, seven batched samples
including output and residual cleanup. The residual keeps only the vectors and
discrete labels. Raw results: `benchmarks/results/plane-permutation-l{64,512}.json`.
The full plane-field operator (128 modes, 4,096 samples) with the same shared
polarization arithmetic takes 1.81 ms against 138.43 ms (76.55x), with 9.77 ms
reverse and 67.5 against 89.5 MiB forward peak RSS
(`plane-transform-check-field-n128.json`).

| Modes | treams us | Rust us | Speedup | Rust reverse us | treams / Rust forward peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 7.52 | 5.41 | 1.39x | 16.78 | 65.0 / 42.2 |
| 1024 | 170.23 | 34.68 | 4.91x | 77.78 | 65.2 / 43.4 |

### Plane translation phases

Compact exp(i k.r) tables at 4,096 displacements with propagating and
evanescent wavevectors, against the treams `pw.translate` ufunc; both packages
receive precomputed vectors. The context keeps only positions and wavevectors,
and the table goes to NumPy without a copy. Reading contiguous cotangents without
copying brings the large forward-plus-reverse peak from 235.1 to 172.3 MiB and
reverse from 79.33 to 64.42 ms; noncontiguous inputs are copied. This covers only
the phase calculation. The timings were not archived. Workload: `--workload plane-phases
--particles 16 --lmax 32 --samples 4096 --threads 4` (plane modes =
2·particles·lmax; the smaller case uses 8 and 8).

| Plane modes | treams ms | Rust ms | Speedup | Rust reverse ms | treams / Rust forward peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 7.81 | 1.81 | 4.31x | 7.95 | 90.5 / 50.2 |
| 1024 | 72.69 | 22.53 | 3.23x | 64.42 | 258.9 / 108.0 |

### Oriented chirality forms

Compact signed-helicity up/down/cross coefficients averaged from -0.2 to 0.7
along x, with propagating and evanescent waves. The reference builds Cartesian
polarizations with `treams.special.vpw_A` and contracts their inner products and
exponential averages; it is the equivalent Cartesian calculation, not treams'
xy-only chirality method. Outputs have shape (3, N). Reverse covers both real
transverse components, complex normal components and the interval ends; the
residual keeps only the input geometry. Raw results:
`benchmarks/results/oriented-chirality-l{64,512}.json`. Workload:
`--workload oriented-chirality --particles 1 --lmax 512 --threads 4`.

| Modes | Reference us | Rust us | Speedup | Rust reverse us | Reference / Rust forward peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 54.72 | 24.45 | 2.24x | 114.67 | 65.3 / 42.8 |
| 1024 | 619.42 | 65.88 | 9.40x | 280.72 | 65.3 / 43.4 |

### Heterogeneous particle clusters and block adjoints

Spheres with alternating cutoffs 3 and 4, radii 0.15–0.25, permittivity
4+0.1j, vacuum wavenumber 1.3 and spacing 0.8; local particle construction is
outside the timings, and both packages return the complete interacting matrix.
The Rust calculation takes the local arrays separately and keeps their block
structure. Its reverse pass returns only the local diagonal-block gradients plus
every position and both embedding-wavenumber cotangents; the 624-mode process
peaks at 107.1 MiB through reverse. For homogeneous clusters with lmax=3, 240
modes take 2.48 ms against 90.32 ms (36.42x), 2.34 ms reverse and 46.1/50.9 MiB;
960 modes take 37.50 ms against 1678.25 ms (44.75x), 63.31 ms reverse and
102.6/177.5 MiB. Raw results: `particle-cluster{-public,}-n{4,16}-l3.json` and
`block-adjoint-cluster-n{8,32}-l3.json`. Workloads: `--workload particle-cluster`
or `--workload particle-cluster-public` with `--particles 16 --lmax 3 --threads 4`.

| Particles | Modes | Path | treams ms | Rust ms | Speedup | Rust reverse ms | treams / Rust forward peak MiB |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 156 | Native block path | 35.43 | 3.24 | 10.95x | 0.91 | 71.3 / 44.0 |
| 16 | 624 | Native block path | 734.82 | 17.51 | 41.96x | 22.74 | 109.2 / 68.5 |
| 4 | 156 | Public cluster + solve | 35.67 | 3.97 | 8.99x | — | 71.7 / 45.1 |
| 16 | 624 | Public cluster + solve | 737.35 | 23.76 | 31.03x | — | 107.9 / 86.3 |

### Broadcast Bessel functions

Cylindrical Hankel H1 of order 3 at 128 or 4,096 complex arguments with real
part 0.6–8.0 and imaginary part 0.2; values and first derivatives are compared
with `treams.special`. Four threads, seven batched samples including cleanup.
Parallel evaluation starts at 64 values. Ordinary forward calls use their inputs
without copying and save no pullback data; a 128-value run without these changes
fails the runtime check (65.59 against 61.08 us). treams-rs peaks at 39.8–41.0 MiB
against 64.9–65.4 MiB. Small differences between recorded and ordinary forward
times come from run-to-run scheduling. The ufunc table in the next section
measures a later build of the same functions. Raw results:
`benchmarks/results/bessel{,-derivative}{,-forward}-n{128,4096}.json`.
Workload: `--workload bessel-forward --samples 4096 --particles 1 --lmax 3
--threads 4`; the other workload names match the file names.

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

### Heterogeneous cylindrical clusters

Alternating azimuthal cutoffs 3 and 4, two fixed axial channels (0.2, 0.4) and
four threads; local cylinder construction is outside the timings, and the
public path includes cluster construction and the interaction solve. The
512-mode native run peaks at 83.0 MiB through reverse. Parallel cylindrical
translation contractions bring its reverse from 59.38 to 25.60 ms and the
128-mode reverse from 3.01 to 1.30 ms. Raw data:
`benchmarks/results/cylindrical-particle-cluster{,-public}-n{4,16}-l3.json`;
`just bench-performance` reruns all four into `benchmarks/results/local/`.

| Particles / modes | Path | Upstream ms | Rust ms | Speedup | Reverse ms | Upstream / Rust forward RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 4 / 128 | Native blocks | 4.161 | 1.373 | 3.03x | 1.298 | 70.5 / 42.5 |
| 16 / 512 | Native blocks | 67.254 | 20.640 | 3.26x | 25.596 | 96.1 / 59.3 |
| 4 / 128 | Public API | 4.255 | 1.336 | 3.18x | — | 70.2 / 43.9 |
| 16 / 512 | Public API | 67.142 | 23.677 | 2.84x | — | 96.8 / 70.2 |

### Scalar overhead and NumPy ufuncs

Scalar Python arguments at z=1.3+0.2i take a direct native path without
temporary arrays; arrays (Re z = 0.6..8, Im z = 0.2) use NumPy ufunc loops, which
keep NumPy's output allocation, masks, broadcasting and overlap handling. Order
is 3. The complex Hankel derivative gets its adjacent orders from one sequence
evaluation, using the [Bessel derivative recurrence](https://dlmf.nist.gov/10.6.ii).
All twelve cases pass their correctness, runtime and forward-RSS checks;
treams-rs peaks at 39.2–41.3 MiB against 64.2–65.2 MiB. The recorded scalar
value is faster by only 1.06x, so recheck it on the target host. Raw files:
`benchmarks/results/ufunc-bessel{,-derivative}{,-forward}-n{1,128,4096}.json`;
`just bench-performance` reruns all twelve into `benchmarks/results/local/`.

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

### Public angular functions

Integer-degree Legendre, pi and tau functions at degree 6 and order 2, with a
scalar argument 0.3+0.1j and arrays over [-0.8,0.8]+0.1j; four threads. They
use NumPy loops, a direct scalar path and the same factored recurrences forward
and in reverse; tau shares one sine-power factor between adjacent Legendre
orders. Recording includes saving the pullback data and freeing the result;
each reverse pass uses a fresh record. All 50 runtime checks of this
`just bench-performance` run pass, and the ordinary forward and recorded
special-function and particle paths pass their peak-RSS checks. treams-rs peaks
at 40.3–41.7 MiB against 64.1–65.2 MiB. The 18 `angular-*.json` files in
[raw results](https://github.com/yaugenst/treams-rs/tree/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/results) record every sample, native-extension
hash, Python and NumPy version, thread setting and reverse peak RSS. Units are
microseconds.

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

### Wigner elements, Euler pullbacks and public Ewald integrals

Wigner degree 6, row 1, column -2, outer Euler angles 0.2 and -0.1; the polar
argument is 0.7+0.1j for scalars and spans [0.3,1.3]+0.1j for arrays. The 3j
case uses degrees (6,6,6) and orders (1,-2,1); the incomplete gamma degree is
1.5; the Kambe integral has order -2 and eta=0.7+0.1j. Individual Wigner
elements use the [Jacobi recurrence](https://dlmf.nist.gov/18.9.E2) without a
full angular matrix; the ufuncs run large strided arrays in parallel and keep
masked and in-place behavior. Wigner residuals keep only labels and Euler
arguments, and the scalar 3j path skips NumPy dispatch. All 68 runtime checks
of this run pass (50 angular and baseline workloads and 18 Wigner and Ewald
cases). The angular table above records the run of commit `4f41708`; this
68-case run refreshed the raw JSON files. treams-rs peaks at 39.9–41.5 MiB
against 64.2–65.8 MiB. Native hashes, samples and reverse measurements are in
the [raw results](https://github.com/yaugenst/treams-rs/tree/e464d5fea94749223e22fe80c2ba5a53b7426950/benchmarks/results). Units are microseconds.

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

### Cylindrical axial field derivatives

Four particles at order 3, four threads. The ordinary reverse and the optional
per-mode axial reverse both pass the accuracy, forward-runtime and forward-RSS
checks. The forward is the same for both, and recording adds no axial
derivative table; the axial reverse computes one more derivative per mode in
about the same time. The build at `ce0bde4` measured 3.172/47.935 ms forward and
4.197/62.601 ms reverse at 128/2,048 samples; successive short runs vary by a
few percent, so these differences show no change. These four cases ran
separately from the 68-case run above. Times in milliseconds:

| Reverse requested | Samples | Upstream forward | Rust forward | Speedup | Rust reverse | Rust / upstream RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing | 128 | 32.071 | 3.120 | 10.28x | 4.045 | 44.8 / 67.7 |
| Existing | 2048 | 503.039 | 49.549 | 10.15x | 66.882 | 44.9 / 96.4 |
| All + axial | 128 | 33.642 | 3.094 | 10.87x | 4.180 | 43.8 / 67.7 |
| All + axial | 2048 | 504.578 | 48.038 | 10.50x | 65.684 | 44.3 / 96.3 |

### Shared axial derivatives of finite and periodic cylindrical expansions

Four or sixteen particles, two axial groups, order 3 and four threads; every
case passes its accuracy, forward-runtime and peak-RSS checks. Periodic cases
use the Ewald split eta=0.7 in both packages: at period 12.8 the automatic
treams split differs from the converged result by up to 0.007466 in this
matrix, while explicit eta=0.5/0.8/1.0 and the treams-rs automatic split agree
with treams-rs to about 1.5e-11. A scalar regression test covers periods 7.2
and 12.8. The `-axial` cases run the same forward and also request the grouped
axial gradient in reverse, which reuses the Ewald derivatives without an axial
Jacobian. These eight cases ran separately from the other runs. The complete
four-cylinder array without and with the grouped axial derivative takes 4.729
and 4.511 ms forward, 12.138 and 11.760 ms reverse, and peaks at 42.5 and
43.9 MiB, in one short run on one host (`cylindrical-array-{before,after}-axial.json`).

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

### Coordinate transformations

All eight point and eight vector-frame conversions use NumPy gufuncs with
analytic pullbacks, support component strides, broadcasting and output buffers,
and apply factored rotations without temporary matrices. The run passes 128
performance checks, 48 of them coordinate comparisons at 1, 128 and 65,536
points; each requires a forward speedup ≥ 1 and no higher forward peak RSS.
The forward-only helpers of both packages return the result directly and free
it inside the timing; recorded operations also return data for the pullback, and the
JSON field `forward_records_adjoint` says which. Sub-microsecond scalar
differences are sensitive to machine noise. Raw samples, RSS and binary
identities: `coordinate-*-n*.json`.

| Transform | 1 point | 128 points | 65,536 points |
| --- | ---: | ---: | ---: |
| car2sph | 1.08x | 1.21x | 2.31x |
| sph2car | 1.04x | 1.33x | 2.87x |
| car2pol | 1.08x | 1.51x | 4.23x |
| vsph2car | 1.01x | 1.31x | 2.96x |
| vpol2car | 1.07x | 1.43x | 5.04x |

### Local wave functions

The run passes all 191 runtime checks and all 187 applicable forward-RSS checks;
the four recorded internal-illumination cases keep owned input copies and are
not compared with the forward-only treams memory. It used
`taskset -c 8-11 just bench-performance`, giving both packages the same four
physical cores and four BLAS/Rayon threads; diagnostic runs on busier cores are
kept next to the final results. When both isolated medians are below 1 ms, the
two packages also alternate in one process over 14 paired samples of at least
20 ms each, and the speedup is the median paired ratio. The smallest speedup is
1.013x (scalar plane M wave). The wave kernels share normalized angular
recurrences, fetch adjacent cylindrical Bessel values together, reuse plane
polarizations for constant directions and normalize with an algebraic complex
square root, with scaled fallbacks for extreme magnitudes. Recorded
illumination copies its eight input blocks as independent tasks from 128 modes
when Rayon has more than one thread (on one thread: serially below 512 modes,
on the worker from there) and finishes the copies before solving; with two
tasks from 512 modes, the 510-mode (order 255) case copied 33 MB serially at
0.85x of the treams forward-only call (`snapshot-*`, `serial-*` and
`parallel-copy-*` files). Final results: `wave-<function>-n<size>.json` and
`wave-adjoint-final-<function>-n<size>.json`, each with the native binary hash.

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

### Coefficient namespaces and integral adjoints

The run passes all 257 runtime checks and all 253 applicable forward-RSS checks
on the same four-core Linux setup, with the four recorded internal-illumination
cases exempt from the memory check. The smallest speedup is 1.006x for scalar
`cw.rotate`. Cylindrical coefficient arrays switch to Rayon at 64 elements,
scalar Python numbers skip NumPy dispatch, and plane phases evaluate one
exponential and one sine/cosine pair. Paired timing calibrates each package's
batch to at least 20 ms on its own, so the slower package is not oversampled;
91 cases kept the batch sizes of the preceding calibration at the same native
SHA-256 and 166 used independent sizes, as the manifest records. Failed
baselines and targeted rechecks stay next to the passing results. The result
manifest and native SHA-256 are in `benchmarks/coefficient-qualification.json`.

| Forward function | 1 sample | 128 samples | 4,096 samples |
| --- | ---: | ---: | ---: |
| tl_vcw | 1.822x | 1.299x | 2.569x |
| tl_vcw_r | 2.427x | 1.368x | 4.606x |
| cw.translate | 2.195x | 1.510x | 4.272x |
| pw.permute_xyz | 3.592x | 1.367x | 10.028x |
| pw.translate | 2.223x | 1.226x | 1.236x |

### Lattice geometry and material branches

The run passes 42 geometry cases and 257 regression cases: 299 runtime checks
and 295 applicable forward-RSS checks, on CPUs 8–11 with four BLAS/Rayon
threads. The smallest margin is 1.005x for scalar `cw.rotate`, and the largest
applicable RSS ratio is 0.912. Circular diffraction enumeration runs 7.915x,
64.034x and 256.860x faster at radius 1, 4 and 16; tests check complete
enumeration in skew cells separately. The geometry results are
`geometry-<operation>-n<size>.json`, and the 257 rechecks of the coefficient
cases use the `geometry-recheck-` prefix; `benchmarks/geometry-qualification.json`
records all 299 paths and their native binary hash. Mac diagnostic results keep
their own file names and platform metadata.

| Forward operation | 1 sample | 128 samples | 4,096 samples |
| --- | ---: | ---: | ---: |
| volume2 | 1.033x | 2.978x | 44.687x |
| volume3 | 1.047x | 2.647x | 25.899x |
| reciprocal2 | 1.026x | 6.004x | 8.677x |
| reciprocal3 | 1.034x | 3.121x | 3.577x |
| refractive_index | 3.213x | 3.466x | 1.955x |
| wave_vec_z | 1.904x | 3.197x | 1.607x |

## Lattice sums at explicit small splits (pre-release timings)

These timings compare `5dadf1d`, a pre-release commit without the cancellation
checks of explicit Ewald splits below every automatic split, with the later
pre-release commit that adds them
([numerical limits](../validation/numerical-limits.md#explicit-splits-below-the-automatic-one)).
They were not rerun.

Timed as a whole against `5dadf1d` (release, one thread, unit cells), 1D
spherical sums on the axis of degree 0 to 8 at k = 0.42, eta = 0.14 and 0.2 and
Bloch vectors of 0, 0.1 and 0.25 take 0.07 to 0.52 ms each, 0.8 to 2.6 times as
long, and up to 9 times at degrees 6 to 8 at the shift 0.1, where `5dadf1d`
returned some in 0.02 ms and up to 7e-8 of max(|S|, 1) off; at a zero Bloch
vector and the shift 0.5, odd degrees vanish exactly at once. 2D spherical
sums of degree up to 6 at k = 1.3 and eta = 0.25 take 0.8 times as long in all.
3D sums of degree up to 3 at k = 1.3 and eta = 0.25 and 0.2, and a quarter of the
2D sums at 0.2, failed with "did not converge" at `5dadf1d`; with the checks,
the 3D sums converge in about 0.1 and 0.24 s each on average.

Sums and Ewald parts that fail to converge sum to the grown shell limit first
(release): 3D sums fail within about 2.4 s, the 3D real-space part of degree 3 at
k = 0.7 + 0.1i after 28 s at eta = 0.03, 2D sums and parts within 2.5 s.
