# CPU improvements after 0.1.0

The October 4 CPU comparison measures batched cluster solves, spherical fields,
periodic arrays and internal slab illumination against separately rebuilt
treams-rs 0.1.0 sources. On one Ryzen 9 9950X host, the largest measured gain in
the four-thread algorithm comparison is
**12.34× for eight requested illuminations of 512 dipole spheres**, and **11.04×
for their complete physical gradient**. This is a comparison between treams-rs
builds, not against Python treams.

The [qualification record](https://github.com/yaugenst/treams-rs/blob/1f3deb62d763a664f7e85b836fa2501c18a51442/benchmarks/cpu-speedups-20261004.json) and
[raw evidence archive](https://github.com/yaugenst/treams-rs/blob/1f3deb62d763a664f7e85b836fa2501c18a51442/benchmarks/results/cpu-speedups-20261004.tar.gz)
retain the measurements, numerical checks, source identities and exceptions.
These measurements precede the separate work on scheduling small dense products;
the [scheduling qualification below](#small-product-scheduling) measures that
later change separately.

## Measured workloads

Four threads, complex128/f64, identical inputs and accuracy settings. A gradient
time includes its forward calculation and analytic pullback. Speedup is the
median of the paired baseline/candidate ratios; it can differ slightly from the
ratio of the two displayed time medians.

| Workload | 0.1.0 | Measured candidate | Speedup |
| --- | ---: | ---: | ---: |
| 512 dipole spheres, one illumination | 157.63 ms | 59.71 ms | 2.64× |
| 512 dipole spheres, eight illuminations | 1235.19 ms | 100.25 ms | 12.34× |
| Same eight illuminations, complete gradient | 2567.39 ms | 232.64 ms | 11.04× |
| 32 stronger-scattering spheres, degree 3, eight illuminations | 212.09 ms | 62.89 ms | 3.37× |
| Prepared 32-dipole field, 4096 points | 12.19 ms | 5.22 ms | 2.34× |
| Same prepared field, forward and field pullback | 48.44 ms | 26.52 ms | 1.83× |
| Degree-12 sphere near field, 4096 points | 13.45 ms | 6.32 ms | 2.13× |
| Same near field, forward and pullback | 67.82 ms | 39.42 ms | 1.72× |
| Four-sphere periodic array, degree 6, size parameter near one | 57.74 ms | 26.72 ms | 2.16× |
| Same periodic array, complete gradient | 238.11 ms | 111.58 ms | 2.13× |
| Internal illumination, 1024 slab modes | 31.65 ms | 15.53 ms | 2.03× |
| Same internal illumination, complete gradient | 63.30 ms | 47.87 ms | 1.33× |

Cluster times include local coefficients, operator construction and the requested
solve, with incident-wave construction outside timing. Field maps start with
prepared scattering coefficients: the dipole map does not include the preceding
cluster solve or its derivative. Slab cases use prepared matrices. The archive's
`compare_builds-final.py` snapshot specifies every case.

The changes share translation work across independently converging right-hand
sides, reuse spherical angular and radial calculations, share Ewald terms, and
apply thin internal-illumination operators without first forming their dense
product. They retain analytic derivatives and the original accuracy gates.

These methods have limits. Dipole order is a model, not a general multipole
convergence claim. The recorded degree-12 near field differs from degree 14 by
`3.18e-9` in relative norm; the degree-6 periodic case differs from degree 8 by
`9.68e-9`. Those are cutoff checks for the stated inputs. In a separate 32-sphere
strong-scattering diagnostic, selected dense LU remains about five times faster
than the improved iterative solver for a fresh solve. That diagnostic tightens
the iterative residual to `1e-12` after a strict componentwise comparison failed
at `1e-10`; both the failure and the successful diagnostic are archived.

## Memory and retained failures

The speedups use additional temporary storage in some cases. For the 512-dipole,
eight-illumination gradient, whole-process peak RSS changes from **57.97 to
60.69 MiB**; growth above the post-setup high-water mark changes from **4.75 to
9.30 MiB**. Eight independent Krylov spaces account for additional storage.

Four periodic degree-6 cases exceed the unchanged 5% call-growth memory limit:
both forward and gradient cases at the two measured size parameters. Gradient
growth increases by **4.68–6.34 MiB**, while whole-process peak increases by
approximately **2–4 MiB**. The size-parameter-near-one gradient changes from
**86.57 to 88.57 MiB** peak RSS. These memory failures are retained; faster
execution is not a claim of universally lower memory. Source-based scratch
bounds are recorded separately from measured process peaks.

All **36 targeted four-thread timing comparisons pass**, with maximum output
difference `1.30e-14` under the `1e-12` comparison gate. The broader 92-case screen
has four original timing failures. Separate same-build controls and longer
seven-round rechecks pass all four without changing the limits. The small
64-mode stack still shows about 60 microseconds of overhead in that recheck,
within the existing 10% limit for submillisecond calls. The original failures
remain in the archive; the rechecks do not establish zero regressions.

The one- and sixteen-thread screens pass all 17 cases. At 32 threads, all 16
optimization targets pass, while the additional small-stack timing fails its
initial screen. Its same-build control and longer paired recheck pass. The
pre-scheduling implementation is inefficient for that small workload at large
thread budgets, and this record makes no claim about the subsequent scheduling
change.

## Measurement and provenance

The host was a Ryzen 9 9950X with 16 physical cores, Linux, Python 3.13.1,
NumPy 2.5.3 and Rust 1.94.0. Both extensions used the repository release profile.
The primary comparison pinned four workers to CPUs 8–11, with seven paired
rounds and three samples of at least 60 ms each. A fresh process runs each tree
and case, sequentially in alternating order, so inactive numerical thread pools
cannot compete. Imports and input setup are excluded. Linux memory measurements
use the worker's `VmHWM`, with three fresh-process repetitions per tree and case.

The baseline source is `8bbcb87aaeaccd9293fc8c09eecb8727c37c4970`; its numerical
and Python sources match tag `v0.1.0`. The measured candidate was an uncommitted
patch on that source, identified by patch SHA-256
`72039ec484ab21902cc034cb12835841eed69360e9de60f385075d1d432f0950` and native-library
SHA-256 `68919ce62085ca79a34719aa03a1171c2da530cab33da4559c3d9b5a1148e4f4`.
The archive includes that patch and the exact comparison-harness snapshot.
Later commits or releases are not the measured revision.

Local `just ci` passed 262 Rust tests and 6625 Python tests, with two ignored Rust
tests and 22 skipped Python tests. Additional physical, reference, gradient and
cutoff diagnostics are included. The full historical 1300-case benchmark grid
was not replayed; no macOS, Windows or GPU performance is established here.

Earlier exploratory runs exposed two harness errors: inherited `ru_maxrss`
invalidated their memory results, and concurrent idle BLAS pools distorted some
timings. The final measurements use corrected isolation, actual minimum-duration
batches and worker-local memory accounting. The archive identifies the rejected
measurements and retains the diagnostic evidence. Privacy-only distribution
changes are recorded in [privacy provenance](https://github.com/yaugenst/treams-rs/blob/1f3deb62d763a664f7e85b836fa2501c18a51442/benchmarks/privacy-provenance.json);
numerical values, timing samples and outcomes are unchanged.

## Reproduce

Extract the archive from the repository root. Reconstruct the measured candidate
from its stated base and `stage6-final.patch`, and use `compare_builds-final.py`
as the `scripts/compare_builds.py` snapshot. Build baseline and candidate package
trees separately with `just build-ext-release`; a rebuilt binary receives its
own identity. On an otherwise idle Linux host, run:

```sh
taskset -c 8-11 uv run --no-sync python scripts/compare_builds.py \
  --baseline /path/to/built-baseline --candidate /path/to/built-candidate \
  --match '^(iterative-|mie-nearfield|prepared-dipole|periodic-array|slab-internal)' \
  --threads 4 --rounds 7 --samples 3 --min-sample 0.06 --memory \
  --output benchmarks/results/local/cpu-comparison.json
```

The archive manifest gives each original and distributed file's digest. Its
README distinguishes replay inputs, final results, exploratory diagnostics and
superseded measurements. New runs belong under `benchmarks/results/local/`.

## Small-product scheduling

A separate follow-up limits the workers used for a dense matrix product to
`min(thread_budget, floor(m*n*k / 65536))`, retaining the existing serial cutoff
and vector-product rules. Small products previously requested the entire thread
budget, so scheduling overhead could dominate at 32 threads. A 64-by-64 product
now requests four workers even when the application allows 32.

The [scheduling qualification](https://github.com/yaugenst/treams-rs/blob/1f3deb62d763a664f7e85b836fa2501c18a51442/benchmarks/cpu-scheduling-20261004.json) and
[separate raw archive](https://github.com/yaugenst/treams-rs/blob/1f3deb62d763a664f7e85b836fa2501c18a51442/benchmarks/results/cpu-scheduling-20261004.tar.gz)
record the selected build against the original 0.1.0 baseline. At 32 threads,
all 12 cases pass the unchanged timing and numerical gates across seven paired
rounds, using three samples of at least 80 ms per round:

| Workload, 32-thread budget | 0.1.0 | Selected candidate | Paired speedup |
| --- | ---: | ---: | ---: |
| 64-mode stack power | 17.376 ms | 1.143 ms | 15.35× |
| Four-sphere dense cluster solve | 6.854 ms | 0.700 ms | 10.00× |
| 16-sphere dense cluster solve | 36.090 ms | 14.082 ms | 2.59× |

These gains remove excessive scheduling overhead at a large thread budget;
they do not imply that 32 threads is the fastest setting. The selected
64-mode stack takes 0.765 ms with four threads. In seven-round comparisons at
four threads, the selected stack and 16-sphere solve have paired time ratios
of 1.017 and 1.020 against 0.1.0, both within their existing limits.

One intermediate regression remains: the selected 16-sphere solve is **29.8%
slower than the preceding stage6 build** at four threads, with medians of
5.623 and 7.159 ms. Against the released-source baseline, its medians are
7.248 and 7.314 ms. A temporary exception preserving the old policy at budgets
of four or fewer did not fix the intermediate regression; that rejected variant
was 32.9% slower than stage6. The archive retains all five failed intermediate
timing gates, the null control, rechecks and rejected implementation. The cause
of the intermediate advantage is unresolved.

A final three-round four-thread spot check retains the main algorithm gains:
512 dipoles with eight illuminations take 1.252 s versus 0.103 s for the selected
build (12.21×), and the complete gradient takes 2.563 s versus 0.235 s (10.90×).
The selected build includes the preceding CPU changes and translation-plan
allocation pruning, so these comparisons measure the combined build. They do
not assign every gain to scheduling.

Six selected-build forward/gradient workflows produce identical output bits
across native budgets of 1, 4, 16 and 32 with BLAS held to one thread. Final
comparative timings cover four and 32 threads; the one- and sixteen-thread
speed screens used an earlier implementation of the same cap. No additional
memory measurement was made for scheduling, and the preceding periodic-memory
exceptions remain. These results still cover one Linux host and a selected
workload set, not the full historical grid.

The selected native-library SHA-256 is
`41748dd504e744b58c14842390c6aa5015591328c3bfe7f3de5e5f4998a3c6dd`.
Its source inventory is **reconstructed from the immediately following rejected
variant**, restoring the selected scheduling implementation; it was not captured
at the selected build's creation. The archive's `selected-reconstructed.patch`
applies to the same `8bbcb87` base and reproduces all 180 recorded native,
Python and Cargo-file hashes. Two supporting build inputs have separately
recorded publication-time hashes. This does not reconstruct the entire as-run
documentation and test tree.

For replay, apply that patch, build separate baseline and candidate package
trees, and restore the archived `compare_builds-as-run.py` as
`scripts/compare_builds.py`, with `_harness.py` beside it. The harness was
simplified after timing, so use the snapshot
whose SHA-256 is
`925aeb12247fc09ae0d7d14e3b9fee26f601b4273e10ce2b5696c321b7c38361`.
The 32-thread qualification uses:

```sh
taskset -c 0-31 uv run --no-sync python scripts/compare_builds.py \
  --baseline /path/to/built-baseline --candidate /path/to/built-candidate \
  --match '^(cluster-solve-n(2|4|16)|stack-power-h(2|16|64|256)|periodic-to-smatrix-l3|periodic-array-n4-l6-size1-(forward|gradient)|slab-internal-h1024-(forward|gradient))$' \
  --threads 32 --rounds 7 --samples 3 --min-sample 0.08 \
  --output benchmarks/results/local/cpu-scheduling-replay.json
```
