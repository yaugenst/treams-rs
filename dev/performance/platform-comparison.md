# Mac and Linux benchmark comparison

This comparison measured treams 0.4.5 and treams-rs on an Apple M3 and an AMD
Ryzen 9 9950X. The Linux results belong to the treams-rs build at
source commit `7421ca9`, the revision of the original run
([`history-provenance.json`](https://github.com/yaugenst/treams-rs/blob/8b442f18ec7a54787ffbc8fe8db0be4d137d89ef/benchmarks/history-provenance.json) maps it
to this repository's history). The median speedups over the 527-case reference
grid are 5.17× on the M3 and 5.04× on the Ryzen. The
Linux-only measurements of 2026-09-19 and 2026-09-20 are summarized in
[Linux core validation](https://github.com/yaugenst/treams-rs/blob/8b442f18ec7a54787ffbc8fe8db0be4d137d89ef/benchmarks/linux-core-qualification.md); macOS
results come only from this comparison and the
[macOS dispatch grid](https://github.com/yaugenst/treams-rs/blob/8b442f18ec7a54787ffbc8fe8db0be4d137d89ef/benchmarks/mac-qualification.json).

The full report (figures, a combined PDF, a searchable case table,
measurement files and source records) is stored outside the
repository. It keeps every planned case, including errors and resource limits.
The repository keeps the per-case results of the Linux reference grid under
`benchmarks/results/final/` and those of the 30-case macOS dispatch grid under
`benchmarks/results/mac-final/`.

## Experimental scope

The comparison reruns every case of the
[527-case reference grid](https://github.com/yaugenst/treams-rs/blob/8b442f18ec7a54787ffbc8fe8db0be4d137d89ef/benchmarks/complete-qualification.json), then the
209 cases of the [scaling and gradient plan](https://github.com/yaugenst/treams-rs/blob/8b442f18ec7a54787ffbc8fe8db0be4d137d89ef/benchmarks/comparison-plan.json).
The numerical code is the same for every case. Hashes of the native library,
Python sources, numerical sources and benchmark scripts identify each result;
every number comes from a fresh measurement, not from the earlier reference
run.

| Environment | CPU | Physical cores | Installed RAM |
| --- | --- | ---: | ---: |
| macOS / arm64 | Apple M3 | 8 | 24 GiB |
| Linux / x86-64 | AMD Ryzen 9 9950X | 16 | 60.4 GiB |

Each platform gets its own runtime, memory, speedup, and scaling plots. A separate
cross-platform figure compares identical cases without pooling their samples.
The results reflect both hardware and software differences; they do not isolate
the effect of the operating system.

The main budget is four threads for each implementation. Thread-scaling cases
request one, two, four, eight, and, on Linux, sixteen threads. Linux workers use
distinct physical cores; macOS controls the budget through the library settings
and leaves CPU placement to the OS. Accelerate's actual active thread count is
not exposed by threadpoolctl. Machine settings, library versions, BLAS details,
available memory, swap, and load observations accompany the raw evidence.

## Accuracy evidence

The [accuracy plan](https://github.com/yaugenst/treams-rs/blob/8b442f18ec7a54787ffbc8fe8db0be4d137d89ef/benchmarks/accuracy-plan.json) adds fresh accuracy
checks on both platforms. Its figures separate three questions: whether
the implementations agree, whether independently known mathematical or physical
relations hold, and whether the selected discretization has converged.

For reference agreement, the script records absolute, normwise relative and
tolerance-scaled errors of every broad reference check, and each check still
asserts its own tolerance. The scaled residual is
`abs(actual-reference)/(atol + rtol*abs(reference))`, with a passing bound of one.
Near-zero values are judged with an absolute scale; a large pointwise relative
error at a cancellation is not automatically a large observable error. Non-finite
outputs and undefined metrics are reported as such.

Independent references use arbitrary-precision functions and direct analytical
formulas. A second evaluation at higher precision checks their stability.
Spherical Bessel definitions and derivative relations follow
[DLMF 10.47](https://dlmf.nist.gov/10.47) and
[DLMF 10.51](https://dlmf.nist.gov/10.51). Physical checks include conservation,
passivity, zero contrast, subdivision of a homogeneous layer, and coordinate
symmetries. Convergence plots vary multipole order or quadrature resolution and
identify the reference cutoff. Agreement with a higher treams-rs cutoff is labelled
as convergence evidence, separate from an independent analytical reference.

Published-workflow checks cover the
[electron-beam spectroscopy paper](https://arxiv.org/abs/2602.12743v1) and
[CPC companion examples](https://doi.org/10.1016/j.cpc.2023.109076). Electron-beam
spectra are compared separately with the authors' 50-point regression tables and
200-point notebook executions; the cylinder notebook requires the explicitly
recorded polarization-basis correction. CPC sphere, chiral-slab and periodic-array
spectra are compared with fresh treams runs of the paper recipes, not with
independent author tables.
The array excludes the singular 350 nm endpoint and retains four nearby samples.
Altogether this adds 853 spectrum samples per package and 15 treams-rs cutoff
refinements. Neither the CPC quasi-BIC figure nor every application in either
paper is claimed reproduced. [Published applications](../validation/published-applications.md)
lists the fixed source links, hashes and tolerances, and the thermal study,
which is separate from these spectra.

Gradient accuracy uses the archived finite-difference step sweeps and full
gradient comparisons. Step-size curves expose both truncation and roundoff;
the exact linear field adjoint supplies an additional independent baseline where
available. The report keeps failed tolerances, exclusions and unconverged
points next to the successful cases. These checks show accuracy only over their
stated parameter ranges.

The Linux size sweep retained four failed reference checks: the eight-sphere
chain at multipole orders 8 and 12, and order-24 rotations with one and two
positions. These cases stopped before timing and contribute no speedup or
kernel-memory comparison. The rotation cases repeat the same coefficient
discrepancies at each position; the differences alone do not identify which
implementation is inaccurate.

Separately labelled post-failure diagnostics compare cluster conditioning,
backward residuals, low-order blocks and illuminated cross-section convergence,
and test the affected rotation coefficients against arbitrary-precision Wigner
formulas. They neither replace the failed cases nor relax their original
tolerances. Small backward residuals and conservation alone do not bound forward
error in an ill-conditioned system.

The Linux high-precision run contains 1,930 original inputs and 120 separately
labelled rotation diagnostics. Eighteen original inputs require values above
float64's finite range and are excluded from the finite-output checks. Of the
remaining 1,912 original inputs, treams-rs passes 1,908 and fails four
large metallic spheres. For size parameter `x=80`, `epsilon=-8+0.4j`, `mu=1`
and orders 1, 3, 80 and 99, this build returns zero Mie blocks where the
references are nonzero and stable. treams passes those four checks. The error
lies in treams-rs, not in the comparison; conservation identities alone can
miss an incorrect zero-scattering solution. The Linux core correctness run at
`2843a70` passes these four inputs ([validation](../validation/index.md),
[`linux-core-qualification.json`](https://github.com/yaugenst/treams-rs/blob/8b442f18ec7a54787ffbc8fe8db0be4d137d89ef/benchmarks/linux-core-qualification.json)).

All 120 rotation diagnostics pass for treams-rs. At the affected order-24
rotation entries, treams-rs has absolute errors of about 1.4e-15 to 2.6e-15
and treams about 2.0e-12. These errors explain the failed rotation
comparisons. For the cluster, treams-rs cross sections change by 2.96e-10 from
order 8 to 12, and treams cross sections by 2.31e-5. This points to an
instability in treams, but it does not prove the treams-rs cluster result
correct.

In three Linux boolean mode-selection cases, the residual collector raises a
TypeError before the numerical assertion runs; those errors stay archived. The
[three-case collector correction](https://github.com/yaugenst/treams-rs/blob/8b442f18ec7a54787ffbc8fe8db0be4d137d89ef/benchmarks/accuracy-boolean-correction-plan.json)
uses exact 0/1 arithmetic for boolean residuals and keeps the assertion as it
is. Its results stay separate from the main suite. macOS uses the corrected
collector throughout, and the solver and timing code are the same in both.

## What is timed

The broad and size/thread sweeps check numerical agreement before timing.
Requested-illumination workers measure first, then validate their outputs before
publishing a successful result. Both implementations use the same physical inputs
and double-precision complex arithmetic where applicable. Native libraries are
release builds. Each package runs in its own worker process, so importing
treams does not add to the memory of the treams-rs worker.

These are workflow comparisons. Their differences combine algorithms, data
layout, parallel execution and language implementation; they do not isolate the
effect of rewriting the same algorithm in Rust.

Small calls use calibrated batches. When both isolated medians are below one
millisecond, the broad benchmark script also runs the two packages in one
process in alternating order; the median of the paired time ratios gives the
speedup. Larger calls keep the isolated samples: treams runs first, then
treams-rs, each in its own process.
Their intervals describe within-run variation, not day-to-day drift or every
possible measurement order. The repetition policy is declared before measurement:
seven samples normally, three for the expensive preidentified cases. Figures
show observed ranges and sample-bootstrap intervals where supported. Three
samples provide limited evidence about timing variability.

Timing categories remain distinct:

- Public forward-only calls.
- treams-rs forward calls that keep data for a later pullback, compared with
  treams forward-only calls. The cost of keeping that data is part of the
  measurement.
- treams-rs reverse passes. treams has no reverse pass, so they have no
  treams timing.
- Complete gradients, including objective evaluation and the reverse pass.

The EBCM comparison uses the treams surface integral, which omits a radial
area factor, so both packages compute the same quantity; the corrected integral
is never timed against a different treams calculation. See [differences from treams](../coming-from-treams/differences.md).

## Memory and large problems

Peak RSS is the highest resident memory observed for the process, including
imports, inputs and memory retained for reuse. Additional RSS is the increase
above the peak after input setup; it is not an allocation count and can be zero
when earlier allocations already set a higher peak. Forward and retained
forward-plus-reverse peaks are labelled separately.

The runner applies an explicit process-group RSS budget and per-case time limit
to avoid exhausting the host. Its sampled group total can count shared pages
more than once, so it is used only to stop excessive memory use, not in plots.
Timeouts and memory-limit stops remain in the case table. They establish a limit of
this run, not a universal limit of either package.

Two Mac cases hit the 8 GiB group-memory guard: the 256-particle dense
cluster and the 262,144-point plane-wave operator. A process-cleanup error kept
their sampled peaks and elapsed times from being saved, so both count as
memory-limit outcomes, with no reconstructed measurement and no retry. The
controller waits for worker processes to exit on macOS and saves limit decisions
before cleanup. The records of both runs, the controller versions and a synthetic
cleanup check are kept with the comparison results outside the repository. The
numerical code and timing scripts are the same in both runs.

Requested-illumination cases distinguish four methods: the full interacting
T-matrix in treams, a dense treams/SciPy solve for only the requested columns,
dense treams-rs methods, and matrix-free iteration in treams-rs. A case
compares the two packages at the same size only where the treams calculation
and its numerical check ran. Beyond the dense cutoff, results labelled
"treams-rs only" show how treams-rs scales. Matrix-free
results include convergence checks and the chosen coupling regime. Above that
cutoff, the saved checks cover equation residuals, physical invariants and one
radius-direction finite difference, not a complete independent gradient.
Illumination JSON retains these residuals and aggregate errors; its temporary
comparison arrays are not archived. The separate gradient suite archives full
gradient arrays.

## Gradient comparisons

The gradient grid covers finite clusters, multilayer spheres, planar stacks,
spherical fields, and cylindrical fields. Parameter sets include geometry,
material properties, wavelength or wavenumber, and complex field coefficients.
Every timed finite-difference gradient evaluates every selected real coordinate.
Directional derivatives are reported separately from complete gradients.

Full treams-rs and finite-difference gradients are compared, with step-size
checks and archived validation arrays. Complex inputs use separate real and
imaginary coordinates. For coefficient-field objectives, the report also
includes an exact dense linear adjoint computed with treams. This keeps the
finite-difference baseline from overstating the benefit where a simple exact
adjoint is available.
The report lists forward recording, reverse execution and their total separately.
These checks differentiate the stated discretized objective; finite-difference
agreement alone does not establish convergence with multipole cutoff.

These timings measure the treams-rs pullbacks called from Python. They do not measure JAX
compilation, PyTorch integration, browser execution, higher derivatives, or GPU
geometry gradients.

## Reproduce the comparison

Build the release extension with `just build-ext-release` in the locked development
environment. Run `scripts/run_benchmark_suite.py --help` for the controller and
consult the archived manifests for each exact command and resource budget.
The high-precision collector also requires mpmath 1.3.0. If it is not already
installed through the optional framework dependencies, invoke the controller with
`uv run --no-sync --with mpmath==1.3.0 python scripts/run_benchmark_suite.py ...`.
Its child processes use the same Python environment. Plotting dependencies are
declared in the plotting script and can run in a separate environment.
The runner resumes an interrupted run only when its source,
protocol, and case selection match. Completed slow or unsuccessful cases are
not retried to select a favorable outcome.

The recorded Git commit identifies the numerical source of each measurement.
File hashes identify the benchmark scripts, which are newer than that commit.
Reproduce from a revision that contains both, and compare the recorded source
and environment identities.

Render saved evidence with `scripts/plot_benchmarks.py`, passing one `--manifest`
per suite and an empty output directory. Plotting does not run either solver.
The report weights sampled cases equally. Its distributions do not predict the
speedup of an arbitrary application.
