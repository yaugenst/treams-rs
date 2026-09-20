# Mac and Linux benchmark comparison

This page describes the historical two-platform campaign. The completed
September 19–20 Linux-only qualification is recorded in
[the current summary](../benchmarks/RESUME.md), including its measured performance
and memory exceptions. macOS has not been rerun for the current core source.

The [generated report](../benchmarks/comparison-report/index.html) compares
upstream treams 0.4.5 with treams-rs on two independently measured platforms.
It includes downloadable figures, a combined PDF, a searchable case ledger,
machine-readable measurements, and the source manifests. The ledger retains
every planned case, including errors and resource limits.

## Experimental scope

The campaign replays every case in the
[527-case reference grid](../benchmarks/complete-qualification.json), then runs
the [additional scaling and gradient plan](../benchmarks/comparison-plan.json).
The numerical implementation stays fixed throughout the campaign. Results are
identified by native-library, Python-source, numerical-source, and harness hashes;
historical performance gates are not substituted for new measurements.

| Environment | CPU | Physical cores | Installed RAM |
| --- | --- | ---: | ---: |
| macOS / arm64 | Apple M3 | 8 | 24 GiB |
| Linux / x86-64 | AMD Ryzen 9 9950X | 16 | 60.4 GiB |

Each platform gets its own runtime, memory, speedup, and scaling plots. A separate
cross-platform figure compares identical cases without pooling their samples.
These are comparisons of the complete hardware/software configurations, not an
isolated experiment on the effect of an operating system.

The main budget is four threads for each implementation. Thread-scaling cases
request one, two, four, eight, and, on Linux, sixteen threads. Linux workers use
distinct physical cores; macOS controls the budget through the library settings
and leaves CPU placement to the OS. Accelerate's actual active thread count is
not exposed by threadpoolctl. Machine settings, library versions, BLAS details,
available memory, swap, and load observations accompany the raw evidence.

## Accuracy evidence

The [accuracy plan](../benchmarks/accuracy-plan.json) adds fresh numerical
qualification on both platforms. Its figures separate three questions: whether
the implementations agree, whether independently known mathematical or physical
relations hold, and whether the selected discretization has converged.

For reference agreement, the collector records absolute, normwise relative, and
tolerance-scaled errors from every original broad reference check. The original
assertion still executes. The scaled residual is
`abs(actual-reference)/(atol + rtol*abs(reference))`, with a passing bound of one.
Near-zero values are judged with an absolute scale; a large pointwise relative
error at a cancellation is not automatically a large observable error. Nonfinite
outputs and undefined metrics remain explicit rather than becoming zero-error
passes.

Independent references use arbitrary-precision functions and direct analytical
formulas, with a second, higher precision evaluation to test reference stability.
Spherical Bessel definitions and derivative relations follow
[DLMF 10.47](https://dlmf.nist.gov/10.47) and
[DLMF 10.51](https://dlmf.nist.gov/10.51). Physical checks include conservation,
passivity, zero contrast, subdivision of a homogeneous layer, and coordinate
symmetries. Convergence plots vary multipole order or quadrature resolution and
identify the reference cutoff. Agreement with a higher native cutoff is labelled
as convergence evidence, separately from an independent analytical reference.

Published-workflow checks cover the
[electron-beam spectroscopy paper](https://arxiv.org/abs/2602.12743v1) and
[CPC companion examples](https://doi.org/10.1016/j.cpc.2023.109076). Electron-beam
spectra are compared separately with the authors' 50-point regression tables and
200-point notebook executions; the cylinder notebook requires the explicitly
recorded polarization-basis correction. CPC sphere, chiral-slab and periodic-array
spectra use fresh upstream recipe comparisons, not independent author tables.
The array excludes the singular 350 nm endpoint and retains four nearby samples.
Altogether this adds 853 spectrum samples per backend and 15 native cutoff
refinements. Neither the CPC quasi-BIC figure nor every application in either
paper is claimed reproduced. The [paper qualification guide](paper-qualification.md)
records immutable source links, hashes, tolerances and the earlier thermal study;
that thermal study is separate from this fresh two-platform spectrum campaign.

Gradient accuracy uses the archived finite-difference step sweeps and full
gradient comparisons. Step-size curves expose both truncation and roundoff;
the exact linear field adjoint supplies an additional independent baseline where
available. The report retains failed tolerances, exclusions, and unconverged
points alongside successful cases. These datasets establish accuracy over their
stated parameter ranges rather than by relying on implementation provenance.

The Linux size sweep retained four failed reference checks: the eight-sphere
chain at multipole orders 8 and 12, and order-24 rotations with one and two
origins. These cases stopped before timing and contribute no speedup or
kernel-memory comparison. The rotation cases repeat the same coefficient
discrepancies at each origin; the differences alone do not identify which
implementation is inaccurate.

Separately labelled post-failure diagnostics compare cluster conditioning,
backward residuals, low-order blocks and illuminated cross-section convergence,
and test the affected rotation coefficients against arbitrary-precision Wigner
formulas. They neither replace the failed cases nor relax their original
tolerances. Small backward residuals and conservation alone do not bound forward
error in an ill-conditioned system.

The Linux high-precision run contains 1,930 original inputs and 120 separately
labelled rotation diagnostics. Eighteen original inputs require values above
float64's finite range and are excluded from finite-output qualification. Of the
remaining 1,912 original inputs, native passes 1,908 and fails four large-metallic-
sphere cases. For size parameter `x=80`, `epsilon=-8+0.4j`, `mu=1`, and orders
1, 3, 80 and 99, it returns zero Mie blocks despite nonzero stable references.
Upstream passes those four checks. This is a native numerical limitation, not
an upstream-agreement ambiguity; conservation identities alone can miss an
incorrect zero-scattering solution.

All 120 rotation diagnostics pass native qualification. At the affected order-24
public rotation entries, native absolute errors are about 1.4e-15 to 2.6e-15,
where upstream errors are about 2.0e-12. These independently explain the strict
rotation comparison failures. For the cluster, native cross sections change
2.96e-10 from order 8 to 12, while upstream changes 2.31e-5. That relative
convergence evidence supports investigating upstream instability but is not an
independent certificate of the native cluster result.

Three Linux boolean mode-selection cases exposed a residual-collector TypeError
before the original numerical assertion. Their initial errors remain archived.
The [three-case collector correction](../benchmarks/accuracy-boolean-correction-plan.json)
uses exact 0/1 arithmetic for boolean residuals and leaves the original assertion
unchanged. Its supplementary results are separate from the original suite;
macOS uses the corrected collector from the start. No solver or timing code
changed for this correction.

## What is timed

The broad and size/thread sweeps check numerical agreement before timing.
Requested-illumination workers measure first, then validate their outputs before
publishing a successful result. Both implementations use the same physical inputs
and double-precision complex arithmetic where applicable. Native libraries are
release builds. Standalone workers measure each backend, so importing the
reference library does not inflate the native worker's memory baseline.

These are workflow comparisons. Their differences combine algorithms, data
layout, parallel execution and language implementation; they do not isolate the
effect of rewriting the same algorithm in Rust.

Small calls use calibrated batches. When both isolated medians are below one
millisecond, the broad harness also alternates the backends in one process; its
paired median of time ratios is authoritative for speedup. Larger calls retain
the isolated samples, with upstream measured before native in separate processes.
Their intervals describe within-run variation, not day-to-day drift or every
possible measurement order. The repetition policy is declared before measurement:
seven samples normally, three for the expensive preidentified cases. Figures
show observed ranges and sample-bootstrap intervals where supported. Three
samples provide limited evidence about timing variability.

Timing categories remain distinct:

- Public forward-only calls.
- Native forwards that retain data for a later pullback, compared with upstream
  forward-only execution. Their retained-state cost is part of the measurement.
- Native reverse passes, with no invented upstream reverse timing.
- Complete gradients, including objective evaluation and the reverse pass.

The EBCM comparison explicitly uses the upstream-compatible legacy surface
integral. It does not time the physically corrected integral against a different
upstream calculation. See [upstream findings](upstream-findings.md).

## Memory and large problems

Peak RSS is the process high-water mark, including imports, inputs, native
allocations, and allocator retention. Additional RSS is the increase above the
high-water mark after input setup; it is not an allocation count and can be zero
when earlier allocations already set a higher peak. Forward and retained
forward-plus-reverse peaks are labelled separately.

The runner applies an explicit process-group RSS budget and per-case time limit
to avoid exhausting the host. Its sampled group total can count shared pages
more than once; it is a cleanup guard, not the plotted memory measurement.
Timeouts and memory-limit stops remain in the ledger. They establish a limit of
this run, not a universal limit of either package.

Two Mac cases triggered the 8 GiB group-memory guard: the 256-particle dense
cluster and the 262,144-point plane-wave operator. Process-cleanup errors
prevented the controller from saving their sampled peaks and elapsed times.
Both remain memory-limit outcomes with no reconstructed measurements or retry.
The controller now waits for macOS exit teardown and saves limit decisions
before cleanup. The two original manifests and controller versions are retained,
along with [synthetic cleanup validation](../benchmarks/results/comparison/mac/controller-recovery/cleanup-validation.json).
Numerical code and timing harnesses were unchanged during both recoveries.

Requested-illumination cases distinguish upstream's full interacting T matrix,
an upstream/SciPy dense solve for only the requested columns, native dense
methods, and native matrix-free iteration. Same-size upstream comparisons are
made only where the upstream calculation and numerical check actually ran.
Beyond the dense cutoff, native-only scaling is labelled explicitly. Matrix-free
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
Every timed finite-difference gradient evaluates every selected real coordinate;
a directional derivative is never presented as a complete gradient.

Full native and finite-difference gradients are compared, with step-size checks
and archived validation arrays. Complex inputs use separate real and imaginary
coordinates. For coefficient-field objectives, the report also includes an
exact upstream dense linear adjoint. This prevents the finite-difference
baseline from overstating the benefit where a simple exact adjoint is available.
Forward recording, reverse execution, and their total remain separately visible.
These checks differentiate the stated discretized objective; finite-difference
agreement alone does not establish convergence with multipole cutoff.

These timings exercise the native pullback boundary. They do not measure JAX
compilation, PyTorch integration, browser execution, higher derivatives, or GPU
geometry gradients. Optional CUDA measurements belong in a separate appendix;
CPU-to-GPU speedups within treams-rs are not upstream-to-treams-rs speedups.

## Reproducing the campaign

Build the release extension with `just build-ext-release` in the locked development
environment. Run `scripts/run_benchmark_suite.py --help` for the controller and
consult the archived manifests for each exact command and resource budget.
The high-precision collector also requires mpmath 1.3.0. If it is not already
installed through the optional framework dependencies, invoke the controller with
`uv run --no-sync --with mpmath==1.3.0 python scripts/run_benchmark_suite.py ...`.
Its child processes use the same Python environment. Plotting dependencies are
declared in the plotting script and can run in a separate environment.
The runner supports resuming an interrupted campaign only when its source,
protocol, and case selection match. Completed slow or unsuccessful cases are
not retried to select a favorable outcome.

The recorded Git commit identifies the numerical checkout used for measurement.
The new benchmark scripts are identified separately by file hashes and are
included with this report's repository revision. Reproduce from that revision
and compare the recorded source and environment identities; the numerical-base
commit alone does not contain the new harnesses.

Render saved evidence with `scripts/plot_benchmarks.py`, passing one `--manifest`
per suite and an empty output directory. Plotting does not run either solver.
The report's distributions weight sampled cases equally; they are not a forecast
of speedup for an arbitrary application or a claim of a universal performance
guarantee.
