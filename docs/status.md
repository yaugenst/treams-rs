# Capabilities and numerical limits

The Rust numerical implementation covers the pinned `treams` 0.4.5 numerical
inventory within the contracts and limits below. The Python API is independently
designed around physical objects; it does not promise upstream source compatibility.
There is no runtime fallback to treams, SciPy, or Cython. The Python interface
uses explicit objects rather than the legacy ndarray annotation engine.

The source reference is `1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39`.
The inventory contains 182 public functions/classes across the package and its
numerical, configuration and I/O modules. Export presence is only a coverage
check: the tests also compare complete workflows and independent physical laws.
Known reference defects are documented in [upstream findings](upstream-findings.md).

## Current branch scope

`main` contains the CPU Rust core, Python API and first-order framework adapters.
The browser/WASM playground is preserved on `experimental/browser`;
CUDA support is preserved on `experimental/gpu`. Neither experiment
is part of the core build, public API catalog, or CI. Historical benchmark records
retain their original build identities; they do not qualify this branch revision.
Linux core correctness, performance and peak-memory qualification completed on
`main` at `0137eca` on this machine. Its results and explicit performance exceptions
remain bound to that revision; the API redesign does not relabel those measurements.
This branch's agent campaign measures workflow success and agent effort, not
solver speed or peak memory.
Its final wheel-only Linux evaluation reached 118/120 full main-task passes,
versus 110/120 for the original API, plus 24/24 confirmation and 16/16 fresh
polarization-transfer passes. Agent time fell 27.4%; the predeclared 40% time
gate was not met. The remaining nonpasses concern a private import and a
requested gradient output structure, not incorrect physical quantities.
See the [complete campaign and retained evidence](../benchmarks/results/agent-api-20260920/REPORT.md).

## Numerical and workflow coverage

| Area | Implemented behavior and derivative boundary |
| --- | --- |
| Materials and geometry | Isotropic lossy, magnetic and chiral materials; outgoing wavenumber branches; Cartesian/polar/spherical transforms and vector-frame changes; native continuous geometry pullbacks. Immutable Lattice and partial WaveVector metadata, reciprocal cells, diffraction orders and coordinate transforms. |
| Special functions | Cylindrical/spherical Bessel and incoming/outgoing Hankel values and derivatives; integer Legendre, pi/tau and harmonics; fractional-degree real Legendre; Wigner 3j and small/full D; incomplete gamma and Kambe integrals. NumPy broadcasting, output buffers and native argument/angle pullbacks. |
| Local waves and coefficients | Spherical, cylindrical and plane waves; helicity/parity bases; regular/outgoing translation coefficients; sw/cw/pw namespaces; analytic axis limits and continuous coordinate/wavenumber pullbacks. |
| Particle scattering | Multilayer chiral sphere and cylinder coefficients and T matrices; radius, permittivity, permeability, chirality, frequency and cylindrical axial-wavenumber pullbacks. |
| Finite interactions | Homogeneous optimized sphere clusters and heterogeneous spherical/cylindrical local T matrices, differing cutoffs and mode subsets. Native block storage, interaction solves and LU-reusing pullbacks for local matrices, positions and embedding wavenumbers. |
| Basis operations | Rotations, translations, polarization changes, finite and periodic expansion, plane coordinate permutations and spherical/cylindrical conversions. Fixed discrete mode labels with native continuous parameter pullbacks. |
| Fields and illumination | Electric, magnetic, displacement, flux and G/F field operators; weighted samples; spherical/cylindrical sources and plane waves, including complex directions. Native amplitude, sample, origin and wavenumber pullbacks. Direct T-matrix illumination, cross sections/widths and source expansion. |
| Periodic sums | Spherical 1D/2D/3D and cylindrical 1D/2D Ewald sums, full/real/reciprocal/direct-shell APIs, shifted geometries and batched inputs. Native k, Bloch vector, cell, shift and split-parameter pullbacks. |
| Custom periodic tables | User-supplied spherical lattice-sum callbacks are evaluated once with broadcast geometry. Native table-to-matrix contraction and its conjugate-transpose pullback compose with differentiable lattice sums. |
| Periodic scattering | Coupling, interactions, spherical 2D and cylindrical 1D particle-to-plane channels, particle-array S matrices and spherical-array-to-cylinder conversion. Complete native pullbacks, including shared cylindrical axial groups. |
| Planar layers | Chiral Fresnel interfaces, propagation, compact multilayer stacks, S-matrix composition and doubling. Native factorization-reusing pullbacks and internal illumination between adjacent stacks. |
| Power and dichroism | Native batched S-matrix transmittance/reflectance includes coherent incident/reflected interference in absorbing media. Pullbacks cover the illuminated S-matrix column, incident amplitudes, both port wavenumbers/impedances and transverse directions. Circular dichroism evaluates both incident polarizations in one batch and has an Advect adapter. |
| Other observables | T-matrix CD, duality breaking and electromagnetic chirality, thin SVD and native pullbacks. Plane chirality-density forms in all coordinate orientations, with interval and geometry pullbacks. |
| Bloch bands | Native transfer matrices, right eigensystems and Bloch wavenumbers/vectors, with S-matrix, period and nondegenerate eigenvector pullbacks. |
| Axisymmetric EBCM | Callable radial surfaces sampled by Gauss-Legendre quadrature, native regular/outgoing Q integrals and radius, slope, complex-wavenumber and impedance pullbacks. Correct surface area by default; explicit legacy mode for upstream comparisons. |
| Python objects | Material, all four basis types, TMatrix/CylindricalTMatrix, SMatrix/ScatteringBlock, typed waves, unsolved clusters and solved periodic responses. Physical scattering, fields, named observables and explicit numerical operators; basis selections, coordinate transforms and read-only port block views. |
| I/O | Optional HDF5 scalar matrices and rectangular sweeps, streamed writes, chirality, local origins, mode indices, units, mesh and reproducibility metadata. Gmsh convenience uses actual boundary surface tags. |

## Python contract

The [user guide](user-guide.md) defines the primary API: keyword particle
constructors, typed scattered waves, distinct unsolved `Cluster` and solved
`PeriodicResponse`, explicit conversion, and named planar/particle observables.
`SMatrix` is the full network and `ScatteringBlock` is a single block.
`operators` owns numerical matrix builders. Complete review decisions are in
[physics](api-physics-map.md) and [autodiff](api-autodiff-map.md) maps.
Old root basis names and operator re-exports have been removed. Unsolved
`TMatrix.cluster` and implicitly solving `SMatrix.from_array` constructors have
been removed from the public API; explicit numerical methods remain available
for expert calculations. Framework namespaces expose the same fixed basis and
lattice types as the root namespace. Fields are evaluated on waves, e.g.
`response.scatter(incident).efield(points)`; bound T-matrix field operators were
removed because applying them to scattered coefficients applies the response twice.

The wheel includes an executable quickstart and focused offline symbol lookup:
`python -m treams_rs`, `python -m treams_rs advect`, and
`python -m treams_rs --search periodic`. Explicit `--format json` or `--format markdown`
exports the complete source-derived catalog. The wheel-only evaluation protocol
and independently replayed numerical tasks are in
[the campaign specification](../benchmarks/agent-api/README.md).

Import as `treams_rs`; the distinct name allows the development oracle to coexist.
Numerical conventions follow treams. This is an
explicit-object API rather than a replica of the ndarray annotation engine:

- `.array` exposes read-only numerical storage. Ordinary indexing/arithmetic on
  PhysicsArray returns NumPy values without inferred physical metadata.
- Selecting a T matrix with a wave basis returns a T matrix in those channels;
  ordinary numerical indexing returns values.
- `SMatrix.block(outgoing, incoming)` shares storage and carries port metadata.
  Numeric S-matrix indexing remains a cheap ndarray view.
- Bound operators expose explicit evaluation and left/right application. T-matrix
  rotation/expansion and S-matrix transformations return physical objects.
  The old `.ann`/`.relax` metadata machinery and every descriptor spelling are
  intentionally not reproduced; use explicit operators for one-sided transforms.
- Polarization defaults deterministically to helicity. Pass parity explicitly;
  there is no mutable global polarization setting.
- Plane-basis z rotations retain lattice and Bloch metadata. Rotations that would
  turn a partial Cartesian constraint into an oblique constraint raise instead
  of discarding it. Full xy planes, z axes and three-dimensional cells are closed
  under these rotations; exact quarter turns also handle other Cartesian spans.

## Adjoint contract

`diff`/`coeffs` forward calls return an array and an opaque native residual.
`residual.pullback(g)` consumes it once, using
`dL = Re(sum(conj(g) * doutput))`. Invalid cotangent shapes are rejected before
consumption. The residual owns retained data; mutating the caller's inputs does
not alter a recorded derivative. Rust reuses factorizations or recomputes local
analytic derivatives without storing dense parameter Jacobians.

Advect, JAX and PyTorch compose these numerical boundaries with user objectives.
Ordinary NumPy object constructors do not trace framework arrays: use the
[framework adapters](adapters.md) for continuous parameters. Static basis labels,
discretization sizes, quadrature nodes, material topology, and eigenvalue ordering
are not differentiated.
Forward mode and higher derivatives are outside this first-order contract. JAX
supports CPU `jit`, sequential `vmap` and checkpointing using host callbacks;
PyTorch supports eager CPU backward and repeated backward through native
recomputation. GPU tensors and `torch.compile` are not supported by these adapters.

At normal incidence, observables with smooth limits have explicit limiting
pullbacks. Polarization-frame derivatives at undefined directions require a
fixed-direction boundary. Exact diffraction thresholds, changing direct-shell
half-cell groupings and individual degenerate eigenmodes do not have a general
smooth derivative.

## Verification and performance

The September 19 API redesign passed `just verify` on Linux with Python 3.13:
93 Rust tests and 2,271 Python tests, including Advect, JAX and Torch, with
14 plotting tests skipped in the base environment. All 40 report tests passed
separately with Matplotlib. The release clean-wheel check (including typed
scattering, an Advect physical objective and optional HDF5) and
warnings-as-errors rustdoc also passed. New API checks cover dense/requested
cluster agreement, periodic conversion without another solve, all six field
families, optical-theorem and scale identities, moving diffraction ports and
first-order gradients. Numerical kernels and dependencies were not changed.

The benchmark campaign, macOS rerun, Python 3.12 rerun and experimental
browser/GPU hardware lanes were not rerun for this redesign. Framework
high-level objects retain the static-direction/static-axial-label limits in
the [autodiff map](api-autodiff-map.md); specialized kernels retain explicit
expert interfaces. No new runtime or memory-performance qualification is claimed.

`just ci` runs Rust proptest/unit tests, Python reference/Hypothesis/workflow and
adjoint tests, rustfmt, warnings-as-errors Clippy/rustdoc, Ruff, strict Pyrefly,
lockfile validation and file hygiene. `just check-wheel` creates an isolated
environment, checks native execution and complete Advect objectives without
SciPy/treams, then checks optional HDF5 separately.

Recorded CPU qualification passed 84 Rust tests and 2,072 Python tests on Linux
and macOS, including JAX and PyTorch. Clean-wheel execution, optional HDF5,
strict types, lint, locks, and rustdoc also passed. CI checks Python 3.12/3.13 on Linux.

The historical [Linux performance manifest](../benchmarks/complete-qualification.json)
records 527 passing runtime gates and 525 passing peak-RSS gates, tied to the
tested native binary, Python-source and benchmark hashes (minimum speedup
1.02x, maximum gated RSS ratio 0.912). The macOS dispatch grid passes all
30 runtime/RSS cases on its recorded build (minimum speedup 1.30x, maximum RSS
ratio 0.695). The requested-illumination, matrix-free and paper reports
include their own executable/source hashes.

The [broader Mac and Linux comparison](benchmark-comparison.md) repeats the full
527-case grid on the same numerical source and retains every outcome. All 527
reference checks pass on each platform. Median sampled speedups are 5.17x on the
Apple M3 and 5.04x on the Ryzen 9 9950X. Eleven Mac cases are slower than upstream;
the Linux grid has no observed median slowdowns. Both have two higher-RSS
recorded-adjoint cases. These results supersede any interpretation of the
historical subsets as a platform-wide performance guarantee.

The [API and adapter qualification](../benchmarks/agent-usability/qualification.json)
checks offline discovery, executable examples, numerical checking helpers, and
adapter diagnostics. Sphere/cylinder constructors explicitly label native
helicity data before conversion to the requested polarization basis;
cross-default Hypothesis regressions cover `config.POLTYPE` independence.
The associated Python change left native binaries unchanged. Earlier full grids
and paper reports retain their measured Python hashes and were not
rerun or relabelled for that change. The two affected public cluster paths were
requalified on macOS: spherical 7.20× and cylindrical 1.56× faster than upstream,
both with lower peak RSS. Each report describes its tested build; these records
are not a claim that every later revision reruns every performance grid.

The [independent fractional Legendre check](../scripts/qualify_legendre.py) covers
530 finite value/argument-derivative cases and two expected-overflow cases against
70-digit hypergeometric calculations. Its largest finite relative error was
2.17e-13; [raw results](../benchmarks/results/fractional-legendre-physical.json)
record the installed native binary hash. Physical checks also cover energy
conservation, reciprocity, translation/rotation identities, scaling, limiting
behavior, and complete
objective derivatives. Finite differences are test oracles, never pullbacks.

Performance qualification uses an optimized extension, matched thread budgets,
isolated correctness/timing/RSS workers and alternating calibrated timings for
small calls. The historical performance gates require no slowdown or higher RSS,
apart from two recorded internal-illumination memory exceptions. The broader
comparison also records cases that miss those targets; it does not discard or
retry slower measurements to select a passing result. Recorded internal
illumination retains owned reverse data while upstream computes only the forward
result. Raw results, exact scope, and historical
measurements are in [benchmarks](benchmarks.md). Finite measurements do not establish a universal
speed or memory guarantee for every input.

## Numerical and platform limits

A supported label bound is not an accuracy certification throughout that range.
Extreme orders, arguments, resonant conditioning and very many layers require
case-specific convergence checks. Real fractional Legendre evaluation is qualified
for 0 < degree <= 128 and |order| <= degree. The public lpmv function preserves the
reference's zero extension at |order| > degree; it does not evaluate the general
Ferrers function in that extended domain. Fractional complex arguments and
fractional pi/tau are not implemented.

The original independent high-precision campaign exposed zero Mie blocks for a
large metallic sphere at size parameter 80, `epsilon=-8+0.4j`, `mu=1` in vacuum.
The current source includes the scaled-inverse correction and its targeted
regression checks. The full corrected accuracy/performance campaign has not run;
its original failures remain archived in the [comparison report](benchmark-comparison.md).

Dense outputs and LU storage remain quadratic in channel dimension, with cubic
factorization work. Requested-illumination factors avoid the full interacting
T-matrix, while the matrix-free sphere path avoids global quadratic storage in
both forward and physical-parameter adjoint evaluation. It recomputes pair
translations each GMRES iteration and checks the actual residual; it can be
slower than a reused dense factor. See [large problems](large-problems.md).
Optimized homogeneous sphere clusters require non-overlapping
nonmagnetic spheres in vacuum; the general local-T-matrix path supports other
materials and cutoffs. The caller must ensure enclosing particle surfaces do not
overlap. HDF5 layout compatibility is not certification against every external
T-matrix database.

The EBCM degree-6 benchmark contains analytically zero entries with severe
cancellation. Both implementations reach a double-precision roundoff floor; the
strict comparison gate is retained and no degree-6 speed claim is made. See the
independent high-precision reproducer in the benchmark documentation.

Python versions outside 3.12/3.13 and broader wheel-platform distribution remain
unqualified. A passing finite test set does not establish correctness or
performance for every parameter choice.
