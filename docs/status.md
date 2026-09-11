# Implementation status

The documented CPU rewrite is complete: the Rust numerical implementation and
Python workflow layer cover the pinned `treams` 0.4.5 API inventory within the
contracts and numerical limits below. The complete Linux performance grid and
isolated Linux/macOS wheels pass. There is no runtime fallback to treams, SciPy
or Cython. This is not exact emulation of the legacy ndarray annotation engine.

The source reference is `1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39`.
The inventory contains 182 public functions/classes across the package and its
numerical, configuration and I/O modules. Export presence is only a coverage
check: the tests also compare complete workflows and independent physical laws.
Known reference defects are documented in [upstream findings](upstream-findings.md).

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
| Python objects | Material, all four basis types, TMatrix/TMatrixC, SMatrix/SMatrices, PhysicsArray, wave sources and reusable bound operators. T-matrix field methods, basis selections and exclusion masks; S-matrix coordinate transforms and read-only port block views. |
| I/O | Optional HDF5 scalar matrices and rectangular sweeps, streamed writes, chirality, local origins, mode indices, units, mesh and reproducibility metadata. Gmsh convenience uses actual boundary surface tags. |

## Python contract

Import as `treams_rs`; the distinct name allows the development oracle to coexist.
The common constructors and numerical conventions follow treams. This is an
explicit-object API rather than a replica of the ndarray annotation engine:

- `.array` exposes read-only numerical storage. Ordinary indexing/arithmetic on
  PhysicsArray returns NumPy values without inferred physical metadata.
- Selecting a T matrix with a wave basis returns a T matrix in those channels;
  ordinary numerical indexing returns values.
- `SMatrices.block(outgoing, incoming)` shares storage and carries port metadata.
  Numeric S-matrix indexing remains a cheap ndarray view.
- Bound operators expose explicit evaluation and left/right application. T-matrix
  rotation/expansion and S-matrix transformations return physical objects.
  The old `.ann`/`.relax` metadata machinery and every descriptor spelling are
  intentionally not reproduced; use explicit operators for one-sided transforms.
- Changing `config.POLTYPE` affects subsequent defaulted calls. Existing objects
  and recorded pullbacks retain the convention with which they were created.
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

Advect composes these numerical boundaries with user objectives. Object
constructors themselves do not trace framework arrays: use `treams_rs.advect` for
continuous parameters. Static basis labels, discretization sizes, quadrature
nodes, material topology and eigenvalue ordering are not differentiated.
Forward mode, higher derivatives, staging, checkpointing and other framework
adapters are outside this first-order contract.

At normal incidence, observables with smooth limits have explicit limiting
pullbacks. Polarization-frame derivatives at undefined directions require a
fixed-direction boundary. Exact diffraction thresholds, changing direct-shell
half-cell groupings and individual degenerate eigenmodes do not have a general
smooth derivative.

## Verification and performance

`just ci` runs Rust proptest/unit tests, Python reference/Hypothesis/workflow and
adjoint tests, rustfmt, warnings-as-errors Clippy/rustdoc, Ruff, strict Pyrefly,
lockfile validation and file hygiene. `just check-wheel` creates an isolated
environment, checks native execution and complete Advect objectives without
SciPy/treams, then checks optional HDF5 separately.

The complete implementation passes 73 Rust tests and 1,934 Python tests on Linux
and macOS. Its [Linux performance manifest](../benchmarks/complete-qualification.json)
records 527 passing runtime gates and 525 passing peak-RSS gates, all tied to the
same native binary, Python-source and benchmark hashes. The separate macOS
dispatch-regression grid passes 30 runtime/RSS cases. Finite test coverage is not
a proof of correctness or performance for all possible inputs.

The [independent fractional Legendre check](../scripts/qualify_legendre.py) covers
530 finite value/argument-derivative cases and two expected-overflow cases against
70-digit hypergeometric calculations. Its largest finite relative error was
2.17e-13; [raw results](../benchmarks/results/fractional-legendre-physical.json)
record the installed native binary hash. Physical checks elsewhere cover energy
conservation, reciprocity,
translation/rotation identities, scaling, limiting behavior and complete
objective derivatives. Finite differences are test oracles, never pullbacks.

Performance qualification uses an optimized extension, matched thread budgets,
isolated correctness/timing/RSS workers and alternating calibrated timings for
small calls. Every measured forward path must be at least as fast and have no
higher peak RSS than upstream. Recorded internal illumination has two explicitly
listed high-dimension RSS exceptions because its owned reverse data is compared
with an upstream forward-only computation; its runtime and all corresponding forward RSS gates
remain strict. Raw results, exact scope and older milestones are in
[benchmarks](benchmarks.md). Finite measurements do not establish a universal
speed or memory guarantee for every input.

## Numerical and platform limits

A supported label bound is not an accuracy certification throughout that range.
Extreme orders, arguments, resonant conditioning and very many layers require
case-specific convergence checks. Real fractional Legendre evaluation is qualified
for 0 < degree <= 128 and |order| <= degree. The public lpmv function preserves the
reference's zero extension at |order| > degree; it does not evaluate the general
Ferrers function in that extended domain. Fractional complex arguments and
fractional pi/tau are not implemented.

Dense outputs and LU storage remain quadratic in channel dimension, with cubic
factorization work. Optimized homogeneous sphere clusters require non-overlapping
nonmagnetic spheres in vacuum; the general local-T-matrix path supports other
materials and cutoffs. The caller must ensure enclosing particle surfaces do not
overlap. HDF5 layout compatibility is not certification against every external
T-matrix database.

The EBCM degree-6 benchmark contains analytically zero entries with severe
cancellation. Both implementations reach a double-precision roundoff floor; the
strict comparison gate is retained and no degree-6 speed claim is made. See the
independent high-precision reproducer in the benchmark documentation.

CPU execution is the target. GPU, WASM/browser bindings, Python versions outside
3.12/3.13 and broader wheel-platform distribution are separate qualification work.
