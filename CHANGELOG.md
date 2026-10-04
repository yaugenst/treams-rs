# Changelog

## Unreleased

## 0.1.1 (2026-10-04)

- Faster CPU sphere-cluster solves for multiple illuminations by sharing pair
  translations across up to eight independent GMRES solves, with bounded
  Krylov storage and unchanged per-column convergence criteria.
- Faster spherical field maps and translations through shared harmonic and radial
  sequences, including a qualified positive-real outgoing Hankel recurrence.
- Reuse Ewald geometry and radial integrals across spherical harmonic orders in
  periodic-array values and analytic gradients, with bounded temporary storage.
- Avoid forming dense reflection products for certified thin internal-field
  solves between layered structures; retain the dense solve otherwise.
- Limit small dense matrix products to workers supported by their work size,
  avoiding excessive scheduling on machines with large thread budgets.
- Record matched-build CPU timing and memory evidence, and document solver
  tradeoffs. Public APIs, convergence tolerances and thread controls are unchanged.

## 0.1.0 (2026-10-03)

First public release of treams-rs: electromagnetic T-matrix scattering with a
Rust numerical core, a Python interface and analytic first-order derivatives.

### Capabilities

- Spherical, cylindrical and plane waves; helicity and parity bases; translations,
  rotations, polarization changes and finite or periodic basis expansions.
- Homogeneous and multilayer spheres and cylinders, including lossy, magnetic
  and chiral materials; finite particle clusters, periodic arrays and planar
  stacks. Dense and matrix-free sphere-cluster solvers support different problem
  sizes.
- Electric and magnetic fields, scattering cross sections and widths, power
  balance, circular dichroism, chirality and Bloch bands. Axisymmetric EBCM
  supports particles described by a sampled radial surface.
- The numerical modules `special`, `sw`, `cw`, `pw`, `lattice`, `coeffs`,
  `misc` and `ebcm` follow treams function names and conventions. The physics
  objects carry their basis, wavenumber and media explicitly.
- Analytic derivatives in `treams_rs.diff`, with optional Advect, JAX, PyTorch
  and HIPS Autograd support through the ordinary API. Constructors, physics
  methods and differentiable numerical functions select the adapter from their
  inputs and load it when needed. Python and NumPy inputs retain NumPy behavior;
  constants compose with framework objects, and mixed frameworks raise before
  conversion. `treams_rs.testing` provides gradient checks.
- JAX and PyTorch accept their default float32/complex64 arrays, preserving
  input gradient dtypes. Native work remains double precision; JAX outputs
  follow its `jax_enable_x64` setting without changing it globally.
- Differentiable numerical functions wrap their native NumPy implementations.
  NumPy ufunc attributes, methods and `out`/`where` behavior remain available on
  NumPy calls, but these public callables need not be instances of `numpy.ufunc`.
  Differentiated calls do not support mutable `out` or masked `where` outputs.
- Optional HDF5 T-matrix input/output through `treams_rs.io`, including matrix
  sweeps and physical metadata.
- An owned CPU worker pool with `set_num_threads`, `get_num_threads`,
  `threads` and `thread_info`, process-pool support and optional threadpoolctl
  integration. Values and gradients repeat bit for bit across thread budgets
  within one build and machine; different builds or processors can differ.
- Real-degree Ferrers functions use an original implementation of DLMF
  formulas; upstream treams attribution and linked-library license notices
  accompany the distributions.
- Reference documentation, executable examples, treams comparisons, independent
  physical checks, numerical limits and benchmark provenance. Offline API help
  is available through `python -m treams_rs` and `support_catalog()`.

### Supported scope

- CPython 3.12–3.15. Optional frameworks and HDF5 support depend on their own
  package availability; see
  [installation](https://yaugenst.github.io/treams-rs/latest/getting-started/install/).
- CPU execution and first-order reverse-mode derivatives. Discrete mode labels,
  cutoffs and topology stay fixed during differentiation. GPU execution and
  higher-order derivatives are unsupported.
- The core Python package requires NumPy at run time; it does not call treams,
  SciPy or Cython. Framework dtype and transform support is listed in
  [framework adapters](https://yaugenst.github.io/treams-rs/latest/differentiation/frameworks/).
- Numerical agreement is subject to the documented conventions, deliberate
  differences and conditioning limits; see
  [validation](https://yaugenst.github.io/treams-rs/latest/validation/).

### Numerical corrections

- NumPy integer inputs and results use 64-bit values on Windows, matching Linux
  and macOS. Oversized wave labels are rejected without truncation.
- Slabs retain both internal polarizations before selecting external ports.
  Partial bases give the corresponding projection of the complete response,
  including in framework adapters and parity ports.
- Planar interfaces and gain-medium fields use consistent tangential fields,
  Maxwell equations and constitutive relations. High-level planar APIs reject
  incompatible strong-chirality and negative-index branches; expert Fresnel
  coefficients retain the upstream convention.
- Clusters reject coincident particle centres before assembly, including
  cylinders on the same transverse axis.
- Ewald sums reject gain wavenumbers (`Im k < 0`) and, for spherical waves,
  negative real wavenumbers (`Re k < 0`). Finite direct shells retain their
  full complex-wavenumber domain.
- Spherical fields and analytic gradients use bounded internal length units,
  avoiding intermediate solid-harmonic overflow for finite fields expressed
  in very small or large length units.

### Migrating from pre-release checkouts

The public API was consolidated before 0.1.0. The table below maps earlier
pre-release names to the release API; it does not promise aliases for old names.
For the complete mapping from upstream treams, including object members and
keywords, see the
[treams name map](https://yaugenst.github.io/treams-rs/latest/coming-from-treams/names/).

| Earlier name or call | Release API |
| --- | --- |
| `SphericalWaveBasis` | `SphericalBasis` |
| `CylindricalWaveBasis` | `CylindricalBasis` |
| `PlaneWaveBasisByUnitVector` | `PlaneWaveBasis` |
| `PlaneWaveBasisByComp` | `PlaneWavePorts` |
| `MultipoleWave` | `Wave` |
| `TMatrixC` | `CylindricalTMatrix` |
| `SMatrices` | `SMatrix`; a single port block is `ScatteringBlock` |
| Top-level operators and `PhysicsArray` | `treams_rs.operators` |
| `treams_rs.support` | `support_catalog()` |
| `coeffs.fresnel_with_context`, `coeffs.mie_with_context`, `coeffs.mie_cyl_with_context` | `diff.fresnel`, `diff.mie`, `diff.mie_cyl` |
| `lattice.expansion_with_context(destination, source, ks, a, kpar)` | `diff.lattice_expansion(destination, source, ks, kpar, a)` |
| `lattice.expansion(...)` | The value from `diff.lattice_expansion(...)[0]`, or `sw.translate_periodic` / `cw.translate_periodic` |
| `diff.interface`, `diff.propagation` | `diff.interface_coefficients`, `diff.propagation_matrix` |
| `diff.cluster`, `diff.cluster_factor` | `diff.sphere_cluster`, `diff.sphere_cluster_factor` |
| `diff.periodic_from_table`, `diff.periodic_conversion`, `diff.wigner` | `diff.lattice_expansion_from_table`, `diff.periodic_to_cw`, `diff.wignerd` |
| `advect.cluster`, `advect.periodic_from_table`, `advect.periodic_conversion`, `advect.wigner` | `advect.sphere_cluster`, `advect.lattice_expansion_from_table`, `advect.periodic_to_cw`, `advect.wignerd` |
| `advect.smatrix_cd` | `advect.smatrix_circular_dichroism` |
| `iterative.Pullback` | `iterative.IterativeContext` |
| `iterative.SphereCluster.solve_with_pullback(...)` | `iterative.SphereCluster.record(...)` |
| `plane_wave(..., polarization=<state>)` | `plane_wave(..., pol=<state>, k0=...)`; `polarization` selects `"helicity"` or `"parity"` |
| Multipole-wave `kind="outgoing"` | `kind="singular"`; framework wave `outgoing=` becomes `singular=` |
| Special-function selector `kind=` | `function=` in `bessel`, `angular`, `vector_wave`, `coordinates` and `vector_coordinates` records and adapters |
| `tmatrix_metric(kind=...)` | `tmatrix_metric(metric=...)` |
| `vector_wave(polarization=...)` | `vector_wave(pol=...)` |
| `PeriodicResponse.to_smatrix(orders=...)` in framework adapters | `diffraction_orders=...` |
| `ebcm.qmat(legacy=True)` and its differentiable variants | `radial_area_factor=False`; the default includes the radial surface-area factor |
| `diff.ebcm_qmat(out=..., in_=...)` and its Advect adapter | `destination=..., source=...` |
| `PlaneWave.amplitudes` | `PlaneWave.coefficients` |
| Catalog key `contract`; adapter key `contract` | `rules`; adapter key `summary` |

Further behavioral changes for pre-release callers:

- Wave constructors require an explicit `k0`. There is no global
  `config.POLTYPE`; pass `polarization=` or `poltype=` where appropriate.
- `TMatrix.translate` and `CylindricalTMatrix.translate` return a T-matrix;
  read `.array` for its NumPy matrix or use `operators.Translate` for the
  translation operator.
- `Cluster(particles, positions=...).solve()` constructs the coupled response.
  A `TMatrix` has the same behavior whether it came from one particle or a
  cluster.
- A record returns `(value, context)`. Its pullback consumes the context and
  returns gradients in forward-argument order. In particular,
  `diff.sphere_cluster` returns gradients `(k0, radii, epsilon, positions)`;
  `iterative.Gradient` adds `incident` and `convergence` after those fields.
- Framework field functions use `positions` and per-mode `kz`; their wave and
  T-matrix objects read axial wavenumbers from `basis.kz`. Framework port waves
  and S-matrices hold their port metadata in `ports`.
- Old class-name pickles are not compatible with the renamed classes. Use the
  public reference for native context types and Rust paths instead of relying
  on pre-release internal names.
