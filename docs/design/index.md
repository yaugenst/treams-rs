---
description: The layers of treams-rs, their responsibilities and their treams counterparts.
---

# Design

treams-rs follows treams' numerical conventions and adds analytic gradients.
Four goals shape the code:

- **treams numerics.** Mode order, normalization and the helicity convention follow
  treams at a pinned commit (see [conventions](../coming-from-treams/conventions.md)).
  Where treams-rs differs, the [differences page](../coming-from-treams/differences.md)
  says how and why.
- **Native gradients.** Every differentiable operation has a hand-derived
  first-order pullback in Rust: a function that turns the gradient with respect to
  an output into gradients with respect to the inputs (see
  [analytic pullbacks](pullbacks.md)).
- **Speed.** Numerical loops run in compiled Rust. Independent items run in
  parallel when this reduces runtime (see [performance](../performance/index.md)).
- **A Python-free core.** The Rust crate `treams-core` depends on faer, nalgebra,
  Rayon and a few special-function crates, and on no Python package. `cargo test`
  checks it without an interpreter.

## Layers

| Layer | Code | Role |
|---|---|---|
| 1. Rust core | `crates/treams-core` | Numerical kernels and their analytic pullbacks |
| 2. Bindings | `crates/treams-py`, imported as `treams_rs._native` | Array conversion, input checks, the global interpreter lock (GIL), the floating-point guard and pullback contexts |
| 3. Python package | `treams_rs` | Physics objects (`TMatrix`, `Cluster`, `SMatrix`, waves), the treams-style numerical namespaces (`special`, `sw`, `cw`, `pw`, `lattice`, `coeffs`, `misc`, `ebcm`, `io`) and the records of `treams_rs.diff` |
| 4. Framework adapters | `treams_rs.advect`, `treams_rs.jax`, `treams_rs.torch`, `treams_rs.autograd` | Physics objects that hold framework arrays, and native records as framework operations |

The Python API selects optional adapters from its inputs. The core never sees a
Python object. The adapters reach native code only through the records of
`treams_rs.diff`. Inside the core, modules form six levels: numerical support
(`linalg`, `fpenv`, `numerics`), special functions, lattice sums, wave families,
scattering (`coeffs`, `tmatrix`, `ebcm`, `cluster`) and planar S-matrices
(`smatrix`). A module uses only its own and lower levels.

## Crosswalk

The table pairs Rust modules with their Python and treams counterparts. A dash
marks a module without a counterpart.

<!-- generated: rust-crosswalk -->

| Rust module | treams_rs namespace | treams | Contents |
|---|---|---|---|
| `special` | `treams_rs.special` | `treams.special` | Bessel, Legendre and Wigner functions, coordinate transforms, incomplete gamma and Kambe integrals |
| `lattice` | `treams_rs.lattice`, `treams_rs.misc.firstbrillouin*` | `treams.lattice`, `treams.misc.firstbrillouin*` | Lattice sums and lattice geometry |
| `sw` | `treams_rs.sw`, `treams_rs.special.tl_vsw_*` | `treams.sw`, `treams.special.tl_vsw_*` | Spherical modes, translation coefficients, expansion matrices and the conversion of chains to cylindrical waves |
| `cw` | `treams_rs.cw`, `treams_rs.special.tl_vcw` | `treams.cw`, `treams.special.tl_vcw` | Cylindrical modes, translation coefficients, expansion matrices and the conversion to spherical waves |
| `pw` | `treams_rs.pw`, `treams_rs.misc.wave_vec_z` | `treams.pw`, `treams.misc.wave_vec_z` | Plane-wave polarizations, fields and multipole expansions |
| `basis` | `treams_rs.SphericalBasis`, `treams_rs.CylindricalBasis` | `treams.SphericalWaveBasis`, `treams.CylindricalWaveBasis` | Multipole bases and the expansion gradients that both families share |
| `rotation` | `treams_rs.sw.rotate`, `treams_rs.cw.rotate`, `treams_rs.operators.Rotate` | `treams.sw.rotate`, `treams.cw.rotate`, `treams.Rotate` | Rotation coefficients and rotation matrices of multipole bases |
| `channels` | `treams_rs.sw.periodic_to_pw`, `treams_rs.cw.periodic_to_pw` | `treams.sw.periodic_to_pw`, `treams.cw.periodic_to_pw` | Plane-wave channels of periodic arrays |
| `vectorwaves` | `treams_rs.special.vsh_*`, `vsw_*`, `vcw_*`, `vpw_*`, `sph_harm` | `treams.special.vsh_*`, `vsw_*`, `vcw_*`, `vpw_*`, `sph_harm` | Vector spherical harmonics and vector spherical, cylindrical and plane waves |
| `fields` | `treams_rs.operators.efield` | `treams.efield` | Electric fields of multipole expansions at sample points |
| `coeffs` | `treams_rs.coeffs`, `treams_rs.Material`, `treams_rs.misc.refractive_index` | `treams.coeffs`, `treams.Material`, `treams.misc.refractive_index` | Mie coefficients of multilayer spheres and cylinders, and materials |
| `tmatrix` | `treams_rs.TMatrix`, `treams_rs.CylindricalTMatrix` | `treams.TMatrix`, `treams.TMatrixC` | T-matrices of single spheres and cylinders, and the cd, db and chi metrics |
| `ebcm` | `treams_rs.ebcm` | `treams.ebcm` | Q-matrices of the extended boundary condition method |
| `cluster` | `treams_rs.Cluster`, `treams_rs.iterative` | `treams.TMatrix.cluster`, `treams.TMatrix.interaction` | Multiple scattering in particle clusters, dense and matrix-free |
| `smatrix` | `treams_rs.SMatrix`, `treams_rs.coeffs.fresnel` | `treams.SMatrices`, `treams.coeffs.fresnel` | Planar S-matrices: interfaces, layer stacks, composition, illumination, transmittance and bands |
| `linalg` | `treams_rs.diff.solve`, `svdvals`, `eig` | - | Dense LU, SVD and eigensystems with pullbacks, and restarted GMRES |
| `numerics` | - | - | Forward-mode jets, broadcasting and parallel thresholds |
| `fpenv` | - | - | A guard that keeps subnormal numbers when the caller flushes them to zero |
| `threads` | `treams_rs.set_num_threads`, `threads`, `thread_info` | - | The thread budget and the fork-safe pool that runs every parallel region |

<!-- end generated -->

## Who owns what

- **Rust core:** all numerical work and every derivative. A Rust function checks
  the physical validity of its inputs and returns a typed `Error`.
- **Bindings:** conversion between NumPy arrays and Rust types, releasing the GIL
  during numerical work, the floating-point guard of every entry point (see
  [floating-point environment](floating-point.md)) and the context objects that hold
  one-use pullback data. A Rust error becomes a `ValueError`, except
  `Error::OutOfMemory`, which becomes a `MemoryError`. Some allocation paths
  still abort on failure ([memory limits](parallelism.md#audit-and-open-work)).
- **Python package:** bases, `k0`, media, polarization conventions, lattices,
  Bloch vectors, named results and the numerical calls behind physics methods.
- **Framework adapters:** composition only. They wrap native records as Advect,
  JAX, PyTorch or HIPS Autograd operations and add no numerical code (see
  [framework adapters](adapters.md)).

The bindings keep the GIL for scalar calls of microsecond-scale functions, such as
one Bessel value: releasing and taking it back costs about 0.09 µs, 5-9% of such a
call.

## What treams-rs leaves out

- **Array annotations.** treams attaches basis, `k0` and material to arrays and
  carries them through NumPy arithmetic (`.ann`, `.relax`). treams-rs objects keep
  them as attributes, and plain arithmetic returns plain NumPy values (see
  [Python API design](python-api.md)).
- **Global configuration.** treams reads `treams.config.POLTYPE`. treams-rs takes
  the polarization convention as an argument, `helicity` by default.
- **Fallback to treams.** treams-rs never imports treams. An unsupported case
  raises an error. treams serves only as a test reference.

## Dense and matrix-free cluster solves

A dense solve with requested illuminations factors `I - T C` once and solves only
the requested right-hand sides. Here `T` holds the particle T-matrices and `C` the
translations between particles. The factor never changes, so any number of
pullback records share it (`cluster::InteractionFactor`).

The matrix-free solver (`cluster::IterativeSphereCluster`, Python
`iterative.SphereCluster`) applies the pair translations on demand and solves
the forward and the adjoint system with restarted GMRES. It sums the gradients
of radii, positions and materials pair by pair, so it never forms the coupling
matrix or its gradient. Both paths share the Mie coefficients, the translation
plans and the gradient convention. [Large clusters](../guide/large-clusters.md)
explains when to use which.
