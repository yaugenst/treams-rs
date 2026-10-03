---
description: Implemented physics, derivatives and limits, compared with treams.
---

# Capabilities

treams-rs implements the numerical functions of treams 0.4.7 in Rust, at the
treams commit named on the [conventions](../coming-from-treams/conventions.md)
page. treams-rs never calls treams, SciPy or Cython at run time. Its Python API
is built around physical objects and does not copy the treams source API.
[Differences from treams](../coming-from-treams/differences.md) lists the
results that differ on purpose, including defects of treams 0.4.7.

The treams inventory has 182 public functions and classes across the package and
its numerical, configuration and I/O modules. Matching those names checks
coverage; tests also compare complete workflows and independent physical laws
([validation](index.md)).

## Numerical and workflow coverage

A pullback maps the gradient of a loss with respect to an output to the gradients
with respect to the inputs ([differentiation](../differentiation/index.md)).

| Area | Implemented behavior and derivatives |
| --- | --- |
| Materials and geometry | Isotropic lossy, magnetic and chiral materials; outgoing wavenumber branches; Cartesian/polar/spherical transforms and vector-frame changes; native continuous geometry pullbacks. Immutable Lattice and partial WaveVector metadata, reciprocal cells, diffraction orders and coordinate transforms. |
| Special functions | Cylindrical/spherical Bessel and incoming/outgoing Hankel values and derivatives; integer Legendre, pi/tau and harmonics; fractional-degree real Legendre; Wigner 3j and small/full D; incomplete gamma and Kambe integrals. NumPy broadcasting, output buffers and native argument/angle pullbacks. |
| Local waves and coefficients | Spherical, cylindrical and plane waves; helicity/parity bases; regular/singular translation coefficients; sw/cw/pw namespaces; analytic axis limits and continuous coordinate/wavenumber pullbacks. |
| Particle scattering | Multilayer chiral sphere and cylinder coefficients and T matrices; radius, permittivity, permeability, chirality, frequency and cylindrical axial-wavenumber pullbacks. |
| Finite interactions | Homogeneous optimized sphere clusters and heterogeneous spherical/cylindrical local T matrices, differing cutoffs and mode subsets. Native block storage, interaction solves and LU-reusing pullbacks for local matrices, positions and embedding wavenumbers. |
| Basis operations | Rotations, translations, polarization changes, finite and periodic expansion, plane coordinate permutations and spherical/cylindrical conversions. Fixed discrete mode labels with native continuous parameter pullbacks. |
| Fields and illumination | Electric, magnetic, displacement, flux and G/F field operators; weighted samples; spherical/cylindrical sources and plane waves, including complex directions. Native amplitude, sample, position and wavenumber pullbacks. Direct T-matrix illumination, cross sections/widths and source expansion. |
| Periodic sums | Spherical 1D/2D/3D and cylindrical 1D/2D Ewald sums, full/real/reciprocal/direct-shell APIs, shifted geometries and batched inputs. Native k, Bloch vector, cell, shift and split-parameter pullbacks. The Ewald parts run over reduced direct and reciprocal bases with the Bloch vector reduced modulo the reciprocal lattice, so they do not depend on the primitive basis or Bloch cell; direct shells keep the given basis. 1D spherical sums off the axis take their spectral series where the Ewald sum cancels ([lattice limits](numerical-limits.md#lattice-sums)). |
| Custom periodic tables | User-supplied spherical lattice-sum callbacks are evaluated once with broadcast geometry. Native table-to-matrix contraction and its conjugate-transpose pullback compose with differentiable lattice sums. |
| Periodic scattering | Coupling, interactions, spherical 2D and cylindrical 1D particle-to-plane channels, particle-array S matrices and spherical-array-to-cylinder conversion. Complete native pullbacks, including shared cylindrical axial groups. Channels agree natively with plane-wave expansions, including evanescent orders, and obey reciprocity; spherical and cylindrical channels reject non-finite entries (for example a strongly evanescent order far from the axis). |
| Planar layers | Chiral Fresnel interfaces, propagation, compact multilayer stacks, S-matrix composition and doubling. Native factorization-reusing pullbacks and internal illumination between adjacent stacks. |
| Power and dichroism | Native batched S-matrix transmittance/reflectance includes coherent incident/reflected interference in absorbing media. Pullbacks cover the illuminated S-matrix column, incident amplitudes, both port wavenumbers/impedances and transverse directions. Circular dichroism evaluates both incident polarizations in one batch and has an Advect adapter. |
| Other observables | T-matrix CD, duality breaking and electromagnetic chirality, thin SVD and native pullbacks. Plane chirality-density forms in all coordinate orientations, with interval and geometry pullbacks. |
| Bloch bands | Native transfer matrices, right eigensystems and Bloch wavenumbers/vectors, with S-matrix, period and nondegenerate eigenvector pullbacks. |
| Axisymmetric EBCM | Callable radial surfaces sampled by Gauss-Legendre quadrature, native regular/singular Q integrals and radius, slope, complex-wavenumber and impedance pullbacks. Correct surface area by default; an explicit option reproduces the treams surface element for comparisons. |
| Python objects | Material, all four basis types, TMatrix/CylindricalTMatrix, SMatrix/ScatteringBlock, typed waves, unsolved clusters and solved periodic responses. Physical scattering, fields, named observables and explicit numerical operators; basis selections, coordinate transforms and read-only port block views. |
| I/O | Optional HDF5 scalar matrices and rectangular sweeps, streamed writes, chirality, positions, mode indices, units, mesh and reproducibility metadata. Gmsh convenience uses actual boundary surface tags. |

## treams inventory

Size of treams at the pinned commit, in physical lines of `src/treams`, without
notebooks, documentation and generated C:

| Source | Files | Lines |
| --- | ---: | ---: |
| Python | 13 | 8,714 |
| Cython (20 `.pyx`, 13 `.pxd`) | 33 | 13,995 |
| Total package source | 46 | 22,709 |
| Tests (730 test definitions) | 16 | 6,005 |

Three repetitive ufunc/gufunc wrapper files account for 6,505 of the source
lines. The package covers spherical, cylindrical and plane waves, multilayer and
chiral materials, translation and rotation algebra, periodic Ewald sums in
several dimensions, S-matrix composition, fields, EBCM, observables and basis
metadata.

Accuracy depends on numerical conventions and conditioning: complex square-root
branches, helicity ordering, normalization, small-argument limits, multipole
cancellation and lattice convergence. Each needs its own tests. The
Python API passes explicit arrays and objects, so physical metadata never travels
silently through NumPy operations ([Python API design](../design/python-api.md)).

## Scope

- **CPU only.** No GPU or other accelerator support.
- **First-order derivatives.** Native pullbacks compute the derivative of the
  numerical solve; the framework adapters for Advect, JAX, PyTorch and HIPS Autograd compose them
  with a user's objective. A native pullback gives no higher-order derivatives,
  batching or accelerator support.
- **Explicit physical metadata.** Physics objects carry their own basis and
  material information. treams-rs does not copy treams' array annotation system.
- **HDF5 files.** Layout compatibility is not certification against every
  external T-matrix database.
- **CPython 3.12–3.15.** The release targets these Python versions; optional
  framework and HDF5 support varies by interpreter. See [Install](../getting-started/install.md).
