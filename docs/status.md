# Implementation status

Active rewrite, not complete treams parity. There is no runtime dependency or
fallback to treams, SciPy, Cython, or a Python autodiff framework.

| Subsystem | Implemented and checked | Remaining |
| --- | --- | --- |
| Project | Cargo/PyO3/maturin/uv, lockfiles, just, Ruff, strict Pyrefly, Clippy, pre-commit, CI definition | Hosted CI and packaged-platform qualification |
| Spherical functions | Complex regular/outgoing radial values and first two derivatives; Legendre functions; Wigner 3j; Cartesian harmonics | Wider extreme-argument/order qualification; full special namespace |
| Sphere coefficients | Multilayer, lossy, magnetic, chiral Mie; all continuous input VJPs | Extreme-layer-conditioning analysis |
| Wave expansion | Regular/outgoing, helicity/parity, arbitrary spherical bases, axial and coincident regular origins; position/complex-wavenumber VJPs | Rotation operators and cylindrical/plane-wave conversions |
| Finite scattering | Dense solve and factorization-reusing adjoint; optimized sphere clusters; heterogeneous local matrices via public API | Native end-to-end heterogeneous-cluster parameter context |
| Python interface | Material, SphericalWaveBasis, TMatrix.sphere, TMatrix.cluster, interaction.solve, changepoltype, expand, xs and averaged cross sections | Full upstream ndarray annotation machinery is not reproduced; explicit .array is used |
| Differentiation | Opaque one-use native contexts in coeffs and diff; arbitrary complex output cotangents; Autograd adapters for spheres, clusters, interactions and sphere/cylinder coefficients | Higher derivatives and other framework adapters |
| Testing | Native proptest invariants and adjoint identities; Hypothesis physical invariants; treams/SciPy reference comparisons; complete Python workflows | Expand qualification with every ported subsystem |
| Performance | Cached angular plans and radial tables, faer LU and matmul, block-diagonal local storage, Rayon coupling assembly | See measured scope and limitations in benchmarks.md |
| Cylindrical scattering | Complex J/H values and first two derivatives; multilayer chiral cylinder coefficients and all continuous input VJPs; reference and invariant checks | Cylindrical basis, T-matrix interface, fields and translation |
| Planar scattering | Not implemented | Plane-wave conversion, interfaces, layered S matrices and VJPs |
| Periodic scattering | Not implemented | 1D/2D/3D lattice sums, periodic coupling and derivatives |
| Remaining public API | Not implemented | Fields, EBCM, band calculations, I/O and remaining observables |

The optimized `diff.cluster` is restricted to non-overlapping homogeneous,
nonmagnetic spheres in vacuum with a common multipole cutoff. Its pullback covers
radii, positions, complex sphere permittivities and vacuum wavenumber. The public
`TMatrix.cluster` accepts general local spherical T-matrices with distinct cutoffs
and a common, possibly chiral, embedding material.

Tests exercise degrees through 30 for radial functions, through 10 for Mie, and
smaller orders for full derivative/cluster checks. An input bound of 128 does not
constitute a claim of accuracy throughout that range. Dense solve storage still
scales quadratically and factorization cubically with the multipole dimension.

The Python package is `treams_rs` during development so the upstream oracle can
coexist in the same environment. It is not yet a drop-in `import treams` replacement.
Python 3.12 and 3.13 are the qualification targets. Numerical observables and linear
basis composition currently use NumPy in the thin Python layer; native kernels own
the special functions, particle scattering and multiple-scattering solve.
