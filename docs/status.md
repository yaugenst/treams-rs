# Implementation status

Active rewrite, not complete treams parity. There is no runtime dependency or
fallback to treams, SciPy, Cython, or a Python autodiff framework.

| Subsystem | Implemented and checked | Remaining |
| --- | --- | --- |
| Project | Cargo/PyO3/maturin/uv, lockfiles, just, Ruff, strict Pyrefly, Clippy, pre-commit; hosted Linux CI passing on Python 3.12 and 3.13 | Broader packaged-platform qualification |
| Spherical functions | Complex regular/outgoing radial values and first two derivatives; Legendre functions; Wigner 3j; Cartesian harmonics | Wider extreme-argument/order qualification; full special namespace |
| Sphere coefficients | Multilayer, lossy, magnetic, chiral Mie; all continuous input VJPs | Extreme-layer-conditioning analysis |
| Wave expansion | Regular/outgoing, helicity/parity, arbitrary spherical bases, axial and coincident regular origins; position/complex-wavenumber VJPs | Rotation operators and conversions between wave families |
| Spherical fields | Cartesian regular/outgoing vector waves, origin and polar-axis limits, position and complex-wavenumber derivatives; batched fields and all continuous-input VJPs with linear residual storage; Maxwell and reference checks | Magnetic fields and upstream field-operator convenience API |
| Finite scattering | Dense solve and factorization-reusing adjoint; optimized sphere clusters; heterogeneous local matrices via public API | Native end-to-end heterogeneous-cluster parameter context |
| Python interface | Material, spherical/cylindrical bases, TMatrix.sphere, TMatrixC.cylinder, clusters, interaction.solve, changepoltype, expand, xs/xw and averaged cross sections | Full upstream ndarray annotation machinery is not reproduced; explicit .array is used |
| Differentiation | Opaque one-use native contexts in coeffs and diff; arbitrary complex output cotangents; Advect adapters for spherical/cylindrical T-matrices, clusters, interactions, expansions, fields and sphere/cylinder coefficients | Higher derivatives and other framework adapters |
| Testing | Native proptest invariants and adjoint identities; Hypothesis physical invariants; treams/SciPy reference comparisons; complete Python workflows | Expand qualification with every ported subsystem |
| Performance | Cached angular plans and radial tables, faer LU and matmul, block-diagonal local storage, Rayon coupling assembly | See measured scope and limitations in benchmarks.md |
| Cylindrical scattering | Complex J/H and derivatives; multilayer chiral coefficients and complete T-matrix with all parameter VJPs; cylindrical bases, translations, clusters and cross widths | Cylindrical fields and conversions to spherical/plane waves |
| Plane-wave illumination | Real/complex directions, scalar/helicity/Cartesian polarization inputs, native spherical/cylindrical conversion, direct T-matrix illumination and cross sections; xy-component plane-wave bases and slab illumination | General basis alignments, diffraction-order generation and plane-wave direction/material VJPs |
| Planar layers | Native chiral Fresnel coefficients and propagation; one-LU S-matrix composition with reused-factor adjoint; interfaces, multilayer slabs, stacking/doubling, polarization conversion and power-flux transmittance/reflectance | Periodic particle-to-plane-wave channels; internal-field convenience; full SMatrix annotation API |
| Periodic scattering | Spherical Ewald sums in 1D/2D/3D and cylindrical sums in 1D/2D, displaced sources, periodic coupling and interaction solves; all continuous-input native pullbacks and Advect composition; direct-sum, reference and Bloch/split/scale invariants | S-matrix channels and reflection/transmission |
| Remaining public API | Not implemented | Field-operator conveniences, EBCM, band calculations, I/O and remaining observables |

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

Cylindrical T-matrix pullbacks differentiate the common input axial wavenumbers.
Cylindrical basis expansion treats axial wavenumbers as fixed mode labels because
unequal labels decouple exactly; its public VJP covers origins and medium
wavenumbers. Exact cylindrical cutoffs require a limiting formulation and are
explicitly unsupported.

Periodic sums currently broadcast multipole indices for one lattice geometry. They
reject exact diffraction thresholds rather than replacing singularities with an
arbitrary finite constant. `latticeinteraction.solve` returns an array of periodic
response coefficients; isolated-particle cross-section formulae do not apply.

Periodic pullbacks cover both sets of expansion origins, complex medium
wavenumbers, the real Bloch wavevector and every lattice-vector component.
The split parameter eta is held fixed because the exact sum is independent of it.
At coincident origins, derivatives use the regular image sum with that lattice
point excluded under perturbation. Cylindrical axial labels remain fixed.
The core propagates analytic local chain rules through at most sixteen continuous
Ewald parameters, with derivative arithmetic compiled out of forward-only calls.
No full output-by-parameter Jacobian is stored in the residual.

Planar interfaces, propagation and S-matrix composition have native pullbacks and
Advect adapters (`fresnel`, `propagation`, `smatrix_add`). Fresnel pullbacks cover all
complex full/axial wavenumbers and impedances for one interface; propagation covers
complex wavevectors and real Cartesian displacement; composition covers both full
four-block arrays. Material-to-wavevector arithmetic can be composed in Advect,
as checked by complete slab gradients. The metadata-bearing SMatrices convenience
class itself accepts ordinary arrays rather than tracked parameters.

The new plane-wave basis fixes the transverse plane to xy. Slab power calculations
accept one incident amplitude vector or PlaneWave object per direction. Arrays are
explicit rather than inheriting upstream's ndarray metadata. Fresnel's low-level
API currently evaluates one (two-media, two-helicity) interface at a time.
