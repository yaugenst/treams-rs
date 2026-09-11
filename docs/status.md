# Implementation status

Active rewrite, not complete treams parity. There is no runtime dependency or
fallback to treams, SciPy, Cython, or a Python autodiff framework.

| Subsystem | Implemented and checked | Remaining |
| --- | --- | --- |
| Project | Cargo/PyO3/maturin/uv, lockfiles, just, Ruff, strict Pyrefly, Clippy, pre-commit; hosted Linux CI passing on Python 3.12 and 3.13 | Broader packaged-platform qualification |
| Spherical functions | Complex regular/outgoing radial values and first two derivatives; Legendre functions; Wigner 3j; Cartesian harmonics | Wider extreme-argument/order qualification; full special namespace |
| Sphere coefficients | Multilayer, lossy, magnetic, chiral Mie; all continuous input VJPs | Extreme-layer-conditioning analysis |
| Wave expansion | Regular/outgoing, helicity/parity, arbitrary spherical bases, axial and coincident regular origins; position/complex-wavenumber VJPs | Rotation operators and conversions between wave families |
| Multipole fields | Spherical and cylindrical Cartesian regular/outgoing vector waves; analytic axis limits, position and complex-wavenumber derivatives; batched fields with native VJPs and linear residual storage; Maxwell and reference checks | Magnetic fields, cylindrical axial-label derivatives and upstream field-operator convenience API |
| Finite scattering | Dense solve and factorization-reusing adjoint; optimized sphere clusters; heterogeneous local matrices via public API | Native end-to-end heterogeneous-cluster parameter context |
| Python interface | Material, spherical/cylindrical bases, TMatrix.sphere, TMatrixC.cylinder, clusters, interaction.solve, changepoltype, expand, xs/xw and averaged cross sections | Full upstream ndarray annotation machinery is not reproduced; explicit .array is used |
| Differentiation | Opaque one-use native contexts in coeffs and diff; arbitrary complex output cotangents; Advect adapters for spherical/cylindrical T-matrices, clusters, interactions, expansions, fields and sphere/cylinder coefficients | Higher derivatives and other framework adapters |
| Testing | Native proptest invariants and adjoint identities; Hypothesis physical invariants; treams/SciPy reference comparisons; complete Python workflows | Expand qualification with every ported subsystem |
| Performance | Cached angular plans and radial tables, faer LU and matmul, block-diagonal local storage, Rayon coupling assembly | See measured scope and limitations in benchmarks.md |
| Cylindrical scattering | Complex J/H and derivatives; multilayer chiral coefficients and complete T-matrix with all parameter VJPs; cylindrical bases, translations, clusters, electric fields and cross widths | Conversions to spherical/plane waves |
| Plane-wave illumination | Real/complex directions, scalar/helicity/Cartesian polarization inputs, native spherical/cylindrical conversion, direct T-matrix illumination and cross sections; xy-component plane-wave bases, diffraction orders and slab illumination | General basis alignments and standalone PlaneWave direction/material VJPs |
| Planar layers | Native chiral Fresnel coefficients and propagation; one-LU S-matrix composition with reused-factor adjoint; interfaces, multilayer slabs, stacking/doubling, polarization conversion and power-flux transmittance/reflectance | Internal-field convenience; full SMatrix annotation API |
| Periodic scattering | Spherical Ewald sums in 1D/2D/3D and cylindrical sums in 1D/2D; periodic coupling and solves; spherical 2D particle-to-plane channels and S matrices; complete native pullbacks and Advect reflectance gradients; direct-sum, reference, energy, Bloch/split/scale invariants | Cylindrical radiation channels; broader combined particle/layer workflows |
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

`diff.field` and `advect.field` accept either spherical or cylindrical bases.
Cylindrical field pullbacks cover amplitudes, sample points, expansion origins and
both medium wavenumbers, holding axial labels fixed. Adjacent cylindrical orders
reuse one Bessel evaluation and its first two radial derivatives; regular near-axis
fields and translations use Cartesian series. Reference, Maxwell, native adjoint
and complete cylinder-to-field Advect checks cover this path. Near-axis gradient
limits are checked through coordinate offsets of 1e-300.

Periodic sums currently broadcast multipole indices for one lattice geometry. They
reject exact diffraction thresholds rather than replacing singularities with an
arbitrary finite constant. `latticeinteraction.solve` returns an array of periodic
response coefficients; isolated-particle cross-section formulae do not apply.

Periodic pullbacks cover both sets of expansion origins, complex medium
wavenumbers, the real Bloch wavevector and every lattice-vector component.
The split parameter eta is held fixed because the exact sum is independent of it.
At coincident origins, derivatives use the regular image sum with that lattice
point excluded under perturbation. Cylindrical axial labels remain fixed.
The core propagates analytic local chain rules through six, ten or sixteen
continuous Ewald parameters for 1D, 2D or 3D, with derivative arithmetic compiled
out of forward-only calls. Equal medium wavenumbers share derivative evaluation
between polarization cotangents.
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

`SMatrices.from_array(tm, basis, lattice=..., kpar=...)` accepts an uncoupled
spherical unit cell and solves its periodic interaction before radiating. This
explicit constructor differs from upstream's annotated, already-interacting
T-matrix input. The native `spherical_channels` and `smatrix_from_array` boundaries
support arbitrary complex cotangents. Advect tests differentiate reflected power
through particle radii, positions, complex permittivities, frequency, Bloch vector
and every 2D cell component, including the moving diffraction orders.

At exactly normal incidence, the upstream plane-wave polarization convention has
no defined azimuth. Forward values preserve its convention; `fixed_q=True` enables
all other channel derivatives while treating transverse directions as constants.
A requested direction derivative there raises an explicit error. Near-normal,
evanescent, chiral and lossy channels are reference checked. Diffraction-order
generation includes all reciprocal vectors inside the cutoff, including skew
cells where upstream's simple iterator can omit orders.
