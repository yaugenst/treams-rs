# Implementation status

Active rewrite, not complete treams parity. There is no runtime dependency or
fallback to treams, SciPy, Cython, or a Python autodiff framework.

| Subsystem | Implemented and checked | Remaining |
| --- | --- | --- |
| Project | Cargo/PyO3/maturin/uv, lockfiles, just, Ruff, strict Pyrefly, Clippy, pre-commit; hosted Linux CI passing on Python 3.12 and 3.13 | Broader packaged-platform qualification |
| Spherical functions | Complex regular/outgoing radial values and first two derivatives; Legendre functions; Wigner 3j; Cartesian harmonics | Wider extreme-argument/order qualification; full special namespace |
| Sphere coefficients | Multilayer, lossy, magnetic, chiral Mie; all continuous input VJPs | Extreme-layer-conditioning analysis |
| Wave expansion | Regular/outgoing, helicity/parity, arbitrary spherical bases, axial and coincident regular origins; position/complex-wavenumber VJPs; spherical Euler and cylindrical axis rotations with native angle pullbacks; regular cylindrical-to-spherical conversion and native origin/wavenumber VJPs | Remaining wave-family conversions; plane-wave basis rotations |
| Multipole fields | Spherical/cylindrical Cartesian waves and analytic axis limits; weighted fields and full field operators with native position/wavenumber VJPs and linear residuals; electric, magnetic, displacement and flux operators; Advect magnetic samples including impedance gradients; native weighted/full plane fields and complex-wavevector VJPs | Riemann-Silberstein fields, cylindrical axial-label derivatives and upstream operator-attribute machinery |
| Finite scattering | Dense solve and factorization-reusing adjoint; optimized sphere clusters; heterogeneous local matrices via public API | Native end-to-end heterogeneous-cluster parameter context |
| Python interface | Material, spherical/cylindrical bases, TMatrix.sphere, TMatrixC.cylinder, clusters, interaction.solve, changepoltype, expand, xs/xw and averaged cross sections | Full upstream ndarray annotation machinery is not reproduced; explicit .array is used |
| Differentiation | Opaque one-use native contexts in coeffs and diff; arbitrary complex output cotangents; Advect adapters for spherical/cylindrical T-matrices, clusters, interactions, expansions, fields and sphere/cylinder coefficients | Higher derivatives and other framework adapters |
| Testing | Native proptest invariants and adjoint identities; Hypothesis physical invariants; treams/SciPy reference comparisons; complete Python workflows | Expand qualification with every ported subsystem |
| Performance | Cached angular plans and radial tables, faer LU and matmul, block-diagonal local storage, Rayon coupling assembly | See measured scope and limitations in benchmarks.md |
| Cylindrical scattering | Complex J/H and derivatives; multilayer chiral coefficients and complete T-matrix with all parameter VJPs; cylindrical bases, translations, clusters, electric fields and cross widths; regular spherical conversion; periodic plane-wave radiation and adjoints | Broader cutoff qualification |
| Plane-wave illumination | Real/complex directions, scalar/helicity/Cartesian polarization inputs, native spherical/cylindrical conversion, direct T-matrix illumination and cross sections; full unit-vector and xy/yz/zx component plane bases, diffraction orders, native Cartesian fields and slab illumination; spherical/cylindrical illumination VJPs | Broader constrained-incidence workflows |
| Planar layers | Native chiral Fresnel coefficients and propagation; one-LU S-matrix composition with reused-factor adjoint; interfaces, multilayer slabs, stacking/doubling, polarization conversion and power-flux transmittance/reflectance | Oriented interfaces, internal-field convenience; full SMatrix annotation API |
| Periodic scattering | Spherical Ewald sums in 1D/2D/3D and cylindrical sums in 1D/2D; periodic coupling and solves; spherical 2D and cylindrical 1D particle-to-plane channels and S matrices; complete native pullbacks and Advect reflectance gradients; direct-sum, reference, energy, Bloch/split/scale invariants | Broader combined particle/layer workflows |
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

Component plane bases support xy, yz and zx alignments for fields and illumination.
Fresnel interfaces currently require xy alignment. Propagation, composition,
illumination and Cartesian power-flux calculations support all three alignments,
with one incident amplitude vector or PlaneWave object per direction. Arrays are
explicit rather than inheriting upstream's ndarray metadata. Fresnel's low-level
API currently evaluates one (two-media, two-helicity) interface at a time.

`SMatrices.from_array(tm, basis, lattice=..., kpar=...)` accepts an uncoupled
spherical 2D xy or cylindrical 1D x unit cell and solves its periodic interaction
before radiating. Cylindrical ports use zx alignment, with up/down along y. This
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

`rotate`, `TMatrix.rotate`, `TMatrixC.rotate`, `diff.rotation` and
`advect.rotation` support local multipole rotations. Spherical rotations use the
full z-y-z Euler convention and differentiate all three angles; cylindrical theta
is constrained to zero. Expansion origins remain fixed. Complete angular blocks
come from a symmetric angular-momentum eigensystem, avoiding factorial-sum
cancellation. The reverse pass uses angular generators without retaining three
full output Jacobians. Tests check reference values, arbitrary rectangular basis
subsets, inverse/composition identities, angle pullbacks, unitarity through degree
60 and first-order terms at angles down to 1e-100.

The public `efield`, `hfield`, `dfield` and `bfield` functions return explicit
Cartesian operator arrays with shape (..., 3, modes) for either multipole family.
Their conventions match treams in lossy, magnetic and chiral media. The native
`diff.field_operator` pullback covers points, origins and complex wavenumbers,
retaining geometry only; `advect.field_operator` composes the same boundary.
`advect.hfield` uses the weighted native field path, composing the elementary
polarization/impedance scaling in Advect. The metadata-bearing convenience
functions themselves accept ordinary arrays. Prefer weighted `diff.field` or
`advect.field`/`hfield` when the full sample-by-mode operator is not needed.

`expand((destination, source), ...)` now exposes multipole addition theorems and
regular cylindrical-to-spherical conversion. The latter includes all explicit
origin pairs by translating the cylindrical wave before its spherical expansion;
upstream's same-origin angular coefficients and reconstructed displaced fields
are independently checked. It does not reproduce upstream's particle-index mask
for this cross-family operator. `diff.expansion` and `advect.expansion` share its
native origin and complex-wavenumber pullback, holding axial labels fixed.
On-axis evaluation skips azimuthal orders that vanish analytically, retaining
adjacent orders in reverse because their first position derivatives can be nonzero.

`diff.plane_field` and `advect.plane_field` evaluate weighted plane-wave samples or
full operators (`coefficients=None`). Pullbacks cover amplitudes, real Cartesian
points and each full complex wavevector, including its polarization dependence.
The full operator and weighted path share their native forward/pullback; the
residual retains inputs only. Polarization adjoints are accumulated over samples
and differentiated once per mode. At exactly axial propagation the upstream
polarization gauge has no direction derivative: `fixed_vectors=True` treats the
vectors as constants while enabling amplitude and point derivatives.

The root E/H/D/B field operators accept all component and unit-vector plane bases and up/down
propagation. `PlaneWave.efield` uses the weighted native kernel. All field
operators now transfer owned Rust buffers to NumPy without copying; their strides
need not be C-contiguous. Cotangent checks cover contiguous, permuted and reversed
views, and malformed cotangents leave the one-use residual available for retry.

`expand` also maps component and unit-vector plane bases into regular spherical bases.
`diff.plane_expansion` and `advect.plane_expansion` expose arbitrary complex plane
wavevectors and native origin/vector pullbacks. A complete Advect test includes
incidence angles, frequency, sphere radius and position through illumination,
scattering and total-field intensity. The angular coefficient uses the same
transverse branch as Cartesian polarization; this fixes a physical reconstruction
failure for complex directions where upstream's additional principal square root
flips that branch. For example, k=(0.2+i, 0.1+0.3i, 1.3-0.8i) reconstructs to
about 1e-13 absolute error, whereas upstream coefficients err by about 1.25 on the
same samples. Tests cover both polarizations, helicity/parity and native adjoint
scale identities. Fixed-vector mode enables origin gradients at axial incidence.

`PlaneWaveBasisByUnitVector` provides complete complex directions with stable
algebraic normalization. `PlaneWaveBasisByComp` supports xy, yz and zx alignments;
`kvecs` is the shared source of truth for the missing component, including evanescent
waves and up/down propagation. `byunitvector`, `bycomp` and cyclic `permute` are
reference- and round-trip checked. The E/H/D/B operators, spherical illumination
and `PlaneWave.expand` accept these bases. Plane-wave directions remain unchanged
under input scales from 1e-300 to 1e300. These Python metadata conveniences compose
the previously checked native operations; tracked parameters use the explicit
Advect boundaries. Arbitrarily oriented slab interfaces are still pending.

Cylindrical radiation channels have native origin, complex-wavenumber, transverse
wavevector and period pullbacks. Axial wavenumbers remain fixed mode labels; exact
label matching selects each channel. Advect tests differentiate complete periodic
cylinder reflectance through radius, frequency, period, Bloch vector and origins.
Independent image-field reconstruction and lossless power conservation cover both
sides and polarization conventions. Equal medium wavenumbers share cylindrical
Ewald evaluation without merging their separate wavenumber gradients. A large-cell
regression checks split independence against a converged reference; upstream's
automatic split loses accuracy for that case.

`diff.plane_expansion` and `advect.plane_expansion` now share the same native
forward/pullback for spherical and cylindrical destinations. Cylindrical vector
axial components are exact, fixed labels with zero cotangents; transverse complex
components and every origin component are differentiated. Tests cover up/down,
helicity/parity, lossy/chiral media, field reconstruction, scale invariants and
complete Advect cylinder illumination, scattering and total-field intensity.
Origin phases are computed once per plane/origin pair in both directions rather
than repeated for every multipole. The matrix buffer transfers directly to NumPy;
the residual holds only inputs.
