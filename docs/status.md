# Implementation status

Active rewrite, not complete treams parity. There is no runtime dependency or
fallback to treams, SciPy, Cython, or a Python autodiff framework.

Known reference defects and accuracy limits are indexed in [upstream findings](upstream-findings.md).

`chirality_density` returns up/down/coherent-cross forms for 2 Re(E* . i Z H)
averaged over an interval along z, for xy plane bases. It fixes upstream's
attenuation average, shifted observation plane and missing coherent cross phase.
The cross form can be complex and contracts as Re(down* X up).
`diff.chirality_density` and `advect.chirality_density` use compact (3, modes)
coefficients before polarization weighting; their native contexts retain only
full/normal wavenumbers and interval endpoints. They are checked against direct
Cartesian E/H quadrature in lossy/chiral and evanescent cases. Other alignments
need their actual polarization vectors in the forms and are explicitly unsupported.

| Subsystem | Implemented and checked | Remaining |
| --- | --- | --- |
| Project | Cargo/PyO3/maturin/uv, lockfiles, just, Ruff, strict Pyrefly, Clippy, pre-commit; hosted Linux CI passing on Python 3.12 and 3.13 | Broader packaged-platform qualification |
| Spherical functions | Complex regular/outgoing radial values and first two derivatives; Legendre functions; Wigner 3j; Cartesian harmonics | Wider extreme-argument/order qualification; full special namespace |
| Sphere coefficients | Multilayer, lossy, magnetic, chiral Mie; all continuous input VJPs | Extreme-layer-conditioning analysis |
| Wave expansion | Regular/outgoing, helicity/parity, arbitrary spherical bases, axial and coincident regular origins; position/complex-wavenumber VJPs; spherical Euler and cylindrical axis rotations with native angle pullbacks; regular cylindrical-to-spherical and periodic spherical-to-cylindrical conversion with native pullbacks; explicit expandlattice dispatch | Remaining wave-family conversions; plane-wave basis rotations |
| Multipole fields | Spherical/cylindrical Cartesian waves and analytic axis limits; weighted fields and full field operators with native position/wavenumber VJPs and linear residuals; electric, magnetic, displacement, flux and Riemann-Silberstein operators; Advect magnetic and G/F samples; native weighted/full plane fields and complex-wavevector VJPs | Cylindrical axial-label derivatives and upstream operator-attribute machinery |
| Finite scattering | Dense solve and factorization-reusing adjoint; optimized sphere clusters; heterogeneous local matrices via public API | Native end-to-end heterogeneous-cluster parameter context |
| Python interface | Material, spherical/cylindrical bases, TMatrix.sphere, TMatrixC.cylinder, clusters, interaction.solve, changepoltype, expand, xs/xw and averaged cross sections; explicit spherical/cylindrical sources with weighted E/H/D/B/G/F fields and direct T-matrix illumination | Full upstream ndarray annotation machinery is not reproduced; explicit .array is used |
| Differentiation | Opaque one-use native contexts in coeffs and diff; arbitrary complex output cotangents; Advect adapters for spherical/cylindrical T-matrices, clusters, interactions, expansions, fields and sphere/cylinder coefficients | Higher derivatives and other framework adapters |
| Testing | Native proptest invariants and adjoint identities; Hypothesis physical invariants; treams/SciPy reference comparisons; complete Python workflows | Expand qualification with every ported subsystem |
| Performance | Cached angular plans and radial tables, faer LU and matmul, block-diagonal local storage, Rayon coupling assembly | See measured scope and limitations in benchmarks.md |
| Cylindrical scattering | Complex J/H and derivatives; multilayer chiral coefficients and complete T-matrix with all parameter VJPs; cylindrical bases, translations, clusters, electric fields and cross widths; regular spherical conversion; periodic plane-wave radiation and adjoints | Broader cutoff qualification |
| Plane-wave illumination | Real/complex directions, scalar/helicity/Cartesian polarization inputs, native spherical/cylindrical conversion, direct T-matrix illumination and cross sections; full unit-vector and xy/yz/zx component plane bases, diffraction orders, native Cartesian fields and slab illumination; spherical/cylindrical illumination VJPs | Broader constrained-incidence workflows |
| Planar layers | Native chiral Fresnel coefficients and propagation; one-LU S-matrix composition with reused-factor adjoint; interfaces, multilayer slabs, stacking/doubling, polarization conversion, power-flux transmittance/reflectance and internal fields between adjacent stacks | Full SMatrix annotation API |
| Periodic scattering | Spherical Ewald sums in 1D/2D/3D and cylindrical sums in 1D/2D; periodic coupling and solves; spherical 2D and cylindrical 1D particle-to-plane channels and S matrices; complete native pullbacks and Advect reflectance gradients; direct-sum, reference, energy, Bloch/split/scale invariants | Broader combined particle/layer workflows |
| Bloch bands | Native periodic transfer matrices, complex right eigensystems, Bloch wavenumbers/vectors; native S-matrix, period and eigenvector adjoints; complete Advect multilayer bands | Wider conditioning and branch-crossing qualification; individual degenerate modes have no derivative |
| Global observables | Native TMatrix cd/db/chi with matrix and CD embedding-wavenumber pullbacks; thin SVD and singular-value VJP; complete Advect chiral-sphere gradients; xy plane chirality-density forms with native wavenumber and interval adjoints; SMatrices.cd with direction-aware polarization swapping | Oriented-plane chirality forms; direct high-level S-matrix observable adapters |
| Axisymmetric EBCM | Native sampled-surface regular/outgoing Q integrals and radius, slope, complex-wavenumber and impedance pullbacks; callable-surface convenience; complete Advect deformed-particle solve | Wider shape/order conditioning and quadrature qualification |
| HDF5 interchange | Optional h5py adapter; scalar matrices and rectangular parameter sweeps; streamed matrix writes; chirality, mode origins/indices and length-unit round trips; legacy treams names and rectangular incident/scattered mode sets | Gmsh mesh helper and extended tmat.h5 v1 submission metadata |
| Remaining public API | Not implemented | Remaining field-operator conveniences and public low-level namespace coverage |

`translate` covers spherical, cylindrical and both plane basis families, including
batched displacements, rectangular mode subsets and masks. Multipoles translate
equal particle indices at fixed local origins; `expand` handles all physical origin
pairs. Both reuse the existing native addition theorems. Plane translations use
`diff.plane_phases` / `advect.plane_phases`: a compact exp(i k.r) kernel with real
displacement and complex wavevector pullbacks, including axial vectors. The native
residual retains only inputs, output storage transfers directly to NumPy, and
large phase tables use Rayon. Reverse recomputes phases in two reductions to avoid
sample-by-mode residuals or per-thread gradient arrays. The binding borrows
contiguous C/F cotangents during reverse and packs noncontiguous inputs. Tests
cover both layouts, strided/reversed cotangents and empty sample sets.

`expand` also maps plane bases by full wavevector and polarization, supports unit/
component conversion and reordered subsets, and preserves fields for real and
evanescent directions. Matching tolerates normalization roundoff relative to the
wavevector scale. This is a discrete basis map; moving unmatched labels through a
match is not differentiable. The implementation fixes upstream's missing
polarization mask rather than reproducing its all-ones identity expansion.

The optimized `diff.cluster` is restricted to non-overlapping homogeneous,
nonmagnetic spheres in vacuum with a common multipole cutoff. Its pullback covers
radii, positions, complex sphere permittivities and vacuum wavenumber. The public
`TMatrix.cluster` accepts general local spherical T-matrices with distinct cutoffs
and a common, possibly chiral, embedding material.

`spherical_wave` and `cylindrical_wave` return an explicit `MultipoleWave` with
owned amplitudes, physical metadata, expansion and weighted E/H/D/B/G/F samples.
`TMatrix @ wave` checks the medium/frequency/polarization and expands the source
into regular incident channels, including displaced outgoing sources.
The same object can hold a superposition through `MultipoleWave(coefficients, ...)`.
It does not track framework values: use the existing `diff`/`advect` field and
expansion boundaries to differentiate its continuous parameters and amplitudes.

`SMatrices.cd` preserves upstream's transmission and total-outgoing-power contrast
formulas. Its second value is normalized by T+R despite upstream's absorption-CD
label. Helicity swapping matches actual transverse directions, including reordered
bases; parity flips the magnetic polarization amplitude. It accepts PlaneWave
illumination and explicit up/down direction through the existing flux calculation.

The public `changepoltype` operator handles all four basis families, rectangular
subsets and masks. T matrices, S matrices and multipole-source objects share this
mode-matching rule. Complete object conversions require both polarization partners;
the explicit operator can project onto partial bases. These are discrete linear
maps, so amplitude derivatives compose through ordinary matrix multiplication.

Tests exercise degrees through 30 for radial functions, through 10 for Mie, and
smaller orders for full derivative/cluster checks. An input bound of 128 does not
constitute a claim of accuracy throughout that range. Dense solve storage still
scales quadratically and factorization cubically with the multipole dimension.

The Python package is `treams_rs` during development so the upstream oracle can
coexist in the same environment. It is not yet a drop-in `import treams` replacement.
Python 3.12 and 3.13 are the qualification targets. Numerical observables and linear
basis composition currently use NumPy in the thin Python layer; native kernels own
the special functions, particle scattering and multiple-scattering solve.

`ebcm.qmat` samples positive radial surfaces on fixed Gauss-Legendre nodes;
`diff.ebcm_qmat` and `advect.ebcm_qmat` accept sampled radii and angular slopes,
holding quadrature nodes, weights and mode labels fixed. The native reverse
recomputes local wave derivatives without a dense output-by-parameter Jacobian.
The default restores a missing radial area factor in upstream's integral;
`legacy=True` explicitly reproduces that integral, including its derivative.
Tests cover homogeneous spheres against Mie, zero scattering for identical media,
lossless deformed-particle convergence through degree 6 and complete geometry,
material and frequency gradients through `-solve(Q_singular, Q_regular)`.
Callers must check quadrature and multipole convergence for their shapes. Sharp
surfaces, shapes that are not positive radial graphs, and extreme conditioning
are not qualified by these tests.

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
Fresnel interfaces, propagation, composition,
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
Advect boundaries. Coordinate-aligned xy, yz and zx interfaces and slabs are supported.

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

`diff.interface` and `advect.interface` match Cartesian tangential E/H fields
for xy, yz and zx interfaces in lossy, magnetic and chiral media. Their eight
parameter groups are the four medium wavenumbers, two impedances and two real
transverse components. The 4-by-4 solve uses pivoted LU; reverse reuses its small
inverse and recomputes local field derivatives without retaining an output
Jacobian. Normal-incidence xy direction derivatives use the analytic zero limit
of the S blocks, avoiding the intermediate polarization azimuth singularity.
`SMatrices.interface` retains the closed-form Fresnel path for xy bases. Oriented
slabs compose with cylindrical array S matrices. Cartesian boundary continuity,
lossless power, slab splitting, native identity adjoints and complete Advect slab
gradients cover the new path. Exact diffraction thresholds remain unsupported.

`diff.layer_stack` and `advect.layer_stack` solve each transverse channel as an
independent two-polarization system. Their compact output has shape
(channels, 2, 2, 2, 2); material and thickness cotangents accumulate across channels.
Storage for the native reverse grows linearly with channels and layers. The public
`SMatrices.slab` uses this path for complete polarization pairs, materializing its
familiar dense matrix only at the boundary. Partial bases retain projected-step
semantics; interfaces convert local polarization pairs before selecting requested
modes, which also supports partial parity bases. Tests cover both orderings,
partial bases, all coordinate normals, all parameter pullbacks, lossless power,
scale identities and complete Advect multilayer gradients.

Near-axis spherical translation tests separate the upstream angular routine's
valid range from its rounded-axis regime. An extrapolated, resolved-angle
reference independently checks values and Cartesian gradients through transverse
offsets of 1e-300; the Rust result preserves terms that upstream rounds to zero.

`diff.periodic_conversion` and `advect.periodic_conversion` map a spherical
z-periodic array into outgoing cylindrical modes. Their native pullback covers
both expansion origins, complex medium wavenumbers, every real output axial
wavenumber and the period. Axial wavenumbers are independent parameters at this
boundary; Advect composes the moving diffraction orders from Bloch vector and
period. The regular and periodic conversions share their angular coefficient.
The residual retains inputs only and transfers the output buffer without copying.
Tests compare common-origin coefficients with upstream, reconstruct independent
spherical image sums at displaced origins, and check all parameter VJPs, native
scaling identities and complete Advect sphere-to-cylindrical radiation gradients.

`expandlattice` exposes same-family periodic coupling, spherical-to-cylindrical
radiation, and spherical/cylindrical-to-plane radiation with explicit geometry.
Radiation ports must match the cell's diffraction orders. `TMatrixC.from_array`
accepts an uncoupled spherical unit cell and explicitly solves its periodic
interaction before converting both incident and outgoing channels. This differs
from upstream's annotated, already-interacting input. Common-origin constructor
references and Hypothesis lossless power and coordinate scaling invariants pass.
Cross-family conversions include displaced origin pairs, rather than reproducing
upstream's matching-particle-index mask.

`TMatrixC.xw` excludes evanescent outgoing modes from radiated power. Its incident
coefficients must be propagating; this normalization does not define an evanescent
incident flux. Averaged widths uniformly average over azimuth and the represented
propagating (kz, polarization) channels, with helicity-dependent cutoffs in chiral
media. Adding closed channels leaves these averages unchanged. Diffraction cutoffs
are explicitly unsupported. Hypothesis tests compare the averages with explicit
plane-wave illumination and enforce lossless conservation and lossy passivity.
Global cross sections avoid constructing an identity overlap matrix. Native plane
illumination tolerates normalization roundoff when matching a static axial label,
with a relative tolerance independent of length units; discrete basis labels stay
fixed for differentiation.

`SMatrices.illuminate(..., smat=upper)` returns outgoing up/down and internal
up/down coefficients between adjacent stacks. It checks the shared medium and
uses the correct outer medium for each PlaneWave input. The native
`diff.smatrix_illuminate` and Advect adapter also accept multiple independent
illuminations as matrix columns. Forward solves only those right-hand sides;
reverse reuses the packed LU and contracts the coupled field equations with rank-P
products, overwriting the no-longer-needed primal blocks with their cotangents. It avoids constructing the complete combined S matrix, dense operator
cotangents and explicit conjugate-transpose copies. Tests cover all four input
pullbacks, treams coefficients, Cartesian E/H boundary continuity for every normal,
amplitude identities and complete lossy/chiral multilayer Advect gradients.
Ordinary illumination borrows contiguous blocks without recording an adjoint;
strided blocks are packed. Hypothesis tests cover C/F/block-F/reversed/strided
layouts and owned pullbacks after all original inputs are overwritten.
`just bench-performance` checks runtime and forward memory against upstream
at 256/1024 modes and one/eight illuminations. The recorded adjoint still owns
input snapshots and has a larger memory footprint than forward alone.

`SMatrices.periodic()` and `bands_kz(period)` require matching outer media and
use the basis normal (z/x/y for xy/yz/zx). The transfer construction solves both
right-hand blocks with one native LU. `diff.bands` and `advect.bands` expose
wavenumbers and right eigenvectors with native S-matrix and period pullbacks.
`diff.solve` and `diff.eig` expose the same underlying numerical boundaries.
Eigenvectors have unit norm with their largest component real positive. Their
pullback includes both normalization and phase; phase-dependent cotangents at a
tied largest component are explicitly rejected. Equal eigenvalue weights and zero
vector cotangents support spectral sums within repeated groups; individual modes
at degeneracy have no supported derivative. Band derivatives hold the principal
logarithm branch fixed. Strongly evanescent transfer matrices can be ill-conditioned;
finite-stack composition continues to use stable S matrices.

Reference eigensystems, uniform propagating/evanescent bands, all input VJPs,
Hermitian spectral invariants, native scale/shift identities, and complete Advect
multilayer band gradients cover this path. Eigensystem values, vectors and vector
pullbacks are checked under common input scales from 1e-200 to 1e200. A uniform
cell's complete frequency/period gradient is checked at polarization degeneracy.

`gfield` and `ffield` support spherical, cylindrical, component-plane and
unit-vector plane bases. They preserve upstream's family-dependent normalization:
spherical G carries an additional sqrt(2), and helicity selection and parity
combinations have different weights. For a convention-independent physical
definition, use `(E +/- i Z H)/sqrt(2)` directly. F includes the chiral refractive
index weights. `advect.gfield` and `advect.ffield` use weighted native multipole
fields and their existing pullbacks; the latter also differentiates those index
weights. Reference values, Hypothesis E/H reconstruction identities and complete
amplitude/point/origin/complex-wavenumber gradients cover these paths.

`TMatrix.cd`, `.db` and `.chi` now evaluate in Rust. The framework-neutral
`diff.tmatrix_metric` and Advect adapter expose matrix and real embedding-wavenumber
pullbacks; the wavenumbers affect only absorption circular dichroism. Normalized
duality breaking and electromagnetic chirality use a scaled matrix to avoid norm
overflow/underflow, checked from 1e-200 to 1e200. A scalar metric retains its matrix
gradient, releasing singular vectors before returning to Python. No full
output-by-parameter Jacobian is needed for the other native kernels.

`diff.svdvals` and `advect.svdvals` expose the thin native singular-value boundary.
Repeated positive singular values require equal weights; zero singular values
require zero weights. Normalized metrics at zero scattering, and CD at zero total
absorption, are undefined. Chi has a forward value at zero contrast, but a nonzero
pullback there is rejected; a zero cotangent allows smooth compositions such as
chi squared. Reference metrics, scale/phase/helicity-swap invariants, rectangular
SVDs, Frobenius-gradient properties and complete chiral-sphere Advect derivatives
cover these cases. Higher derivatives remain unsupported.
