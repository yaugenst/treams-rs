//! The numerical core of treams-rs: special functions, lattice sums, wave expansions,
//! T-matrices and S-matrices, each with an analytic pullback, in Rust and without Python.
//!
//! A pullback turns the gradient of a real loss with respect to a result into the
//! gradients with respect to the inputs; the [glossary](#glossary) defines it with the
//! related terms.
//!
//! `treams-core` is an internal crate. The Python package `treams_rs` calls it through
//! the bindings crate `treams-py`, which Python imports as `treams_rs._native`; neither
//! crate is published on crates.io. The user documentation, with examples and the
//! design rationale, is at <https://yaugenst.github.io/treams-rs/latest/>.
//!
//! Upstream: [treams](https://github.com/tfp-photonics/treams) 0.4.7 at commit `1f5d0d6`.
//! The kernels follow its formulas, conventions and mode order. Each module names the
//! treams functions it mirrors, and the [module map](#module-map) pairs every module
//! with its `treams_rs` and treams namespaces.
//!
//! # Layers
//!
//! The modules form six layers:
//!
//! | Layer | Modules | Contents |
//! |---|---|---|
//! | L0 | `numerics`, [`linalg`], [`fpenv`], [`threads`] | Numerical support |
//! | L1 | [`special`] | Special functions |
//! | L2 | [`lattice`] | Lattice sums and lattice geometry |
//! | L3 | [`basis`], [`sw`], [`cw`], [`pw`], [`rotation`], [`channels`], [`vectorwaves`], [`fields`] | Wave families |
//! | L4 | [`coeffs`], [`tmatrix`], [`ebcm`], [`cluster`] | Scattering by particles |
//! | L5 | [`smatrix`] | Planar and periodic S-matrices |
//!
//! Production code of a module uses its own layer and the layers below it, never a
//! layer above. Modules of one layer may use each other: [`cw::to_sw()`] builds
//! spherical modes, and [`tmatrix::sphere()`] takes the Mie coefficients of [`coeffs`].
//! Any module may use the root items [`Complex`], [`Error`], [`Result`] and
//! [`MAX_DEGREE`]. Tests may use any module.
//!
//! [`rotation`] and [`channels`] share one residual or gradient type across both
//! multipole families ([`rotation::RotationResidual`], [`channels::ChannelGradient`]),
//! so they stay at the top level instead of inside [`sw`] and [`cw`].
//!
//! A directory module keeps its files private and re-exports their items, so callers
//! write `sw::expansion` and `smatrix::tr`. Three modules keep their own path:
//! [`special::coordinates`], `numerics::broadcast` and `numerics::parallel`.
//!
//! # Conventions
//!
//! ## Numbers and labels
//!
//! - [`Complex`] is `num_complex::Complex64`, which `NumPy` calls `complex128`. Every
//!   kernel takes and returns double precision.
//! - The degree is `l` and the order is `m`; `lmax` and `mmax` bound them. "Order" also
//!   names the order of a Bessel function, never a degree.
//! - [`MAX_DEGREE`] = 128 bounds the degree `l` and the size of cylindrical orders `m`.
//!   Translations couple degrees up to `special::MAX_ORDER` = 256, and Wigner and Kambe
//!   labels reach [`special::MAX_LABEL`] = 260. treams has no such limits.
//! - `pol: u8` is the polarization index, 0 or 1, and `helicity: bool` selects the
//!   convention that treams calls `poltype`. With `helicity` true, pol 1 is positive and
//!   pol 0 negative helicity; with `helicity` false, pol 1 is TM (the N multipoles) and
//!   pol 0 TE (the M multipoles). Bases list pol 1 before pol 0, as treams does.
//! - [`special::Radial`] is the treams `modetype`: `Regular` is `'regular'`, and
//!   `Singular` is `'singular'`, the outgoing Hankel function of the first kind. The
//!   bindings pass it as `singular: bool`.
//! - A basis entry `(pidx, mode)` holds the position index (treams `pidx`) and the mode
//!   label, [`sw::Mode`] or [`cw::Mode`]. `positions` are Cartesian expansion centres.
//! - A matrix between two bases maps coefficients of the `source` basis to the
//!   `destination` basis. Translations take the displacement `destination - source`;
//!   lattice sums run over `source - destination`, and each lattice path says where it
//!   negates the displacement.
//! - Lattice sums take the Bloch vector `kpar`, the Ewald split parameter `eta`, the
//!   lattice `vectors` (treams `a`) and the `shift` (treams `r`).
//! - Planar inputs: `ks` holds the two helicity wavenumbers of each medium, `kzs` the
//!   normal wavenumbers, `zs` the impedances and `q` the transverse wavevectors.
//!   [`smatrix::chirality_density`] and [`smatrix::oriented_chirality`] use the
//!   argument names of their native functions: `normal` for the normal wavenumbers
//!   and, in [`smatrix::oriented_chirality`], `transverse` for the transverse
//!   wavevectors. The [`smatrix`] module docs give the order of the sides and of the
//!   four S-matrix blocks.
//!
//! ## Names
//!
//! - Items inside [`sw`], [`cw`] and [`pw`] carry no family prefix; other modules name
//!   them by path, as in `sw::Mode`, `cw::Basis` and `pw::expansion`. A module that
//!   serves both families prefixes the family: `rotation::sw_rotate`,
//!   `channels::cw_periodic_to_pw`.
//! - A forward function carries the treams name or the name of its operation:
//!   `coeffs::mie`, `cluster::interaction`, `smatrix::tr`. Suffixes mark the variants:
//!   `_array` for an elementwise forward over arrays that returns a residual
//!   (`special::bessel_array`), `_matrix` for a matrix that changes the wave family
//!   (`cw::to_sw_matrix`), and `_value` for the forward that returns only the value of
//!   a forward with a residual (`smatrix::tr_value`). Matrices between two bases of one
//!   family are nouns: `sw::expansion`, `sw::lattice_expansion`, `rotation::sw_rotation`.
//! - A forward `X` that supports gradients returns `(value, XResidual)` when its
//!   pullback does not read the value. When the pullback reads it, the forward returns
//!   only the residual, which lends the value through an accessor such as `value()`:
//!   [`coeffs::mie`], [`smatrix::tr`] and [`linalg::solve`] work this way.
//!   `XResidual::pullback(&self, cotangent)` returns an `XGradient` with one field per
//!   differentiable input, or an array of gradients in argument order. The residual
//!   name drops the suffixes `_array`, `_matrix` and `_value`:
//!   `sw::polar_translation_array` returns a [`sw::PolarTranslationResidual`], and
//!   `cw::to_sw_matrix` a [`cw::ToSwResidual`].
//! - The `record` methods of [`cluster::InteractionFactor`] and
//!   [`cluster::IterativeSphereCluster`] return residuals named after their native
//!   context classes. [`cluster::IlluminateResidual`] matches `IlluminateContext`, which
//!   `diff.illuminate` also returns. [`cluster::IterativeResidual`] matches
//!   `IterativeContext`.
//! - Family twins share one residual: [`rotation::sw_rotation`] and
//!   [`rotation::cw_rotation`] return [`rotation::RotationResidual`], and
//!   [`cluster::particle_cluster`] and [`cluster::cylindrical_particle_cluster`] return
//!   [`cluster::ParticleClusterResidual`]. [`linalg::solve_owned`], which reuses the
//!   operator's buffer, returns the [`linalg::SolveResidual`] of [`linalg::solve`].
//! - The pullbacks of [`linalg::EigResidual`] and [`smatrix::BandsResidual`] take one
//!   cotangent per output and name each after its output: `values` and `vectors`, and
//!   `wavenumbers` and `vectors`.
//!
//! ## Gradients of complex values
//!
//! A loss `L` changes by `dL = Re Σ conj(g)·dx` when a value changes by `dx`, where `g`
//! is the cotangent of that value. For a complex value `x = a + i b`, the cotangent is
//! `∂L/∂a + i ∂L/∂b`, and real inputs get real gradients. A pullback through a
//! holomorphic map `y = f(x)` therefore multiplies by the conjugate derivative:
//! `g_x = conj(f'(x)) g_y`.
//!
//! ## Errors
//!
//! Every fallible function returns [`Result`], whose [`Error`] says why it failed:
//! [`InvalidInput`](Error::InvalidInput) for input outside the domain,
//! [`SpecialFunction`](Error::SpecialFunction), [`NonFinite`](Error::NonFinite) and
//! [`NotConverged`](Error::NotConverged) for numerical failures,
//! [`Singular`](Error::Singular) for a singular linear system, and
//! [`OutOfMemory`](Error::OutOfMemory) when the system refuses the memory of a large
//! output or workspace. The bindings raise `OutOfMemory` as Python `MemoryError` and
//! every other variant as `ValueError`, with the displayed message. treams returns NaN
//! or emits `NumPy` warnings in the numerical cases.
//!
//! Dense coupling, T-matrix, expansion and field outputs, decomposition vectors,
//! and LU, SVD and eigenvalue workspaces are reserved fallibly, through
//! `numerics::zeros`, `numerics::filled`, `numerics::reserve` and the LU workspace of
//! [`linalg`], so a refused request returns `OutOfMemory` instead of aborting the
//! process. This is not a process-wide guarantee: input copies, matrix products,
//! gradient buffers and other allocations can still abort when refused.
//!
//! ## Parallelism and reproducibility
//!
//! Rayon runs independent items in parallel, on the pool of [`threads`], once their
//! number reaches a threshold; [`numerics::parallel`](numerics/parallel/index.html)
//! lists every threshold. The elementwise evaluations and the helpers there keep every
//! output in index order. A pullback that adds over items splits them into chunks
//! fixed by the item count, or into blocks fixed by the shape, and adds the partial
//! sums in order, so the thread count sets how many workers run the chunks but never
//! the order of the additions. The docs of these residuals, and of
//! `numerics::broadcast` for the elementwise ones, say how they add; the thread count
//! changes the result of none of them:
//!
//! - Outputs in index order, or sums in a fixed order: the elementwise residuals of
//!   [`special`] and [`lattice::SumResidual`], [`sw::PolarTranslationResidual`],
//!   [`sw::ExpansionResidual`], [`sw::LatticeExpansionFromTableResidual`],
//!   [`cw::PolarTranslationResidual`], [`cw::ExpansionResidual`],
//!   [`cw::LatticeExpansionResidual`], [`rotation::RotationResidual`],
//!   [`pw::PhasesResidual`], [`pw::PermutationResidual`],
//!   [`vectorwaves::VectorWaveResidual`], [`ebcm::QmatResidual`],
//!   [`coeffs::MieResidual`], [`coeffs::MieCylResidual`],
//!   [`tmatrix::SphereResidual`], [`tmatrix::CylinderResidual`],
//!   [`tmatrix::MetricResidual`], [`cluster::InteractionResidual`],
//!   [`cluster::IlluminateResidual`] and [`cluster::SphereClusterResidual`].
//! - Partial sums over fixed chunks or blocks, added in order:
//!   [`sw::LatticeExpansionResidual`], [`sw::PeriodicToCwResidual`],
//!   [`cw::ToSwResidual`], [`pw::ExpansionResidual`], [`pw::FieldResidual`],
//!   [`channels::SphericalChannelsResidual`], [`channels::CylindricalChannelsResidual`],
//!   [`fields::FieldResidual`], [`fields::OperatorResidual`],
//!   [`cluster::IterativeResidual`] and [`smatrix::LayerStackResidual`].
//!
//! These residuals give the same bits at every thread budget. With a budget of one
//! thread ([`threads::set_num_threads`] or `TREAMS_RS_NUM_THREADS=1`), every result
//! repeats bit for bit.
//!
//! ## Floating-point environment
//!
//! The kernels assume IEEE 754 gradual underflow: subnormal numbers stay subnormal.
//! JAX (through XLA) and `torch.set_flush_denormal(True)` make the calling thread flush
//! them to zero, which breaks algorithms that scale through the subnormal range, such
//! as faer's complex reciprocal of an LU pivot. [`fpenv::ieee`] runs a closure with
//! subnormals on the calling thread and restores the caller's mode afterwards. Its name
//! says what the closure gets, the IEEE 754 default arithmetic, and stays short because
//! the bindings wrap the body of every Python entry point in it: `ieee(|| ...)`.
//! `fpenv::flushing` does the opposite, for tests. The [`fpenv`] module docs explain
//! the control registers and why the guard is sound.
//!
//! ## Lints
//!
//! The workspace keeps clippy's default lints, adds `pedantic` and `nursery`, requires
//! docs on every public item, and the checks treat every warning as an error.
//! `clippy::indexing_slicing` stays on, so new code indexes through iterators or checked
//! access. A numerical module that validates its indices on entry opts out once, with
//! `#![allow(clippy::indexing_slicing)] // <why the indices are in range>`; the reason
//! comment is required. Tests may index freely (`clippy.toml`).
//!
//! # Glossary
//!
//! A record is a function that returns a value and a reusable context. The context stores what is needed to compute derivatives later: `context.pullback(g)` takes the gradient `g` of a real-valued loss with respect to the value and returns the gradients with respect to the inputs, one for each differentiable input, in the order of the arguments. Gradients follow the convention dL = Re Σ conj(g)·dx.
//!
//! In the Rust core, a function returns `(value, XResidual)`, and `XResidual::pullback(&self, cotangent)` returns the input gradients as `XGradient`.
//!
//! | Term | Meaning |
//! |---|---|
//! | forward | A function that computes a value. One that supports gradients returns a residual, beside the value or holding it. |
//! | residual | What a forward saves for its pullback: `XResidual` for the forward `X`. Not the `b - A x` of a linear system, which the GMRES solver of [`linalg`] calls the residual. |
//! | pullback | `XResidual::pullback`: turns the gradient with respect to the value into the gradients with respect to the inputs. It borrows the reusable residual. |
//! | cotangent | The gradient of a real-valued loss with respect to one value: the `g` of the definition above. |
//! | gradient | `XGradient`: the input gradients that a pullback returns, in the order of the forward's arguments. |
//! | record | The Python name of a forward with a residual: a function of `treams_rs.diff` that returns `(value, context)`. |
//! | context | The Python object that holds a residual, named after its record: `MieContext` for `diff.mie`. |
//! | jet | A value carried through the arithmetic with its first derivatives along `N` directions (`numerics::Jet<N>`). |
//! | [`RadialJet`](special::RadialJet) | A radial function with its first and second derivatives in its one complex argument. |
//! | split (`eta`) | The Ewald split parameter, which divides a lattice sum into a real-space and a reciprocal-space part; `eta = 0` selects it automatically. |
//! | Bloch vector (`kpar`) | The wavevector components along a lattice: the term of lattice point `R` carries the phase `exp(i kpar · R)`. |
//! | sheet | The branch of the complex logarithm on which a reciprocal-space term of a lattice sum is evaluated. |
//!
//! The [`lattice`](lattice#glossary) module docs define the other terms of the lattice
//! sums.
//!
//! # Tests
//!
//! The tests have two tiers:
//!
//! - A module's inline `tests` module checks how that module computes its results:
//!   fast paths against reference paths, tables and plans, dispatch thresholds, error
//!   paths, accuracy against reference tables, and private helpers.
//! - `src/properties/<domain>.rs` checks what the public operations of one domain
//!   promise: physical laws, analytic identities, and pullbacks against finite
//!   differences and adjoint pairings. The domains are `special`, `linalg`, `lattice`,
//!   `waves`, `plane`, `smatrix` and `tmatrix`, as in the Python suite; `lattice` is
//!   the directory `src/properties/lattice/`.
//!
//! A test of one module's implementation goes inline; a physical, analytic or adjoint
//! identity goes into its domain file. Both tiers use crate-private items, so the crate
//! has no `tests/` directory. `cargo test -p treams-core` runs both.
//! [Testing](https://yaugenst.github.io/treams-rs/latest/development/testing/) on the docs site
//! gives the helpers and conventions of both tiers.
//!
//! # Module map
//!
//! Each module mirrors one treams namespace where one exists. A dash marks a module
//! without a counterpart.
//!
//! <!-- crosswalk:start -->
//! | Rust module | treams_rs namespace | treams | Contents |
//! |---|---|---|---|
//! | [`special`] | `treams_rs.special` | `treams.special` | Bessel, Legendre and Wigner functions, coordinate transforms, incomplete gamma and Kambe integrals |
//! | [`lattice`] | `treams_rs.lattice`, `treams_rs.misc.firstbrillouin*` | `treams.lattice`, `treams.misc.firstbrillouin*` | Lattice sums and lattice geometry |
//! | [`sw`] | `treams_rs.sw`, `treams_rs.special.tl_vsw_*` | `treams.sw`, `treams.special.tl_vsw_*` | Spherical modes, translation coefficients, expansion matrices and the conversion of chains to cylindrical waves |
//! | [`cw`] | `treams_rs.cw`, `treams_rs.special.tl_vcw` | `treams.cw`, `treams.special.tl_vcw` | Cylindrical modes, translation coefficients, expansion matrices and the conversion to spherical waves |
//! | [`pw`] | `treams_rs.pw`, `treams_rs.misc.wave_vec_z` | `treams.pw`, `treams.misc.wave_vec_z` | Plane-wave polarizations, fields and multipole expansions |
//! | [`basis`] | `treams_rs.SphericalBasis`, `treams_rs.CylindricalBasis` | `treams.SphericalWaveBasis`, `treams.CylindricalWaveBasis` | Multipole bases and the expansion gradients that both families share |
//! | [`rotation`] | `treams_rs.sw.rotate`, `treams_rs.cw.rotate`, `treams_rs.operators.Rotate` | `treams.sw.rotate`, `treams.cw.rotate`, `treams.Rotate` | Rotation coefficients and rotation matrices of multipole bases |
//! | [`channels`] | `treams_rs.sw.periodic_to_pw`, `treams_rs.cw.periodic_to_pw` | `treams.sw.periodic_to_pw`, `treams.cw.periodic_to_pw` | Plane-wave channels of periodic arrays |
//! | [`vectorwaves`] | `treams_rs.special.vsh_*`, `vsw_*`, `vcw_*`, `vpw_*`, `sph_harm` | `treams.special.vsh_*`, `vsw_*`, `vcw_*`, `vpw_*`, `sph_harm` | Vector spherical harmonics and vector spherical, cylindrical and plane waves |
//! | [`fields`] | `treams_rs.operators.efield` | `treams.efield` | Electric fields of multipole expansions at sample points |
//! | [`coeffs`] | `treams_rs.coeffs`, `treams_rs.Material`, `treams_rs.misc.refractive_index` | `treams.coeffs`, `treams.Material`, `treams.misc.refractive_index` | Mie coefficients of multilayer spheres and cylinders, and materials |
//! | [`tmatrix`] | `treams_rs.TMatrix`, `treams_rs.CylindricalTMatrix` | `treams.TMatrix`, `treams.TMatrixC` | T-matrices of single spheres and cylinders, and the cd, db and chi metrics |
//! | [`ebcm`] | `treams_rs.ebcm` | `treams.ebcm` | Q-matrices of the extended boundary condition method |
//! | [`cluster`] | `treams_rs.Cluster`, `treams_rs.iterative` | `treams.TMatrix.cluster`, `treams.TMatrix.interaction` | Multiple scattering in particle clusters, dense and matrix-free |
//! | [`smatrix`] | `treams_rs.SMatrix`, `treams_rs.coeffs.fresnel` | `treams.SMatrices`, `treams.coeffs.fresnel` | Planar S-matrices: interfaces, layer stacks, composition, illumination, transmittance and bands |
//! | [`linalg`] | `treams_rs.diff.solve`, `svdvals`, `eig` | - | Dense LU, SVD and eigensystems with pullbacks, and restarted GMRES |
//! | `numerics` | - | - | Forward-mode jets, broadcasting and parallel thresholds |
//! | [`fpenv`] | - | - | A guard that keeps subnormal numbers when the caller flushes them to zero |
//! | [`threads`] | `treams_rs.set_num_threads`, `threads`, `thread_info` | - | The thread budget and the fork-safe pool that runs every parallel region |
//! <!-- crosswalk:end -->

// The crosswalk table ends the crate docs: its header names treams_rs without backticks,
// as the docs generator expects, and clippy's doc_markdown lint flags that header when
// a paragraph follows the table.

// Modules by layer; production code imports only from its own and lower layers.

// L0: numerical support.
pub mod fpenv;
pub mod linalg;
pub(crate) mod numerics;
pub mod threads;

// L1: special functions (treams.special).
pub mod special;

// L2: lattice sums and lattice geometry (treams.lattice).
pub mod lattice;

// L3: wave families (bases, translations, conversions, rotations, channels, fields).
pub mod basis;
pub mod channels;
pub mod cw;
pub mod fields;
pub mod pw;
pub mod rotation;
pub mod sw;
pub mod vectorwaves;

// L4: scattering (Mie coefficients, T-matrices, EBCM, clusters).
pub mod cluster;
pub mod coeffs;
pub mod ebcm;
pub mod saved;
pub mod tmatrix;

// L5: planar and periodic S-matrices (treams SMatrices).
pub mod smatrix;

/// Property tests by domain. An inline `tests` module checks how one module computes;
/// a file here checks a physical, analytic or adjoint identity of a domain's public
/// operations.
#[cfg(test)]
mod properties;
/// Case budgets, pairings, finite differences, strategies and fixtures that the tests
/// share.
#[cfg(test)]
pub(crate) mod test_support;

/// Complex double precision used throughout the numerical core.
pub type Complex = num_complex::Complex64;

/// The largest multipole degree `l`, and the largest `|m|` of a cylindrical mode.
///
/// Translations couple degrees up to `2 MAX_DEGREE` (`special::MAX_ORDER`): the
/// log-factorial cache, the AMOS Bessel orders and the solid-harmonic tables cover that
/// range. treams has no such limit.
///
/// Error messages write the numbers out, with a comment naming the constant: the checks
/// run once per element, and a literal message keeps formatting code out of them.
pub const MAX_DEGREE: i32 = 128;

/// A numerical failure, invalid physical input or refused memory.
///
/// The Python bindings raise [`OutOfMemory`](Self::OutOfMemory) as `MemoryError` and
/// every other variant as `ValueError`, each with the displayed message.
#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    /// The operation does not accept the input, for example a negative degree or a
    /// non-finite wavenumber.
    #[error("{0}")]
    InvalidInput(String),
    /// An AMOS Bessel, Legendre, Ferrers, Wigner, incomplete gamma or Kambe evaluation
    /// failed.
    #[error("special-function evaluation failed: {0}")]
    SpecialFunction(String),
    /// A finite input produced a non-finite value, for example an overflowing phase.
    #[error("{0}")]
    NonFinite(String),
    /// A sum, series, iteration or decomposition did not reach its tolerance, or
    /// cancellation lost the accuracy.
    #[error("{0}")]
    NotConverged(String),
    /// A linear system (an LU solve, a 2 x 2 Mie inverse or an S-matrix solve) is
    /// singular at the supplied parameters.
    #[error("singular linear system")]
    Singular,
    /// The system refused memory for an output or workspace. The bindings raise it as
    /// `MemoryError`.
    #[error("{0}")]
    OutOfMemory(String),
}

impl Error {
    /// [`OutOfMemory`](Self::OutOfMemory) for `count` elements of `element_bytes` bytes
    /// each: "cannot allocate 4096 bytes", or "cannot allocate more than
    /// 18446744073709551615 bytes" (`usize::MAX` on 64-bit targets) when the product
    /// overflows.
    #[must_use]
    pub fn out_of_memory(count: usize, element_bytes: usize) -> Self {
        Self::OutOfMemory(count.checked_mul(element_bytes).map_or_else(
            || format!("cannot allocate more than {} bytes", usize::MAX),
            |bytes| format!("cannot allocate {bytes} bytes"),
        ))
    }

    /// Whether the evaluation failed numerically ([`SpecialFunction`], [`NonFinite`] or
    /// [`NotConverged`]) rather than on its input, a singular system or memory.
    ///
    /// Another method may succeed where one fails numerically: the one-dimensional
    /// spherical lattice sums then fall back to their spectral series.
    ///
    /// [`SpecialFunction`]: Self::SpecialFunction
    /// [`NonFinite`]: Self::NonFinite
    /// [`NotConverged`]: Self::NotConverged
    pub(crate) const fn is_numerical(&self) -> bool {
        matches!(
            self,
            Self::SpecialFunction(_) | Self::NonFinite(_) | Self::NotConverged(_)
        )
    }
}

/// Numerical result with a typed error.
pub type Result<T> = std::result::Result<T, Error>;
