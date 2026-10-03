//! Lattice sums of outgoing spherical and cylindrical waves, and lattice geometry.
//!
//! Upstream: `treams.lattice` (`lattice/_esum.pyx`, `_dsum.pyx` and `_misc.pyx` of
//! tfp-photonics/treams, MIT), whose formulae and conventions the sums follow, and the
//! Brillouin-zone helpers of `treams.misc`.
//!
//! # Definition
//!
//! A lattice of dimension `d` holds the points `R = n_1 a_1 + ... + n_d a_d` with integer
//! `n_i`. For a wavenumber `k`, a real Bloch vector `kpar` (the wavevector components
//! along the lattice) and a shift `r`, [`sum`] returns
//!
//! ```text
//! D_lm(k, kpar, r) = Σ'_R h_l^(1)(k |r + R|) Y_lm(-r - R) exp(i kpar · R)        spherical
//! D_m(k, kpar, r)  = Σ'_R H_m^(1)(k |r + R|) exp(i m φ(-r - R)) exp(i kpar · R)   cylindrical
//! ```
//!
//! as in `treams.lattice`. `Y_lm(-r - R)` is the spherical harmonic in the direction of
//! `-r - R` and `φ` its azimuth. The prime drops the point with `r + R = 0`, if there is
//! one. [`Family`] selects the sum and its labels: degree `l` and order `m` for spherical
//! sums (`d <= 3`), order `m` for cylindrical sums (`d <= 2`).
//!
//! # Frame
//!
//! The lattice lies along fixed Cartesian axes:
//!
//! | Sum | Lattice axes |
//! |---|---|
//! | 1D spherical | z |
//! | 1D cylindrical | x |
//! | 2D spherical and cylindrical | x and y (the plane `z = 0`) |
//! | 3D spherical | x, y and z |
//!
//! [`BlochLattice`] takes the lattice vectors and `kpar` as components along these axes.
//! The shift is a Cartesian 3-vector; cylindrical shifts lie in the xy plane.
//!
//! # Branch and Bloch sign
//!
//! The waves are outgoing: `h_l^(1)` and `H_m^(1)`, which decay for `Im k > 0`. The Bloch
//! factor is `exp(+i kpar · R)`, so the sum at `r + R` is `exp(-i kpar · R)` times the
//! sum at `r`. The sum depends on the lattice and on `kpar` modulo the reciprocal
//! lattice, but not on the primitive basis or on the Bloch cell. Exact diffraction
//! thresholds (`|kpar + G| = k` for a reciprocal lattice vector `G`) are singular.
//!
//! # Ewald split
//!
//! The Ewald split parameter `eta` divides each sum into a real-space part, whose terms
//! fall like `exp(-Re((k eta)^2) |R|^2 / 2)`, and a reciprocal-space part, whose terms
//! fall like `exp(-Re(1 / (k eta)^2) |kpar + G|^2 / 2)`. The exact sum does not depend
//! on `eta`. `eta = 0` selects the automatic split
//!
//! ```text
//! eta = sqrt(2 pi) / (k L) max(|k L| / 8, 1),    L = V^(1/d)
//! ```
//!
//! with the cell measure `V` (length, area or volume). Derivatives hold the split fixed.
//! The automatic split has the modulus `sqrt(2 pi) / 8 = 0.31` from `|k L| = 8` on, and
//! more below. A split is small when it lies below every automatic one,
//! `Re(1 / (2 eta^2)) > 16 / pi`: there the two parts grow like `exp(Re(1 / (2 eta^2)))`
//! and cancel.
//!
//! For `Im k >= 0` every sum has the same value at every split where its Ewald parts
//! converge. Spherical Ewald sums take `Re k > 0`. For `Im k < 0`, 2D spherical and 1D
//! cylindrical sums keep the principal branches: on or near their plane or axis they
//! depend on the split, and farther off they fail.
//!
//! # Crosswalk
//!
//! Each `treams.lattice` function is one call of [`sum_part`] (or [`sum`] for `Full`).
//! `r` is the shift, `i` the shell of a direct sum, and `dim` the lattice dimension of
//! [`BlochLattice`].
//!
//! | treams | Family | `dim` | Shift |
//! |---|---|---|---|
//! | `lsumsw1d(l, k, kpar, a, r, eta)` | `Spherical { l, m: 0 }` | 1 | `[0, 0, r]` |
//! | `lsumsw1d_shift(l, m, ...)` | `Spherical { l, m }` | 1 | `r` |
//! | `lsumsw2d(l, m, ...)` | `Spherical { l, m }` | 2 | `[r_x, r_y, 0]` |
//! | `lsumsw2d_shift(l, m, ...)` | `Spherical { l, m }` | 2 | `r` |
//! | `lsumsw3d(l, m, ...)` | `Spherical { l, m }` | 3 | `r` |
//! | `lsumcw1d(m, ...)` | `Cylindrical { m }` | 1 | `[r, 0, 0]` |
//! | `lsumcw1d_shift(m, ...)` | `Cylindrical { m }` | 1 | `[r_x, r_y, 0]` |
//! | `lsumcw2d(m, ...)` | `Cylindrical { m }` | 2 | `[r_x, r_y, 0]` |
//!
//! The prefix selects the [`SumPart`]:
//!
//! | treams prefix | [`SumPart`] |
//! |---|---|
//! | `lsum*` | `Full` |
//! | `realsum*` | `Real` |
//! | `recsum*`, with `zero2d` or `zero3d` added at `r = 0` | `Reciprocal`, which adds the self term at `r = 0` for every degree |
//! | `dsum*(..., i)` | `Direct(i)`; `eta` is not used |
//!
//! The geometry helpers map one to one:
//!
//! | treams | treams-rs |
//! |---|---|
//! | `lattice.volume`, `lattice.area` | [`volume`] |
//! | `lattice.reciprocal` | [`reciprocal()`] |
//! | `lattice.cube(d, n)`, `lattice.cubeedge(d, n)` | [`cube`] with `edge` false or true |
//! | `lattice.diffr_orders_circle(b, rmax)` | [`diffraction_orders`] |
//! | `misc.firstbrillouin1d` | [`first_brillouin_1d`] |
//! | `misc.firstbrillouin2d`, `misc.firstbrillouin3d` | [`first_brillouin`] |
//!
//! [`derivatives`], [`derivatives_part`] and [`SumResidual`] have no upstream
//! counterpart: they give the derivatives and gradients of the sums.
//!
//! # Differences from `treams.lattice`
//!
//! - **Automatic split in 3D.** The cell length is `L = V^(1/3)`. Upstream `_check_eta`
//!   divides by `k V^(2/3)`. Both choices give the same sum wherever the parts converge;
//!   they differ in cost and in which sums converge.
//! - **Convergence.** Each Ewald part adds integer cube shells until two consecutive
//!   shells add less than `2e-13 max(|P|, 1)` of the part `P`, and fails after 200, 32
//!   or 16 shells in one, two or three dimensions (up to four times that where a large
//!   `|k| L` or a small `|k eta|` needs it). Upstream stops once two shells change a part
//!   by less than `1e-10`, or after 200, 20 or 10 shells, and reports no failure.
//! - **Reduced bases.** Both Ewald parts sum over shells of reduced bases, the shift is
//!   reduced into a cell of the direct lattice and `kpar` into a cell of the reciprocal
//!   lattice. Upstream sums over the given basis, whose shells can miss short lattice
//!   vectors of a skewed basis.
//! - **Outgoing sheets.** For `Im k >= 0` the reciprocal integrals at a complex split
//!   take the sheet that continues the sum from the automatic split, so the value does
//!   not depend on the split. Upstream takes the principal branches.
//! - **Small splits.** Sums at small splits predict the rounding their cancelling parts
//!   leave and fail where it exceeds `1e-3` of the scale of a component. Upstream returns
//!   the cancelled sum.
//! - **Exact zeros.** For odd `l` (spherical) or odd `m` (cylindrical), `kpar = 0` and a
//!   shift on the lattice line or plane with `2 r` a lattice vector, inversion maps the
//!   sum onto its negative. The sum is then exactly zero at every split. Upstream
//!   returns exact zeros only at `r = 0` (and, in 1D, at `r = ±a / 2`); at the other
//!   such shifts it returns the rounding of its parts.
//! - **Spectral series.** Off the axis the Ewald parts of 1D spherical sums cancel by up
//!   to `exp(w^2)` (see the glossary). There each component whose rounding exceeds
//!   `1e-12 max(|S|, 1)` comes from the spectral series, a sum over diffraction orders
//!   of cylindrical waves, where that is more accurate.
//! - **Derivatives.** [`derivatives`] returns the analytic derivatives in `k`, `eta`,
//!   the shift, `kpar` and the lattice vectors, from jets (see the glossary) and the
//!   derivative identities of the integrals.
//!
//! # Glossary
//!
//! | Term | Meaning |
//! |---|---|
//! | shell | The integer points `n` with `max_i \|n_i\| = s` of shell `s`; sums add shell by shell |
//! | split | The Ewald split parameter `eta` |
//! | small split | A split below every automatic one, `Re(1 / (2 eta^2)) > 16 / pi` |
//! | diffraction order | `q = kpar + G` for a reciprocal lattice vector `G` |
//! | `v` | `(q · q / k^2 - 1) / (2 eta^2)` of a diffraction order `q` |
//! | `w` | `k rho eta`, with the distance `rho` of the shift from the lattice plane or axis |
//! | `t` | `w^2` |
//! | `F_n(v, t)` | The reduced integral `(-1)^n ∫_1^∞ s^(n-1) exp(-v s - t / (2 s)) ds` of the reciprocal terms; `F_n(v, 0) = Γ(n, v) / (-v)^n` (upstream `_redincgamma`, `_redintkambe`) |
//! | `I_n(z, eta)` | The Kambe integral `∫_eta^∞ t^n exp(-z^2 t^2 / 2 + 1 / (2 t^2)) dt` of the real-space terms, at `z = k \|r + R\|` (upstream `intkambe`) |
//! | sheet | The branch of `ln v` on which a reciprocal term takes `F_n` |
//! | self term | The correction at `r = 0` that removes the missing `R = 0` term from the reciprocal part; upstream adds it as `zero2d` or `zero3d` |
//! | jet | A value carried through the arithmetic together with its first derivatives |
//! | settled | A real-space shell sum that reaches its shell limit while the Bloch phases cancel each of its last two shells; it is kept only if it agrees with the sum at the automatic split |
//!
//! # Failure messages
//!
//! | Message | Error | Cause |
//! |---|---|---|
//! | `Ewald sum did not converge within the shell limit` | [`NotConverged`] | A part needs more shells than its limit |
//! | ... `; use a larger split (eta = 0 selects one)` | [`NotConverged`] | The same, at a small split |
//! | ... `, neither at this split nor at the automatic one` | [`NotConverged`] | A settled sum whose check at the automatic split does not converge |
//! | `Ewald split too small: its cancelling parts lose about ...; use a larger split (eta = 0 selects one)` | [`NotConverged`] | The parts at a small split cancel by more than about `1e-3` of the scale of the value or a derivative |
//! | `Ewald sum lost its accuracy to cancelling Kambe integrals; reduce the split parameter` | [`NotConverged`] | A 1D spherical sum off the axis rounds by more than `1e-3 max(\|S\|, 1)`, and its spectral series does not help |
//! | `non-finite Ewald summand; change the split parameter or reduce the order` | [`NonFinite`] | A term is not finite |
//! | `non-finite direct lattice term` | [`NonFinite`] | A term of a direct sum is not finite |
//! | `lattice sum is at a diffraction threshold; supply a limiting complex wavenumber` | [`InvalidInput`] | A diffraction order lies on its threshold, `\|v\| <= 1e-14` |
//! | `component adjoints require an explicit nonzero Ewald split` | [`InvalidInput`] | Derivatives of the real or reciprocal part at `eta = 0` |
//! | `direct half-cell shell grouping changes discontinuously; use the Ewald sum for adjoints` | [`InvalidInput`] | Derivatives of a direct 1D shell at a half-cell shift |
//!
//! Invalid labels, dimensions and non-finite inputs give [`InvalidInput`] with a message
//! that names the input. A value-only call can return a sum whose derivatives fail.
//!
//! # Tuning constants
//!
//! | Name | Value | Meaning | Why |
//! |---|---|---|---|
//! | `resolve_split` | `1 / 8`, `sqrt(2 pi)` | Automatic split `sqrt(2 pi) / (k L) max(\|k L\| / 8, 1)` | As upstream `_check_eta`, except for `L` in 3D |
//! | `SHELL_TOLERANCE` | `2e-13` | Part of `max(\|P\|, 1)` that each of the last two shells may add | About 900 ulps of the scale |
//! | `shells` | 200, 32, 16 | Shell limit in 1D, 2D, 3D, raised up to four times where the terms need it (`shells_within`) | 200 in 1D as upstream; above upstream's 20 and 10 in 2D and 3D |
//! | `TERM_DECAY` | `50` | Shell limits reach every term above `exp(-50)` | `exp(-50) = 2e-22`, far below `SHELL_TOLERANCE` |
//! | `AUTOMATIC_GROWTH` | `16 / pi (1 + 1e-12)` | Largest `Re(1 / (2 eta^2))` of the automatic split; larger values mark a small split | Exact, with a margin for rounding |
//! | `MAX_LOSS` | `1e-3` | Largest predicted loss of a sum, relative to the scale of a component, before it fails | Calibrated: the bounds exceed the errors by a median factor of 17 to 87 |
//! | `EARLY_FAILURE_LOSS` | `2e-3` | Loss at which a complete sum at a small split fails before its far shells | Calibrated: without it, a 3D sum at the split 0.03 takes 29 s to fail |
//! | `TERM_ULPS` | `8` | Rounding of each Ewald term and of the self term, in ulps | The `MAX_LOSS` calibration compares the bounds built from it with the errors |
//! | `REAL_ULPS` | `64` | Rounding of the real-space part of a 1D spherical sum, in ulps | Its Kambe integrals of even order keep their relative accuracy |
//! | `SPECTRAL_SWITCH_LOSS` | `1e-12` | Rounding of a 1D spherical component, relative to its scale, from which the spectral series replaces it | Far below `MAX_LOSS`, so the series takes over long before a sum fails |
//! | `SPECTRAL_FIRST_W` | `2.5` | `w = \|k\| rho \|eta\|` from which a 1D spherical sum tries the spectral series first | The reciprocal part cancels by `exp(w^2) = 5e2` there |
//! | `SERIES_SHELLS` | `512` | Diffraction orders the spectral series may take on each side | Caps the cost of one series; beyond it the Ewald sum is used |
//! | `SERIES_DECAY` | `45` | The spectral series runs only where its terms fall by `exp(-45)` within `SERIES_SHELLS` | `exp(-45) = 3e-20`, below `QUIET_SHELL` |
//! | `QUIET_SHELL` | `1e-18` | Part of the summed moduli below which a shell of the spectral series is quiet | The series stops after two quiet shells past its peak; `1e-18` is below the rounding of the sum |
//! | `SERIES_T` | `1e-2` | Largest `\|t\|` at which integer orders of `F_n` come from their series in `t` | Below about `\|w\| = 1e-50` the Kambe integrals at `eta = -i / w` leave the range of `f64`; the series loses only a few ulps |
//! | `SERIES_REACH` | `1` | The same, for the orders below the Kambe base pair | Each of these orders would need a gamma-function series of its own |
//! | `HALF_INTEGER_SERIES_T` | `1` | Largest `\|t\|` at which half-integer orders come from their series in `t` | Calibrated: within `1e-13` of 30-digit references for `\|t\|` from 0.3 to 1 |
//! | `SERIES_TERMS` | `64` | Terms of the series in `t` at most | `\|t\| <= 1` stops within 20 |
//! | `NEAR_THRESHOLD` | `1 / 16` | Bound on the real and imaginary parts of `q · q / k^2 - 1` below which the zeroth order takes compensated sums | Rounded sums lose 50 ulps of `v` there |
//! | `THRESHOLD_DISTANCE` | `1e-14` | `\|v\|` (reciprocal terms) or `\|k_q^2 / k^2\|` (spectral series) at or below which a diffraction order counts as on its threshold | The sums are singular on the threshold `\|q\| = k` |
//! | `SHEET_MARGIN` | `0.5` | Angle margin within which a diffraction order needs the argument of its sheet | Far above the rounding of the angle (`1e-2` next to a threshold) |
//! | `axial_cylindrical_reciprocal` | `4` | `\|v\|` above which the gamma functions of value-only 1D cylindrical sums on the axis recur upward | A downward step there subtracts the leading asymptotic term |
//! | `kambe_arguments` | `1e-100` | Imaginary part that puts a real Kambe argument on the outgoing side | Any positive part selects the upper half plane |
//! | `reduce_basis` | `1e-8`, `2^20`, 64 | Relative shortening a reduction step needs, largest coefficient, most passes | Only strict decreases count, so ties such as those of a hexagonal basis keep it as given |
//! | `first_brillouin` | `1e-13`, `1e-14` | Relative and absolute tolerance on squared lengths of a reduced 2D basis | The absolute one is upstream's |
//! | `PARALLEL` | 8 | Batched sums run in parallel from this many outputs | Fewer outputs run on the calling thread |
//!
//! # References
//!
//! - K. Kambe, Z. Naturforsch. A 22 (1967), <https://doi.org/10.1515/zna-1967-0305>
//! - K. Kambe, Z. Naturforsch. A 23 (1968), <https://doi.org/10.1515/zna-1968-0908>
//! - C. M. Linton, SIAM Rev. 52, 630 (2010), <https://doi.org/10.1137/09075130X>
//! - D. Beutel et al., J. Opt. Soc. Am. B 38 (2021), <https://doi.org/10.1364/JOSAB.419645>
//! - NIST DLMF, chapter 8 (incomplete gamma functions), <https://dlmf.nist.gov/8>
//!
//! [`NotConverged`]: crate::Error::NotConverged
//! [`NonFinite`]: crate::Error::NonFinite
//! [`InvalidInput`]: crate::Error::InvalidInput
#![allow(clippy::float_cmp, clippy::indexing_slicing)] // Validated dimensions, exact branch points.

use crate::{
    Complex, Error, Result,
    numerics::{Jet, finite},
};

mod accuracy;
mod batch;
mod cell;
mod direct;
mod ewald;
mod geometry;
mod inputs;
mod real;
mod reciprocal;
mod reduced;
mod sheets;
mod shells;
mod spectral;
mod wave;

// Cell helpers of treams.lattice (volume, area, reciprocal, cube, cubeedge,
// diffr_orders_circle) and treams.misc (firstbrillouin1d/2d/3d).
pub use geometry::{
    cube, diffraction_orders, first_brillouin, first_brillouin_1d, reciprocal, volume,
};

pub use batch::{SumResidual, sum_array};
pub use cell::BlochLattice;
pub(crate) use cell::resolve_split;
pub use wave::Family;
// Read by the property tests in properties/lattice/.
#[cfg(test)]
pub(crate) use ewald::probes;
#[cfg(test)]
pub(crate) use reduced::HALF_INTEGER_SERIES_T;
#[cfg(test)]
pub(crate) use shells::SHELL_TOLERANCE;

use accuracy::below_automatic;
use direct::direct_shell;
use ewald::ewald;
use inputs::{Evaluation, Inputs, unpack};
use shells::{NOT_CONVERGED, is_shell_limit};

/// The lattice sum of `wave` at the wavenumber `k`, the shift `shift` and the Ewald split
/// parameter `eta`, as `treams.lattice.lsum*` returns it.
///
/// `shift` is the `r` of treams, and the lattice vectors `a` and the Bloch vector `kpar`
/// come with `lattice`. `eta = 0` selects the automatic split. The module docs give the
/// definition, the frame, the crosswalk to treams and the differences from it.
///
/// Upstream: the `lsum*` functions of `treams.lattice`, from `lsumsw1d` to `lsumcw2d`.
///
/// # Errors
///
/// [`Error::InvalidInput`] for invalid labels, dimensions or inputs and at a diffraction
/// threshold, [`Error::NotConverged`] where an Ewald part does not converge or the parts
/// cancel beyond use, and [`Error::NonFinite`] for a non-finite term. The module docs
/// list the messages.
pub fn sum(
    wave: Family,
    k: Complex,
    lattice: &BlochLattice,
    shift: [f64; 3],
    eta: Complex,
) -> Result<Complex> {
    sum_part(wave, k, lattice, shift, eta, SumPart::Full)
}

/// Part of an Ewald sum, or one unaccelerated integer cube shell: the `lsum*`,
/// `realsum*`, `recsum*` and `dsum*` functions of `treams.lattice`.
#[derive(Clone, Copy, Debug)]
pub enum SumPart {
    /// Complete outgoing lattice sum.
    Full,
    /// Real-space Ewald contribution.
    Real,
    /// Reciprocal-space Ewald contribution, including the self correction.
    Reciprocal,
    /// One cube boundary of the given basis (which is not reduced); axial half-cell
    /// shifts pair equidistant images instead.
    Direct(i64),
}

/// Evaluate an Ewald component or one direct-summation shell.
///
/// Upstream: the `lsum*`, `realsum*`, `recsum*` and `dsum*` functions of
/// `treams.lattice`, one per [`SumPart`].
///
/// # Errors
///
/// As [`sum`].
pub fn sum_part(
    wave: Family,
    k: Complex,
    lattice: &BlochLattice,
    shift: [f64; 3],
    eta: Complex,
    part: SumPart,
) -> Result<Complex> {
    Ok(evaluate::<0>(wave, k, lattice, shift, eta, Evaluation::Part(part))?.value)
}

/// Local derivatives of one scalar Ewald sum.
///
/// Where the shift is a lattice point, the derivatives are those of the sum without that
/// point, which stays excluded as the inputs move.
#[derive(Clone, Copy, Debug)]
pub struct Derivatives {
    /// Sum value.
    pub value: Complex,
    /// Complex wavenumber derivative.
    pub k: Complex,
    /// Ewald split derivative; zero for the full and direct sums.
    pub eta: Complex,
    /// Cartesian shift derivatives.
    #[doc(alias = "r")]
    pub shift: [Complex; 3],
    /// Bloch derivatives in the lattice frame of [`BlochLattice`] (unused components are zero).
    pub kpar: [Complex; 3],
    /// Row lattice-vector derivatives in the lattice frame (unused components are zero).
    #[doc(alias = "a")]
    pub vectors: [[Complex; 3]; 3],
}

/// Differentiate an Ewald sum using analytic integral identities and Cartesian polynomials.
///
/// Derivatives of 1D spherical sums carry rounding bounds of their own, from the Kambe
/// orders one above and one below the value's. Each derivative that loses more than
/// `1e-12` of its scale comes from the spectral series (see the module docs). Where only
/// derivatives switch, the value stays the Ewald sum's, whose bound is then within
/// `1e-12 max(|S|, 1)`.
///
/// # Errors
///
/// As [`sum`].
pub fn derivatives(
    wave: Family,
    k: Complex,
    lattice: &BlochLattice,
    shift: [f64; 3],
    eta: Complex,
) -> Result<Derivatives> {
    derivatives_part(wave, k, lattice, shift, eta, SumPart::Full)
}

/// Differentiate the selected component, holding a direct shell's index fixed.
///
/// # Errors
///
/// As [`sum`], and [`Error::InvalidInput`] for the real or reciprocal part at `eta = 0`,
/// whose split derivative needs an explicit split.
pub fn derivatives_part(
    wave: Family,
    k: Complex,
    lattice: &BlochLattice,
    shift: [f64; 3],
    eta: Complex,
    part: SumPart,
) -> Result<Derivatives> {
    let component = matches!(part, SumPart::Real | SumPart::Reciprocal);
    if component && eta == Complex::default() {
        return Err(Error::InvalidInput(
            "component adjoints require an explicit nonzero Ewald split".into(),
        ));
    }
    let evaluation = Evaluation::Part(part);
    let dim = lattice.dim;
    let mut result = match dim {
        1 => unpack(
            evaluate::<6>(wave, k, lattice, shift, eta, evaluation)?,
            dim,
        ),
        2 => unpack(
            evaluate::<10>(wave, k, lattice, shift, eta, evaluation)?,
            dim,
        ),
        _ => unpack(
            evaluate::<16>(wave, k, lattice, shift, eta, evaluation)?,
            dim,
        ),
    };
    if component {
        // The complete sum is independent of the split: the reciprocal part
        // carries the negative split derivative of the real part.
        let real =
            evaluate::<1>(wave, k, lattice, shift, eta, Evaluation::EtaDerivative)?.derivative[0];
        result.eta = if matches!(part, SumPart::Real) {
            real
        } else {
            -real
        };
    }
    Ok(result)
}

/// Validates the inputs and evaluates what `evaluation` selects: a direct shell, or the
/// Ewald sum at the resolved split. `N` is the number of jet slots: 0 for a value, 1 for
/// the split derivative of the real-space part, and for all derivatives 6, 10 or 16 in
/// one, two or three dimensions (`k`, three shift components, `dim` Bloch components and
/// `dim * dim` lattice-vector components).
fn evaluate<const N: usize>(
    wave: Family,
    k: Complex,
    lattice: &BlochLattice,
    r: [f64; 3],
    eta: Complex,
    evaluation: Evaluation,
) -> Result<Jet<N>> {
    wave.validate(lattice.dim)?;
    if let Evaluation::Part(SumPart::Direct(shell)) = evaluation
        && !(0..=i64::from(i32::MAX)).contains(&shell)
    {
        return Err(Error::InvalidInput(
            "direct shell index must be in [0, i32::MAX]".into(),
        ));
    }
    if !finite(k) || k == Complex::default() || !finite(eta) || r.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "lattice wavenumber must be finite and nonzero; shift and eta must be finite".into(),
        ));
    }
    if matches!(wave, Family::Cylindrical { .. }) && r[2] != 0.0 {
        return Err(Error::InvalidInput(
            "scalar cylindrical lattice shifts lie in the xy plane".into(),
        ));
    }
    let direct = matches!(evaluation, Evaluation::Part(SumPart::Direct(_)));
    let variable = !matches!(evaluation, Evaluation::EtaDerivative);
    let inputs = Inputs::new(wave, k, lattice, r, variable, !direct);
    if let Evaluation::Part(SumPart::Direct(shell)) = evaluation {
        return direct_shell(wave, lattice, &inputs, shell);
    }
    let eta = resolve_split(k, lattice, eta);
    ewald(wave, lattice, inputs, eta, evaluation).map_err(|error| larger_split(error, eta))
}

/// `error`, advising a larger split where a sum at a split `eta` below every automatic
/// one does not converge: its real-space shells converge sooner at a larger one. A sum
/// whose shells settle but whose automatic split, which verifies it, does not converge
/// either reports that instead (see [`ewald`]).
///
/// [`ewald`]: ewald::ewald
fn larger_split(error: Error, eta: Complex) -> Error {
    if is_shell_limit(&error) && below_automatic(eta) {
        Error::NotConverged(format!(
            "{NOT_CONVERGED}; use a larger split (eta = 0 selects one)"
        ))
    } else {
        error
    }
}

/// Cotangents of the continuous Ewald parameters under the real Hermitian pairing.
///
/// The fields run (k, eta, shift, kpar, vectors); `treams_rs.diff.lattice_sum` returns
/// the same cotangents in the treams argument order (k, kpar, a, r, eta).
#[derive(Clone, Copy, Debug, Default)]
pub struct SumGradient {
    /// Complex wavenumber cotangent.
    pub k: Complex,
    /// Complex split cotangent; zero for the full and direct sums.
    pub eta: Complex,
    /// Real shift cotangents.
    #[doc(alias = "r")]
    pub shift: [f64; 3],
    /// Real Bloch cotangents in the lattice frame of [`BlochLattice`].
    pub kpar: [f64; 3],
    /// Real row lattice-vector cotangents in the lattice frame.
    #[doc(alias = "a")]
    pub vectors: [[f64; 3]; 3],
}

impl SumGradient {
    pub(crate) fn add(&mut self, other: Self) {
        self.k += other.k;
        self.eta += other.eta;
        for i in 0..3 {
            self.shift[i] += other.shift[i];
            self.kpar[i] += other.kpar[i];
            for j in 0..3 {
                self.vectors[i][j] += other.vectors[i][j];
            }
        }
    }
}

impl Derivatives {
    pub(crate) fn pullback(self, cotangent: Complex) -> SumGradient {
        SumGradient {
            k: self.k.conj() * cotangent,
            eta: self.eta.conj() * cotangent,
            shift: self.shift.map(|x| (cotangent.conj() * x).re),
            kpar: self.kpar.map(|x| (cotangent.conj() * x).re),
            vectors: self
                .vectors
                .map(|row| row.map(|x| (cotangent.conj() * x).re)),
        }
    }
}
