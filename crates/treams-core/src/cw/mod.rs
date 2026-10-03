//! Cylindrical vector waves: mode labels, translation coefficients of mode pairs,
//! expansion matrices between bases and over lattices, and the conversion to spherical
//! waves.
//!
//! Upstream: `treams.cw`, and `treams.special.tl_vcw` and `tl_vcw_r`.
//!
//! | File | Contents | Upstream |
//! |---|---|---|
//! | `polar.rs` | [`polar_translation`] and [`polar_translation_array`] of one order difference, [`tl_vcw`] and [`translate`] of one mode pair | `special.tl_vcw`, `tl_vcw_r`, `cw.translate` |
//! | `cartesian.rs` | [`cartesian_translation`]: one coefficient at a Cartesian displacement, with its derivatives | treams-rs extension |
//! | `expansion.rs` | [`expansion()`] and [`lattice_expansion`] | `Expand`, `ExpandLattice`, `cw.translate_periodic` |
//! | `to_sw.rs` | [`to_sw()`] and [`to_sw_matrix`] | `cw.to_sw`, `Expand` to a spherical basis |
//!
//! Conventions:
//!
//! - A translation from a source position to a destination position takes the
//!   displacement `destination - source`. Lattice sums run over `source - destination`,
//!   the argument of [`crate::lattice::sum`].
//! - Two modes couple only at equal axial wavenumbers `kz` and equal polarization
//!   indices `pol`. `kz` is an exact label: the coupling compares it bit for bit.
//! - `pol` is the polarization index, 0 or 1, with the meanings of [`crate::sw`].
//! - [`Radial::Regular`](crate::special::Radial::Regular) and
//!   [`Radial::Singular`](crate::special::Radial::Singular) are the treams `modetype`
//!   values 'regular' and 'singular'.
//! - In [`CartesianTranslation`], every field other than `value` holds the derivative of
//!   `value` with respect to the input it is named after; `position` holds the
//!   derivatives with respect to the displacement components.
//!
//! The inline `tests` modules check how each file computes its results; the physical
//! and adjoint identities of translations and expansions are in `properties/waves.rs`.

mod cartesian;
mod expansion;
mod polar;
mod to_sw;

pub(crate) use cartesian::regular_harmonic;
pub use cartesian::{CartesianTranslation, cartesian_translation};
pub use expansion::{ExpansionResidual, LatticeExpansionResidual, expansion, lattice_expansion};
pub use polar::{
    PolarTranslationResidual, polar_translation, polar_translation_array, tl_vcw, translate,
};
pub(crate) use to_sw::coefficient;
pub use to_sw::{ToSwResidual, to_sw, to_sw_matrix};

use crate::{Complex, Error, MAX_DEGREE, Result, basis::ModeLabel, numerics::Jet};

/// One cylindrical-wave mode: axial wavenumber `kz`, order `m` and polarization index
/// `pol`.
///
/// Upstream: the `(kz, m, pol)` of one entry of `treams.CylindricalWaveBasis`; [`Basis`]
/// adds the position index `pidx`.
#[derive(Clone, Copy, Debug)]
pub struct Mode {
    /// Axial wavenumber (a fixed mode label for basis expansion).
    pub kz: f64,
    /// Integer azimuthal order.
    pub m: i32,
    /// Polarization index: 1 or 0, as in treams.
    pub pol: u8,
}

impl ModeLabel for Mode {
    /// Require a finite `kz`, `|m| <= MAX_DEGREE` and `pol` 0 or 1.
    fn validate(self) -> Result<()> {
        if !self.kz.is_finite() || self.m.unsigned_abs() > MAX_DEGREE.unsigned_abs() || self.pol > 1
        {
            return Err(Error::InvalidInput(
                // 128 is MAX_DEGREE.
                "require finite kz, |m| <= 128 and polarization 0 or 1".into(),
            ));
        }
        Ok(())
    }

    fn pol(self) -> u8 {
        self.pol
    }
}

/// Transverse wavenumber `sqrt(k² - kz²)` on the branch with nonnegative imaginary part.
pub(crate) fn transverse<const N: usize>(k: Jet<N>, kz: Jet<N>) -> Jet<N> {
    let root = (k * k - kz * kz).sqrt();
    if root.value.im < 0.0 { -root } else { root }
}

/// [`transverse`] of a complex wavenumber and a real axial wavenumber.
fn transverse_wavenumber(k: Complex, kz: f64) -> Complex {
    transverse(Jet::<0>::constant(k), Jet::constant(kz)).value
}

/// Cylindrical modes at Cartesian expansion positions.
///
/// Upstream: `treams.CylindricalWaveBasis`.
pub type Basis = crate::basis::Basis<Mode>;
