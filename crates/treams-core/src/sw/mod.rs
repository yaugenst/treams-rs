//! Spherical vector waves: mode labels, translation coefficients of mode pairs, expansion
//! matrices between bases and over lattices, and the cylindrical waves radiated by a
//! z-periodic chain of spheres.
//!
//! Upstream: `treams.sw`, and `treams.special.tl_vsw_A`, `tl_vsw_B`, `tl_vsw_rA` and
//! `tl_vsw_rB`.
//!
//! | File | Contents | Upstream |
//! |---|---|---|
//! | `coupling.rs` | degree terms `(p, weight)` of one mode pair, from Wigner 3j symbols | `special._tl_vsw_helper`, the sums of `tl_vsw_A` and `tl_vsw_B` |
//! | `polar.rs` | [`PolarTranslation`] and [`polar_translation_array`]: coefficients at polar arguments `(kr, theta, phi)` | `sw.translate`, `special.tl_vsw_*` |
//! | `cartesian.rs` | [`cartesian_translation`]: one coefficient at a Cartesian displacement, with its derivatives | treams-rs extension |
//! | `plan.rs` | the couplings of whole mode blocks, shared by every pair of positions | the loops of `sw.translate` over a basis |
//! | `expansion.rs` | [`expansion()`], [`lattice_expansion`] and [`lattice_expansion_from_table`] | `Expand`, `ExpandLattice`, `sw.translate_periodic` |
//! | `periodic_to_cw.rs` | [`periodic_to_cw()`] and [`periodic_to_cw_matrix`] | `sw.periodic_to_cw`, `ExpandLattice` to a cylindrical basis |
//!
//! Conventions:
//!
//! - A translation from a source position to a destination position takes the
//!   displacement `destination - source`. Lattice sums run over `source - destination`,
//!   the argument of [`crate::lattice::sum`]; the lattice paths negate the displacement
//!   where they call it.
//! - `pol` is the polarization index, 0 or 1. With `helicity` true, 1 is positive and 0
//!   negative helicity; with `helicity` false, 1 is TM (the N multipoles) and 0 TE (the M
//!   multipoles). Bases list pol 1 before pol 0, as treams does.
//! - [`Radial::Regular`](crate::special::Radial::Regular) and
//!   [`Radial::Singular`](crate::special::Radial::Singular) are the treams `modetype`
//!   values 'regular' and 'singular'; singular waves use the outgoing Hankel function.
//! - In a derivative struct such as [`CartesianTranslation`], every field other than
//!   `value` holds the derivative of `value` with respect to the input it is named after;
//!   `position` holds the derivatives with respect to the displacement components.
//!
//! The inline `tests` modules check how each file computes its results; the physical
//! and adjoint identities of translations and expansions are in `properties/waves.rs`.

mod cartesian;
mod coupling;
mod expansion;
mod periodic_to_cw;
mod plan;
mod polar;

#[cfg(test)]
pub(crate) use cartesian::harmonic;
pub use cartesian::{CartesianTranslation, cartesian_translation};
pub(crate) use coupling::terms;
pub use expansion::{
    ExpansionResidual, LatticeExpansionFromTableResidual, LatticeExpansionResidual, expansion,
    lattice_expansion, lattice_expansion_from_table,
};
pub use periodic_to_cw::{
    PeriodicToCwGradient, PeriodicToCwResidual, periodic_to_cw, periodic_to_cw_matrix,
};
pub(crate) use plan::TranslationPlan;
pub use polar::{PolarTranslation, PolarTranslationResidual, polar_translation_array};

use std::collections::BTreeMap;

use crate::{Error, MAX_DEGREE, Result, basis::ModeLabel};

/// One spherical-wave mode: degree `l`, order `m` and polarization index `pol`.
///
/// Upstream: the `(l, m, pol)` of one entry of `treams.SphericalWaveBasis`; [`Basis`]
/// adds the position index `pidx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Mode {
    /// Multipole degree, `1 <= l <= MAX_DEGREE`.
    pub l: i32,
    /// Azimuthal order, `|m| <= l`.
    pub m: i32,
    /// Polarization index: 1 or 0, as in treams.
    pub pol: u8,
}

impl ModeLabel for Mode {
    /// Require `1 <= l <= MAX_DEGREE`, `|m| <= l` and `pol` 0 or 1.
    fn validate(self) -> Result<()> {
        if !(1..=MAX_DEGREE).contains(&self.l)
            || self.m.unsigned_abs() > self.l.unsigned_abs()
            || self.pol > 1
        {
            return Err(Error::InvalidInput("invalid spherical mode".into()));
        }
        Ok(())
    }

    fn pol(self) -> u8 {
        self.pol
    }
}

/// Spherical modes at Cartesian expansion positions.
///
/// Upstream: `treams.SphericalWaveBasis`.
pub type Basis = crate::basis::Basis<Mode>;

impl crate::basis::Basis<Mode> {
    /// The basis entries and modes at each position index, in basis order.
    pub(crate) fn groups(&self) -> BTreeMap<usize, (Vec<usize>, Vec<Mode>)> {
        let mut groups = BTreeMap::<usize, (Vec<usize>, Vec<Mode>)>::new();
        for (index, &(pidx, mode)) in self.modes.iter().enumerate() {
            let group = groups.entry(pidx).or_default();
            group.0.push(index);
            group.1.push(mode);
        }
        groups
    }
}

/// The modes of every degree up to `lmax`, ordered by degree, then order, then pol 1
/// before pol 0.
///
/// Upstream: the modes of `treams.SphericalWaveBasis.default(lmax)` at one position.
pub fn modes(lmax: u32) -> Result<Vec<Mode>> {
    if !(1..=MAX_DEGREE.unsigned_abs()).contains(&lmax) {
        return Err(Error::InvalidInput(
            // 128 is MAX_DEGREE.
            "require 1 <= lmax <= 128".into(),
        ));
    }
    let lmax = i32::try_from(lmax).map_err(|_| Error::InvalidInput("invalid lmax".into()))?;
    Ok((1..=lmax)
        .flat_map(|l| (-l..=l).flat_map(move |m| [Mode { l, m, pol: 1 }, Mode { l, m, pol: 0 }]))
        .collect())
}
