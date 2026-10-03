//! Polarization conventions of vector waves: how the polarization index `pol` combines
//! the two multipole parts M and N of a wave.
//!
//! - Parity: `pol = 0` is M and `pol = 1` is N.
//! - Helicity: `(N + (2 pol - 1) M) / sqrt(2)`, so `pol = 0` is negative and `pol = 1`
//!   positive helicity.
//!
//! Upstream: the convention of `treams.special.vsw_A` and `vcw_A`. [`polarized_wave`]
//! applies it to vector waves, and `legendre::polarized_angular` applies the same sign
//! `2 pol - 1` ([`helicity_sign`]) to the angular factors `tau + (2 pol - 1) pi` of
//! plane-wave expansions.
#![allow(clippy::indexing_slicing)] // Fixed Cartesian component arrays.

use std::f64::consts::FRAC_1_SQRT_2;

use crate::{Error, Result, numerics::Jet};

/// The sign `2 pol - 1` of polarization index `pol` in the helicity convention: -1 for
/// negative helicity (0), +1 for positive helicity (1).
#[inline]
pub(crate) fn helicity_sign(pol: u8) -> f64 {
    2.0 * f64::from(pol) - 1.0
}

/// The polarization index of an integer label, which must be 0 or 1.
#[inline]
pub fn pol_index(label: i64) -> Result<u8> {
    match label {
        0 => Ok(0),
        1 => Ok(1),
        _ => Err(Error::InvalidInput("polarization must be 0 or 1".into())),
    }
}

/// Checks that `pol` is a polarization index: 0 or 1.
#[inline]
pub(crate) fn check_pol(pol: u8) -> Result<()> {
    if pol > 1 {
        return Err(Error::InvalidInput("polarization must be 0 or 1".into()));
    }
    Ok(())
}

/// The vector wave of polarization `pol` from its M and N parts: M (0) or N (1) in
/// parity polarization, `(N -+ M) / sqrt(2)` for negative (0) or positive (1) helicity.
pub(crate) fn polarized_wave<const N: usize>(
    m: [Jet<N>; 3],
    n: [Jet<N>; 3],
    pol: u8,
    helicity: bool,
) -> [Jet<N>; 3] {
    if helicity {
        std::array::from_fn(|i| (n[i] + helicity_sign(pol) * m[i]) * FRAC_1_SQRT_2)
    } else if pol == 0 {
        m
    } else {
        n
    }
}
