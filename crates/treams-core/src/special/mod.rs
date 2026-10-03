//! Special functions: Bessel, Legendre, Ferrers and Wigner functions, solid harmonics,
//! incomplete gamma and Kambe integrals, and coordinate transforms.
//!
//! Upstream: `treams.special`.
//!
//! | File | Contents | treams source |
//! |---|---|---|
//! | `bessel` | Cylindrical and spherical Bessel functions and radial jets | `jv`, `yv`, `hankel1`, `hankel2` and `spherical_*` (scipy), `_bessel.pyx` |
//! | `legendre` | Associated Legendre functions and the angular functions `pi` and `tau` | `lpmv`, `pi_fun`, `tau_fun` of `_waves.pyx` |
//! | `ferrers` | Ferrers functions of real, non-integer degree | `treams.special.lpmv` |
//! | `wigner` | Wigner 3j symbols and Wigner small-d and D functions | `_wigner3j.pyx`, `_wignerd.pyx` |
//! | `harmonics` | Cartesian solid harmonics and the spherical-harmonic normalization | none: a treams-rs addition for translations and lattice sums |
//! | [`coordinates`] | Point and vector-component transforms between charts | `_coord.pyx` |
//! | `integrals` | Upper incomplete gamma function and Kambe integrals | `_integrals.pyx` |
//! | `polarization` | Helicity and parity combinations of vector waves | the `vsw_A` convention of `_waves.pyx` |
//!
//! ## Label bounds
//!
//! treams accepts any degree. Here the multipole degree, the integer-order radial jets
//! and the Kambe and Wigner labels have fixed bounds, so the caches and tables that
//! serve them have a fixed size:
//!
//! - [`MAX_DEGREE`] = 128 bounds the multipole degree `l`.
//! - `MAX_ORDER` = 256 = `2 MAX_DEGREE` bounds the integer-order radial jets and the
//!   log-factorial cache, since translations couple degrees `l1 + l2`.
//! - [`MAX_LABEL`] = 260 bounds Kambe orders, and Wigner labels at the Python
//!   interface. Lattice sums and their derivatives take Kambe orders `n` up to
//!   `|n| = 2 MAX_DEGREE + 3` = 259, which the derivatives of 1D spherical sums reach;
//!   the bound keeps one order of margin.
//! - `SERIES_RADIUS` = 0.5 is the argument modulus below which regular radial functions
//!   use their power series instead of AMOS evaluations.
//!
//! ## Two kinds of jet
//!
//! A jet is a value together with some of its derivatives. [`RadialJet`] holds a radial
//! function and its first two derivatives in its one complex argument, as the vector
//! waves need them. `numerics::Jet<N>` holds a value and its first derivatives along `N`
//! inputs; kernels generic in `N` use it to evaluate (`N = 0`) and to differentiate with
//! one code path. [`spherical_radial`] and [`cylindrical_radial`] return a `RadialJet`,
//! and `radial_jet` turns one into a `Jet<N>` by the chain rule.

mod bessel;
pub mod coordinates;
mod ferrers;
mod harmonics;
mod integrals;
mod legendre;
mod polarization;
mod wigner;

pub use bessel::{
    Bessel, BesselResidual, Radial, RadialJet, bessel, bessel_array, cylindrical_radial,
    spherical_radial,
};
pub(crate) use bessel::{hankel_below, radial_jet, spherical_hankels, spherical_radial_sequence};
#[cfg(test)]
pub(crate) use ferrers::ferrers_real_degree;
#[cfg(test)]
pub(crate) use ferrers::ferrers_route_switches;
pub(crate) use harmonics::{
    Solid, SolidTable, direction, harmonic_normalization, on_sphere, solid, tangent,
};
pub(crate) use integrals::{
    EvenKambe, ScaledGammaLadder, SmallSplitKambe, kambe, kambe_argument, kambe_with_bound,
    upper_gamma,
};
pub use integrals::{
    IncgammaResidual, IntkambeResidual, incgamma, incgamma_array, intkambe, intkambe_array,
};
pub use legendre::{Angular, AngularResidual, angular_array, angular_value, lpmv_real};
pub(crate) use legendre::{legendre_factor, legendre_factors, polar_trig, polarized_angular};
pub use polarization::pol_index;
pub(crate) use polarization::{check_pol, helicity_sign, polarized_wave};
pub(crate) use wigner::{Wigner3jRow, index, ladder, wigner_small_d_matrix};
pub use wigner::{WignerDResidual, wigner_d, wigner_d_array, wigner_small_d, wigner3j};

use crate::MAX_DEGREE;

/// The largest order of the integer-order radial jets ([`spherical_radial`],
/// `spherical_radial_sequence`, [`cylindrical_radial`]) and of the order difference of
/// cylindrical translations: translations couple degrees `l1 + l2` up to `2 MAX_DEGREE`,
/// and cylindrical translations reach orders `m1 - m2` of the same size. [`bessel()`]
/// takes any finite order.
pub(crate) const MAX_ORDER: i32 = 2 * MAX_DEGREE;

/// The largest `|n|` of an integer special-function label (Kambe orders, Wigner labels).
///
/// Lattice sums of degree `l` up to [`MAX_DEGREE`] use Kambe orders `n` with `|n|` up
/// to `2 l + 1`, and their derivatives up to `2 l + 3`; the reciprocal terms of 1D
/// spherical sums reach both. Translations use Wigner labels up to
/// `l1 + l2 = 2 MAX_DEGREE`. The bound adds one order of margin.
pub const MAX_LABEL: i32 = 2 * MAX_DEGREE + 4;

/// The radius `|z|` below which regular radial functions use their power series instead
/// of AMOS Bessel evaluations.
///
/// Kernels that expand small arguments themselves (vector waves, polar translations,
/// fields) switch at the same radius, so values and derivatives stay continuous there.
pub(crate) const SERIES_RADIUS: f64 = 0.5;

/// `ln n!`: libm's `lgamma(n + 1)`, cached for `0 <= n <= MAX_ORDER`. The factorials of
/// wave labels recur in every normalization, lattice image and angular closed form, so
/// the cache computes each one once.
pub(crate) fn log_factorial(n: i32) -> f64 {
    const CACHED: usize = MAX_ORDER.unsigned_abs() as usize + 1;
    static LOG_FACTORIAL: std::sync::OnceLock<[f64; CACHED]> = std::sync::OnceLock::new();
    let values = LOG_FACTORIAL.get_or_init(|| {
        std::array::from_fn(|i| libm::lgamma(f64::from(u32::try_from(i).unwrap_or_default()) + 1.0))
    });
    usize::try_from(n)
        .ok()
        .and_then(|i| values.get(i))
        .copied()
        .unwrap_or_else(|| libm::lgamma(f64::from(n) + 1.0))
}

/// `sqrt((l - m)! / (l + m)!)`, the order factor of spherical-harmonic normalizations.
pub(crate) fn factorial_ratio_sqrt(l: i32, m: i32) -> f64 {
    (0.5 * (log_factorial(l - m) - log_factorial(l + m))).exp()
}
