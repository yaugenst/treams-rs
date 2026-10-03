//! The family of a sum, the scalar wave it adds up, and its angular factors.
//!
//! Upstream: the `lsumsw*` (spherical) and `lsumcw*` (cylindrical) families of
//! `treams.lattice`.

use crate::{Complex, Error, MAX_DEGREE, Result, numerics::Jet, special};

/// The family of a lattice sum: spherical or cylindrical.
///
/// Each variant is the scalar outgoing wave the sum adds up. `vectorwaves::Family`
/// selects vector waves instead; code outside each module names both with their module.
///
/// Upstream: the `lsumsw*` and `lsumcw*` functions of `treams.lattice`.
#[derive(Clone, Copy, Debug)]
pub enum Family {
    /// `h_l^(1)(kr) Y_lm(theta, phi)`.
    Spherical {
        /// Degree.
        l: i32,
        /// Order.
        m: i32,
    },
    /// `H_m^(1)(kr) exp(i m phi)`.
    Cylindrical {
        /// Signed order.
        m: i32,
    },
}

impl Family {
    /// Checks the labels, and for cylindrical sums the lattice dimension: spherical sums
    /// take `0 <= l <= MAX_DEGREE` and `|m| <= l`, cylindrical sums take
    /// `|m| <= MAX_DEGREE` and `dim <= 2`.
    ///
    /// The bound keeps the Kambe integrals of the sums within the orders that
    /// `special::MAX_LABEL = 2 MAX_DEGREE + 4` admits: spherical real-space terms and
    /// their derivatives take orders `2 l` and `2 l + 2`, and the reciprocal terms of 1D
    /// spherical sums orders down to `-(2 l + 3)`.
    pub(super) fn validate(self, dim: usize) -> Result<()> {
        let valid = match self {
            Self::Spherical { l, m } => {
                (0..=MAX_DEGREE).contains(&l) && m.unsigned_abs() <= l.unsigned_abs()
            }
            Self::Cylindrical { m } => (-MAX_DEGREE..=MAX_DEGREE).contains(&m) && dim <= 2,
        };
        if !valid {
            return Err(Error::InvalidInput(
                "invalid lattice wave degree or dimension".into(),
            ));
        }
        Ok(())
    }

    /// The Cartesian axis of each lattice coordinate: z for 1D spherical sums, and x, y,
    /// z in order otherwise (x for 1D cylindrical sums, xy for 2D sums).
    pub(super) fn axes(self, dim: usize) -> [usize; 3] {
        if matches!(self, Self::Spherical { .. }) && dim == 1 {
            [2, 0, 1]
        } else {
            [0, 1, 2]
        }
    }
}

/// `x + i y`, or `x - i y` for negative `m`: the azimuthal factor whose `|m|`-th power
/// carries the order of cylindrical waves and of the lattice polynomials.
pub(super) fn azimuthal<const N: usize>(x: Jet<N>, y: Jet<N>, m: i32) -> Jet<N> {
    x + Complex::i() * if m < 0 { -y } else { y }
}

/// `(-1)^m` for negative orders, relating `H_m` to `H_|m|`, and one otherwise.
pub(super) const fn negative_order_sign(m: i32) -> f64 {
    if m < 0 && m % 2 != 0 { -1.0 } else { 1.0 }
}

/// The solid harmonic `r^l P_l^m(cos theta) e^(i m phi)` of the jet vector `r`, with its
/// derivatives by the chain rule through the Cartesian gradient. Times
/// `harmonic_normalization(l, m)` it is `|r|^l Y_lm(r / |r|)`, a polynomial in `r` that
/// stays regular at `r = 0`.
pub(super) fn solid_jet<const N: usize>(l: i32, m: i32, r: [Jet<N>; 3]) -> Jet<N> {
    let solid = special::solid::<false>(l, m, r.map(|r| r.value.re));
    Jet {
        value: solid.value,
        derivative: std::array::from_fn(|i| {
            (0..3).map(|j| solid.gradient[j] * r[j].derivative[i]).sum()
        }),
    }
}
