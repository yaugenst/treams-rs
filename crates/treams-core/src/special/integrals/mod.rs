//! Incomplete gamma and Kambe integrals used by the Ewald lattice sums.
//! Recurrences follow DLMF 8.8 and Kambe (1968), as in treams.
//!
//! Upstream: `treams.special.incgamma` and `intkambe` (`_integrals.pyx`).
//!
//! [`incgamma`] and [`intkambe`] evaluate one integral after checking its inputs.
//! [`incgamma_array`] and [`intkambe_array`] evaluate them over arrays and return an
//! [`IncgammaResidual`] or [`IntkambeResidual`], which keeps the arguments for the
//! gradient.
//! The private submodules hold the numerics:
//!
//! - `gamma`: the upper incomplete gamma function and its scaled ladder;
//! - `kambe`: the Kambe integrals;
//! - `double`: double-double arithmetic for the small-split Kambe series.
//!
//! The Kambe integrals have several entry points, one per kind of caller:
//!
//! | Entry point | Returns | Caller |
//! |---|---|---|
//! | [`intkambe`] | `I_n(z, eta)` after checking the inputs | `treams_rs.special.intkambe`, [`intkambe_array`] |
//! | `kambe` | `I_n(z, eta)` without checks | [`intkambe`], the [`IntkambeResidual`] pullback (order `n + 2`), real-space terms of the Ewald sums (`lattice::real`) |
//! | `kambe_with_bound` | `I_n` and a bound on its rounding error | integer-order reduced integrals of the 1D spherical sums (`lattice::reduced`) |
//! | `kambe_series`, `kambe_series_with` | the series over scaled incomplete gamma functions | `kambe`, `kambe_with_bound` and `SmallSplitKambe` only |
//! | `EvenKambe` | the even orders of one argument from one recurrence | half-integer reduced integrals of the 2D spherical and 1D cylindrical sums (`lattice::reduced`) |
//! | `SmallSplitKambe` | the orders of real-space terms at a split below every automatic one, with bounds | real-space terms at small splits (`lattice::real`) |
#![allow(clippy::float_cmp, clippy::while_float, clippy::indexing_slicing)] // Half-integer degrees step exactly by one; order tables cover each requested order.

mod double;
mod gamma;
mod kambe;

pub(crate) use gamma::{ScaledGammaLadder, upper_gamma};
pub(crate) use kambe::{EvenKambe, SmallSplitKambe, kambe, kambe_argument, kambe_with_bound};

use super::MAX_LABEL;
use crate::{
    Complex, Error, MAX_DEGREE, Result,
    numerics::{
        broadcast::{self, element},
        finite,
        parallel::Parallel,
    },
};
use gamma::gamma_derivative;

/// Upper incomplete gamma function `Gamma(n, z) = integral_z^infinity t^(n-1) e^-t dt`.
///
/// The degree `n` is an integer or half-integer. The branch cut lies on the negative
/// real axis, and the sign of a zero `Im z` selects its side.
///
/// Upstream: `treams.special.incgamma`. Differences: `|n|` above 128 and non-finite `z`
/// give an error; treams runs one degree recurrence, while this function chooses among
/// four methods the one that keeps relative accuracy at `z`.
pub fn incgamma(n: f64, z: Complex) -> Result<Complex> {
    if !n.is_finite() || (2.0 * n).fract() != 0.0 || n.abs() > f64::from(MAX_DEGREE) || !finite(z) {
        return Err(Error::InvalidInput(
            // 128 is MAX_DEGREE.
            "gamma degree must be an integer or half-integer in [-128, 128], and z finite".into(),
        ));
    }
    Ok(upper_gamma(n, z))
}

/// The Kambe integral `I_n(z, eta) = integral_eta^infinity t^n exp(-z^2 t^2/2 + 1/(2t^2)) dt`.
///
/// Upstream: `treams.special.intkambe`. Differences: `|n|` above 260 and non-finite
/// arguments give an error. The odd orders -3 and -1 come from a series over scaled
/// incomplete gamma functions, and lower orders from that series or the downward
/// recurrence of treams, whichever bounds the rounding error more tightly.
pub fn intkambe(n: i32, z: Complex, eta: Complex) -> Result<Complex> {
    if !(-MAX_LABEL..=MAX_LABEL).contains(&n) || !finite(z) || !finite(eta) {
        return Err(Error::InvalidInput(
            // 260 is MAX_LABEL.
            "Kambe order must be in [-260, 260] and arguments finite".into(),
        ));
    }
    Ok(kambe(n, z, eta))
}

/// Elementwise integral evaluations and pullbacks run in parallel from this many elements.
const PARALLEL: Parallel = Parallel::AtLeast(1024);
const BROADCAST: &str = "arrays must have equal lengths or scalar inputs";
const COTANGENT: &str = "integral cotangent must be finite and match output";

/// [`incgamma`] over arrays, each input one value or one value per element, with the
/// residual of their argument gradient.
///
/// Upstream: `treams.special.incgamma` over arrays.
pub fn incgamma_array(
    degrees: Vec<f64>,
    arguments: Vec<Complex>,
) -> Result<(Vec<Complex>, IncgammaResidual)> {
    let residual = IncgammaResidual::new(degrees, arguments)?;
    let value = broadcast::map(residual.size, PARALLEL, |i| {
        incgamma(
            element(&residual.degrees, i),
            element(&residual.arguments, i),
        )
    })?;
    Ok((value, residual))
}

/// What [`incgamma_array`] saves for its pullback: the degrees and the arguments. Only
/// the arguments get gradients.
#[derive(Debug)]
pub struct IncgammaResidual {
    degrees: Vec<f64>,
    arguments: Vec<Complex>,
    size: usize,
}
impl IncgammaResidual {
    /// Keep the broadcast inputs needed by derivatives without evaluating values.
    pub fn new(degrees: Vec<f64>, arguments: Vec<Complex>) -> Result<Self> {
        let size = broadcast::size(&[degrees.len(), arguments.len()], BROADCAST)?;
        Ok(Self {
            degrees,
            arguments,
            size,
        })
    }

    /// Directional derivative with respect to the complex argument.
    pub fn pushforward(&self, tangents: [&[Complex]; 1]) -> Result<Vec<Complex>> {
        broadcast::pushforward(
            tangents,
            self.size,
            "gamma tangent must be finite and broadcast to output",
            PARALLEL,
            |i, [tangent]| {
                if tangent == Complex::default() {
                    return Ok(tangent);
                }
                let value = tangent
                    * gamma_derivative(element(&self.degrees, i), element(&self.arguments, i));
                if !finite(value) {
                    return Err(Error::SpecialFunction(
                        "gamma derivative is singular or overflowed".into(),
                    ));
                }
                Ok(value)
            },
        )
    }

    /// The gradient with respect to the arguments, given the gradient `cotangent` with
    /// respect to the values: `cotangent` times the conjugate of `dGamma/dz`.
    pub fn pullback(&self, cotangent: &[Complex]) -> Result<Vec<Complex>> {
        let [gradient] = broadcast::pullback(
            cotangent,
            self.size,
            COTANGENT,
            [self.arguments.len()],
            PARALLEL,
            |i, &g| {
                if g == Complex::default() {
                    return Ok([g]);
                }
                let gradient =
                    g * gamma_derivative(element(&self.degrees, i), element(&self.arguments, i))
                        .conj();
                if !finite(gradient) {
                    return Err(Error::SpecialFunction(
                        "gamma derivative is singular or overflowed".into(),
                    ));
                }
                Ok([gradient])
            },
        )?;
        Ok(gradient)
    }
}

/// [`intkambe`] over arrays, each input one value or one value per element, with the
/// residual of their `z` and `eta` gradients; the integer order has none.
///
/// Upstream: `treams.special.intkambe` over arrays.
pub fn intkambe_array(
    orders: Vec<i32>,
    arguments: [Vec<Complex>; 2],
) -> Result<(Vec<Complex>, IntkambeResidual)> {
    let residual = IntkambeResidual::new(orders, arguments)?;
    let value = broadcast::map(residual.size, PARALLEL, |i| {
        intkambe(
            element(&residual.orders, i),
            element(&residual.arguments[0], i),
            element(&residual.arguments[1], i),
        )
    })?;
    Ok((value, residual))
}

/// What [`intkambe_array`] saves for its pullback: the orders and the arguments. The
/// pullback uses the adjacent-order recurrence and the endpoint integrand.
#[derive(Debug)]
pub struct IntkambeResidual {
    orders: Vec<i32>,
    arguments: [Vec<Complex>; 2],
    size: usize,
}
impl IntkambeResidual {
    /// Keep the broadcast inputs needed by derivatives without evaluating values.
    pub fn new(orders: Vec<i32>, arguments: [Vec<Complex>; 2]) -> Result<Self> {
        let size = broadcast::size(
            &[orders.len(), arguments[0].len(), arguments[1].len()],
            BROADCAST,
        )?;
        Ok(Self {
            orders,
            arguments,
            size,
        })
    }

    /// Directional derivative in `z` and `eta`. Inactive arguments do not evaluate
    /// their derivative, so a direction in `eta` remains valid when only `dz` is singular.
    pub fn pushforward(&self, tangents: [&[Complex]; 2]) -> Result<Vec<Complex>> {
        broadcast::pushforward(
            tangents,
            self.size,
            "Kambe tangents must be finite and broadcast to output",
            PARALLEL,
            |i, [tz, teta]| {
                if tz == Complex::default() && teta == Complex::default() {
                    return Ok(Complex::default());
                }
                let n = element(&self.orders, i);
                let [z, eta] = self.arguments.each_ref().map(|v| element(v, i));
                let dz = if tz == Complex::default() {
                    Complex::default()
                } else {
                    tz * kambe_z_derivative(n, z, eta)
                };
                let deta = if teta == Complex::default() {
                    Complex::default()
                } else {
                    teta * kambe_eta_derivative(n, z, eta)
                };
                let value = dz + deta;
                if !finite(value) {
                    return Err(Error::SpecialFunction(
                        "Kambe derivative is singular or overflowed".into(),
                    ));
                }
                Ok(value)
            },
        )
    }

    /// The gradients with respect to `z` and `eta`, given the gradient `cotangent` with
    /// respect to the values. At `z = 0` the `z` gradient of orders `n <= -3` is its
    /// limit, zero. Orders `n >= -2` have no `z` gradient at `z = 0` and give an error.
    pub fn pullback(&self, cotangent: &[Complex]) -> Result<[Vec<Complex>; 2]> {
        let lengths = self.arguments.each_ref().map(Vec::len);
        broadcast::pullback(
            cotangent,
            self.size,
            COTANGENT,
            lengths,
            PARALLEL,
            |i, &g| {
                if g == Complex::default() {
                    return Ok([g; 2]);
                }
                let n = element(&self.orders, i);
                let [z, eta] = self.arguments.each_ref().map(|v| element(v, i));
                let dz = kambe_z_derivative(n, z, eta);
                let deta = kambe_eta_derivative(n, z, eta);
                let result = [dz, deta].map(|d| g * d.conj());
                if result.iter().any(|&v| !finite(v)) {
                    return Err(Error::SpecialFunction(
                        "Kambe derivative is singular or overflowed".into(),
                    ));
                }
                Ok(result)
            },
        )
    }
}

fn kambe_z_derivative(n: i32, z: Complex, eta: Complex) -> Complex {
    if z == Complex::default() && n <= -3 {
        Complex::default()
    } else {
        -z * kambe(n + 2, z, eta)
    }
}

fn kambe_eta_derivative(n: i32, z: Complex, eta: Complex) -> Complex {
    -eta.powi(n) * (0.5 * (1.0 / (eta * eta) - z * z * eta * eta)).exp()
}

/// Reference cases of a `n re(z) im(z) [...]: re im` table with `#` comments, for the
/// gamma and Kambe tests.
#[cfg(test)]
fn reference(text: &str) -> Vec<(Vec<f64>, Complex)> {
    crate::test_support::table::<f64, f64>(text)
        .into_iter()
        .map(|(key, value)| (key, Complex::new(value[0], value[1])))
        .collect()
}
