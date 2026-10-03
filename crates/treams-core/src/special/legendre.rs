//! Associated Legendre functions of integer degree and the vector-wave angular functions
//! `pi` and `tau`, with analytic argument derivatives. Real-degree Legendre values come
//! from the Ferrers functions in `ferrers`.
//!
//! Upstream: `treams.special.lpmv`, `pi_fun` and `tau_fun` (`_waves.pyx`).

use std::f64::consts::PI;

use super::helicity_sign;
use crate::{
    Complex, Error, MAX_DEGREE, Result,
    numerics::{
        Jet,
        broadcast::{self, element},
        finite,
        parallel::Parallel,
    },
};

/// Elementwise angular evaluations and pullbacks run in parallel from this many elements.
const PARALLEL: Parallel = Parallel::AtLeast(1024);

/// Associated Legendre functions `F_l^m` after factoring out `sin(theta)^|m|`, visited
/// for every `l` in `|m|..=max` in order from one run of the degree recurrence. Keeping
/// the factor explicit avoids 0/0 and cancellation in angular functions near the poles.
pub(crate) fn legendre_factors<const N: usize>(
    max: i32,
    m: i32,
    z: Jet<N>,
    mut visit: impl FnMut(Jet<N>),
) {
    let order = m.abs();
    if max < order {
        return;
    }
    let mut p = Jet::constant(1.0);
    if m >= 0 {
        for k in 1..=m {
            p *= -f64::from(2 * k - 1);
        }
    } else {
        for k in 1..=-m {
            p *= 1.0 / f64::from(2 * k);
        }
    }
    visit(p);
    if max == order {
        return;
    }
    let mut prev = p;
    p *= (f64::from(2 * order + 1) / f64::from(order - m + 1)) * z;
    visit(p);
    for k in (order + 2)..=max {
        let next =
            (f64::from(2 * k - 1) * z * p - f64::from(k + m - 1) * prev) * (1.0 / f64::from(k - m));
        prev = p;
        p = next;
        visit(p);
    }
}

/// `F_l^m` and `F_(l-1)^m` of [`legendre_factors`], with `F_(l-1)^m = 0` for `l = |m|`
/// and both zero for `|m| > l`.
fn legendre_factor_pair<const N: usize>(l: i32, m: i32, z: Jet<N>) -> [Jet<N>; 2] {
    let mut pair = [Jet::default(); 2];
    legendre_factors(l, m, z, |p| pair = [p, pair[0]]);
    pair
}

/// Associated Legendre polynomial `F_l^m` after factoring out `sin(theta)^|m|`.
pub(crate) fn legendre_factor<const N: usize>(l: i32, m: i32, z: Jet<N>) -> Jet<N> {
    let [value, _] = legendre_factor_pair(l, m, z);
    value
}

/// `tau_l^m / sin(theta)^(|m|-1)` for `m != 0` from one Legendre recurrence, by DLMF
/// 14.10.5: `sin(theta) tau = l cos(theta) P_l^m - (l + m) P_(l-1)^m`.
fn tau_factor<const N: usize>(
    l: i32,
    m: i32,
    [upper, lower]: [Jet<N>; 2],
    cosine: Jet<N>,
) -> Jet<N> {
    f64::from(l) * cosine * upper - f64::from(l + m) * lower
}

/// The angular functions `[pi, tau]` from the polar cosine and sine, with the sine
/// factor kept explicit at the poles.
fn angular_jets<const N: usize>(l: i32, m: i32, cosine: Jet<N>, sine: Jet<N>) -> [Jet<N>; 2] {
    if m == 0 {
        // tau_l^0 = P_l^1.
        return [Jet::default(), sine * legendre_factor(l, 1, cosine)];
    }
    let pair = legendre_factor_pair(l, m, cosine);
    let power = sine.powi(m.abs() - 1);
    [
        f64::from(m) * power * pair[0],
        power * tau_factor(l, m, pair, cosine),
    ]
}

/// Angular factor coupling the vector spherical wave `(l, m)` of polarization
/// `mode_pol` to a plane wave of polarization `pol` whose direction has the polar
/// cosine and sine given: `tau -+ pi` for negative (0) or positive (1) helicity; in
/// parity polarization `tau` for the mode's own polarization and `pi` for the other.
/// The polarization module states this rule for the vector waves themselves.
pub(crate) fn polarized_angular<const N: usize>(
    l: i32,
    m: i32,
    [cosine, sine]: [Jet<N>; 2],
    [mode_pol, pol]: [u8; 2],
    helicity: bool,
) -> Jet<N> {
    let [pi, tau] = angular_jets(l, m, cosine, sine);
    if helicity {
        tau + helicity_sign(pol) * pi
    } else if mode_pol == pol {
        tau
    } else {
        pi
    }
}

/// Associated Legendre polynomial or one of the two vector-wave angular functions.
///
/// Upstream: `treams.special.lpmv`, `pi_fun` and `tau_fun`.
#[derive(Clone, Copy, Debug)]
pub enum Angular {
    /// Associated Legendre P, including its Condon-Shortley phase.
    Legendre,
    /// m P / sqrt(1-z^2), with analytic polar limits.
    Pi,
    /// Polar-angle derivative of P(cos(theta)).
    Tau,
}

/// `sin(theta)^power = (1 - z^2)^(power / 2)` at the cosine `z`.
fn sine_power<const N: usize>(z: Jet<N>, power: i32) -> Jet<N> {
    let w = 1.0 - z * z;
    if power % 2 == 0 {
        w.powi(power / 2)
    } else if power > 1 && w.value == Complex::default() {
        // Odd powers >=3 have zero value and first derivative at the poles.
        Jet::default()
    } else {
        w.powi(power / 2) * w.sqrt()
    }
}

/// `F_l^m(z) sin(theta)^power`: the Legendre function for `power = |m|`, and `pi / m`
/// for `power = |m| - 1`.
fn angular_legendre<const N: usize>(l: i32, m: i32, z: Jet<N>, power: i32) -> Jet<N> {
    if m.abs() > l {
        return Jet::default();
    }
    legendre_factor(l, m, z) * sine_power(z, power)
}

/// The angular function `kind` of degree `l` and order `m` at `z`, with its derivative
/// for `N = 1`, after checking the labels and the argument.
#[allow(clippy::cast_possible_truncation)] // Integer labels are checked and bounded before conversion.
fn angular_jet<const N: usize>(l: f64, m: f64, z: Complex, kind: Angular) -> Result<Jet<N>> {
    if !finite(z)
        || !l.is_finite()
        || !m.is_finite()
        || m.fract() != 0.0
        || !(0.0..=f64::from(MAX_DEGREE)).contains(&l)
    {
        return Err(Error::InvalidInput(
            // 128 is MAX_DEGREE.
            "angular functions require 0 <= degree <= 128, integer order and finite argument"
                .into(),
        ));
    }
    if l.fract() != 0.0 && (!matches!(kind, Angular::Legendre) || z.im != 0.0) {
        return Err(Error::InvalidInput(
            "noninteger degrees require real Legendre arguments".into(),
        ));
    }
    if m.abs() > l {
        return Ok(Jet::default());
    }
    if l.fract() != 0.0 {
        let (value, derivative) = if N == 0 {
            super::ferrers::ferrers_real_degree::<false>(l, m as i32, z.re)?
        } else {
            super::ferrers::ferrers_real_degree::<true>(l, m as i32, z.re)?
        };
        return Ok(Jet::<N>::variable(z, 0).chain(value.into(), derivative.into()));
    }
    let (l, m) = (l as i32, m as i32);
    let z = Jet::<N>::variable(z, 0);
    let value = match kind {
        Angular::Legendre => angular_legendre(l, m, z, m.abs()),
        Angular::Pi if m == 0 => Jet::default(),
        Angular::Pi => f64::from(m) * angular_legendre(l, m, z, m.abs() - 1),
        Angular::Tau if m == 0 => angular_legendre(l, 1, z, 1),
        Angular::Tau => {
            tau_factor(l, m, legendre_factor_pair(l, m, z), z) * sine_power(z, m.abs() - 1)
        }
    };
    if !value.finite() {
        return Err(Error::SpecialFunction(
            if N == 0 {
                "non-finite angular result"
            } else {
                "angular argument derivative is undefined or non-finite"
            }
            .into(),
        ));
    }
    Ok(value)
}

/// Angular function at a cosine argument; real Legendre values also accept real degrees.
///
/// Upstream: `treams.special.lpmv` (which takes `(order, degree, x)`), `pi_fun` and
/// `tau_fun`. Differences: degrees outside `[0, 128]`, non-integer orders, and
/// non-integer degrees with `pi`, `tau` or a complex argument give an error.
pub fn angular_value(l: f64, m: f64, z: Complex, kind: Angular) -> Result<Complex> {
    Ok(angular_jet::<0>(l, m, z, kind)?.value)
}

/// Associated Legendre function `P_degree^order(x)` of a real argument, as a real value.
///
/// Outside `[-1, 1]` the odd orders with `|order| <= degree` have imaginary values: this
/// function rejects them, and [`angular_value`] returns them for a complex argument.
/// Orders with `|order| > degree` give zero.
///
/// Upstream: `treams.special.lpmv` with real arguments, which takes `(order, degree, x)`.
/// Differences: for an integer degree, `x` outside `[-1, 1]` and `0 < |order| <= degree`,
/// treams returns NaN, while this function returns the real value for even orders and an
/// error for odd orders. Degrees outside `[0, 128]`, non-integer orders, non-finite `x`
/// and non-integer degrees with `x` outside `(-1, 1]` give an error, where treams returns
/// a value or NaN.
#[inline]
pub fn lpmv_real(degree: f64, order: f64, x: f64) -> Result<f64> {
    let value = angular_value(degree, order, x.into(), Angular::Legendre)?;
    if value.im == 0.0 {
        Ok(value.re)
    } else {
        Err(Error::InvalidInput(
            "real lpmv requires -1 <= x <= 1 for odd orders |order| <= degree; \
             pass complex z for the continuation"
                .into(),
        ))
    }
}

/// Borrowed angular values with scalar-or-equal-length broadcasting.
pub(crate) fn angular_values(
    degrees: &[f64],
    orders: &[f64],
    arguments: &[Complex],
    kind: Angular,
) -> Result<Vec<Complex>> {
    let size = broadcast::size(
        &[degrees.len(), orders.len(), arguments.len()],
        "angular arrays must have equal lengths or scalar inputs",
    )?;
    broadcast::map(size, PARALLEL, |i| {
        angular_value(
            element(degrees, i),
            element(orders, i),
            element(arguments, i),
            kind,
        )
    })
}

/// What [`angular_array`] saves for its pullback: the degrees, orders, arguments and
/// function. The pullback recomputes the local derivatives.
#[derive(Debug)]
pub struct AngularResidual {
    degrees: Vec<f64>,
    orders: Vec<f64>,
    arguments: Vec<Complex>,
    kind: Angular,
    size: usize,
}

/// Broadcast angular values and a complex-argument residual; labels stay fixed.
///
/// Upstream: the functions of [`angular_value`] over arrays.
pub fn angular_array(
    degrees: Vec<f64>,
    orders: Vec<f64>,
    arguments: Vec<Complex>,
    kind: Angular,
) -> Result<(Vec<Complex>, AngularResidual)> {
    let values = angular_values(&degrees, &orders, &arguments, kind)?;
    let size = values.len();
    Ok((
        values,
        AngularResidual {
            degrees,
            orders,
            arguments,
            kind,
            size,
        },
    ))
}

impl AngularResidual {
    /// The gradient with respect to the arguments, given the gradient `cotangent` with
    /// respect to the values. A zero cotangent skips derivatives that are singular at
    /// the poles.
    pub fn pullback(self, cotangent: &[Complex]) -> Result<Vec<Complex>> {
        let [gradient] = broadcast::pullback(
            cotangent,
            self.size,
            "angular cotangent must be finite and match output",
            [self.arguments.len()],
            PARALLEL,
            |i, &g| {
                if g == Complex::default() {
                    return Ok([g]);
                }
                let jet = angular_jet::<1>(
                    element(&self.degrees, i),
                    element(&self.orders, i),
                    element(&self.arguments, i),
                    self.kind,
                )?;
                Ok([g * jet.derivative.first().copied().unwrap_or_default().conj()])
            },
        )?;
        Ok(gradient)
    }
}

/// `[cos(theta), sin(theta)]` with their derivatives. The sine takes the principal
/// branch (nonnegative real part) and is exactly zero at the poles, where `theta` is a
/// floating-point multiple of `PI`.
#[allow(clippy::float_cmp)] // Exact coordinate pole labels.
pub(crate) fn polar_trig<const N: usize>(theta: Jet<N>) -> [Jet<N>; 2] {
    let cosine = theta.chain(theta.value.cos(), -theta.value.sin());
    let mut sine = theta.chain(theta.value.sin(), theta.value.cos());
    // Associated Legendre functions use the principal sine factor. Computing
    // its magnitude from sin(theta) preserves small angles lost in 1-cos²(theta).
    if sine.value.re < 0.0 {
        sine = -sine;
    }
    // Multipole translations amplify sin(PI)'s rounding residue. An exact
    // floating-point multiple of the PI constant denotes the coordinate pole.
    if theta.value.im == 0.0 && theta.value.re % PI == 0.0 {
        sine.value = Complex::default();
    }
    [cosine, sine]
}
