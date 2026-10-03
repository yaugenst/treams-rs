//! Reciprocal-space terms of the five sums (family and lattice dimension: sw3d, sw2d,
//! sw1d, cw2d, cw1d) and the self term.
//!
//! Upstream: `recsum*`, `_recsw*`, `_reccw*`, `_n_sum_*`, `_s_sum_*`, `zero2d` and
//! `zero3d` of `lattice/_esum.pyx`. The compensated threshold distance and the outgoing
//! sheets are treams-rs extensions.

use std::f64::consts::{PI, TAU};

use super::{
    accuracy::Rounding,
    reduced::{Reduced, reduced_gamma},
    sheets::{Sheet, Split, lower_side, self_sheet, sheet_turns, split_sheet},
    wave::{Family, azimuthal, negative_order_sign, solid_jet},
};
use crate::{
    Complex, Error, Result,
    numerics::Jet,
    special::{self, harmonic_normalization, log_factorial},
};

/// `F_1(v) = Gamma(1, v) / (-v) = exp(-v) / (-v)` of the 3D spherical and 2D
/// cylindrical reciprocal sums, in closed form with `dF_1/dv = F_2 = (1 + v) exp(-v) / v^2`.
fn exponential_kernel<const N: usize>(v: Jet<N>) -> Jet<N> {
    let z = lower_side(v.value);
    let exponential = (-z).exp();
    let value = exponential / (-z);
    if N == 0 {
        return Jet::constant(value);
    }
    v.chain(value, (1.0 + z) * exponential / (z * z))
}

/// The inner sum `S_lmn` of the 2D spherical summand of [`reciprocal_term`] (upstream
/// `_s_sum_sw2d` times `e^(i m phi)`), with `z = k r_z`, `beta^2 = beta2` and
/// `beta e^(i phi) = (q_x + i q_y) / k`:
/// `sum_s (-z)^(2n - s) beta^(l - s) e^(i m phi) / ((2n - s)! (s - n)! ((l + |m| - s) / 2)!
/// ((l - |m| - s) / 2)!)` over `s = n..=min(l - |m|, 2n)` with even `l - |m| - s`.
fn polynomial_sw2d<const N: usize>(
    l: i32,
    m: i32,
    n: i32,
    z: Jet<N>,
    beta2: Jet<N>,
    azimuth: Jet<N>,
) -> Jet<N> {
    let mut sum = Jet::default();
    for s in n..=(l - m.abs()).min(2 * n) {
        if (l - m.abs() - s) % 2 != 0 {
            continue;
        }
        sum += (-z).powi(2 * n - s)
            * beta2.powi((l - s - m.abs()) / 2)
            * (-log_factorial(2 * n - s)
                - log_factorial(s - n)
                - log_factorial((l + m.abs() - s) / 2)
                - log_factorial((l - m.abs() - s) / 2))
            .exp();
    }
    sum * azimuth.powi(m.abs())
}

/// The inner sum `S_lmn` of the 1D spherical summand of [`reciprocal_term`] (upstream
/// `_s_sum_sw1d` times `(-1)^m e^(i m phi)`), with `rho2 = (k rho)^2`, `beta = q_z / k`
/// and the azimuth `phi` of the shift: `sum_s (k rho)^(2n - s) beta^(l - s) (-1)^m
/// e^(i m phi) / ((n - (s + |m|) / 2)! (n - (s - |m|) / 2)! (l - s)! (s - n)!)` over
/// `s = n..=min(2n - |m|, l)` with even `s - |m|`.
fn polynomial_sw1d<const N: usize>(
    l: i32,
    m: i32,
    n: i32,
    rho2: Jet<N>,
    beta: Jet<N>,
    azimuth: Jet<N>,
) -> Jet<N> {
    let mut sum = Jet::default();
    for s in n..=(2 * n - m.abs()).min(l) {
        if (m.abs() - s) % 2 != 0 {
            continue;
        }
        sum += rho2.powi((2 * n - s - m.abs()) / 2)
            * beta.powi(l - s)
            * (-log_factorial(n - s.midpoint(m.abs()))
                - log_factorial(n - (s - m.abs()) / 2)
                - log_factorial(l - s)
                - log_factorial(s - n))
            .exp();
    }
    sum * azimuth.powi(m.abs())
}

/// The inner sum `S_ln` of the 1D cylindrical summand of [`reciprocal_term`] (upstream
/// `_s_sum_cw1d`, whose docstring formula omits the factor `l!` that its code includes)
/// for `l = |m|`, with `y = k r_y` (`-k r_y` for negative `m`) and `beta = q_x / k`:
/// `sum_s (-y)^(2n - s) beta^(l - s) l! / (2^s (2n - s)! (l - s)! (s - n)!)` over
/// `s = n..=min(2n, l)`.
fn polynomial_cw1d<const N: usize>(l: i32, n: i32, y: Jet<N>, beta: Jet<N>) -> Jet<N> {
    let mut sum = Jet::default();
    for s in n..=(2 * n).min(l) {
        sum += (-y).powi(2 * n - s)
            * beta.powi(l - s)
            * (log_factorial(l)
                - log_factorial(2 * n - s)
                - log_factorial(l - s)
                - log_factorial(s - n)
                - f64::from(s) * 2.0_f64.ln())
            .exp();
    }
    sum
}

/// The reciprocal term of a value-only 1D cylindrical sum on its axis, which needs half
/// of the polynomial terms: adjacent half-integer gamma functions share one evaluation,
/// recurring downward near zero and upward at large arguments, where a downward step
/// would subtract the leading asymptotic term. On a `sheet` ([`reduced_gamma`]) the
/// endpoint term takes the phase `e^(-i pi n)` of the lower side.
fn axial_cylindrical_reciprocal(
    order: i32,
    beta: Complex,
    z: Complex,
    eta: Complex,
    sheet: Option<Sheet>,
) -> Complex {
    let maximum = order / 2;
    let upward = z.norm() > 4.0;
    let mut gamma = reduced_gamma(1 - 2 * if upward { maximum } else { 0 }, z, sheet);
    let endpoint = if maximum > 0 {
        (if sheet.is_none() && z.im > 0.0 {
            Complex::i()
        } else {
            -Complex::i()
        }) * (-z).exp()
    } else {
        Complex::default()
    };
    let mut sum = Complex::default();
    for step in 0..=maximum {
        let n = if upward { maximum - step } else { step };
        let weight = beta.powi(order - 2 * n)
            * (0.5 * eta * eta).powi(n)
            * (log_factorial(order) - log_factorial(n) - log_factorial(order - 2 * n)).exp()
            / (2.0_f64.sqrt() * eta);
        sum += weight * gamma;
        if step < maximum {
            let phase = if n % 2 == 0 { endpoint } else { -endpoint };
            gamma = if upward {
                ((0.5 - f64::from(n)) * gamma + phase) / (-z)
            } else {
                (-z * gamma + phase) / (-0.5 - f64::from(n))
            };
        }
    }
    sum
}

/// `q . q - k^2` of a diffraction order to a few ulps of itself, from exact products and
/// compensated additions: next to a threshold the rounded `q . q / k^2 - 1` loses about
/// `eps k^2 / |k_q^2|` of it. Only the zeroth order of an unreduced Bloch vector takes it
/// (within [`NEAR_THRESHOLD`]): every other `q` carries the rounding of its reciprocal
/// vector, as large as the rounding this removes.
fn compensated_threshold_distance(q: [Complex; 3], k: Complex) -> Complex {
    let (mut re, mut im) = (Compensated::default(), Compensated::default());
    for q in q {
        re.add_product(q.re, q.re);
        re.add_product(-q.im, q.im);
        im.add_product(2.0 * q.re, q.im);
    }
    re.add_product(-k.re, k.re);
    re.add_product(k.im, k.im);
    im.add_product(-2.0 * k.re, k.im);
    Complex::new(re.total(), im.total())
}

/// Largest moduli of the real and imaginary parts of `q . q / k^2 - 1` of the zeroth
/// diffraction order at which the reciprocal sums take
/// [`compensated_threshold_distance`]: there the rounded form loses 50 ulps of `v` or
/// more (about 3 ulps of `q . q / k^2` over the difference).
const NEAR_THRESHOLD: f64 = 1.0 / 16.0;

/// Largest `|v|` (for the reciprocal terms) or `|k_q^2 / k^2|` (for the spectral series)
/// at which a diffraction order counts as on its threshold `|q| = k`, where the sums
/// are singular.
pub(super) const THRESHOLD_DISTANCE: f64 = 1e-14;

/// A sum of products that keeps the rounding errors of its products and additions
/// apart until the end.
#[derive(Clone, Copy, Debug, Default)]
struct Compensated {
    sum: f64,
    error: f64,
}

impl Compensated {
    fn add_product(&mut self, a: f64, b: f64) {
        let product = a * b;
        let total = self.sum + product;
        let back = total - self.sum;
        self.error += a.mul_add(b, -product) + (self.sum - (total - back)) + (product - back);
        self.sum = total;
    }

    fn total(self) -> f64 {
        self.sum + self.error
    }
}

/// The wavevector `q` of one diffraction order of a reciprocal summand, `kpar + G`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Diffraction<const N: usize> {
    pub(super) q: [Jet<N>; 3],
    /// Whether `q` carries no rounding: the zeroth order of an unreduced Bloch vector,
    /// which takes [`compensated_threshold_distance`].
    pub(super) exact: bool,
}

/// One reciprocal summand `T(q)` of the sum, without the Bloch phase `exp(-i q . r)`
/// that the caller multiplies in. Every sum shares the argument `v = (q . q / k^2 - 1) /
/// (2 eta^2)`; `wave` and `dim` select the sum. 1D spherical sums, whose odd Kambe
/// orders cancel by up to `e^(w^2)` at `w = k rho eta`, add the rounding bounds of its
/// value and derivatives in units of epsilon to `bound`; the other sums keep their
/// relative accuracy.
pub(super) fn reciprocal_term<const N: usize>(
    wave: Family,
    dim: usize,
    k: Jet<N>,
    Diffraction { q, exact }: Diffraction<N>,
    r: [Jet<N>; 3],
    eta: Complex,
    split: &Split,
    measure: Jet<N>,
    bound: &mut Rounding<N>,
) -> Result<Jet<N>> {
    let beta2 = q.into_iter().map(|q| q * q).sum::<Jet<N>>() / (k * k);
    let mut value = (beta2 - 1.0) / (2.0 * eta * eta);
    let distance = beta2.value - 1.0;
    if exact && distance.re.abs().max(distance.im.abs()) < NEAR_THRESHOLD {
        value.value = compensated_threshold_distance(q.map(|q| q.value), k.value)
            / (k.value * k.value)
            / (2.0 * eta * eta);
    }
    value.value = lower_side(value.value);
    if value.value.norm() <= THRESHOLD_DISTANCE {
        return Err(Error::InvalidInput(
            "lattice sum is at a diffraction threshold; supply a limiting complex wavenumber"
                .into(),
        ));
    }
    let i = Complex::i();
    // The five summands share this one match: the term loop of every sum inlines
    // this function, and in separate functions the summands make the forward 3D sums
    // up to 13% slower.
    Ok(match (wave, dim) {
        // 3D spherical sum (upstream `recsumsw3d`, summand `_recsw3d`), with the cell
        // volume `V = measure` and `F_1` of `exponential_kernel`:
        // `2 i (-i)^l pi Y_lm(q / |q|) (|q| / k)^l F_1(v) / (V k^3 eta^2)`.
        (Family::Spherical { l, m }, 3) => {
            let harmonic = harmonic_normalization(l, m) * solid_jet(l, m, q);
            2.0 * i * (-i).powi(l) * PI / (measure * k.powi(3)) * harmonic / k.powi(l)
                * exponential_kernel(value)
                / (eta * eta)
        }
        // 2D cylindrical sum (upstream `recsumcw2d`, summand `_reccw2d`), with the cell
        // area `A = measure`, `beta e^(i phi) = (q_x + i q_y) / k` and `F_1` of
        // `exponential_kernel`: `2 i (-i)^m beta^|m| e^(i m phi) F_1(v) / (A k^2 eta^2)`.
        (Family::Cylindrical { m }, 2) => {
            let xy = azimuthal(q[0], q[1], m);
            2.0 * i * (-i).powi(m) / (measure * k * k)
                * (xy / k).powi(m.abs())
                * exponential_kernel(value)
                / (eta * eta)
        }
        // 2D spherical sum for any shift (upstream `recsumsw2d_shift`, summand
        // `_n_sum_sw2d`; at `r_z = 0` it also gives upstream `recsumsw2d`), with the cell
        // area `A = measure`:
        //
        // `sqrt(2l + 1) (-i)^m sqrt((l + m)! (l - m)!) / (A k^2 (-2)^l)
        // sum_(n = 0..=l - |m|) (2 eta^2)^n / (sqrt(2) eta) F_(1/2 - n)(v, (k r_z eta)^2) S_lmn`
        //
        // with the reduced integrals `F` of `Reduced` and the inner sums `S_lmn` of
        // `polynomial_sw2d`, summed by `half_integer_orders`.
        (Family::Spherical { l, m }, 2) => {
            let z = k * r[2];
            let azimuth = azimuthal(q[0], q[1], m) / k;
            let factor = 1.0 / (2.0_f64.sqrt() * eta);
            // For Im k < 0 the sums in the plane depend on the split, and with
            // Re(k eta) < 0 the real parts do not converge to the sum: both keep the
            // principal branches.
            let sheet = (k.value.im >= 0.0 && (k.value * eta).re > 0.0).then(|| {
                let q2 = q[0].value * q[0].value + q[1].value * q[1].value;
                split_sheet(split, q2)
            });
            let mut reduced = Reduced::new(value, (z * eta).powi(2), sheet);
            let sum = half_integer_orders(&mut reduced, split, (factor, eta), l - m.abs(), |n| {
                polynomial_sw2d(l, m, n, z, beta2, azimuth)
            });
            f64::from(2 * l + 1).sqrt()
                * (-i).powi(m)
                * (0.5 * (log_factorial(l + m) + log_factorial(l - m))).exp()
                / (measure * k * k * (-2.0_f64).powi(l))
                * sum
        }
        // 1D spherical sum along `z` (upstream `recsumsw1d_shift`, summand
        // `_n_sum_sw1d`), with the period `a = measure` and `rho^2 = r_x^2 + r_y^2`:
        //
        // `-i sqrt((2l + 1) / pi) (-i)^(l - m) sqrt((l + m)! (l - m)!) / (2 a k)
        // sum_(n = |m|..=l) (eta^2 / 2)^n F_(-n)(v, (k rho eta)^2) S_lmn`
        //
        // with the reduced integrals `F` of `Reduced` and the inner sums `S_lmn` of
        // `polynomial_sw1d`. Adds the rounding bounds of the integer orders `F_(-n)` to
        // `bound`.
        (Family::Spherical { l, m }, 1) => {
            let beta = q[2] / k;
            let rho2 = k * k * (r[0] * r[0] + r[1] * r[1]);
            let azimuth = -k * azimuthal(r[0], r[1], m);
            let mut sum = Jet::default();
            let mult = 0.5 * eta * eta;
            let mut factor = mult.powi(m.abs());
            let sheet = split_sheet(split, q[2].value * q[2].value);
            let mut reduced = Reduced::new(value, rho2 * eta * eta, Some(sheet));
            let mut sum_bound = Rounding::default();
            for n in m.abs()..=l {
                let polynomial = polynomial_sw1d(l, m, n, rho2, beta, azimuth);
                if !polynomial.is_zero() {
                    let (integral, integral_bound) = reduced.get(-2 * n);
                    sum += factor * integral * polynomial;
                    sum_bound += integral_bound.times(&(factor * polynomial));
                }
                factor *= mult;
            }
            let prefactor = -i
                * (f64::from(2 * l + 1) / PI).sqrt()
                * (-i).powi(l - m)
                * (0.5 * (log_factorial(l + m) + log_factorial(l - m))).exp()
                / (2.0 * measure * k);
            *bound += sum_bound.times(&prefactor);
            prefactor * sum
        }
        // 1D cylindrical sum along `x` (upstream `recsumcw1d_shift`, summand
        // `_n_sum_cw1d`), with the period `a = measure`:
        //
        // `2 (-i)^m / (sqrt(pi) a k)
        // sum_(n = 0..=|m|) (2 eta^2)^n / (sqrt(2) eta') F_(1/2 - n)(v, (k r_y eta)^2) S_(|m|n)`
        //
        // with `eta' = ±eta`, the root of `eta^2` with `Re(k eta') >= 0`, the reduced
        // integrals `F` of `Reduced` and the inner sums `S_ln` of `polynomial_cw1d`,
        // summed by `half_integer_orders`. Value-only sums on the axis (`r_y = 0`) take
        // `axial_cylindrical_reciprocal`.
        (Family::Cylindrical { m }, 1) => {
            let beta = q[0] / k;
            let y = k * if m < 0 { -r[1] } else { r[1] };
            // The real part and the reduced integrals depend on eta^2 alone, the sum on
            // the sign of eta only through the factor 1 / eta: splits with Re(k eta) < 0
            // take the root -eta of eta^2, with which the Ewald parts converge to the
            // sum. For Im k < 0 the sums on the axis depend on the split: they keep the
            // principal branches.
            let root = if (k.value * eta).re < 0.0 { -eta } else { eta };
            let factor = 1.0 / (2.0_f64.sqrt() * root);
            let sheet = (k.value.im >= 0.0).then(|| split_sheet(split, q[0].value * q[0].value));
            if N == 0 && r[1].value == Complex::default() {
                return Ok(Jet::constant(
                    2.0 * (-i).powi(m) / (PI.sqrt() * measure.value * k.value)
                        * axial_cylindrical_reciprocal(
                            m.abs(),
                            beta.value,
                            value.value,
                            root,
                            sheet,
                        ),
                ));
            }
            let mut reduced = Reduced::new(value, (y * eta).powi(2), sheet);
            let sum = half_integer_orders(&mut reduced, split, (factor, eta), m.abs(), |n| {
                polynomial_cw1d(m.abs(), n, y, beta)
            });
            2.0 * (-i).powi(m) / (PI.sqrt() * measure * k) * sum
        }
        _ => return Err(Error::InvalidInput("invalid lattice dimension".into())),
    })
}

/// `sum_n factor (2 eta^2)^n F_(1/2 - n) P(n)` over the half-integer orders
/// `n = 0..=orders` of one reciprocal point with the degree polynomials `P` (zero ones
/// skipped). Below every automatic split the orders weigh their terms, and are summed
/// again from the single Kambe integrals where those bound the sum more tightly
/// ([`Reduced::prefers_single`]).
fn half_integer_orders<const N: usize>(
    reduced: &mut Reduced<N>,
    split: &Split,
    (factor, eta): (Complex, Complex),
    orders: i32,
    polynomial: impl Fn(i32) -> Jet<N>,
) -> Jet<N> {
    let sum = |reduced: &mut Reduced<N>| {
        let (mut sum, mut factor) = (Jet::default(), factor);
        for n in 0..=orders {
            let polynomial = polynomial(n);
            if !polynomial.is_zero() {
                if split.small {
                    reduced.weigh(1 - 2 * n, factor * polynomial.value);
                }
                sum += factor * reduced.get(1 - 2 * n).0 * polynomial;
            }
            factor *= 2.0 * eta * eta;
        }
        sum
    };
    let from_recurrence = sum(reduced);
    if split.small && reduced.prefers_single() {
        sum(reduced)
    } else {
        from_recurrence
    }
}

/// The self term: the Cartesian Taylor term of `real_Ewald(-r) - outgoing(-r)` at the
/// origin. Upstream `zero2d` (cylindrical) and `zero3d` (spherical) give its value at
/// degree and order zero. Keeping its first-order polynomial gives the regular
/// image-sum derivative at self.
///
/// # Branch
///
/// Its incomplete gamma function of `v = -1 / (2 eta^2)` takes the sheet
/// ([`self_sheet`]) of the principal `ln((k eta)^2)` of the real-space Kambe integrals
/// less `2 ln k` of the outgoing waves. For a sheet `j` turns off the principal one,
/// DLMF 8.2.10 gives `2 Gamma(a) - Gamma(a, v)` for half-integer `a` and odd `j`, and
/// `Gamma(-n, v) - 2 pi i j (-1)^n / n!` for `a = -n`.
pub(super) fn self_term<const N: usize>(
    wave: Family,
    k: Jet<N>,
    r: [Jet<N>; 3],
    eta: Complex,
) -> Jet<N> {
    let value = lower_side(-0.5 / (eta * eta));
    let turns = sheet_turns(self_sheet(k.value, eta), value);
    match wave {
        Family::Spherical { l, m } => {
            let degree = -f64::from(l) - 0.5;
            let mut gamma = special::upper_gamma(degree, value);
            if turns % 2.0 != 0.0 {
                gamma = 2.0 * libm::tgamma(degree) - gamma;
            }
            k.powi(l)
                * solid_jet(l, m, r)
                * harmonic_normalization(l, m)
                * (2.0_f64.powi(-l - 1) / PI.sqrt())
                * gamma
        }
        Family::Cylindrical { m } => {
            let (xy, sign) = (azimuthal(r[0], r[1], m), negative_order_sign(m));
            let n = m.abs();
            let parity = if n % 2 == 0 { 1.0 } else { -1.0 };
            let gamma = special::upper_gamma(-f64::from(n), value)
                - Complex::new(0.0, TAU * turns) * parity * (-log_factorial(n)).exp();
            Complex::i() / PI * sign * (0.5 * k * xy).powi(n) * gamma
        }
    }
}
