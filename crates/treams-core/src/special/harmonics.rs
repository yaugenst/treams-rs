//! Cartesian solid harmonics `r^l P_l^m(z/r) e^(i m phi)` with their gradients and Hessians,
//! singly or tabulated for every degree and order, their values and tangential gradients
//! on the unit sphere, and the normalization of spherical harmonics.
//!
//! treams-rs extension. Solid harmonics are polynomials in `x, y, z`, so they and their
//! derivatives stay finite on the polar axis and at the origin. Spherical translations,
//! lattice sums and field evaluations use them.
#![allow(clippy::indexing_slicing)] // Fixed Cartesian triples, never caller indices.

use std::f64::consts::PI;

use super::{factorial_ratio_sqrt, log_factorial};
use crate::{Complex, numerics::parity};

/// A solid harmonic and its Cartesian derivatives.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Solid {
    /// The value `S`.
    pub(crate) value: Complex,
    /// The derivatives `dS/dx_i`.
    pub(crate) gradient: [Complex; 3],
    /// The second derivatives `d^2 S / dx_i dx_j`; zero unless requested.
    pub(crate) hessian: [[Complex; 3]; 3],
}

/// `r^l P_l^m(z/r) exp(i m phi)`, zero for `|m| > l`. The second derivative is compiled
/// out unless requested.
pub(crate) fn solid<const SECOND: bool>(l: i32, m: i32, position: [f64; 3]) -> Solid {
    if m.abs() > l {
        return Solid::default();
    }
    let [x, y, z] = position;
    let r2 = x * x + y * y + z * z;
    let order = m.abs();
    let xy = Complex::new(x, y);
    let e = [Complex::new(1.0, 0.0), Complex::i(), Complex::default()];
    let mut p = Complex::new(1.0, 0.0);
    let mut grad = [Complex::default(); 3];
    let mut hessian = [[Complex::default(); 3]; 3];
    for k in 1..=order {
        let factor = -f64::from(2 * k - 1);
        if SECOND {
            hessian = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    factor * (hessian[i][j] * xy + grad[i] * e[j] + grad[j] * e[i])
                })
            });
        }
        grad = std::array::from_fn(|i| factor * (grad[i] * xy + p * e[i]));
        p *= factor * xy;
    }
    let (mut prev, mut prev_grad, mut prev_hessian) = (
        Complex::default(),
        [Complex::default(); 3],
        [[Complex::default(); 3]; 3],
    );
    for degree in order + 1..=l {
        let a = f64::from(2 * degree - 1) / f64::from(degree - order);
        let b = f64::from(degree + order - 1) / f64::from(degree - order);
        let next = a * z * p - b * r2 * prev;
        let next_grad = std::array::from_fn(|i| {
            a * (z * grad[i] + if i == 2 { p } else { Complex::default() })
                - b * (r2 * prev_grad[i] + 2.0 * position[i] * prev)
        });
        if SECOND {
            let next_hessian = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    a * (z * hessian[i][j]
                        + if i == 2 { grad[j] } else { Complex::default() }
                        + if j == 2 { grad[i] } else { Complex::default() })
                        - b * (r2 * prev_hessian[i][j]
                            + 2.0 * position[i] * prev_grad[j]
                            + 2.0 * position[j] * prev_grad[i]
                            + if i == j {
                                2.0 * prev
                            } else {
                                Complex::default()
                            })
                })
            });
            prev_hessian = hessian;
            hessian = next_hessian;
        }
        (prev, prev_grad) = (p, grad);
        (p, grad) = (next, next_grad);
    }
    if m < 0 {
        let factor = parity(order) * (log_factorial(l - order) - log_factorial(l + order)).exp();
        p = factor * p.conj();
        grad = grad.map(|g| factor * g.conj());
        if SECOND {
            hessian = hessian.map(|row| row.map(|g| factor * g.conj()));
        }
    }
    Solid {
        value: p,
        gradient: grad,
        hessian,
    }
}

/// Length `r` and direction `x / r` of a displacement `x`. Scaling by the largest
/// component first keeps both accurate where `|x|^2` under- or overflows.
///
/// `None` at the origin, and also where every component is subnormal: `1 / r` would
/// overflow there, while regular translations and their derivatives differ from their
/// origin limits by `O(|k| r)` relative to their size, below rounding for `|k| < 1e290`.
pub(crate) fn direction(position: [f64; 3]) -> Option<(f64, [f64; 3])> {
    let scale = position.iter().fold(0.0_f64, |scale, x| scale.max(x.abs()));
    if scale < f64::MIN_POSITIVE {
        return None;
    }
    let scaled = position.map(|x| x / scale);
    let norm = scaled.iter().map(|x| x * x).sum::<f64>().sqrt();
    Some((scale * norm, scaled.map(|x| x / norm)))
}

/// `Y = P_l^m(cos theta) exp(i m phi)` at a unit vector `u` and its tangential gradient
/// `grad S(u) - l S(u) u`, where `S` is the degree-`l` solid harmonic with `S(u) = Y`.
/// The tangential gradient is `r` times the Cartesian gradient of `Y` at `r u`.
pub(crate) fn on_sphere(l: i32, m: i32, unit: [f64; 3]) -> (Complex, [Complex; 3]) {
    let solid = solid::<false>(l, m, unit);
    (solid.value, tangent(l, solid.value, solid.gradient, unit))
}

/// The tangential part `grad S(u) - l S(u) u` of a degree-`l` solid harmonic's gradient.
pub(crate) fn tangent(
    l: i32,
    value: Complex,
    gradient: [Complex; 3],
    unit: [f64; 3],
) -> [Complex; 3] {
    let radial = f64::from(l) * value;
    std::array::from_fn(|axis| gradient[axis] - radial * unit[axis])
}

/// `sqrt((2l + 1) / 4 pi) sqrt((l - m)! / (l + m)!)`, which turns `P_l^m(cos theta)
/// exp(i m phi)` into the orthonormal spherical harmonic `Y_lm`.
pub(crate) fn harmonic_normalization(l: i32, m: i32) -> f64 {
    (f64::from(2 * l + 1) / (4.0 * PI)).sqrt() * factorial_ratio_sqrt(l, m)
}

/// Solid harmonics `r^p P_p^m e^(i m phi)` of every degree `p <= order` and order
/// `|m| <= p`, from one degree recurrence per `|m|` rather than one per entry, with the
/// order-reflection factors of negative orders precomputed.
#[derive(Clone, Debug)]
pub(crate) struct SolidTable {
    order: i32,
    reflection: Vec<f64>,
}

/// Position `p * p + p + m` of `(p, m)` in a harmonic table.
fn table_index(p: i32, m: i32) -> usize {
    usize::try_from(p * p + p + m).unwrap_or_default()
}

impl SolidTable {
    /// The table of every degree up to `order`.
    pub(crate) fn new(order: i32) -> Self {
        let reflection = (0..=order)
            .flat_map(|l| (-l..=l).map(move |m| (l, m)))
            .map(|(l, m)| {
                let order = m.abs();
                parity(order) * (log_factorial(l - order) - log_factorial(l + order)).exp()
            })
            .collect();
        Self { order, reflection }
    }

    /// Number of table entries, `(order + 1)^2`.
    pub(crate) fn len(&self) -> usize {
        self.reflection.len()
    }

    /// Visits `(p * p + p + m, p, S, grad S)` for every entry. Each value and gradient
    /// equals `solid::<false>(p, m, position)` bit for bit; gradients stay zero unless
    /// `GRADIENT`.
    pub(crate) fn visit<const GRADIENT: bool>(
        &self,
        position: [f64; 3],
        mut emit: impl FnMut(usize, i32, Complex, [Complex; 3]),
    ) {
        self.visit_impl::<GRADIENT, false>(position, |index, degree, solid| {
            emit(index, degree, solid.value, solid.gradient);
        });
    }

    /// Visits every solid harmonic with its gradient and, if `SECOND`, Hessian.
    /// Entries equal the corresponding single [`solid`] evaluation bit for bit.
    pub(crate) fn visit_solid<const SECOND: bool>(
        &self,
        position: [f64; 3],
        emit: impl FnMut(usize, i32, Solid),
    ) {
        self.visit_impl::<true, SECOND>(position, emit);
    }

    fn visit_impl<const GRADIENT: bool, const SECOND: bool>(
        &self,
        position: [f64; 3],
        mut emit: impl FnMut(usize, i32, Solid),
    ) {
        let [x, y, z] = position;
        let r2 = x * x + y * y + z * z;
        let xy = Complex::new(x, y);
        let e = [Complex::new(1.0, 0.0), Complex::i(), Complex::default()];
        let (mut diagonal, mut diagonal_gradient) =
            (Complex::new(1.0, 0.0), [Complex::default(); 3]);
        let mut diagonal_hessian = [[Complex::default(); 3]; 3];
        for order in 0..=self.order {
            if order > 0 {
                // One step of `solid`'s ladder to P_order^order.
                let factor = -f64::from(2 * order - 1);
                if SECOND {
                    diagonal_hessian = std::array::from_fn(|i| {
                        std::array::from_fn(|j| {
                            factor
                                * (diagonal_hessian[i][j] * xy
                                    + diagonal_gradient[i] * e[j]
                                    + diagonal_gradient[j] * e[i])
                        })
                    });
                }
                if GRADIENT {
                    diagonal_gradient = std::array::from_fn(|i| {
                        factor * (diagonal_gradient[i] * xy + diagonal * e[i])
                    });
                }
                diagonal *= factor * xy;
            }
            let (mut p, mut grad) = (diagonal, diagonal_gradient);
            let mut hessian = diagonal_hessian;
            let (mut prev, mut prev_grad) = (Complex::default(), [Complex::default(); 3]);
            let mut prev_hessian = [[Complex::default(); 3]; 3];
            for degree in order..=self.order {
                if degree > order {
                    let a = f64::from(2 * degree - 1) / f64::from(degree - order);
                    let b = f64::from(degree + order - 1) / f64::from(degree - order);
                    let next = a * z * p - b * r2 * prev;
                    let next_grad = if GRADIENT {
                        std::array::from_fn(|i| {
                            a * (z * grad[i] + if i == 2 { p } else { Complex::default() })
                                - b * (r2 * prev_grad[i] + 2.0 * position[i] * prev)
                        })
                    } else {
                        grad
                    };
                    if SECOND {
                        let next_hessian = std::array::from_fn(|i| {
                            std::array::from_fn(|j| {
                                a * (z * hessian[i][j]
                                    + if i == 2 { grad[j] } else { Complex::default() }
                                    + if j == 2 { grad[i] } else { Complex::default() })
                                    - b * (r2 * prev_hessian[i][j]
                                        + 2.0 * position[i] * prev_grad[j]
                                        + 2.0 * position[j] * prev_grad[i]
                                        + if i == j {
                                            2.0 * prev
                                        } else {
                                            Complex::default()
                                        })
                            })
                        });
                        prev_hessian = hessian;
                        hessian = next_hessian;
                    }
                    (prev, prev_grad) = (p, grad);
                    (p, grad) = (next, next_grad);
                }
                emit(
                    table_index(degree, order),
                    degree,
                    Solid {
                        value: p,
                        gradient: grad,
                        hessian,
                    },
                );
                if order > 0 {
                    let index = table_index(degree, -order);
                    let factor = self.reflection[index];
                    let gradient = if GRADIENT {
                        grad.map(|g| factor * g.conj())
                    } else {
                        grad
                    };
                    let hessian = if SECOND {
                        hessian.map(|row| row.map(|g| factor * g.conj()))
                    } else {
                        hessian
                    };
                    emit(
                        index,
                        degree,
                        Solid {
                            value: factor * p.conj(),
                            gradient,
                            hessian,
                        },
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Solid harmonics, their tables and angular values against the Legendre recurrence;
    //! the translations and lattice sums built on them are checked in `properties/`.

    use std::f64::consts::PI;

    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::{SolidTable, direction, on_sphere, solid};
    use crate::{
        Complex,
        special::{Angular, angular_value},
        test_support::{ALGEBRA_CASES, degree_order, prop_assert_close},
    };

    /// Points in a box, on the polar axis and in the equatorial plane.
    fn position() -> impl Strategy<Value = [f64; 3]> {
        prop_oneof![
            (-2.0_f64..2.0, -2.0_f64..2.0, -2.0_f64..2.0).prop_map(<[f64; 3]>::from),
            (-2.0_f64..2.0).prop_map(|z| [0.0, 0.0, z]),
            (-2.0_f64..2.0, -2.0_f64..2.0).prop_map(|(x, y)| [x, y, 0.0]),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn solid_harmonics_are_harmonic_and_homogeneous(
            (l, m) in degree_order(0..31),
            position in position(),
        ) {
            check_solid_harmonic(l, m, position)?;
        }

        #[test]
        fn solid_tables_match_single_harmonics(order in 0_i32..=30, position in position()) {
            check_solid_table(order, position)?;
        }

        #[test]
        fn angular_functions_match_legendre_and_ignore_the_length_unit(
            (l, m) in degree_order(0..31),
            theta in prop_oneof![Just(0.0), Just(PI), 0.0..PI],
            phi in -PI..PI,
            radius in 0.1_f64..10.0,
            exponent in -150_i32..=150,
        ) {
            check_angular(l, m, [theta, phi], radius, exponent)?;
        }

        /// Within `1e-12` to `0.1` of either pole, where the cosine the reference sees
        /// fixes the polar angle worst.
        #[test]
        fn angular_functions_match_legendre_near_the_poles(
            (l, m) in degree_order(0..31),
            offset in -12.0_f64..-1.0,
            south in any::<bool>(),
            phi in -PI..PI,
            radius in 0.1_f64..10.0,
            exponent in -150_i32..=150,
        ) {
            let offset = 10_f64.powf(offset);
            let theta = if south { PI - offset } else { offset };
            check_angular(l, m, [theta, phi], radius, exponent)?;
        }
    }

    /// A table visits every index once, and each entry equals [`solid`] bit for bit, with
    /// and without gradients.
    fn check_solid_table(order: i32, position: [f64; 3]) -> Result<(), TestCaseError> {
        let table = SolidTable::new(order);
        for with_gradient in [false, true] {
            let mut entries = Vec::new();
            let push = |index, p, value, gradient| entries.push((index, p, value, gradient));
            if with_gradient {
                table.visit::<true>(position, push);
            } else {
                table.visit::<false>(position, push);
            }
            let mut indices: Vec<usize> = entries.iter().map(|entry| entry.0).collect();
            indices.sort_unstable();
            prop_assert_eq!(indices, (0..table.len()).collect::<Vec<_>>());
            for (index, p, value, gradient) in entries {
                let m = i32::try_from(index).unwrap() - p * p - p;
                let expected = solid::<false>(p, m, position);
                prop_assert_eq!(value, expected.value, "{} {}", p, m);
                if with_gradient {
                    prop_assert_eq!(gradient, expected.gradient, "{} {}", p, m);
                }
            }
        }
        let mut second = vec![super::Solid::default(); table.len()];
        table.visit_solid::<true>(position, |index, _, solid| second[index] = solid);
        for p in 0..=order {
            for m in -p..=p {
                let expected = solid::<true>(p, m, position);
                let actual = second[super::table_index(p, m)];
                prop_assert_eq!(actual.value, expected.value, "{} {}", p, m);
                prop_assert_eq!(actual.gradient, expected.gradient, "{} {}", p, m);
                prop_assert_eq!(actual.hessian, expected.hessian, "{} {}", p, m);
            }
        }
        Ok(())
    }

    /// Relative tolerance with a floor for exact zeros, since values scale down to about
    /// `1e-40` at extreme orders.
    fn tolerance(scale: f64) -> f64 {
        1e-12 * scale + f64::MIN_POSITIVE
    }

    /// `r^l P_l^m e^(i m phi)` solves Laplace's equation (`tr H = 0`) and is homogeneous
    /// of degree `l`, so Euler's identity holds for the value (`x · grad S = l S`) and
    /// for each gradient component (`H x = (l - 1) grad S`). It vanishes identically
    /// for `|m| > l`.
    fn check_solid_harmonic(l: i32, m: i32, x: [f64; 3]) -> Result<(), TestCaseError> {
        for m in [l + 1, -l - 1] {
            let beyond = solid::<true>(l, m, x);
            prop_assert_eq!(beyond.value, Complex::default(), "m = {}", m);
            prop_assert_eq!(beyond.gradient, [Complex::default(); 3], "m = {}", m);
            prop_assert_eq!(beyond.hessian, [[Complex::default(); 3]; 3], "m = {}", m);
        }
        let s = solid::<true>(l, m, x);
        let laplacian: Complex = (0..3).map(|i| s.hessian[i][i]).sum();
        let scale: f64 = (0..3).map(|i| s.hessian[i][i].norm()).sum();
        prop_assert_close!(laplacian, Complex::default(), tolerance(scale));
        let radial: Complex = (0..3).map(|i| x[i] * s.gradient[i]).sum();
        let scale = f64::from(l) * s.value.norm()
            + (0..3).map(|i| (x[i] * s.gradient[i]).norm()).sum::<f64>();
        prop_assert_close!(radial, f64::from(l) * s.value, tolerance(scale));
        for i in 0..3 {
            let radial: Complex = (0..3).map(|j| x[j] * s.hessian[i][j]).sum();
            let scale = f64::from(l) * s.gradient[i].norm()
                + (0..3).map(|j| (x[j] * s.hessian[i][j]).norm()).sum::<f64>();
            let expected = f64::from(l - 1) * s.gradient[i];
            prop_assert_close!(radial, expected, tolerance(scale), "row {}", i);
        }
        Ok(())
    }

    /// `P_l^m(z/r) e^(i m phi)` at `x` and its Cartesian gradient, from the solid
    /// harmonic on the unit sphere as translations evaluate it.
    fn angular(l: i32, m: i32, x: [f64; 3]) -> (Complex, [Complex; 3]) {
        let (r, unit) = direction(x).unwrap();
        let (value, tangent) = on_sphere(l, m, unit);
        (value, tangent.map(|t| t / r))
    }

    /// The Cartesian angular function at `r (sin theta cos phi, sin theta sin phi,
    /// cos theta)` equals the Legendre recurrence `P_l^m(cos theta) e^(i m phi)` (in
    /// the direction the rounded `cos theta` represents), and it is invariant under
    /// `r -> 10^e r` with its gradient scaling as `10^-e`, far past the range where
    /// `r^l` is representable.
    fn check_angular(
        l: i32,
        m: i32,
        [theta, phi]: [f64; 2],
        radius: f64,
        exponent: i32,
    ) -> Result<(), TestCaseError> {
        let (sin, cos) = theta.sin_cos();
        let x = [
            radius * sin * phi.cos(),
            radius * sin * phi.sin(),
            radius * cos,
        ];
        check_legendre(l, m, cos, phi, radius)?;
        let (value, gradient) = angular(l, m, x);
        // Rounding of the direction moves values by ulps times the tangential gradient.
        let tangent: f64 = gradient.iter().map(|g| g.norm()).sum::<f64>() * radius;
        let scale = value.norm() + tangent;
        let s = 10_f64.powi(exponent);
        let (scaled, scaled_gradient) = angular(l, m, x.map(|x| x * s));
        prop_assert_close!(scaled, value, tolerance(scale));
        // At large scales the gradient, of order 1/r, may be subnormal.
        let subnormal = 8.0 * f64::MIN_POSITIVE * f64::EPSILON * s * radius;
        prop_assert_close!(
            scaled_gradient.map(|g| g * s * radius),
            gradient.map(|g| g * radius),
            tolerance((1.0 + f64::from(l)) * scale) + subnormal
        );
        Ok(())
    }

    /// The Cartesian angular function in the direction whose polar cosine is `z` equals
    /// the Legendre recurrence `P_l^m(z) e^(i m phi)`.
    ///
    /// The reference sees only `z`, and `cos theta` rounds to it with an error that moves
    /// the polar angle by up to `eps / sin theta` (`theta` itself is lost below `1e-8`,
    /// where `z = 1`), which moves values by that times the tangential gradient, beyond
    /// any fixed number of ulps near the poles. So the Cartesian side takes the direction
    /// that `z` represents, whose sine `sqrt((1 - z)(1 + z))` is accurate to ulps.
    fn check_legendre(l: i32, m: i32, z: f64, phi: f64, radius: f64) -> Result<(), TestCaseError> {
        let sine = ((1.0 - z) * (1.0 + z)).sqrt();
        let represented = [
            radius * sine * phi.cos(),
            radius * sine * phi.sin(),
            radius * z,
        ];
        let (value, gradient) = angular(l, m, represented);
        let legendre = angular_value(
            f64::from(l),
            f64::from(m),
            Complex::new(z, 0.0),
            Angular::Legendre,
        )
        .unwrap();
        let expected = legendre * Complex::from_polar(1.0, f64::from(m) * phi);
        // Rounding of the direction moves values by ulps times the tangential gradient.
        let tangent: f64 = gradient.iter().map(|g| g.norm()).sum::<f64>() * radius;
        prop_assert_close!(value, expected, tolerance(value.norm() + tangent));
        Ok(())
    }
}
