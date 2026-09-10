//! Integer Wigner symbols and Cartesian spherical harmonics.
#![allow(clippy::indexing_slicing)] // Fixed Cartesian triples, never caller indices.

use crate::Complex;

fn parity(n: i32) -> f64 {
    if n % 2 == 0 { 1.0 } else { -1.0 }
}
fn lg(n: i32) -> f64 {
    libm::lgamma(f64::from(n))
}
fn c(j1: i32, j2: i32, j3: i32, m3: i32) -> f64 {
    ((f64::from(j3).powi(2) - f64::from(j1 - j2).powi(2))
        * (f64::from(j1 + j2 + 1).powi(2) - f64::from(j3).powi(2))
        * (f64::from(j3).powi(2) - f64::from(m3).powi(2)))
    .sqrt()
}
fn d(j1: i32, j2: i32, j3: i32, m1: i32, m2: i32, m3: i32) -> f64 {
    f64::from(2 * j3 + 1)
        * (f64::from(j3 * (j3 + 1)) * f64::from(m2 - m1)
            + f64::from(j2 * (j2 + 1) - j1 * (j1 + 1)) * f64::from(m3))
}
fn initial_j(j1: i32, j2: i32, m1: i32, m2: i32) -> f64 {
    parity(j1 + m1)
        * (0.5
            * (lg(j1 - m1 + 1) + lg(j1 + m1 + 1) + lg(2 * j1 - 2 * j2 + 1) + lg(2 * j2 + 1)
                - lg(j2 - m2 + 1)
                - lg(j2 + m2 + 1)
                - lg(j1 - j2 - m1 - m2 + 1)
                - lg(j1 - j2 + m1 + m2 + 1)
                - lg(2 * j1 + 2)))
        .exp()
}
fn initial_m(j1: i32, j2: i32, m1: i32, m2: i32) -> f64 {
    parity(j2 + m2)
        * (0.5
            * (lg(j1 + m1 + 1)
                + lg(j2 + m2 + 1)
                + lg(j1 + j2 - m1 - m2 + 1)
                + lg(2 * m1 + 2 * m2 + 1)
                - lg(j1 - m1 + 1)
                - lg(j2 - m2 + 1)
                - lg(j1 - j2 + m1 + m2 + 1)
                - lg(j2 - j1 + m1 + m2 + 1)
                - lg(j1 + j2 + m1 + m2 + 2)))
        .exp()
}

/// Wigner 3j symbol, using treams' two-sided recurrence.
#[must_use]
pub fn wigner3j(j1: i32, j2: i32, j3: i32, m1: i32, m2: i32, m3: i32) -> f64 {
    if j1 < 0
        || j2 < 0
        || j3 < 0
        || j3 < (j1 - j2).abs()
        || j3 > j1 + j2
        || m1.abs() > j1
        || m2.abs() > j2
        || m3.abs() > j3
        || m1 + m2 + m3 != 0
    {
        return 0.0;
    }
    let minimum = (j1 - j2).abs().max((m1 + m2).abs());
    if (j1 + j2 - (j1 - j2).abs()) / 4 + (j1 - j2).abs() > j3 {
        let mut prev = if minimum == j1 - j2 {
            initial_j(j1, j2, m1, m2)
        } else if minimum == j2 - j1 {
            initial_j(j2, j1, m2, m1)
        } else if minimum == m1 + m2 {
            initial_m(j1, j2, m1, m2)
        } else {
            initial_m(j2, j1, -m2, -m1)
        };
        let mut prevprev = 0.0;
        for j in minimum + 1..=j3 {
            let value = if minimum == 0 && j == 1 {
                -prev * f64::from(m2 - m1) / c(j1, j2, 1, m3)
            } else {
                (-d(j1, j2, j - 1, m1, m2, m3) * prev
                    - f64::from(j) * c(j1, j2, j - 1, m3) * prevprev)
                    / (f64::from(j - 1) * c(j1, j2, j, m3))
            };
            prevprev = prev;
            prev = value;
        }
        return prev;
    }
    let maximum = j1 + j2;
    let mut prev = parity(maximum - m3)
        * (0.5
            * (lg(2 * j1 + 1) + lg(2 * j2 + 1) + lg(maximum + m3 + 1) + lg(maximum - m3 + 1)
                - lg(2 * maximum + 2)
                - lg(j1 - m1 + 1)
                - lg(j1 + m1 + 1)
                - lg(j2 - m2 + 1)
                - lg(j2 + m2 + 1)))
        .exp();
    let mut prevprev = 0.0;
    for j in (j3..maximum).rev() {
        // At the upper endpoint the second term is identically zero. Evaluating
        // its coefficient outside the triangle would take sqrt of a negative.
        let value = if j == maximum - 1 {
            -d(j1, j2, j + 1, m1, m2, m3) * prev / (f64::from(j + 2) * c(j1, j2, j + 1, m3))
        } else {
            (-d(j1, j2, j + 1, m1, m2, m3) * prev
                - f64::from(j + 1) * c(j1, j2, j + 2, m3) * prevprev)
                / (f64::from(j + 2) * c(j1, j2, j + 1, m3))
        };
        prevprev = prev;
        prev = value;
    }
    prev
}

/// A solid harmonic and its Cartesian derivatives.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Solid {
    pub(crate) value: Complex,
    pub(crate) gradient: [Complex; 3],
    pub(crate) hessian: [[Complex; 3]; 3],
}

/// `r^l P_l^m(z/r) exp(i m phi)`. The second derivative is compiled out unless requested.
pub(crate) fn solid<const SECOND: bool>(l: i32, m: i32, position: [f64; 3]) -> Solid {
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
        let factor = parity(order) * (lg(l - order + 1) - lg(l + order + 1)).exp();
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

/// `P_l^m(z/r) exp(i m phi)` and its Cartesian position derivatives.
/// Solid harmonics avoid a coordinate singularity on the polar axis.
#[must_use]
pub fn angular(l: i32, m: i32, position: [f64; 3]) -> (Complex, [Complex; 3]) {
    let jet = solid::<false>(l, m, position);
    let r2 = position.iter().map(|v| v * v).sum::<f64>();
    let scale = r2.sqrt().powi(l);
    (
        jet.value / scale,
        std::array::from_fn(|axis| {
            (jet.gradient[axis] - f64::from(l) * jet.value * position[axis] / r2) / scale
        }),
    )
}
