//! Wigner 3j symbols from the Schulten-Gordon recurrence in `j3`, continued in each
//! direction while the symbols grow, and the Wigner small-d and D functions in the z-y-z
//! convention.
//!
//! Upstream: `treams.special.wigner3j` (`_wigner3j.pyx`), `wignersmalld` and `wignerd`
//! (`_wignerd.pyx`).
//!
//! Two algorithms compute small-d values, because their callers need different things:
//!
//! - [`wigner_small_d_matrix`] gives the whole real `(2l + 1) x (2l + 1)` block of one
//!   degree at a real angle, from the eigenvectors of the symmetric angular-momentum
//!   generator. Multipole rotations need every element of a block, and the eigensystem
//!   avoids the cancellation of the factorial sums.
//! - [`wigner_small_d`] gives one element at a real or complex angle from a Jacobi
//!   polynomial recurrence in `O(l)` time. Its kernel takes a `numerics::Jet` angle, so
//!   the Euler-angle pullbacks of [`wigner_d_array`] differentiate the same code.
//!
//! The two agree to `2e-12 + 1e-10 |d|` per element up to degree 128: the property
//! `wigner_small_d_matches_the_generator_exponential` in `properties/special.rs` checks
//! this bound.
#![allow(clippy::indexing_slicing)] // Complete small-d matrices and fixed Euler-angle triples.

use nalgebra::{DMatrix, SymmetricEigen};

use super::log_factorial;
use crate::{
    Complex, Error, MAX_DEGREE, Result,
    linalg::product,
    numerics::{Jet, broadcast, finite, parallel::Parallel, parity},
};

/// The coefficient `A(j3)` of the three-term recurrence of Schulten and Gordon (1975)
/// in `j3`, `j3 A(j3 + 1) f(j3 + 1) + B(j3) f(j3) + (j3 + 1) A(j3) f(j3 - 1) = 0` for
/// `f(j3) = (j1 j2 j3; m1 m2 m3)`: Eq. (7) of Y.-L. Xu, J. Comput. Phys. 139, 137
/// (1998), as cited by treams (`_coeffc`).
fn c(j1: i32, j2: i32, j3: i32, m3: i32) -> f64 {
    ((f64::from(j3).powi(2) - f64::from(j1 - j2).powi(2))
        * (f64::from(j1 + j2 + 1).powi(2) - f64::from(j3).powi(2))
        * (f64::from(j3).powi(2) - f64::from(m3).powi(2)))
    .sqrt()
}

/// The coefficient `B(j3)` of the recurrence of [`c`]: Eq. (7) of Xu (1998), treams
/// `_coeffd`.
fn d(j1: i32, j2: i32, j3: i32, m1: i32, m2: i32, m3: i32) -> f64 {
    f64::from(2 * j3 + 1)
        * (f64::from(j3 * (j3 + 1)) * f64::from(m2 - m1)
            + f64::from(j2 * (j2 + 1) - j1 * (j1 + 1)) * f64::from(m3))
}

/// The closed form of the symbol at the lower end `j3 = j1 - j2` of a row: Eq. (14) of
/// Xu (1998), treams `_initforwardj`.
fn initial_j(j1: i32, j2: i32, m1: i32, m2: i32) -> f64 {
    parity(j1 + m1)
        * (0.5
            * (log_factorial(j1 - m1)
                + log_factorial(j1 + m1)
                + log_factorial(2 * j1 - 2 * j2)
                + log_factorial(2 * j2)
                - log_factorial(j2 - m2)
                - log_factorial(j2 + m2)
                - log_factorial(j1 - j2 - m1 - m2)
                - log_factorial(j1 - j2 + m1 + m2)
                - log_factorial(2 * j1 + 1)))
        .exp()
}

/// The closed form of the symbol at the lower end `j3 = m1 + m2` of a row: Eq. (16) of
/// Xu (1998), treams `_initforwardm`.
fn initial_m(j1: i32, j2: i32, m1: i32, m2: i32) -> f64 {
    parity(j2 + m2)
        * (0.5
            * (log_factorial(j1 + m1)
                + log_factorial(j2 + m2)
                + log_factorial(j1 + j2 - m1 - m2)
                + log_factorial(2 * m1 + 2 * m2)
                - log_factorial(j1 - m1)
                - log_factorial(j2 - m2)
                - log_factorial(j1 - j2 + m1 + m2)
                - log_factorial(j2 - j1 + m1 + m2)
                - log_factorial(j1 + j2 + m1 + m2 + 1)))
        .exp()
}

/// The symbol at the lower end `j3 = minimum` of a row, in closed form: [`initial_j`]
/// or [`initial_m`], with labels swapped or negated by the symmetries of the symbol.
fn lowest(j1: i32, j2: i32, m1: i32, m2: i32, minimum: i32) -> f64 {
    if minimum == j1 - j2 {
        initial_j(j1, j2, m1, m2)
    } else if minimum == j2 - j1 {
        initial_j(j2, j1, m2, m1)
    } else if minimum == m1 + m2 {
        initial_m(j1, j2, m1, m2)
    } else {
        initial_m(j2, j1, -m2, -m1)
    }
}

/// The symbol at the upper end `j3 = j1 + j2` of a row, in closed form, as treams starts
/// its downward recurrence (`_wigner3jbackward`).
fn highest(j1: i32, j2: i32, m1: i32, m2: i32) -> f64 {
    let (maximum, m3) = (j1 + j2, -m1 - m2);
    parity(maximum - m3)
        * (0.5
            * (log_factorial(2 * j1)
                + log_factorial(2 * j2)
                + log_factorial(maximum + m3)
                + log_factorial(maximum - m3)
                - log_factorial(2 * maximum + 1)
                - log_factorial(j1 - m1)
                - log_factorial(j1 + m1)
                - log_factorial(j2 - m2)
                - log_factorial(j2 + m2)))
        .exp()
}

/// The first `j` in `minimum + 1..=stop` at which the upward recurrence of the row
/// `(j1 j2 j; m1 m2 -m1-m2)` stops growing in magnitude, or `stop + 1`. Iterates the
/// ratio of consecutive symbols, which needs no closed-form start value.
fn growth_end(j1: i32, j2: i32, m1: i32, m2: i32, minimum: i32, stop: i32) -> i32 {
    let m3 = -m1 - m2;
    let mut ratio = f64::INFINITY;
    for j in minimum + 1..=stop {
        ratio = if minimum == 0 && j == 1 {
            -f64::from(m2 - m1) / c(j1, j2, 1, m3)
        } else {
            (-d(j1, j2, j - 1, m1, m2, m3) - f64::from(j) * c(j1, j2, j - 1, m3) / ratio)
                / (f64::from(j - 1) * c(j1, j2, j, m3))
        };
        if ratio.abs() < 1.0 {
            return j;
        }
    }
    stop + 1
}

/// The lowest `j` in `stop..=maximum` down to which the downward recurrence of the row
/// from `maximum = j1 + j2` grows in magnitude at every step, so that the symbols decay
/// upward from `j` on; `stop` if they do down to `stop`. The mirror of [`growth_end`].
fn decay_start(j1: i32, j2: i32, m1: i32, m2: i32, maximum: i32, stop: i32) -> i32 {
    let m3 = -m1 - m2;
    let mut ratio = f64::INFINITY;
    for j in (stop..maximum).rev() {
        ratio = if j == maximum - 1 {
            -d(j1, j2, maximum, m1, m2, m3) / (f64::from(maximum + 1) * c(j1, j2, maximum, m3))
        } else {
            (-d(j1, j2, j + 1, m1, m2, m3) - f64::from(j + 1) * c(j1, j2, j + 2, m3) / ratio)
                / (f64::from(j + 2) * c(j1, j2, j + 1, m3))
        };
        if ratio.abs() < 1.0 {
            return j + 1;
        }
    }
    stop
}

/// The first degree [`visit`] recurs downward, as far as `first..=last` can tell:
/// `max(growth_end, min(switch, decay_start))` with each scan cut off where it no
/// longer decides a degree in the range. Whether a `j3 >= first` lies below the turn
/// is therefore the same for every range.
fn turn(
    j1: i32,
    j2: i32,
    m1: i32,
    m2: i32,
    [minimum, maximum]: [i32; 2],
    [first, last]: [i32; 2],
) -> i32 {
    let switch = (j1 + j2 - (j1 - j2).abs()) / 4 + (j1 - j2).abs();
    let growth_end = growth_end(j1, j2, m1, m2, minimum, last);
    if growth_end > last {
        return last + 1;
    }
    let bound = growth_end.max(first);
    if switch <= bound {
        bound
    } else {
        switch.min(decay_start(j1, j2, m1, m2, maximum, bound))
    }
}

/// Visits `(j3, (j1 j2 j3; m1 m2 -m1-m2))` for every `j3` in `first..=last`, a subrange of
/// the admissible `max(|j1 - j2|, |m1 + m2|)..=j1 + j2`, with treams' two-sided
/// recurrence in `j3` (Schulten and Gordon).
///
/// The upward recurrence from the closed form at the lower end is stable while the
/// symbols grow, in the classically forbidden region above the lower end; the downward
/// recurrence from the closed form at `j1 + j2` is stable while they grow downward,
/// through the upper forbidden region; both are neutral where the symbols oscillate.
/// treams switches from upward to downward at `|j1 - j2| + (j1 + j2 - |j1 - j2|) / 4`,
/// whatever the orders. For extreme orders either forbidden region can reach past that
/// switch, and recurring into it loses all accuracy (errors up to 1e-2 below `j = 90`,
/// a factor of 1800 at `j = 260`). Here the upward recurrence therefore
/// continues past the switch while the symbols grow, and the downward recurrence
/// continues below it while they grow downward: the turn is
/// `max(growth_end, min(switch, decay_start))`. Whether a `j3` is recurred upward or
/// downward does not depend on `first` or `last`, so rows and single symbols agree bit
/// for bit.
fn visit(
    j1: i32,
    j2: i32,
    m1: i32,
    m2: i32,
    [first, last]: [i32; 2],
    mut emit: impl FnMut(i32, f64),
) {
    let m3 = -m1 - m2;
    let minimum = (j1 - j2).abs().max(m3.abs());
    let maximum = j1 + j2;
    // The first degree recurred downward.
    let turn = turn(j1, j2, m1, m2, [minimum, maximum], [first, last]);
    if first < turn {
        let (mut value, mut previous) = (lowest(j1, j2, m1, m2, minimum), 0.0);
        for j in minimum..turn.min(last + 1) {
            if j > minimum {
                let next = if minimum == 0 && j == 1 {
                    -value * f64::from(m2 - m1) / c(j1, j2, 1, m3)
                } else {
                    (-d(j1, j2, j - 1, m1, m2, m3) * value
                        - f64::from(j) * c(j1, j2, j - 1, m3) * previous)
                        / (f64::from(j - 1) * c(j1, j2, j, m3))
                };
                (previous, value) = (value, next);
            }
            if j >= first {
                emit(j, value);
            }
        }
    }
    if turn > last {
        return;
    }
    let (mut value, mut previous) = (highest(j1, j2, m1, m2), 0.0);
    if last == maximum {
        emit(maximum, value);
    }
    for j in (turn.max(first)..maximum).rev() {
        // At the upper endpoint the second term is identically zero. Evaluating
        // its coefficient outside the triangle would take sqrt of a negative.
        let next = if j == maximum - 1 {
            -d(j1, j2, j + 1, m1, m2, m3) * value / (f64::from(j + 2) * c(j1, j2, j + 1, m3))
        } else {
            (-d(j1, j2, j + 1, m1, m2, m3) * value
                - f64::from(j + 1) * c(j1, j2, j + 2, m3) * previous)
                / (f64::from(j + 2) * c(j1, j2, j + 1, m3))
        };
        (previous, value) = (value, next);
        if j <= last {
            emit(j, value);
        }
    }
}

/// Wigner 3j symbol `(j1 j2 j3; m1 m2 m3)`, zero outside the admissible labels.
///
/// Evaluated with treams' two-sided recurrence, continued in each direction for as long
/// as the symbols grow (see `visit`).
///
/// Upstream: `treams.special.wigner3j`. Differences: more accurate for large `j` with
/// extreme orders, and equal up to rounding elsewhere; negative `j` gives zero, where
/// treams raises `ValueError`.
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
    let mut symbol = 0.0;
    visit(j1, j2, m1, m2, [j3, j3], |_, value| symbol = value);
    symbol
}

/// Wigner 3j symbols `(j1 j2 j3; m1 m2 -m1-m2)` for every admissible `j3` from one
/// recurrence, equal bit for bit to [`wigner3j`] at each `j3`.
#[derive(Debug, Default)]
pub(crate) struct Wigner3jRow {
    first: i32,
    values: Vec<f64>,
}

impl Wigner3jRow {
    /// The row `(j1 j2 j3; m1 m2 -m1-m2)` over every admissible `j3`; empty for
    /// inadmissible `j1, j2, m1, m2`.
    pub(crate) fn new(j1: i32, j2: i32, m1: i32, m2: i32) -> Self {
        if j1 < 0 || j2 < 0 || m1.abs() > j1 || m2.abs() > j2 {
            return Self::default();
        }
        let first = (j1 - j2).abs().max((m1 + m2).abs());
        let mut values = vec![0.0; usize::try_from(j1 + j2 + 1 - first).unwrap_or_default()];
        visit(j1, j2, m1, m2, [first, j1 + j2], |j3, value| {
            if let Some(slot) = usize::try_from(j3 - first)
                .ok()
                .and_then(|index| values.get_mut(index))
            {
                *slot = value;
            }
        });
        Self { first, values }
    }

    /// The symbol at `j3`; zero outside the admissible range.
    pub(crate) fn get(&self, j3: i32) -> f64 {
        usize::try_from(j3 - self.first)
            .ok()
            .and_then(|index| self.values.get(index))
            .copied()
            .unwrap_or_default()
    }
}

/// Ladder coefficient `sqrt(l(l + 1) - m(m + 1)) / 2` of the angular-momentum generator.
pub(crate) fn ladder(l: i32, m: i32) -> f64 {
    0.5 * f64::from(l * (l + 1) - m * (m + 1)).max(0.0).sqrt()
}

/// Position `l + m` of order `m` in a degree-`l` block ordered `-l..=l`.
pub(crate) fn index(l: i32, m: i32) -> usize {
    usize::try_from(l + m).unwrap_or_default()
}

/// Complete real Wigner small-d matrix `d^l(theta)` of one degree, rows and columns
/// ordered `-l..=l`, for multipole rotations. A symmetric angular-momentum eigensystem
/// avoids factorial-sum cancellation.
pub(crate) fn wigner_small_d_matrix(l: i32, theta: f64) -> Result<DMatrix<f64>> {
    if !(0..=MAX_DEGREE).contains(&l) || !theta.is_finite() {
        return Err(Error::InvalidInput(
            // 128 is MAX_DEGREE.
            "require 0 <= l <= 128 and a finite rotation angle".into(),
        ));
    }
    let n = index(l, l) + 1;
    if theta == 0.0 {
        return Ok(DMatrix::identity(n, n));
    }
    let mut generator = DMatrix::zeros(n, n);
    for m in -l..l {
        let i = index(l, m);
        generator[(i, i + 1)] = ladder(l, m);
        generator[(i + 1, i)] = generator[(i, i + 1)];
    }
    let eigen = SymmetricEigen::new(generator);
    let vectors = eigen.eigenvectors.map(|x| Complex::new(x, 0.0));
    let mut weighted = vectors.clone();
    for j in 0..n {
        // The exact eigenvalues are integers; rounding prevents accumulated phase error.
        let phase = (-Complex::i() * theta * eigen.eigenvalues[j].round()).exp();
        for i in 0..n {
            weighted[(i, j)] *= phase;
        }
    }
    let exponential = product(&weighted, &vectors.transpose());
    Ok(DMatrix::from_fn(n, n, |i, j| {
        let power = i32::try_from(i).unwrap_or_default() - i32::try_from(j).unwrap_or_default();
        ((-Complex::i()).powi(power) * exponential[(i, j)]).re
    }))
}

/// One small-d element `d^l_(mk)(theta)` at a jet angle, from the Jacobi recurrence
/// (DLMF 18.9.1-2) in `O(l)` time and `O(1)` storage. Half-angle powers keep the
/// off-diagonal limits where `cos(theta)` rounds to one. Symmetries keep both Jacobi
/// parameters nonnegative.
fn wigner_small_d_jet<const N: usize>(l: i32, mut m: i32, mut k: i32, theta: Jet<N>) -> Jet<N> {
    let mut sign = 1.0;
    if k.abs() > m.abs() {
        sign *= parity(m - k);
        std::mem::swap(&mut m, &mut k);
    }
    if m < 0 {
        sign *= parity(m - k);
        m = -m;
        k = -k;
    }
    let (a, b) = (f64::from(m - k), f64::from(m + k));
    let x = theta.chain(theta.value.cos(), -theta.value.sin());
    let half = theta * 0.5;
    let sine = half.chain(half.value.sin(), half.value.cos());
    let cosine = half.chain(half.value.cos(), -half.value.sin());
    let normalization = (0.5
        * (log_factorial(l + m) + log_factorial(l - m)
            - log_factorial(l + k)
            - log_factorial(l - k)))
    .exp();
    let n = l - m;
    let mut prev = Jet::constant(1.0);
    let mut polynomial = if n == 0 {
        prev
    } else {
        0.5 * ((a - b) + (a + b + 2.0) * x)
    };
    for j in 2..=n {
        let j = f64::from(j);
        let t = 2.0 * j + a + b;
        let next = ((t - 1.0) * (t * (t - 2.0) * x + a * a - b * b) * polynomial
            - 2.0 * (j + a - 1.0) * (j + b - 1.0) * t * prev)
            * (1.0 / (2.0 * j * (j + a + b) * (t - 2.0)));
        prev = polynomial;
        polynomial = next;
    }
    sign * parity(m - k) * normalization * sine.powi(m - k) * cosine.powi(m + k) * polynomial
}

/// Wigner small-d element `d^l_(mk)(theta)` at a real or complex angle, without a matrix.
/// Orders with `|m| > l` or `|k| > l` give zero.
///
/// Upstream: `treams.special.wignersmalld`. Differences: degrees above 128 and
/// non-finite angles give an error.
pub fn wigner_small_d(l: i32, m: i32, k: i32, theta: Complex) -> Result<Complex> {
    if !(0..=MAX_DEGREE).contains(&l) || !finite(theta) {
        return Err(Error::InvalidInput(
            // 128 is MAX_DEGREE.
            "require 0 <= l <= 128 and finite Wigner angle".into(),
        ));
    }
    if m.unsigned_abs() > l.unsigned_abs() || k.unsigned_abs() > l.unsigned_abs() {
        return Ok(Complex::default());
    }
    let value = wigner_small_d_jet(l, m, k, Jet::<0>::constant(theta)).value;
    if !finite(value) {
        return Err(Error::SpecialFunction("non-finite Wigner result".into()));
    }
    Ok(value)
}

/// Wigner D element `D^l_(mk)(phi, theta, psi) = e^(-i m phi) d^l_(mk)(theta) e^(-i k psi)`
/// in the z-y-z convention, allowing complex Euler angles.
///
/// Upstream: `treams.special.wignerd`. Differences: as [`wigner_small_d`].
pub fn wigner_d(l: i32, m: i32, k: i32, angles: [Complex; 3]) -> Result<Complex> {
    if angles.iter().any(|&z| !finite(z)) {
        return Err(Error::InvalidInput("Wigner angles must be finite".into()));
    }
    let value = (-Complex::i() * (f64::from(m) * angles[0] + f64::from(k) * angles[2])).exp()
        * wigner_small_d(l, m, k, angles[1])?;
    if !finite(value) {
        return Err(Error::SpecialFunction("non-finite Wigner result".into()));
    }
    Ok(value)
}

/// Wigner arrays evaluate in parallel from this many elements.
const PARALLEL: Parallel = Parallel::AtLeast(1024);

/// What [`wigner_d_array`] saves for its pullback: the labels and the Euler angles. The
/// pullback applies the local chain rules.
#[derive(Debug)]
pub struct WignerDResidual {
    labels: Vec<[i32; 3]>,
    angles: [Vec<Complex>; 3],
    size: usize,
}

impl WignerDResidual {
    /// Keep the broadcast inputs needed by derivatives without evaluating values.
    pub fn new(labels: Vec<[i32; 3]>, angles: [Vec<Complex>; 3]) -> Result<Self> {
        let [a, b, c] = angles.each_ref().map(Vec::len);
        let size = broadcast::size(
            &[labels.len(), a, b, c],
            "Wigner arrays must have equal lengths or scalar inputs",
        )?;
        Ok(Self {
            labels,
            angles,
            size,
        })
    }

    fn element(&self, i: usize) -> ([i32; 3], [Complex; 3]) {
        (
            broadcast::element(&self.labels, i),
            self.angles.each_ref().map(|a| broadcast::element(a, i)),
        )
    }

    /// Apply one direction in the Euler angles, carrying one jet slot per value.
    pub fn pushforward(&self, tangents: [&[Complex]; 3]) -> Result<Vec<Complex>> {
        broadcast::pushforward(
            tangents,
            self.size,
            "Wigner tangents must be finite and broadcast to output",
            PARALLEL,
            |i, tangent| {
                let ([l, m, k], angles) = self.element(i);
                if tangent == [Complex::default(); 3]
                    || m.unsigned_abs() > l.unsigned_abs()
                    || k.unsigned_abs() > l.unsigned_abs()
                {
                    return Ok(Complex::default());
                }
                let angles: [Jet<1>; 3] = std::array::from_fn(|axis| Jet {
                    value: angles[axis],
                    derivative: [tangent[axis]],
                });
                Ok(wigner_d_jet(l, m, k, angles)?.derivative[0])
            },
        )
    }

    /// The gradients with respect to the three Euler angles, given the gradient
    /// `cotangent` with respect to the values. An angle given as one value for all
    /// elements receives the sum of its element gradients.
    pub fn pullback(&self, cotangent: &[Complex]) -> Result<[Vec<Complex>; 3]> {
        broadcast::pullback(
            cotangent,
            self.size,
            "Wigner cotangent must be finite and match output",
            self.angles.each_ref().map(Vec::len),
            PARALLEL,
            |i, &g| {
                let ([l, m, k], angles) = self.element(i);
                if g == Complex::default()
                    || m.unsigned_abs() > l.unsigned_abs()
                    || k.unsigned_abs() > l.unsigned_abs()
                {
                    return Ok([Complex::default(); 3]);
                }
                let angles: [Jet<3>; 3] =
                    std::array::from_fn(|axis| Jet::variable(angles[axis], axis));
                let value = wigner_d_jet(l, m, k, angles)?;
                Ok(value.derivative.map(|derivative| g * derivative.conj()))
            },
        )
    }
}

/// The same Euler-angle chain rule serves both one tangent and three adjoint
/// partials, with the derivative width selected by the caller.
fn wigner_d_jet<const N: usize>(l: i32, m: i32, k: i32, angles: [Jet<N>; 3]) -> Result<Jet<N>> {
    let value = (-Complex::i() * (f64::from(m) * angles[0] + f64::from(k) * angles[2])).exp()
        * wigner_small_d_jet(l, m, k, angles[1]);
    if !value.finite() {
        return Err(Error::SpecialFunction(
            "non-finite Wigner derivative".into(),
        ));
    }
    Ok(value)
}

/// Broadcast [`wigner_d`] values with all three Euler angles differentiable: each input
/// is either one value or one value per element.
///
/// Upstream: `treams.special.wignerd` over arrays.
pub fn wigner_d_array(
    labels: Vec<[i32; 3]>,
    angles: [Vec<Complex>; 3],
) -> Result<(Vec<Complex>, WignerDResidual)> {
    let residual = WignerDResidual::new(labels, angles)?;
    let values = broadcast::map(residual.size, PARALLEL, |i| {
        let ([l, m, k], a) = residual.element(i);
        wigner_d(l, m, k, a)
    })?;
    Ok((values, residual))
}

#[cfg(test)]
mod tests {
    //! 3j rows against single symbols and exact values; the Wigner identities of the
    //! public functions are in `properties/special.rs`.

    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::{Wigner3jRow, wigner3j};
    use crate::test_support::ALGEBRA_CASES;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn wigner3j_rows_match_single_symbols(
            (j1, j2, m1, m2) in (0_i32..=90, 0_i32..=90).prop_flat_map(|(j1, j2)| {
                let order = |j: i32| prop_oneof![Just(j), Just(-j), -j..=j];
                (Just(j1), Just(j2), order(j1), order(j2))
            }),
        ) {
            check_wigner3j_row(j1, j2, m1, m2)?;
        }
    }

    /// A row from one recurrence equals [`wigner3j`] bit for bit at every `j3`, including
    /// zero just outside the admissible range.
    fn check_wigner3j_row(j1: i32, j2: i32, m1: i32, m2: i32) -> Result<(), TestCaseError> {
        let row = Wigner3jRow::new(j1, j2, m1, m2);
        for j3 in -1..=j1 + j2 + 1 {
            let single = wigner3j(j1, j2, j3, m1, m2, -m1 - m2);
            prop_assert_eq!(row.get(j3).to_bits(), single.to_bits(), "j3 = {}", j3);
        }
        Ok(())
    }

    #[test]
    fn wigner3j_matches_exact_symbols_in_the_forbidden_regions() {
        // Exact values of the Racah formula in rational arithmetic. The first four lie in
        // the lower classically forbidden region, where the downward recurrence of treams
        // errs by 2e-8, 2e-10, 2e-3 and 4e-12. The last three lie in the upper one, where
        // its upward recurrence below the switch gives -6.91248e-8, -7.8e-10 and -2.8e-9.
        for (symbol, exact) in [
            ([47, 60, 39, -47, 8, 39], 1.329_555_988_816_024_3e-11),
            ([53, 69, 46, -52, 10, 42], -1.230_181_842_452_470_3e-9),
            ([88, 66, 56, -10, 66, -56], 5.787_402_366_675_233e-17),
            ([78, 71, 64, 0, 64, -64], -2.792_414_470_253_1e-8),
            ([128, 64, 95, -128, 64, 64], -6.912_539_260_950_578e-8),
            ([245, 195, 144, -245, 194, 51], -1.884_797_170_508_145_5e-10),
            (
                [260, 130, 190, -260, 129, 131],
                -1.512_351_282_082_983_2e-12,
            ),
        ] {
            let [j1, j2, j3, m1, m2, m3] = symbol;
            let value = wigner3j(j1, j2, j3, m1, m2, m3);
            assert!(
                (value - exact).abs() <= 1e-12 * exact.abs(),
                "{symbol:?}: {value:e} != {exact:e}"
            );
        }
    }
}
