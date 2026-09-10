//! Ewald sums of outgoing spherical and cylindrical waves.
// Formulae and conventions: tfp-photonics/treams, lattice/_esum.pyx (MIT).
#![allow(clippy::float_cmp, clippy::indexing_slicing)] // Validated dimensions, exact branch points.

use std::f64::consts::PI;

use nalgebra::Matrix3;

use crate::{Complex, Error, Result, angular, finite, integrals};

/// Scalar outgoing wave in the Ewald sum.
#[derive(Clone, Copy, Debug)]
pub enum Wave {
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

impl Wave {
    fn validate(self, dim: usize) -> Result<()> {
        let valid = match self {
            Self::Spherical { l, m } => {
                (0..=128).contains(&l) && m.unsigned_abs() <= l.unsigned_abs()
            }
            Self::Cylindrical { m } => (-128..=128).contains(&m) && dim <= 2,
        };
        if !valid {
            return Err(Error::InvalidInput(
                "invalid lattice wave degree or dimension".into(),
            ));
        }
        Ok(())
    }
    fn axes(self, dim: usize) -> [usize; 3] {
        if matches!(self, Self::Spherical { .. }) && dim == 1 {
            [2, 0, 1]
        } else {
            [0, 1, 2]
        }
    }
}

/// Row lattice vectors and a real Bloch wavevector in one, two or three dimensions.
#[derive(Clone, Debug)]
pub struct Lattice {
    dim: usize,
    direct: Matrix3<f64>,
    reciprocal: Matrix3<f64>,
    bloch: [f64; 3],
    measure: f64,
}

impl Lattice {
    /// Validate and precompute the reciprocal lattice. Rows are primitive vectors.
    pub fn new(vectors: &[Vec<f64>], bloch: &[f64]) -> Result<Self> {
        let dim = vectors.len();
        if !(1..=3).contains(&dim)
            || bloch.len() != dim
            || vectors
                .iter()
                .any(|row| row.len() != dim || row.iter().any(|v| !v.is_finite()))
            || bloch.iter().any(|v| !v.is_finite())
        {
            return Err(Error::InvalidInput(
                "lattice must be a finite square 1D, 2D or 3D matrix, with matching Bloch vector"
                    .into(),
            ));
        }
        let mut direct = Matrix3::identity();
        for i in 0..dim {
            for j in 0..dim {
                direct[(i, j)] = vectors[i][j];
            }
        }
        let inverse = direct
            .try_inverse()
            .ok_or_else(|| Error::InvalidInput("lattice vectors must be independent".into()))?;
        let reciprocal = 2.0 * PI * inverse.transpose();
        let measure = direct.determinant().abs();
        if !measure.is_finite() || reciprocal.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidInput(
                "lattice is numerically singular".into(),
            ));
        }
        let mut parallel = [0.0; 3];
        parallel[..dim].copy_from_slice(bloch);
        Ok(Self {
            dim,
            direct,
            reciprocal,
            bloch: parallel,
            measure,
        })
    }

    fn vector(&self, matrix: &Matrix3<f64>, point: [i32; 3]) -> [f64; 3] {
        std::array::from_fn(|j| {
            (0..self.dim)
                .map(|i| f64::from(point[i]) * matrix[(i, j)])
                .sum()
        })
    }
}

fn factorial(n: i32) -> f64 {
    libm::lgamma(f64::from(n + 1))
}
pub(crate) fn normalization(l: i32, m: i32) -> f64 {
    (f64::from(2 * l + 1) / (4.0 * PI)).sqrt() * (0.5 * (factorial(l - m) - factorial(l + m))).exp()
}
fn below(mut z: Complex) -> Complex {
    if z.im == 0.0 {
        z.im = -0.0;
    }
    z
}
fn reduced_gamma(n: f64, z: Complex) -> Complex {
    let z = below(z);
    integrals::gamma(n, z) / (-z).powf(n)
}
fn reduced_kambe(twice_n: i32, value: Complex, w: Complex) -> Complex {
    if w == Complex::default() {
        return reduced_gamma(0.5 * f64::from(twice_n), value);
    }
    let w = if w.re < 0.0 { -w } else { w };
    let mut x = (-2.0 * below(value) * w * w).sqrt();
    // The outgoing limit approaches the real x-axis from the upper half plane.
    if x.im == 0.0 {
        x.im = 1e-100;
    }
    2.0 * integrals::kambe(twice_n - 1, x, -Complex::i() / w) * w.powi(twice_n)
}

fn real_term(wave: Wave, k: Complex, r: [f64; 3], eta: Complex) -> Complex {
    let radius = r[0].hypot(r[1]).hypot(r[2]);
    match wave {
        Wave::Spherical { l, m } => {
            let angular = angular::solid::<false>(l, m, r).value * normalization(l, m);
            -Complex::i()
                * (2.0 / PI).sqrt()
                * k.powi(l)
                * angular
                * integrals::kambe(2 * l, k * radius, eta)
        }
        Wave::Cylindrical { m } => {
            let xy = Complex::new(r[0], if m < 0 { -r[1] } else { r[1] });
            let sign = if m < 0 && m % 2 != 0 { -1.0 } else { 1.0 };
            -2.0 * Complex::i() / PI
                * sign
                * (k * xy).powi(m.abs())
                * integrals::kambe(2 * m.abs() - 1, k * radius, eta)
        }
    }
}

fn inner_sw2(l: i32, m: i32, n: i32, z: Complex, beta: Complex) -> Complex {
    let mut sum = Complex::default();
    for s in n..=(l - m.abs()).min(2 * n) {
        if (l - m.abs() - s) % 2 != 0 {
            continue;
        }
        sum += (-z).powi(2 * n - s)
            * beta.powi(l - s)
            * (-factorial(2 * n - s)
                - factorial(s - n)
                - factorial((l + m.abs() - s) / 2)
                - factorial((l - m.abs() - s) / 2))
            .exp();
    }
    sum
}
fn inner_sw1(l: i32, m: i32, n: i32, rho: Complex, beta: Complex) -> Complex {
    let mut sum = Complex::default();
    for s in n..=(2 * n - m.abs()).min(l) {
        if (m.abs() - s) % 2 != 0 {
            continue;
        }
        sum += rho.powi(2 * n - s)
            * beta.powi(l - s)
            * (-factorial(n - s.midpoint(m.abs()))
                - factorial(n - (s - m.abs()) / 2)
                - factorial(l - s)
                - factorial(s - n))
            .exp();
    }
    sum
}
fn inner_cw1(l: i32, n: i32, y: Complex, beta: Complex) -> Complex {
    let mut sum = Complex::default();
    for s in n..=(2 * n).min(l) {
        sum += (-y).powi(2 * n - s)
            * beta.powi(l - s)
            * (factorial(l)
                - factorial(2 * n - s)
                - factorial(l - s)
                - factorial(s - n)
                - f64::from(s) * 2.0_f64.ln())
            .exp();
    }
    sum
}

fn reciprocal_term(
    wave: Wave,
    dim: usize,
    k: Complex,
    q: [f64; 3],
    r: [f64; 3],
    eta: Complex,
    measure: f64,
) -> Result<Complex> {
    let qnorm = q[0].hypot(q[1]).hypot(q[2]);
    let beta = qnorm / k;
    let value = below((beta * beta - 1.0) / (2.0 * eta * eta));
    if value.norm() <= 1e-14 {
        return Err(Error::InvalidInput(
            "lattice sum is at a diffraction threshold; supply a limiting complex wavenumber"
                .into(),
        ));
    }
    let i = Complex::i();
    Ok(match (wave, dim) {
        (Wave::Spherical { l, m }, 3) => {
            let harmonic = normalization(l, m) * angular::solid::<false>(l, m, q).value;
            2.0 * i * (-i).powi(l) * PI / (measure * k.powi(3)) * harmonic / k.powi(l)
                * reduced_gamma(1.0, value)
                / (eta * eta)
        }
        (Wave::Cylindrical { m }, 2) => {
            let xy = Complex::new(q[0], if m < 0 { -q[1] } else { q[1] });
            2.0 * i * (-i).powi(m) / (measure * k * k)
                * (xy / k).powi(m.abs())
                * reduced_gamma(1.0, value)
                / (eta * eta)
        }
        (Wave::Spherical { l, m }, 2) => {
            let z = k * r[2];
            let mut sum = Complex::default();
            let mut factor = 1.0 / (2.0_f64.sqrt() * eta);
            for n in 0..=l - m.abs() {
                let polynomial = inner_sw2(l, m, n, z, beta);
                if polynomial != Complex::default() {
                    sum += factor * reduced_kambe(1 - 2 * n, value, z * eta) * polynomial;
                }
                factor *= 2.0 * eta * eta;
            }
            f64::from(2 * l + 1).sqrt()
                * (-i).powi(m)
                * (0.5 * (factorial(l + m) + factorial(l - m))).exp()
                / (measure * k * k * (-2.0_f64).powi(l))
                * (i * f64::from(m) * q[1].atan2(q[0])).exp()
                * sum
        }
        (Wave::Spherical { l, m }, 1) => {
            let beta = q[2] / k;
            let rho = k * r[0].hypot(r[1]);
            let mut sum = Complex::default();
            let mult = 0.5 * eta * eta;
            let mut factor = mult.powi(m.abs());
            for n in m.abs()..=l {
                let polynomial = inner_sw1(l, m, n, rho, beta);
                if polynomial != Complex::default() {
                    sum += factor * reduced_kambe(-2 * n, value, rho * eta) * polynomial;
                }
                factor *= mult;
            }
            -i * (f64::from(2 * l + 1) / PI).sqrt()
                * (-i).powi(l - m)
                * (0.5 * (factorial(l + m) + factorial(l - m))).exp()
                / (2.0 * measure * k)
                * (i * f64::from(m) * (-r[1]).atan2(-r[0])).exp()
                * sum
        }
        (Wave::Cylindrical { m }, 1) => {
            let beta = q[0] / k;
            let y = k * if m < 0 { -r[1] } else { r[1] };
            let mut sum = Complex::default();
            let mut factor = 1.0 / (2.0_f64.sqrt() * eta);
            for n in 0..=m.abs() {
                let polynomial = inner_cw1(m.abs(), n, y, beta);
                if polynomial != Complex::default() {
                    sum += factor * reduced_kambe(1 - 2 * n, value, y * eta) * polynomial;
                }
                factor *= 2.0 * eta * eta;
            }
            2.0 * (-i).powi(m) / (PI.sqrt() * measure * k) * sum
        }
        _ => return Err(Error::InvalidInput("invalid lattice dimension".into())),
    })
}

fn shell_sum(dim: usize, mut term: impl FnMut([i32; 3]) -> Result<Complex>) -> Result<Complex> {
    let maximum: i32 = match dim {
        1 => 200,
        2 => 32,
        _ => 16,
    };
    let mut sum = Complex::default();
    let mut quiet = 0;
    for radius in 0..maximum {
        let mut magnitude = 0.0;
        let yrange = if dim > 1 { -radius..=radius } else { 0..=0 };
        let zrange = if dim > 2 { -radius..=radius } else { 0..=0 };
        for x in -radius..=radius {
            for y in yrange.clone() {
                for z in zrange.clone() {
                    if x.abs().max(y.abs()).max(z.abs()) != radius {
                        continue;
                    }
                    let value = term([x, y, z])?;
                    if !finite(value) {
                        return Err(Error::SpecialFunction("non-finite Ewald summand; change the split parameter or reduce the order".into()));
                    }
                    sum += value;
                    magnitude += value.norm();
                }
            }
        }
        quiet = if magnitude <= 2e-13 * sum.norm().max(1.0) {
            quiet + 1
        } else {
            0
        };
        if quiet >= 2 {
            return Ok(sum);
        }
    }
    Err(Error::SpecialFunction(
        "Ewald sum did not converge within the shell limit".into(),
    ))
}

/// Sum outgoing waves at `-r-R` with Bloch factor `exp(i kpar.R)`, excluding zero distance.
/// `eta=0` selects the Ewald split automatically. Exact diffraction thresholds are singular.
pub fn sum(
    wave: Wave,
    k: Complex,
    lattice: &Lattice,
    mut r: [f64; 3],
    eta: Complex,
) -> Result<Complex> {
    wave.validate(lattice.dim)?;
    if !finite(k) || k == Complex::default() || !finite(eta) || r.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "lattice wavenumber must be finite and nonzero; shift and eta must be finite".into(),
        ));
    }
    if matches!(wave, Wave::Cylindrical { .. }) && r[2] != 0.0 {
        return Err(Error::InvalidInput(
            "scalar cylindrical lattice shifts lie in the xy plane".into(),
        ));
    }
    let axes = wave.axes(lattice.dim);
    // Reduce shifts to the primitive cell; this both enforces Bloch periodicity
    // and prevents a distant source from exhausting a fixed shell budget.
    let mut lattice_shift = [0.0; 3];
    for i in 0..lattice.dim {
        let cells = ((0..lattice.dim)
            .map(|j| lattice.reciprocal[(i, j)] * r[axes[j]])
            .sum::<f64>()
            / (2.0 * PI))
            .round();
        for (j, shift) in lattice_shift.iter_mut().enumerate().take(lattice.dim) {
            *shift += cells * lattice.direct[(i, j)];
        }
    }
    for j in 0..lattice.dim {
        r[axes[j]] -= lattice_shift[j];
    }
    let phase = (-Complex::i()
        * (0..lattice.dim)
            .map(|j| lattice.bloch[j] * lattice_shift[j])
            .sum::<f64>())
    .exp();
    let eta = if eta == Complex::default() {
        let length = match lattice.dim {
            1 => lattice.measure,
            2 => lattice.measure.sqrt(),
            _ => lattice.measure.cbrt(),
        };
        let e = 1.0 / (k * length);
        (2.0 * PI).sqrt() * e * (0.125 / e.norm()).max(1.0)
    } else {
        eta
    };
    let real = shell_sum(lattice.dim, |point| {
        let vector = lattice.vector(&lattice.direct, point);
        let mut shift = r.map(|x| -x);
        for j in 0..lattice.dim {
            shift[axes[j]] -= vector[j];
        }
        if shift == [0.0; 3] {
            return Ok(Complex::default());
        }
        let bloch = (0..lattice.dim)
            .map(|j| lattice.bloch[j] * vector[j])
            .sum::<f64>();
        Ok(real_term(wave, k, shift, eta) * (Complex::i() * bloch).exp())
    })?;
    let reciprocal = shell_sum(lattice.dim, |point| {
        let vector = lattice.vector(&lattice.reciprocal, point);
        let mut q = [0.0; 3];
        for j in 0..lattice.dim {
            q[axes[j]] = vector[j] + lattice.bloch[j];
        }
        let phase = (-Complex::i() * q.iter().zip(r).map(|(q, r)| q * r).sum::<f64>()).exp();
        Ok(reciprocal_term(wave, lattice.dim, k, q, r, eta, lattice.measure)? * phase)
    })?;
    let self_term = if r == [0.0; 3] {
        let value = below(-0.5 / (eta * eta));
        match wave {
            Wave::Spherical { l: 0, .. } => integrals::gamma(-0.5, value) / (4.0 * PI),
            Wave::Cylindrical { m: 0 } => Complex::i() * integrals::gamma(0.0, value) / PI,
            _ => Complex::default(),
        }
    } else {
        Complex::default()
    };
    Ok(phase * (real + reciprocal + self_term))
}
