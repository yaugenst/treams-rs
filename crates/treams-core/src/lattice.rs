//! Ewald sums of outgoing spherical and cylindrical waves.
// Formulae and conventions: tfp-photonics/treams, lattice/_esum.pyx (MIT).
#![allow(clippy::float_cmp, clippy::indexing_slicing)] // Validated dimensions, exact branch points.

use std::f64::consts::PI;

use crate::jet::Jet;
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
    /// Number of independent lattice vectors.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dim
    }

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

fn kambe_jet<const N: usize>(n: i32, z: Jet<N>, eta: Complex) -> Jet<N> {
    let value = integrals::kambe(n, z.value, eta);
    let derivative = if N == 0 {
        Complex::default()
    } else {
        -z.value * integrals::kambe(n + 2, z.value, eta)
    };
    z.map(value, derivative)
}

fn reduced<const N: usize>(twice_n: i32, z: Jet<N>, t: Jet<N>) -> Jet<N> {
    let w = t.value.sqrt();
    let value = reduced_kambe(twice_n, z.value, w);
    // Analytic integral identities: dF_n/dz = F_(n+1), dF_n/d(w^2) = F_(n-1)/2.
    // Using w^2 keeps the derivative regular on the lattice plane and axis.
    let dz = if N == 0 {
        Complex::default()
    } else {
        reduced_kambe(twice_n + 2, z.value, w)
    };
    let dt = if N == 0 {
        Complex::default()
    } else {
        0.5 * reduced_kambe(twice_n - 2, z.value, w)
    };
    Jet {
        value,
        derivative: std::array::from_fn(|i| dz * z.derivative[i] + dt * t.derivative[i]),
    }
}

fn solid_jet<const N: usize>(l: i32, m: i32, r: [Jet<N>; 3]) -> Jet<N> {
    let solid = angular::solid::<false>(l, m, r.map(|r| r.value.re));
    Jet {
        value: solid.value,
        derivative: std::array::from_fn(|i| {
            (0..3).map(|j| solid.gradient[j] * r[j].derivative[i]).sum()
        }),
    }
}

fn real_term<const N: usize>(wave: Wave, k: Jet<N>, r: [Jet<N>; 3], eta: Complex) -> Jet<N> {
    let radius = r.into_iter().map(|r| r * r).sum::<Jet<N>>().sqrt();
    match wave {
        Wave::Spherical { l, m } => {
            let harmonic = solid_jet(l, m, r) * normalization(l, m);
            -Complex::i()
                * (2.0 / PI).sqrt()
                * k.powi(l)
                * harmonic
                * kambe_jet(2 * l, k * radius, eta)
        }
        Wave::Cylindrical { m } => {
            let xy = r[0] + Complex::i() * if m < 0 { -r[1] } else { r[1] };
            let sign = if m < 0 && m % 2 != 0 { -1.0 } else { 1.0 };
            -2.0 * Complex::i() / PI
                * sign
                * (k * xy).powi(m.abs())
                * kambe_jet(2 * m.abs() - 1, k * radius, eta)
        }
    }
}

fn inner_sw2<const N: usize>(
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
            * (-factorial(2 * n - s)
                - factorial(s - n)
                - factorial((l + m.abs() - s) / 2)
                - factorial((l - m.abs() - s) / 2))
            .exp();
    }
    sum * azimuth.powi(m.abs())
}
fn inner_sw1<const N: usize>(
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
            * (-factorial(n - s.midpoint(m.abs()))
                - factorial(n - (s - m.abs()) / 2)
                - factorial(l - s)
                - factorial(s - n))
            .exp();
    }
    sum * azimuth.powi(m.abs())
}
fn inner_cw1<const N: usize>(l: i32, n: i32, y: Jet<N>, beta: Jet<N>) -> Jet<N> {
    let mut sum = Jet::default();
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

fn reciprocal_term<const N: usize>(
    wave: Wave,
    dim: usize,
    k: Jet<N>,
    q: [Jet<N>; 3],
    r: [Jet<N>; 3],
    eta: Complex,
    measure: Jet<N>,
) -> Result<Jet<N>> {
    let beta2 = q.into_iter().map(|q| q * q).sum::<Jet<N>>() / (k * k);
    let mut value = (beta2 - 1.0) / (2.0 * eta * eta);
    value.value = below(value.value);
    if value.value.norm() <= 1e-14 {
        return Err(Error::InvalidInput(
            "lattice sum is at a diffraction threshold; supply a limiting complex wavenumber"
                .into(),
        ));
    }
    let i = Complex::i();
    Ok(match (wave, dim) {
        (Wave::Spherical { l, m }, 3) => {
            let harmonic = normalization(l, m) * solid_jet(l, m, q);
            2.0 * i * (-i).powi(l) * PI / (measure * k.powi(3)) * harmonic / k.powi(l)
                * reduced(2, value, Jet::default())
                / (eta * eta)
        }
        (Wave::Cylindrical { m }, 2) => {
            let xy = q[0] + i * if m < 0 { -q[1] } else { q[1] };
            2.0 * i * (-i).powi(m) / (measure * k * k)
                * (xy / k).powi(m.abs())
                * reduced(2, value, Jet::default())
                / (eta * eta)
        }
        (Wave::Spherical { l, m }, 2) => {
            let z = k * r[2];
            let azimuth = (q[0] + i * if m < 0 { -q[1] } else { q[1] }) / k;
            let mut sum = Jet::default();
            let mut factor = 1.0 / (2.0_f64.sqrt() * eta);
            for n in 0..=l - m.abs() {
                let polynomial = inner_sw2(l, m, n, z, beta2, azimuth);
                if polynomial.norm() != 0.0 {
                    sum += factor * reduced(1 - 2 * n, value, (z * eta).powi(2)) * polynomial;
                }
                factor *= 2.0 * eta * eta;
            }
            f64::from(2 * l + 1).sqrt()
                * (-i).powi(m)
                * (0.5 * (factorial(l + m) + factorial(l - m))).exp()
                / (measure * k * k * (-2.0_f64).powi(l))
                * sum
        }
        (Wave::Spherical { l, m }, 1) => {
            let beta = q[2] / k;
            let rho2 = k * k * (r[0] * r[0] + r[1] * r[1]);
            let azimuth = -k * (r[0] + i * if m < 0 { -r[1] } else { r[1] });
            let mut sum = Jet::default();
            let mult = 0.5 * eta * eta;
            let mut factor = mult.powi(m.abs());
            for n in m.abs()..=l {
                let polynomial = inner_sw1(l, m, n, rho2, beta, azimuth);
                if polynomial.norm() != 0.0 {
                    sum += factor * reduced(-2 * n, value, rho2 * eta * eta) * polynomial;
                }
                factor *= mult;
            }
            -i * (f64::from(2 * l + 1) / PI).sqrt()
                * (-i).powi(l - m)
                * (0.5 * (factorial(l + m) + factorial(l - m))).exp()
                / (2.0 * measure * k)
                * sum
        }
        (Wave::Cylindrical { m }, 1) => {
            let beta = q[0] / k;
            let y = k * if m < 0 { -r[1] } else { r[1] };
            let mut sum = Jet::default();
            let mut factor = 1.0 / (2.0_f64.sqrt() * eta);
            for n in 0..=m.abs() {
                let polynomial = inner_cw1(m.abs(), n, y, beta);
                if polynomial.norm() != 0.0 {
                    sum += factor * reduced(1 - 2 * n, value, (y * eta).powi(2)) * polynomial;
                }
                factor *= 2.0 * eta * eta;
            }
            2.0 * (-i).powi(m) / (PI.sqrt() * measure * k) * sum
        }
        _ => return Err(Error::InvalidInput("invalid lattice dimension".into())),
    })
}

fn shell_sum<const N: usize>(
    dim: usize,
    mut term: impl FnMut([i32; 3]) -> Result<Jet<N>>,
) -> Result<Jet<N>> {
    let maximum: i32 = match dim {
        1 => 200,
        2 => 32,
        _ => 16,
    };
    let mut sum = Jet::default();
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
                    if !value.finite() {
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
    r: [f64; 3],
    eta: Complex,
) -> Result<Complex> {
    Ok(sum_impl::<0>(wave, k, lattice, r, eta)?.value)
}

/// Local derivatives of one scalar Ewald sum. At coincidence, derivatives refer to
/// the regular image sum with that same lattice point excluded under perturbation.
#[derive(Clone, Copy, Debug)]
pub struct Derivatives {
    /// Sum value.
    pub value: Complex,
    /// Complex wavenumber derivative.
    pub k: Complex,
    /// Cartesian shift derivatives.
    pub position: [Complex; 3],
    /// Bloch components in lattice coordinates (unused components are zero).
    pub bloch: [Complex; 3],
    /// Row lattice-vector derivatives (unused components are zero).
    pub vectors: [[Complex; 3]; 3],
}

/// Differentiate an Ewald sum using analytic integral identities and Cartesian polynomials.
pub fn derivatives(
    wave: Wave,
    k: Complex,
    lattice: &Lattice,
    r: [f64; 3],
    eta: Complex,
) -> Result<Derivatives> {
    let result = sum_impl::<16>(wave, k, lattice, r, eta)?;
    Ok(Derivatives {
        value: result.value,
        k: result.derivative[0],
        position: std::array::from_fn(|i| result.derivative[1 + i]),
        bloch: std::array::from_fn(|i| result.derivative[4 + i]),
        vectors: std::array::from_fn(|i| std::array::from_fn(|j| result.derivative[7 + 3 * i + j])),
    })
}

fn sum_impl<const N: usize>(
    wave: Wave,
    k: Complex,
    lattice: &Lattice,
    r: [f64; 3],
    eta: Complex,
) -> Result<Jet<N>> {
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
    // The exact Ewald sum is independent of eta: choose it once and hold it fixed
    // during differentiation, avoiding derivatives of a purely numerical heuristic.
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
    let dim = lattice.dim;
    let axes = wave.axes(dim);
    let k = Jet::<N>::variable(k, 0);
    let mut r: [Jet<N>; 3] = std::array::from_fn(|i| Jet::variable(r[i], 1 + i));
    let bloch: [Jet<N>; 3] = std::array::from_fn(|i| {
        if i < dim {
            Jet::variable(lattice.bloch[i], 4 + i)
        } else {
            Jet::default()
        }
    });
    let direct: [[Jet<N>; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| Jet::variable(lattice.direct[(i, j)], 7 + 3 * i + j))
    });
    let reciprocal: [[Jet<N>; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| Jet {
            value: Complex::new(lattice.reciprocal[(i, j)], 0.0),
            derivative: std::array::from_fn(|index| {
                if index < 7 {
                    return Complex::default();
                }
                let (p, q) = ((index - 7) / 3, (index - 7) % 3);
                Complex::new(
                    if p < dim && q < dim {
                        -lattice.reciprocal[(i, q)] * lattice.reciprocal[(p, j)] / (2.0 * PI)
                    } else {
                        0.0
                    },
                    0.0,
                )
            }),
        })
    });
    let measure = Jet {
        value: Complex::new(lattice.measure, 0.0),
        derivative: std::array::from_fn(|index| {
            if index < 7 {
                return Complex::default();
            }
            let (p, q) = ((index - 7) / 3, (index - 7) % 3);
            Complex::new(
                if p < dim && q < dim {
                    lattice.measure * lattice.reciprocal[(p, q)] / (2.0 * PI)
                } else {
                    0.0
                },
                0.0,
            )
        }),
    };
    let vector = |matrix: &[[Jet<N>; 3]; 3], point: [i32; 3]| -> [Jet<N>; 3] {
        std::array::from_fn(|j| (0..dim).map(|i| f64::from(point[i]) * matrix[i][j]).sum())
    };
    let mut lattice_shift = [Jet::default(); 3];
    for (i, row) in direct.iter().enumerate().take(dim) {
        let cells = ((0..dim)
            .map(|j| lattice.reciprocal[(i, j)] * r[axes[j]].value.re)
            .sum::<f64>()
            / (2.0 * PI))
            .round();
        for j in 0..dim {
            lattice_shift[j] += cells * row[j];
        }
    }
    for j in 0..dim {
        r[axes[j]] -= lattice_shift[j];
    }
    let phase = (-Complex::i()
        * (0..dim)
            .map(|j| bloch[j] * lattice_shift[j])
            .sum::<Jet<N>>())
    .exp();
    let real = shell_sum(dim, |point| {
        let vector = vector(&direct, point);
        let mut shift = r.map(|x| -x);
        for j in 0..dim {
            shift[axes[j]] -= vector[j];
        }
        if shift.iter().all(|r| r.value == Complex::default()) {
            return Ok(Jet::default());
        }
        let phase = (0..dim).map(|j| bloch[j] * vector[j]).sum::<Jet<N>>();
        Ok(real_term(wave, k, shift, eta) * (Complex::i() * phase).exp())
    })?;
    let reciprocal = shell_sum(dim, |point| {
        let vector = vector(&reciprocal, point);
        let mut q = [Jet::default(); 3];
        for j in 0..dim {
            q[axes[j]] = vector[j] + bloch[j];
        }
        let phase =
            (-Complex::i() * q.into_iter().zip(r).map(|(q, r)| q * r).sum::<Jet<N>>()).exp();
        Ok(reciprocal_term(wave, dim, k, q, r, eta, measure)? * phase)
    })?;
    let self_term = if r.iter().all(|r| r.value == Complex::default()) {
        let value = below(-0.5 / (eta * eta));
        // Cartesian Taylor term of real_Ewald(-r)-outgoing(-r). Keeping its
        // first-order polynomial gives the regular image-sum derivative at self.
        match wave {
            Wave::Spherical { l, m } => {
                k.powi(l)
                    * solid_jet(l, m, r)
                    * normalization(l, m)
                    * (2.0_f64.powi(-l - 1) / PI.sqrt())
                    * integrals::gamma(-f64::from(l) - 0.5, value)
            }
            Wave::Cylindrical { m } => {
                let sign = if m < 0 && m % 2 != 0 { -1.0 } else { 1.0 };
                let xy = r[0] + Complex::i() * if m < 0 { -r[1] } else { r[1] };
                Complex::i() / PI
                    * sign
                    * (0.5 * k * xy).powi(m.abs())
                    * integrals::gamma(-f64::from(m.abs()), value)
            }
        }
    } else {
        Jet::default()
    };
    Ok(phase * (real + reciprocal + self_term))
}

/// Cotangents of the continuous Ewald parameters under the real Hermitian pairing.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gradient {
    /// Complex wavenumber cotangent.
    pub k: Complex,
    /// Real shift cotangents.
    pub position: [f64; 3],
    /// Real Bloch cotangents.
    pub bloch: [f64; 3],
    /// Real row lattice-vector cotangents.
    pub vectors: [[f64; 3]; 3],
}
impl Gradient {
    pub(crate) fn add(&mut self, other: Self) {
        self.k += other.k;
        for i in 0..3 {
            self.position[i] += other.position[i];
            self.bloch[i] += other.bloch[i];
            for j in 0..3 {
                self.vectors[i][j] += other.vectors[i][j];
            }
        }
    }
}

/// Contract a batch of scalar-wave cotangents, recomputing local derivatives per wave.
/// Only sixteen local derivatives are live per Rayon worker; no mode Jacobian is retained.
pub fn pullback(
    waves: &[Wave],
    k: Complex,
    lattice: &Lattice,
    r: [f64; 3],
    eta: Complex,
    cotangent: &[Complex],
) -> Result<Gradient> {
    use rayon::prelude::*;
    if waves.len() != cotangent.len() || cotangent.iter().any(|&g| !finite(g)) {
        return Err(Error::InvalidInput("invalid lattice cotangent".into()));
    }
    waves
        .par_iter()
        .zip(cotangent)
        .map(|(&wave, &g)| {
            if g == Complex::default() {
                return Ok(Gradient::default());
            }
            let d = derivatives(wave, k, lattice, r, eta)?;
            Ok(Gradient {
                k: d.k.conj() * g,
                position: d.position.map(|x| (g.conj() * x).re),
                bloch: d.bloch.map(|x| (g.conj() * x).re),
                vectors: d.vectors.map(|row| row.map(|x| (g.conj() * x).re)),
            })
        })
        .try_reduce(Gradient::default, |mut a, b| {
            a.add(b);
            Ok(a)
        })
}
