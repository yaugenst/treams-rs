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
        Self::from_array(
            std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    if i < dim && j < dim {
                        vectors[i][j]
                    } else {
                        0.0
                    }
                })
            }),
            std::array::from_fn(|i| if i < dim { bloch[i] } else { 0.0 }),
            dim,
        )
    }

    /// Construct from fixed storage without allocating intermediate row vectors.
    pub fn from_array(vectors: [[f64; 3]; 3], bloch: [f64; 3], dim: usize) -> Result<Self> {
        if !(1..=3).contains(&dim) || bloch[..dim].iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidInput(
                "invalid lattice dimension or Bloch vector".into(),
            ));
        }
        let direct = Matrix3::from_fn(|i, j| {
            if i < dim && j < dim {
                vectors[i][j]
            } else if i == j {
                1.0
            } else {
                0.0
            }
        });
        let reciprocal = crate::geometry::reciprocal(
            std::array::from_fn(|i| std::array::from_fn(|j| direct[(i, j)])),
            dim,
        )?;
        let reciprocal = Matrix3::from_fn(|i, j| reciprocal[i][j]);
        let measure = direct.determinant().abs();
        if !measure.is_finite() || reciprocal.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidInput(
                "lattice is numerically singular".into(),
            ));
        }
        Ok(Self {
            dim,
            direct,
            reciprocal,
            bloch,
            measure,
        })
    }
}

fn factorial(n: i32) -> f64 {
    // Wave labels are fixed through the lattice shells. Cache the identical
    // libm values instead of repeating gamma evaluations at every image.
    static LOG_FACTORIAL: std::sync::OnceLock<[f64; 257]> = std::sync::OnceLock::new();
    let values = LOG_FACTORIAL.get_or_init(|| {
        std::array::from_fn(|i| libm::lgamma(f64::from(u32::try_from(i).unwrap_or_default()) + 1.0))
    });
    usize::try_from(n)
        .ok()
        .and_then(|i| values.get(i))
        .copied()
        .unwrap_or_else(|| libm::lgamma(f64::from(n) + 1.0))
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
fn reduced_gamma(twice_n: i32, z: Complex) -> Complex {
    let z = below(z);
    let mut power = (-z).powi(twice_n.div_euclid(2));
    if twice_n % 2 != 0 {
        power *= crate::complex_sqrt(-z);
    }
    integrals::gamma(0.5 * f64::from(twice_n), z) / power
}
fn reduced_kambe(twice_n: i32, value: Complex, w: Complex) -> Complex {
    if w == Complex::default() {
        return reduced_gamma(twice_n, value);
    }
    let w = if w.re < 0.0 { -w } else { w };
    let mut x = (-2.0 * below(value) * w * w).sqrt();
    // The outgoing limit approaches the real x-axis from the upper half plane.
    if x.im == 0.0 {
        x.im = 1e-100;
    }
    2.0 * integrals::kambe(twice_n - 1, x, -Complex::i() / w) * w.powi(twice_n)
}

fn kambe_jet<const N: usize, const ETA: bool>(n: i32, z: Jet<N>, eta: Complex) -> Jet<N> {
    let value = integrals::kambe(n, z.value, eta);
    if ETA {
        let derivative =
            -eta.powi(n) * (0.5 * (1.0 / (eta * eta) - z.value * z.value * eta * eta)).exp();
        return Jet {
            value,
            derivative: std::array::from_fn(|i| {
                if i == 0 {
                    derivative
                } else {
                    Complex::default()
                }
            }),
        };
    }
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

fn direct_term<const N: usize>(wave: Wave, k: Jet<N>, r: [Jet<N>; 3]) -> Result<Jet<N>> {
    use crate::special::{self, Bessel, Radial};
    let radius = r.into_iter().map(|v| v * v).sum::<Jet<N>>().sqrt();
    let argument = k * radius;
    let (degree, spherical) = match wave {
        Wave::Spherical { l, .. } => (l, true),
        Wave::Cylindrical { m } => (m, false),
    };
    let radial = if N == 0 {
        Jet::constant(special::bessel(
            f64::from(degree),
            argument.value,
            Bessel::H1,
            spherical,
            0,
        )?)
    } else {
        let radial = if spherical {
            special::spherical(
                u32::try_from(degree).map_err(|_| Error::InvalidInput("invalid degree".into()))?,
                argument.value,
                Radial::Outgoing,
            )?
        } else {
            special::cylindrical(degree, argument.value, Radial::Outgoing)?
        };
        argument.map(radial.value, radial.first)
    };
    let angular = match wave {
        Wave::Spherical { l, m } => normalization(l, m) * solid_jet(l, m, r.map(|v| v / radius)),
        Wave::Cylindrical { m } => ((r[0] + Complex::i() * r[1]) / radius).powi(m),
    };
    let result = radial * angular;
    if !result.finite() {
        return Err(Error::SpecialFunction(
            "nonfinite direct lattice term".into(),
        ));
    }
    Ok(result)
}

fn real_term<const N: usize, const ETA: bool>(
    wave: Wave,
    k: Jet<N>,
    r: [Jet<N>; 3],
    eta: Complex,
) -> Jet<N> {
    let radius = r.into_iter().map(|r| r * r).sum::<Jet<N>>().sqrt();
    match wave {
        Wave::Spherical { l, m } => {
            let harmonic = solid_jet(l, m, r) * normalization(l, m);
            -Complex::i()
                * (2.0 / PI).sqrt()
                * k.powi(l)
                * harmonic
                * kambe_jet::<N, ETA>(2 * l, k * radius, eta)
        }
        Wave::Cylindrical { m } => {
            let xy = r[0] + Complex::i() * if m < 0 { -r[1] } else { r[1] };
            let sign = if m < 0 && m % 2 != 0 { -1.0 } else { 1.0 };
            -2.0 * Complex::i() / PI
                * sign
                * (k * xy).powi(m.abs())
                * kambe_jet::<N, ETA>(2 * m.abs() - 1, k * radius, eta)
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

// On-axis values need only half of the polynomial terms. Adjacent half-order
// gamma functions share one evaluation: recurse downward near zero, upward at
// large arguments to avoid subtracting the leading asymptotic term.
fn axial_cylindrical_reciprocal(order: i32, beta: Complex, z: Complex, eta: Complex) -> Complex {
    let maximum = order / 2;
    let upward = z.norm() > 4.0;
    let mut gamma = reduced_gamma(1 - 2 * if upward { maximum } else { 0 }, z);
    let endpoint = if maximum > 0 {
        (if z.im > 0.0 {
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
            * (factorial(order) - factorial(n) - factorial(order - 2 * n)).exp()
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
            if N == 0 && r[1].value == Complex::default() {
                return Ok(Jet::constant(
                    2.0 * (-i).powi(m) / (PI.sqrt() * measure.value * k.value)
                        * axial_cylindrical_reciprocal(m.abs(), beta.value, value.value, eta),
                ));
            }
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
    mut term: impl FnMut([i64; 3]) -> Result<Jet<N>>,
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
        crate::geometry::visit_cube(dim, i64::from(radius), true, |point| {
            let value = term(point)?;
            if !value.finite() {
                return Err(Error::SpecialFunction(
                    "non-finite Ewald summand; change the split parameter or reduce the order"
                        .into(),
                ));
            }
            sum += value;
            magnitude += value.norm();
            Ok(())
        })?;
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
    Ok(sum_impl::<0, 0>(wave, k, lattice, r, eta, 0)?.value)
}

/// Part of an Ewald sum, or one unaccelerated integer cube shell.
#[derive(Clone, Copy, Debug)]
pub enum SumPart {
    /// Complete outgoing lattice sum.
    Full,
    /// Real-space Ewald contribution.
    Real,
    /// Reciprocal-space Ewald contribution, including the self correction.
    Reciprocal,
    /// One cube boundary; axial half-cell shifts pair equidistant images instead.
    Direct(i64),
}
/// Evaluate an Ewald component or one direct-summation shell.
pub fn sum_part(
    wave: Wave,
    k: Complex,
    lattice: &Lattice,
    r: [f64; 3],
    eta: Complex,
    part: SumPart,
) -> Result<Complex> {
    Ok(match part {
        SumPart::Full => sum_impl::<0, 0>(wave, k, lattice, r, eta, 0)?,
        SumPart::Real => sum_impl::<0, 1>(wave, k, lattice, r, eta, 0)?,
        SumPart::Reciprocal => sum_impl::<0, 2>(wave, k, lattice, r, eta, 0)?,
        SumPart::Direct(shell) => sum_impl::<0, 3>(wave, k, lattice, r, eta, shell)?,
    }
    .value)
}

/// Local derivatives of one scalar Ewald sum. At coincidence, derivatives refer to
/// the regular image sum with that same lattice point excluded under perturbation.
#[derive(Clone, Copy, Debug)]
pub struct Derivatives {
    /// Sum value.
    pub value: Complex,
    /// Complex wavenumber derivative.
    pub k: Complex,
    /// Ewald split derivative; zero for the full and direct sums.
    pub eta: Complex,
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
    derivatives_part(wave, k, lattice, r, eta, SumPart::Full)
}
/// Differentiate the selected component, holding a direct shell's index fixed.
pub fn derivatives_part(
    wave: Wave,
    k: Complex,
    lattice: &Lattice,
    r: [f64; 3],
    eta: Complex,
    part: SumPart,
) -> Result<Derivatives> {
    if matches!(part, SumPart::Real | SumPart::Reciprocal) && eta == Complex::default() {
        return Err(Error::InvalidInput(
            "component adjoints require an explicit nonzero Ewald split".into(),
        ));
    }
    let mut result = match part {
        SumPart::Full => derivatives_impl::<0>(wave, k, lattice, r, eta, 0),
        SumPart::Real => derivatives_impl::<1>(wave, k, lattice, r, eta, 0),
        SumPart::Reciprocal => derivatives_impl::<2>(wave, k, lattice, r, eta, 0),
        SumPart::Direct(shell) => derivatives_impl::<3>(wave, k, lattice, r, eta, shell),
    }?;
    if matches!(part, SumPart::Real | SumPart::Reciprocal) {
        result.eta = sum_impl::<1, 4>(wave, k, lattice, r, eta, 0)?.derivative[0]
            * if matches!(part, SumPart::Real) {
                1.0
            } else {
                -1.0
            };
    }
    Ok(result)
}
fn derivatives_impl<const PART: usize>(
    wave: Wave,
    k: Complex,
    lattice: &Lattice,
    r: [f64; 3],
    eta: Complex,
    shell: i64,
) -> Result<Derivatives> {
    fn unpack<const N: usize>(result: Jet<N>, dim: usize) -> Derivatives {
        Derivatives {
            value: result.value,
            k: result.derivative[0],
            eta: Complex::default(),
            position: std::array::from_fn(|i| result.derivative[1 + i]),
            bloch: std::array::from_fn(|i| {
                if i < dim {
                    result.derivative[4 + i]
                } else {
                    Complex::default()
                }
            }),
            vectors: std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    if i < dim && j < dim {
                        result.derivative[4 + dim + dim * i + j]
                    } else {
                        Complex::default()
                    }
                })
            }),
        }
    }
    Ok(match lattice.dim {
        1 => unpack(sum_impl::<6, PART>(wave, k, lattice, r, eta, shell)?, 1),
        2 => unpack(sum_impl::<10, PART>(wave, k, lattice, r, eta, shell)?, 2),
        _ => unpack(sum_impl::<16, PART>(wave, k, lattice, r, eta, shell)?, 3),
    })
}

#[allow(clippy::cast_precision_loss)] // Shell indices are bounded by i32::MAX, exactly representable as f64.
fn sum_impl<const N: usize, const PART: usize>(
    wave: Wave,
    k: Complex,
    lattice: &Lattice,
    r: [f64; 3],
    eta: Complex,
    shell: i64,
) -> Result<Jet<N>> {
    wave.validate(lattice.dim)?;
    if PART == 3 && !(0..=i64::from(i32::MAX)).contains(&shell) {
        return Err(Error::InvalidInput(
            "direct shell index must be in [0, i32::MAX]".into(),
        ));
    }

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
    let variable = |value: Complex, index| {
        if PART == 4 {
            Jet::<N>::constant(value)
        } else {
            Jet::<N>::variable(value, index)
        }
    };
    let k = variable(k, 0);
    let mut r: [Jet<N>; 3] = std::array::from_fn(|i| variable(r[i].into(), 1 + i));
    let bloch: [Jet<N>; 3] = std::array::from_fn(|i| {
        if i < dim {
            variable(lattice.bloch[i].into(), 4 + i)
        } else {
            Jet::default()
        }
    });
    let direct: [[Jet<N>; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i < dim && j < dim {
                variable(lattice.direct[(i, j)].into(), 4 + dim + dim * i + j)
            } else {
                Jet::constant(lattice.direct[(i, j)])
            }
        })
    });
    let reciprocal: [[Jet<N>; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| Jet {
            value: Complex::new(lattice.reciprocal[(i, j)], 0.0),
            derivative: std::array::from_fn(|index| {
                if index < 4 + dim {
                    return Complex::default();
                }
                let (p, q) = ((index - 4 - dim) / dim, (index - 4 - dim) % dim);
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
            if index < 4 + dim {
                return Complex::default();
            }
            let (p, q) = ((index - 4 - dim) / dim, (index - 4 - dim) % dim);
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
    let vector = |matrix: &[[Jet<N>; 3]; 3], point: [i64; 3]| -> [Jet<N>; 3] {
        std::array::from_fn(|j| (0..dim).map(|i| point[i] as f64 * matrix[i][j]).sum())
    };
    if PART == 3 {
        let mut result = Jet::default();
        let half_cell = dim == 1
            && r.iter()
                .enumerate()
                .all(|(j, v)| j == axes[0] || v.value == Complex::default())
            && (r[axes[0]].value.re / lattice.direct[(0, 0)]).abs() == 0.5;
        if N > 0 && half_cell {
            return Err(Error::InvalidInput("direct half-cell shell grouping changes discontinuously; use the Ewald sum for adjoints".into()));
        }
        let mut term = |point| {
            let vector = vector(&direct, point);
            let mut shift = r.map(|v| -v);
            for j in 0..dim {
                shift[axes[j]] -= vector[j];
            }
            if shift.iter().all(|v| v.value == Complex::default()) {
                return Ok(());
            }
            let phase =
                (Complex::i() * (0..dim).map(|j| bloch[j] * vector[j]).sum::<Jet<N>>()).exp();
            result += phase * direct_term(wave, k, shift)?;
            Ok(())
        };
        if half_cell {
            let direction = if r[axes[0]].value.re / lattice.direct[(0, 0)] > 0.0 {
                1
            } else {
                -1
            };
            term([direction * shell, 0, 0])?;
            term([-direction * (shell + 1), 0, 0])?;
        } else {
            crate::geometry::visit_cube(dim, shell, true, term)?;
        }
        return Ok(result);
    }
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
    let real = if PART == 2 {
        Jet::default()
    } else {
        shell_sum(dim, |point| {
            let vector = vector(&direct, point);
            let mut shift = r.map(|x| -x);
            for j in 0..dim {
                shift[axes[j]] -= vector[j];
            }
            if shift.iter().all(|r| r.value == Complex::default()) {
                return Ok(Jet::default());
            }
            let phase = (0..dim).map(|j| bloch[j] * vector[j]).sum::<Jet<N>>();
            Ok((if PART == 4 {
                real_term::<N, true>(wave, k, shift, eta)
            } else {
                real_term::<N, false>(wave, k, shift, eta)
            }) * (Complex::i() * phase).exp())
        })?
    };
    let reciprocal = if PART == 1 || PART == 4 {
        Jet::default()
    } else {
        shell_sum(dim, |point| {
            let vector = vector(&reciprocal, point);
            let mut q = [Jet::default(); 3];
            for j in 0..dim {
                q[axes[j]] = vector[j] + bloch[j];
            }
            let phase =
                (-Complex::i() * q.into_iter().zip(r).map(|(q, r)| q * r).sum::<Jet<N>>()).exp();
            Ok(reciprocal_term(wave, dim, k, q, r, eta, measure)? * phase)
        })?
    };
    let self_term = if PART != 1 && PART != 4 && r.iter().all(|r| r.value == Complex::default()) {
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
    /// Complex split cotangent; zero for the full and direct sums.
    pub eta: Complex,
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
        self.eta += other.eta;
        for i in 0..3 {
            self.position[i] += other.position[i];
            self.bloch[i] += other.bloch[i];
            for j in 0..3 {
                self.vectors[i][j] += other.vectors[i][j];
            }
        }
    }
}

impl Derivatives {
    pub(crate) fn pullback(self, g: Complex) -> Gradient {
        Gradient {
            k: self.k.conj() * g,
            eta: self.eta.conj() * g,
            position: self.position.map(|x| (g.conj() * x).re),
            bloch: self.bloch.map(|x| (g.conj() * x).re),
            vectors: self.vectors.map(|row| row.map(|x| (g.conj() * x).re)),
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
            Ok(derivatives(wave, k, lattice, r, eta)?.pullback(g))
        })
        .try_reduce(Gradient::default, |mut a, b| {
            a.add(b);
            Ok(a)
        })
}

/// Owned scalar-or-element inputs for batched lattice-sum pullbacks.
#[derive(Debug)]
pub struct Residual {
    waves: Vec<Wave>,
    wavenumbers: Vec<Complex>,
    lattices: Vec<Lattice>,
    shifts: Vec<[f64; 3]>,
    etas: Vec<Complex>,
    parts: Vec<SumPart>,
    size: usize,
}
impl Residual {
    /// Record native inputs only; local derivatives are recomputed during reverse.
    pub fn new(
        waves: Vec<Wave>,
        wavenumbers: Vec<Complex>,
        lattices: Vec<Lattice>,
        shifts: Vec<[f64; 3]>,
        etas: Vec<Complex>,
        parts: Vec<SumPart>,
    ) -> Result<(Vec<Complex>, Self)> {
        use crate::integrals::{broadcast_size, element};
        use rayon::prelude::*;
        let size = broadcast_size(&[
            waves.len(),
            wavenumbers.len(),
            lattices.len(),
            shifts.len(),
            etas.len(),
            parts.len(),
        ])?;
        if parts
            .iter()
            .any(|p| matches!(p, SumPart::Real | SumPart::Reciprocal))
            && etas.contains(&Complex::default())
        {
            return Err(Error::InvalidInput(
                "component adjoints require an explicit nonzero Ewald split".into(),
            ));
        }
        let residual = Self {
            waves,
            wavenumbers,
            lattices,
            shifts,
            etas,
            parts,
            size,
        };
        let evaluate = |i| {
            sum_part(
                element(&residual.waves, i),
                element(&residual.wavenumbers, i),
                &residual.lattices[if residual.lattices.len() == 1 { 0 } else { i }],
                element(&residual.shifts, i),
                element(&residual.etas, i),
                element(&residual.parts, i),
            )
        };
        let values = if size >= 8 {
            (0..size)
                .into_par_iter()
                .map(evaluate)
                .collect::<Result<_>>()?
        } else {
            (0..size).map(evaluate).collect::<Result<_>>()?
        };
        Ok((values, residual))
    }
    /// One contracted gradient per output. No per-mode Jacobian or solve tape is retained.
    pub fn pullback(self, cotangent: &[Complex]) -> Result<Vec<Gradient>> {
        use crate::integrals::element;
        use rayon::prelude::*;
        if cotangent.len() != self.size || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "cotangent must be finite and match lattice output".into(),
            ));
        }
        let evaluate = |(i, &g): (usize, &Complex)| {
            if g == Complex::default() {
                return Ok(Gradient::default());
            }
            Ok(derivatives_part(
                element(&self.waves, i),
                element(&self.wavenumbers, i),
                &self.lattices[if self.lattices.len() == 1 { 0 } else { i }],
                element(&self.shifts, i),
                element(&self.etas, i),
                element(&self.parts, i),
            )?
            .pullback(g))
        };
        if self.size >= 8 {
            cotangent.par_iter().enumerate().map(evaluate).collect()
        } else {
            cotangent.iter().enumerate().map(evaluate).collect()
        }
    }
}

#[cfg(test)]
mod decomposition_tests {
    #![allow(clippy::unwrap_used)] // Test failures should preserve proptest shrinking.
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(12))]
        #[test]
        fn decomposition_derivatives_and_scale(spherical in any::<bool>(), dim in 1_usize..3, order in 0_i32..4, pitch in 1.4_f64..1.8, eta in 0.8_f64..1.1) {
            let vectors=std::array::from_fn(|i|std::array::from_fn(|j|if i==j {pitch}else{0.0}));
            let bloch=[0.13;3];
            let lattice=Lattice::from_array(vectors,bloch,dim).unwrap();
            let wave=if spherical {Wave::Spherical{l:order,m:order}}else{Wave::Cylindrical{m:order}};
            let k=Complex::new(2.0,0.4);
            let r=[0.17,0.11,if spherical {0.09}else{0.0}];
            let full=derivatives(wave,k,&lattice,r,eta.into()).unwrap();
            let real=derivatives_part(wave,k,&lattice,r,eta.into(),SumPart::Real).unwrap();
            let reciprocal=derivatives_part(wave,k,&lattice,r,eta.into(),SumPart::Reciprocal).unwrap();
            for (a,(b,c)) in [full.value,full.k].into_iter().chain(full.position).chain(full.bloch).chain(full.vectors.into_iter().flatten()).zip(
                [real.value,real.k].into_iter().chain(real.position).chain(real.bloch).chain(real.vectors.into_iter().flatten()).zip(
                [reciprocal.value,reciprocal.k].into_iter().chain(reciprocal.position).chain(reciprocal.bloch).chain(reciprocal.vectors.into_iter().flatten()))) {
                prop_assert!((a-b-c).norm()<1e-10*(1.0+a.norm()));
            }
            for part in [SumPart::Real,SumPart::Reciprocal,SumPart::Direct(2)] {
                let jet=derivatives_part(wave,k,&lattice,r,eta.into(),part).unwrap();
                let spatial:Complex=jet.position.into_iter().zip(r).map(|(g,r)|g*r).sum::<Complex>()+jet.vectors.into_iter().flatten().zip(vectors.into_iter().flatten()).map(|(g,a)|g*a).sum::<Complex>();
                let spectral=jet.k*k+jet.bloch.into_iter().zip(bloch).map(|(g,q)|g*q).sum::<Complex>();
                prop_assert!((spatial-spectral).norm()<2e-9*(1.0+jet.value.norm()));
            }
        }
        #[test]
        fn direct_shells_converge_to_ewald(spherical in any::<bool>(), dim in 1_usize..4, order in 0_i32..3) {
            let dim=if spherical {dim}else{dim.min(2)};
            let lattice=Lattice::from_array([[1.7,0.0,0.0],[0.0,1.7,0.0],[0.0,0.0,1.7]],[0.13;3],dim).unwrap();
            let wave=if spherical {Wave::Spherical{l:order,m:-order}}else{Wave::Cylindrical{m:-order}};
            let r=[0.17,0.11,if spherical {0.09}else{0.0}];
            let k=Complex::new(2.0,1.2);
            let total:Complex=(0..15).map(|shell|sum_part(wave,k,&lattice,r,0.0.into(),SumPart::Direct(shell)).unwrap()).sum();
            let ewald=sum(wave,k,&lattice,r,0.0.into()).unwrap();
            prop_assert!((total-ewald).norm()<2e-10*(1.0+ewald.norm()));
        }
    }
}
