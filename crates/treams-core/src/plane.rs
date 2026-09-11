//! Plane-wave illumination in spherical and cylindrical bases.
#![allow(clippy::indexing_slicing)] // Validated three-component vectors and polarizations.

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::{
    Complex, Error, Result, finite,
    jet::Jet,
    special::{pi_fun, tau_fun},
    waves::Mode,
};

// Scale before squaring or dividing: nearly axial directions may have transverse
// components small enough that their squares underflow while their azimuth matters.
fn algebraic_norm(values: &[Complex]) -> Complex {
    let scale = values.iter().map(|v| v.norm()).fold(0.0, f64::max);
    if scale == 0.0 {
        return Complex::default();
    }
    values
        .iter()
        .map(|v| (v / scale).powu(2))
        .sum::<Complex>()
        .sqrt()
        * scale
}
fn ratio(numerator: Complex, denominator: Complex) -> Complex {
    let scale = denominator.re.abs().max(denominator.im.abs());
    (numerator / scale) / (denominator / scale)
}

fn wavenumbers(vector: [Complex; 3]) -> Result<(Complex, Complex, [Complex; 2])> {
    if vector.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput("wavevector must be finite".into()));
    }
    let scale = vector[..2].iter().map(|v| v.norm()).fold(0.0, f64::max);
    let (transverse, xy) = if scale == 0.0 {
        (Complex::default(), [Complex::default(); 2])
    } else {
        let scaled = [vector[0] / scale, vector[1] / scale];
        let norm = algebraic_norm(&scaled);
        if norm == Complex::default() {
            return Err(Error::InvalidInput(
                "undefined polarization for a null transverse vector".into(),
            ));
        }
        (norm * scale, scaled.map(|v| ratio(v, norm)))
    };
    let k = algebraic_norm(&vector);
    if k == Complex::default() || !finite(k) {
        return Err(Error::InvalidInput(
            "wavevector must have nonzero algebraic norm".into(),
        ));
    }
    Ok((k, transverse, xy))
}

/// Plane-wave electric vector at the origin in treams normalization.
fn polarization_jet<const N: usize>(
    vector: [Complex; 3],
    pol: u8,
    helicity: bool,
) -> Result<[Jet<N>; 3]> {
    if pol > 1 {
        return Err(Error::InvalidInput("polarization must be 0 or 1".into()));
    }
    let (k, transverse, xy) = wavenumbers(vector)?;
    if N != 0 && transverse == Complex::default() {
        return Err(Error::InvalidInput("plane-wave direction derivative is undefined on the polarization axis; fix the wavevectors".into()));
    }
    let wave = Jet {
        value: k,
        derivative: std::array::from_fn(|a| ratio(vector[a], k)),
    };
    let radial = Jet {
        value: transverse,
        derivative: std::array::from_fn(|a| if a < 2 { xy[a] } else { Complex::default() }),
    };
    let xy_jet: [Jet<N>; 2] = std::array::from_fn(|a| Jet {
        value: xy[a],
        derivative: std::array::from_fn(|b| {
            if b < 2 {
                ratio(
                    Complex::new(if a == b { 1.0 } else { 0.0 }, 0.0) - xy[a] * xy[b],
                    transverse,
                )
            } else {
                Complex::default()
            }
        }),
    });
    let z = vector[2];
    let (m, n) = if xy == [Complex::default(); 2] {
        let sign = if z.im == 0.0 {
            if z.re >= 0.0 { 1.0 } else { -1.0 }
        } else if z.im >= 0.0 {
            1.0
        } else {
            -1.0
        };
        (
            [Jet::default(), Jet::constant(-Complex::i()), Jet::default()],
            [Jet::constant(-sign), Jet::default(), Jet::default()],
        )
    } else {
        (
            [
                Complex::i() * xy_jet[1],
                -Complex::i() * xy_jet[0],
                Jet::default(),
            ],
            [
                -xy_jet[0] * Jet::variable(z, 2) / wave,
                -xy_jet[1] * Jet::variable(z, 2) / wave,
                radial / wave,
            ],
        )
    };
    Ok(if helicity {
        std::array::from_fn(|a| {
            (n[a] + (2.0 * f64::from(pol) - 1.0) * m[a]) * std::f64::consts::FRAC_1_SQRT_2
        })
    } else if pol == 0 {
        m
    } else {
        n
    })
}

/// Plane-wave electric vector at the origin in treams normalization.
pub fn polarization(vector: [Complex; 3], pol: u8, helicity: bool) -> Result<[Complex; 3]> {
    Ok(polarization_jet::<0>(vector, pol, helicity)?.map(|p| p.value))
}

/// Spherical expansion coefficient for a unit-amplitude plane wave.
pub fn to_spherical(mode: Mode, vector: [Complex; 3], pol: u8, helicity: bool) -> Result<Complex> {
    mode.validate()?;
    if pol > 1 {
        return Err(Error::InvalidInput("polarization must be 0 or 1".into()));
    }
    let (k, _, xy) = wavenumbers(vector)?;
    if helicity && mode.pol != pol {
        return Ok(Complex::default());
    }
    let (l, m) = (mode.l, mode.m);
    let azimuth = if xy == [Complex::default(); 2] {
        Complex::new(1.0, 0.0)
    } else {
        (xy[0] - Complex::i() * xy[1]).powi(m)
    };
    let normalization = 2.0
        * (std::f64::consts::PI * f64::from(2 * l + 1) / f64::from(l * (l + 1))).sqrt()
        * (0.5 * (libm::lgamma(f64::from(l - m + 1)) - libm::lgamma(f64::from(l + m + 1)))).exp();
    let z = if xy == [Complex::default(); 2] {
        Complex::new(if (vector[2] / k).re >= 0.0 { 1.0 } else { -1.0 }, 0.0)
    } else {
        vector[2] / k
    };
    let angular = if helicity {
        tau_fun(l, m, z) + (2.0 * f64::from(pol) - 1.0) * pi_fun(l, m, z)
    } else if mode.pol == pol {
        tau_fun(l, m, z)
    } else {
        pi_fun(l, m, z)
    };
    Ok(normalization * Complex::i().powi(l) * azimuth * angular)
}

/// Regular spherical multipole amplitudes of one plane wave.
pub fn spherical(
    basis: &crate::basis::Basis,
    vector: [Complex; 3],
    pol: u8,
    helicity: bool,
) -> Result<Vec<Complex>> {
    basis.validate()?;
    basis
        .modes
        .iter()
        .map(|&(origin, mode)| {
            let phase = (Complex::i()
                * vector
                    .iter()
                    .zip(basis.positions[origin])
                    .map(|(k, r)| k * r)
                    .sum::<Complex>())
            .exp();
            Ok(phase * to_spherical(mode, vector, pol, helicity)?)
        })
        .collect()
}

/// Regular cylindrical multipole amplitudes of one plane wave.
#[allow(clippy::float_cmp)] // kz is an exact basis mode label, not a tolerance match.
pub fn cylindrical(
    basis: &crate::cylwaves::Basis,
    vector: [Complex; 3],
    pol: u8,
) -> Result<Vec<Complex>> {
    basis.validate()?;
    let (_, _, xy) = wavenumbers(vector)?;
    if pol > 1 || vector[2].im != 0.0 {
        return Err(Error::InvalidInput(
            "cylindrical expansion requires real axial wavenumber and polarization 0 or 1".into(),
        ));
    }
    basis
        .modes
        .iter()
        .map(|&(origin, mode)| {
            if mode.pol != pol || mode.kz != vector[2].re {
                return Ok(Complex::default());
            }
            let angular = if xy == [Complex::default(); 2] {
                Complex::i().powi(mode.m)
            } else {
                (Complex::i() * xy[0] + xy[1]).powi(mode.m)
            };
            let phase = (Complex::i()
                * vector
                    .iter()
                    .zip(basis.positions[origin])
                    .map(|(k, r)| k * r)
                    .sum::<Complex>())
            .exp();
            Ok(phase * angular)
        })
        .collect()
}

/// Geometry retained for weighted plane fields or their full sampling operator.
#[derive(Debug)]
pub struct FieldResidual {
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    points: Vec<[f64; 3]>,
    coefficients: Option<Vec<Complex>>,
    helicity: bool,
}
/// Cotangents in the real Hermitian pairing.
#[derive(Debug)]
pub struct FieldGradient {
    /// Amplitude cotangents; empty for a field operator.
    pub coefficients: Vec<Complex>,
    /// Real sample-point cotangents.
    pub points: Vec<[f64; 3]>,
    /// Complex full-wavevector cotangents; zero when wavevectors are held fixed.
    pub vectors: Vec<[Complex; 3]>,
}
fn phase(vector: [Complex; 3], point: [f64; 3]) -> Complex {
    (Complex::i()
        * vector
            .iter()
            .zip(point)
            .map(|(k, r)| k * r)
            .sum::<Complex>())
    .exp()
}
/// Evaluate weighted plane fields, or the full operator when coefficients are absent.
pub fn field(
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    points: Vec<[f64; 3]>,
    coefficients: Option<Vec<Complex>>,
    helicity: bool,
) -> Result<(DMatrix<Complex>, FieldResidual)> {
    if vectors.is_empty()
        || vectors.len() != polarizations.len()
        || points.iter().flatten().any(|r| !r.is_finite())
        || coefficients
            .as_ref()
            .is_some_and(|c| c.len() != vectors.len() || c.iter().any(|&v| !finite(v)))
    {
        return Err(Error::InvalidInput(
            "require nonempty plane modes, finite points, matching finite amplitudes and polarizations"
                .into(),
        ));
    }
    let electric: Vec<_> = vectors
        .iter()
        .zip(&polarizations)
        .map(|(&k, &p)| polarization(k, p, helicity))
        .collect::<Result<_>>()?;
    let mut value = DMatrix::zeros(
        3 * points.len(),
        if coefficients.is_some() {
            1
        } else {
            vectors.len()
        },
    );
    if let Some(c) = &coefficients {
        value
            .as_mut_slice()
            .par_chunks_mut(3)
            .zip(&points)
            .for_each(|(out, &point)| {
                for ((&k, e), &c) in vectors.iter().zip(&electric).zip(c) {
                    let weight = c * phase(k, point);
                    for (out, &e) in out.iter_mut().zip(e) {
                        *out += weight * e;
                    }
                }
            });
    } else if !points.is_empty() {
        value
            .as_mut_slice()
            .par_chunks_mut(3 * points.len())
            .enumerate()
            .for_each(|(j, column)| {
                for (out, &point) in column.chunks_exact_mut(3).zip(&points) {
                    let phase = phase(vectors[j], point);
                    for (out, &e) in out.iter_mut().zip(&electric[j]) {
                        *out = phase * e;
                    }
                }
            });
    }
    Ok((
        value,
        FieldResidual {
            vectors,
            polarizations,
            points,
            coefficients,
            helicity,
        },
    ))
}
impl FieldResidual {
    /// Flattened output dimensions (3 * samples, modes or one weighted column).
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (
            3 * self.points.len(),
            if self.coefficients.is_some() {
                1
            } else {
                self.vectors.len()
            },
        )
    }
    /// Recompute and contract polarization/phase derivatives without a dense field Jacobian.
    pub fn pullback(self, g: &DMatrix<Complex>, fixed_vectors: bool) -> Result<FieldGradient> {
        if g.shape() != self.shape() || g.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput("invalid plane-field cotangent".into()));
        }
        let zero = || FieldGradient {
            coefficients: vec![Complex::default(); self.coefficients.as_ref().map_or(0, Vec::len)],
            points: vec![[0.0; 3]; self.points.len()],
            vectors: vec![[Complex::default(); 3]; self.vectors.len()],
        };
        if self.points.is_empty() {
            return Ok(zero());
        }
        self.vectors
            .par_iter()
            .enumerate()
            .try_fold(zero, |mut result, (j, &vector)| -> Result<_> {
                let electric = if fixed_vectors {
                    polarization(vector, self.polarizations[j], self.helicity)?
                        .map(Jet::<3>::constant)
                } else {
                    polarization_jet::<3>(vector, self.polarizations[j], self.helicity)?
                };
                let coefficient = self
                    .coefficients
                    .as_ref()
                    .map_or(Complex::new(1.0, 0.0), |c| c[j]);
                let column = if self.coefficients.is_some() { 0 } else { j };
                let mut polarization_cotangent = [Complex::default(); 3];
                for (p, &point) in self.points.iter().enumerate() {
                    let phase = phase(vector, point);
                    let weighted_phase = coefficient * phase;
                    let paired: Complex = (0..3)
                        .map(|a| g[(3 * p + a, column)].conj() * electric[a].value)
                        .sum();
                    if self.coefficients.is_some() {
                        result.coefficients[j] += (phase * paired).conj();
                    }
                    let paired_field = weighted_phase * paired * Complex::i();
                    for axis in 0..3 {
                        result.points[p][axis] += (paired_field * vector[axis]).re;
                        if !fixed_vectors {
                            result.vectors[j][axis] += (paired_field * point[axis]).conj();
                            polarization_cotangent[axis] +=
                                g[(3 * p + axis, column)] * weighted_phase.conj();
                        }
                    }
                }
                // Polarization is constant across samples: contract its adjoint once per mode.
                if !fixed_vectors {
                    for axis in 0..3 {
                        result.vectors[j][axis] += (0..3)
                            .map(|a| {
                                polarization_cotangent[a] * electric[a].derivative[axis].conj()
                            })
                            .sum::<Complex>();
                    }
                }
                Ok(result)
            })
            .try_reduce(zero, |mut a, b| {
                for (a, b) in a
                    .coefficients
                    .iter_mut()
                    .chain(a.vectors.iter_mut().flatten())
                    .zip(b.coefficients.iter().chain(b.vectors.iter().flatten()))
                {
                    *a += b;
                }
                for (a, b) in a.points.iter_mut().flatten().zip(b.points.iter().flatten()) {
                    *a += b;
                }
                Ok(a)
            })
    }
}
