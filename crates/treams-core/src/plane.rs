//! Plane-wave illumination in spherical and cylindrical bases.
#![allow(clippy::indexing_slicing)] // Validated three-component vectors and polarizations.

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::{
    Complex, Error, Result, fields::FieldBasis, finite, jet::Jet, ratio, special::angular_jets,
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

struct Direction<const N: usize> {
    k: Jet<N>,
    transverse: Jet<N>,
    xy: [Jet<N>; 2],
}
impl<const N: usize> Direction<N> {
    fn new(vector: [Complex; 3]) -> Result<Self> {
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
        let xy: [Jet<N>; 2] = std::array::from_fn(|a| Jet {
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
        Ok(Self {
            k: wave,
            transverse: radial,
            xy,
        })
    }
}

fn polarization_jet<const N: usize>(
    vector: [Complex; 3],
    pol: u8,
    helicity: bool,
) -> Result<[Jet<N>; 3]> {
    if pol > 1 {
        return Err(Error::InvalidInput("polarization must be 0 or 1".into()));
    }
    let Direction {
        k: wave,
        transverse: radial,
        xy: xy_jet,
    } = Direction::new(vector)?;
    let z = vector[2];
    let (m, n) = if radial.value == Complex::default() {
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

/// Chain the local Cartesian polarization derivative into solver parameters.
pub(crate) fn polarization_from_inputs<const N: usize>(
    vector: [Jet<N>; 3],
    pol: u8,
) -> Result<[Jet<N>; 3]> {
    let values = vector.map(|k| k.value);
    if N == 0
        || (values[0] == Complex::default()
            && values[1] == Complex::default()
            && vector[..2]
                .iter()
                .flat_map(|k| k.derivative)
                .all(|d| d == Complex::default()))
    {
        return Ok(polarization(values, pol, true)?.map(Jet::constant));
    }
    Ok(polarization_jet::<3>(values, pol, true)?.map(|e| Jet {
        value: e.value,
        derivative: std::array::from_fn(|a| {
            (0..3)
                .map(|b| e.derivative[b] * vector[b].derivative[a])
                .sum()
        }),
    }))
}

fn spherical_coefficient<const N: usize>(
    mode: Mode,
    vector: [Complex; 3],
    direction: &Direction<N>,
    pol: u8,
    helicity: bool,
) -> Jet<N> {
    if helicity && mode.pol != pol {
        return Jet::default();
    }
    let (l, m) = (mode.l, mode.m);
    let axis = direction.transverse.value == Complex::default();
    let azimuth = if axis {
        Jet::constant(1.0)
    } else {
        (direction.xy[0] - Complex::i() * direction.xy[1]).powi(m)
    };
    let cosine = if axis {
        Jet::constant(if (vector[2] / direction.k.value).re >= 0.0 {
            1.0
        } else {
            -1.0
        })
    } else {
        Jet::variable(vector[2], 2) / direction.k
    };
    // Keep the same transverse branch as the Cartesian polarization. Taking a
    // second principal square root of 1-cos(theta)^2 can flip complex directions.
    let sine = direction.transverse / direction.k;
    let [pi, tau] = angular_jets(l, m, cosine, sine);
    let angular = if helicity {
        tau + (2.0 * f64::from(pol) - 1.0) * pi
    } else if mode.pol == pol {
        tau
    } else {
        pi
    };
    let normalization = 2.0
        * (std::f64::consts::PI * f64::from(2 * l + 1) / f64::from(l * (l + 1))).sqrt()
        * (0.5 * (libm::lgamma(f64::from(l - m + 1)) - libm::lgamma(f64::from(l + m + 1)))).exp();
    normalization * Complex::i().powi(l) * azimuth * angular
}

/// Spherical expansion coefficient for a unit-amplitude plane wave.
pub fn to_spherical(mode: Mode, vector: [Complex; 3], pol: u8, helicity: bool) -> Result<Complex> {
    mode.validate()?;
    if pol > 1 {
        return Err(Error::InvalidInput("polarization must be 0 or 1".into()));
    }
    Ok(spherical_coefficient(mode, vector, &Direction::<0>::new(vector)?, pol, helicity).value)
}

/// Regular spherical multipole amplitudes of one plane wave.
pub fn spherical(
    basis: &crate::basis::Basis,
    vector: [Complex; 3],
    pol: u8,
    helicity: bool,
) -> Result<Vec<Complex>> {
    basis.validate()?;
    if pol > 1 {
        return Err(Error::InvalidInput("polarization must be 0 or 1".into()));
    }
    let direction = Direction::<0>::new(vector)?;
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
            Ok(phase * spherical_coefficient(mode, vector, &direction, pol, helicity).value)
        })
        .collect()
}

/// Match a continuous plane vector to a static axial label after normalization.
/// Keep the tolerance relative so a change of length units cannot merge orders.
pub(crate) fn cylindrical_mode_matches(mode: crate::cylwaves::Mode, kz: Complex, pol: u8) -> bool {
    mode.pol == pol
        && kz.im == 0.0
        && (mode.kz - kz.re).abs() <= 16.0 * f64::EPSILON * mode.kz.abs().max(kz.re.abs())
}

fn cylindrical_coefficient<const N: usize>(
    mode: crate::cylwaves::Mode,
    vector: [Complex; 3],
    direction: &Direction<N>,
    pol: u8,
) -> Jet<N> {
    if !cylindrical_mode_matches(mode, vector[2], pol) {
        Jet::default()
    } else if direction.transverse.value == Complex::default() {
        Jet::constant(Complex::i().powi(mode.m))
    } else {
        (Complex::i() * direction.xy[0] + direction.xy[1]).powi(mode.m)
    }
}

/// Regular cylindrical multipole amplitudes of one plane wave.
pub fn cylindrical(
    basis: &crate::cylwaves::Basis,
    vector: [Complex; 3],
    pol: u8,
) -> Result<Vec<Complex>> {
    basis.validate()?;
    let direction = Direction::<0>::new(vector)?;
    if pol > 1 || vector[2].im != 0.0 {
        return Err(Error::InvalidInput(
            "cylindrical expansion requires real axial wavenumber and polarization 0 or 1".into(),
        ));
    }
    Ok(basis
        .modes
        .iter()
        .map(|&(origin, mode)| {
            phase(vector, basis.positions[origin])
                * cylindrical_coefficient(mode, vector, &direction, pol).value
        })
        .collect())
}

impl FieldBasis {
    fn plane_coefficient<const N: usize>(
        &self,
        i: usize,
        vector: [Complex; 3],
        direction: &Direction<N>,
        pol: u8,
        helicity: bool,
    ) -> Jet<N> {
        match self {
            Self::Spherical(b) => {
                spherical_coefficient(b.modes[i].1, vector, direction, pol, helicity)
            }
            Self::Cylindrical(b) => cylindrical_coefficient(b.modes[i].1, vector, direction, pol),
        }
    }
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

/// Inputs retained for exp(i k.r); no sample-by-mode values or Jacobian are kept.
#[derive(Debug)]
pub struct PhaseResidual {
    points: Vec<[f64; 3]>,
    vectors: Vec<[Complex; 3]>,
}
/// Cotangents of real displacements and complex full wavevectors.
#[derive(Debug)]
pub struct PhaseGradient {
    /// Real displacement cotangents.
    pub points: Vec<[f64; 3]>,
    /// Complex wavevector cotangents in the real Hermitian pairing.
    pub vectors: Vec<[Complex; 3]>,
}

/// Plane-wave translation phases with shape (displacements, wavevectors).
pub fn phases(
    points: Vec<[f64; 3]>,
    vectors: Vec<[Complex; 3]>,
) -> Result<(DMatrix<Complex>, PhaseResidual)> {
    if vectors.is_empty()
        || vectors.iter().flatten().any(|&k| !finite(k))
        || points.iter().flatten().any(|r| !r.is_finite())
    {
        return Err(Error::InvalidInput(
            "require finite displacements and nonempty finite wavevectors".into(),
        ));
    }
    let mut value = DMatrix::zeros(points.len(), vectors.len());
    let fill = |(j, column): (usize, &mut [Complex])| {
        for (out, &point) in column.iter_mut().zip(&points) {
            *out = phase(vectors[j], point);
        }
    };
    if value.len() >= 4096 {
        value
            .as_mut_slice()
            .par_chunks_mut(points.len().max(1))
            .enumerate()
            .for_each(fill);
    } else {
        value
            .as_mut_slice()
            .chunks_mut(points.len().max(1))
            .enumerate()
            .for_each(fill);
    }
    if value.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput(
            "plane translation phase overflow".into(),
        ));
    }
    Ok((value, PhaseResidual { points, vectors }))
}
impl PhaseResidual {
    /// Number of displacements and wavevectors.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.points.len(), self.vectors.len())
    }
    /// Recompute local phases in two reductions, avoiding per-thread gradient arrays.
    pub fn pullback<S: nalgebra::Storage<Complex, nalgebra::Dyn, nalgebra::Dyn> + Sync>(
        self,
        g: &nalgebra::Matrix<Complex, nalgebra::Dyn, nalgebra::Dyn, S>,
    ) -> Result<PhaseGradient> {
        if g.shape() != self.shape() || g.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput("invalid plane-phase cotangent".into()));
        }
        let point_gradient = |i: usize| {
            let mut result = [0.0; 3];
            for (j, &vector) in self.vectors.iter().enumerate() {
                let factor = g[(i, j)].conj() * Complex::i() * phase(vector, self.points[i]);
                for axis in 0..3 {
                    result[axis] += (factor * vector[axis]).re;
                }
            }
            result
        };
        let vector_gradient = |j: usize| {
            let mut result = [Complex::default(); 3];
            for (i, &point) in self.points.iter().enumerate() {
                let factor = g[(i, j)] * (Complex::i() * phase(self.vectors[j], point)).conj();
                for axis in 0..3 {
                    result[axis] += factor * point[axis];
                }
            }
            result
        };
        let (points, vectors) = if g.len() >= 4096 {
            (
                (0..self.points.len())
                    .into_par_iter()
                    .map(point_gradient)
                    .collect(),
                (0..self.vectors.len())
                    .into_par_iter()
                    .map(vector_gradient)
                    .collect(),
            )
        } else {
            (
                (0..self.points.len()).map(point_gradient).collect(),
                (0..self.vectors.len()).map(vector_gradient).collect(),
            )
        };
        Ok(PhaseGradient { points, vectors })
    }
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

/// Inputs retained for plane-to-multipole conversion; no output Jacobian is stored.
#[derive(Debug)]
pub struct ExpansionResidual {
    basis: FieldBasis,
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    helicity: bool,
}
/// Plane-expansion cotangents.
#[derive(Debug)]
pub struct ExpansionGradient {
    /// Multipole origin cotangents.
    pub origins: Vec<[f64; 3]>,
    /// Full complex plane-wavevector cotangents.
    pub vectors: Vec<[Complex; 3]>,
}
/// Expand multiple plane waves in a regular multipole basis.
pub fn expansion(
    basis: impl Into<FieldBasis>,
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    helicity: bool,
) -> Result<(DMatrix<Complex>, ExpansionResidual)> {
    let basis = basis.into();
    basis.validate()?;
    if vectors.is_empty()
        || vectors.len() != polarizations.len()
        || polarizations.iter().any(|&p| p > 1)
        || (matches!(basis, FieldBasis::Cylindrical(_)) && vectors.iter().any(|k| k[2].im != 0.0))
    {
        return Err(Error::InvalidInput(
            "require nonempty plane vectors and matching polarizations 0/1; cylindrical axial labels must be real".into(),
        ));
    }
    let mut value = DMatrix::zeros(basis.len(), vectors.len());
    value
        .as_mut_slice()
        .par_chunks_mut(basis.len())
        .enumerate()
        .try_for_each(|(j, column)| -> Result<()> {
            let direction = Direction::<0>::new(vectors[j])?;
            let phases: Vec<_> = basis
                .origins()
                .iter()
                .map(|&p| phase(vectors[j], p))
                .collect();
            for (i, out) in column.iter_mut().enumerate() {
                let p = basis.origin_pol(i).0;
                *out = basis
                    .plane_coefficient(i, vectors[j], &direction, polarizations[j], helicity)
                    .value
                    * phases[p];
            }
            Ok(())
        })?;
    Ok((
        value,
        ExpansionResidual {
            basis,
            vectors,
            polarizations,
            helicity,
        },
    ))
}
impl ExpansionResidual {
    /// Multipole output and plane input mode counts.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.basis.len(), self.vectors.len())
    }
    /// Differentiate origin phases and the full direction-dependent angular coefficient.
    pub fn pullback(self, g: &DMatrix<Complex>, fixed_vectors: bool) -> Result<ExpansionGradient> {
        if g.shape() != self.shape() || g.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput(
                "invalid plane-expansion cotangent".into(),
            ));
        }
        let zero = || ExpansionGradient {
            origins: vec![[0.0; 3]; self.basis.origins().len()],
            vectors: vec![[Complex::default(); 3]; self.vectors.len()],
        };
        self.vectors
            .par_iter()
            .enumerate()
            .try_fold(zero, |mut result, (j, &vector)| -> Result<_> {
                let direction = if fixed_vectors {
                    let d = Direction::<0>::new(vector)?;
                    Direction {
                        k: Jet::constant(d.k.value),
                        transverse: Jet::constant(d.transverse.value),
                        xy: d.xy.map(|v| Jet::constant(v.value)),
                    }
                } else {
                    Direction::<3>::new(vector)?
                };
                let phases: Vec<_> = self
                    .basis
                    .origins()
                    .iter()
                    .map(|&p| phase(vector, p))
                    .collect();
                for i in 0..self.basis.len() {
                    let p = self.basis.origin_pol(i).0;
                    let position = self.basis.origins()[p];
                    let angular = self.basis.plane_coefficient(
                        i,
                        vector,
                        &direction,
                        self.polarizations[j],
                        self.helicity,
                    );
                    let phase = phases[p];
                    let value = phase * angular.value;
                    for axis in 0..3 {
                        result.origins[p][axis] +=
                            (g[(i, j)].conj() * value * Complex::i() * vector[axis]).re;
                        if !fixed_vectors
                            && (axis < 2 || matches!(self.basis, FieldBasis::Spherical(_)))
                        {
                            result.vectors[j][axis] += g[(i, j)]
                                * (phase
                                    * (angular.derivative[axis]
                                        + Complex::i() * position[axis] * angular.value))
                                    .conj();
                        }
                    }
                }
                Ok(result)
            })
            .try_reduce(zero, |mut a, b| {
                for (a, b) in a
                    .origins
                    .iter_mut()
                    .flatten()
                    .zip(b.origins.iter().flatten())
                {
                    *a += b;
                }
                for (a, b) in a
                    .vectors
                    .iter_mut()
                    .flatten()
                    .zip(b.vectors.iter().flatten())
                {
                    *a += b;
                }
                Ok(a)
            })
    }
}
