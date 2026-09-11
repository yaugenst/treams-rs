//! Plane-wave illumination in spherical and cylindrical bases.
#![allow(clippy::indexing_slicing)] // Validated three-component vectors and polarizations.

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::{
    Complex, Error, Result, complex_sqrt, fields::FieldBasis, finite, jet::Jet, ratio,
    special::angular_jets, waves::Mode,
};

// Scale before squaring or dividing: nearly axial directions may have transverse
// components small enough that their squares underflow while their azimuth matters.
#[inline]
fn algebraic_norm(values: &[Complex]) -> Complex {
    let scale = values
        .iter()
        .map(|v| v.re.abs().max(v.im.abs()))
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return Complex::default();
    }
    // Ordinary magnitudes need no scaling. Keep the scaled formulation when
    // squaring could underflow or overflow (including near-axis regression cases).
    if (1e-150..=1e150).contains(&scale) {
        return complex_sqrt(values.iter().map(|v| v * v).sum::<Complex>());
    }
    complex_sqrt(values.iter().map(|v| (v / scale).powu(2)).sum::<Complex>()) * scale
}
#[inline]
fn transverse_values(vector: [Complex; 2]) -> Result<(Complex, [Complex; 2])> {
    let scale = vector
        .iter()
        .map(|v| v.re.abs().max(v.im.abs()))
        .fold(0.0, f64::max);
    Ok(if scale == 0.0 {
        (Complex::default(), [Complex::default(); 2])
    } else {
        let (scaled, factor) = if (1e-150..=1e150).contains(&scale) {
            ([vector[0], vector[1]], 1.0)
        } else {
            ([vector[0] / scale, vector[1] / scale], scale)
        };
        let norm = complex_sqrt(scaled[0] * scaled[0] + scaled[1] * scaled[1]);
        if norm == Complex::default() {
            return Err(Error::InvalidInput(
                "undefined polarization for a null transverse vector".into(),
            ));
        }
        (norm * factor, scaled.map(|v| ratio(v, norm)))
    })
}
fn wavenumbers(vector: [Complex; 3]) -> Result<(Complex, Complex, [Complex; 2])> {
    if vector.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput("wavevector must be finite".into()));
    }
    let (transverse, xy) = transverse_values([vector[0], vector[1]])?;
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
        Self::from_parts(vector, k, transverse, xy)
    }
    #[inline]
    fn from_parts(
        vector: [Complex; 3],
        k: Complex,
        transverse: Complex,
        xy: [Complex; 2],
    ) -> Result<Self> {
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

#[inline]
pub(crate) fn polarization_jet<const N: usize>(
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
    let m = if radial.value == Complex::default() {
        [Jet::default(), Jet::constant(-Complex::i()), Jet::default()]
    } else {
        [
            Complex::i() * xy_jet[1],
            -Complex::i() * xy_jet[0],
            Jet::default(),
        ]
    };
    if !helicity && pol == 0 {
        return Ok(m);
    }
    let n = if radial.value == Complex::default() {
        let sign = if z.im == 0.0 {
            if z.re >= 0.0 { 1.0 } else { -1.0 }
        } else if z.im >= 0.0 {
            1.0
        } else {
            -1.0
        };
        [Jet::constant(-sign), Jet::default(), Jet::default()]
    } else {
        let longitudinal = Jet::variable(z, 2) / wave;
        [
            -xy_jet[0] * longitudinal,
            -xy_jet[1] * longitudinal,
            radial / wave,
        ]
    };
    Ok(if helicity {
        std::array::from_fn(|a| {
            (n[a] + (2.0 * f64::from(pol) - 1.0) * m[a]) * std::f64::consts::FRAC_1_SQRT_2
        })
    } else {
        n
    })
}

/// Plane-wave electric vector at the origin in treams normalization.
#[inline]
pub fn polarization(vector: [Complex; 3], pol: u8, helicity: bool) -> Result<[Complex; 3]> {
    Ok(polarization_jet::<0>(vector, pol, helicity)?.map(|p| p.value))
}

/// Evaluate a plane-wave field from its already normalized polarization.
#[inline]
pub fn field_value(
    polarization: [Complex; 3],
    k: [Complex; 3],
    position: [Complex; 3],
) -> Result<[Complex; 3]> {
    if position.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput("field position must be finite".into()));
    }
    let phase = (Complex::i() * (0..3).map(|a| k[a] * position[a]).sum::<Complex>()).exp();
    let values = polarization.map(|p| p * phase);
    if values.iter().any(|&v| !finite(v)) {
        return Err(Error::SpecialFunction("nonfinite plane-wave field".into()));
    }
    Ok(values)
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

/// One plane-to-cylindrical coefficient with exact axial-label matching.
#[allow(clippy::float_cmp)] // A direct coefficient preserves exact discrete labels.
pub fn to_cylindrical(
    mode: crate::cylwaves::Mode,
    vector: [Complex; 3],
    pol: u8,
) -> Result<Complex> {
    mode.validate()?;
    if pol > 1 || vector.iter().any(|&v| !finite(v)) || vector[2].im != 0.0 {
        return Err(Error::InvalidInput(
            "finite wavevector, real axial component and polarization 0/1 required".into(),
        ));
    }
    if mode.pol != pol || mode.kz != vector[2].re {
        return Ok(Complex::default());
    }
    Ok(cylindrical_coefficient(mode, vector, &Direction::<0>::new(vector)?, pol).value)
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
#[inline]
fn phase(vector: [Complex; 3], point: [f64; 3]) -> Complex {
    let angle = vector[0].re * point[0] + vector[1].re * point[1] + vector[2].re * point[2];
    let attenuation = vector[0].im * point[0] + vector[1].im * point[1] + vector[2].im * point[2];
    let (sine, cosine) = angle.sin_cos();
    let scale = (-attenuation).exp();
    Complex::new(scale * cosine, scale * sine)
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

/// Inputs retained for a cyclic Cartesian-axis permutation of plane polarizations.
#[derive(Debug)]
pub struct PermutationResidual {
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    turns: usize,
    helicity: bool,
}

fn permutation_pair<const N: usize>(vector: [Complex; 3], turns: usize) -> Result<[Jet<N>; 2]> {
    if turns == 0 {
        wavenumbers(vector)?;
        return Ok([Jet::constant(1.0), Jet::default()]);
    }
    // Both coefficients have one denominator. Avoid separately normalizing four
    // transverse components; retain the scaled gauge path at axes/extreme scales.
    let scale = vector
        .iter()
        .map(|v| v.re.abs().max(v.im.abs()))
        .fold(0.0, f64::max);
    if (1e-70..=1e70).contains(&scale) && vector.iter().all(|&v| finite(v)) {
        let [x, y, z] = std::array::from_fn(|i| Jet::<N>::variable(vector[i], i));
        let transverse = (x * x + y * y).sqrt();
        let destination = if turns == 1 {
            (z * z + x * x).sqrt()
        } else {
            (y * y + z * z).sqrt()
        };
        let denominator = transverse * destination;
        let k = (x * x + y * y + z * z).sqrt();
        if (1e-140..=1e140).contains(&denominator.value.norm_sqr()) && k.value != Complex::default()
        {
            let inverse = Jet::constant(1.0) / denominator;
            return Ok(if turns == 1 {
                [-y * z * inverse, -Complex::i() * x * k * inverse]
            } else {
                [-x * z * inverse, Complex::i() * y * k * inverse]
            });
        }
    }
    let rotated = std::array::from_fn(|axis| vector[(axis + 3 - turns) % 3]);
    let source_direction = Direction::<N>::new(vector)?;
    let (transverse, xy) = transverse_values([rotated[0], rotated[1]])?;
    // Cyclic rotation preserves the full norm; only the transverse frame changes.
    let destination =
        Direction::<N>::from_parts(rotated, source_direction.k.value, transverse, xy)?;
    if source_direction.transverse.value != Complex::default()
        && destination.transverse.value != Complex::default()
    {
        let unpermute = |mut jet: Jet<N>| {
            jet.derivative = std::array::from_fn(|j| jet.derivative[(j + turns) % 3]);
            jet
        };
        let pair = if turns == 1 {
            (
                -source_direction.xy[1] * unpermute(destination.xy[0]),
                -Complex::i()
                    * source_direction.xy[0]
                    * (source_direction.k / unpermute(destination.transverse)),
            )
        } else {
            (
                -source_direction.xy[0] * unpermute(destination.xy[1]),
                Complex::i()
                    * source_direction.xy[1]
                    * (source_direction.k / unpermute(destination.transverse)),
            )
        };
        return Ok(pair.into());
    }
    // At an axis, evaluate the defined Cartesian polarization gauges directly.
    let source = polarization_jet::<N>(vector, 0, false)?;
    let mut result = [Jet::default(); 2];
    for (p, output) in result.iter_mut().enumerate() {
        // The algebraic dual of the parity pair is (-M, N).
        let dual_pol = u8::from(p == 1);
        let sign = if p == 0 { -1.0 } else { 1.0 };
        let dual = polarization_jet::<N>(rotated, dual_pol, false)?;
        for (axis, mut component) in dual.into_iter().enumerate() {
            component.derivative = std::array::from_fn(|j| component.derivative[(j + turns) % 3]);
            *output += sign * component * source[(axis + 3 - turns) % 3];
        }
    }
    Ok(result)
}

fn permutation_polarization<const N: usize>(
    pair: [Jet<N>; 2],
    pol: u8,
    helicity: bool,
) -> [Jet<N>; 2] {
    let [same, cross] = pair;
    std::array::from_fn(|p| {
        if helicity {
            if p == usize::from(pol) {
                same + (2.0 * f64::from(pol) - 1.0) * cross
            } else {
                Jet::default()
            }
        } else if p == usize::from(pol) {
            same
        } else {
            cross
        }
    })
}

/// Polarization coefficients for cyclic xyz -> zxy permutations, shape (2, modes).
///
/// Each column contains both output polarizations for one input mode; matching
/// direction labels and construction of a dense basis operator are separate.
pub fn permutation(
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    turns: usize,
    helicity: bool,
) -> Result<(DMatrix<Complex>, PermutationResidual)> {
    if vectors.is_empty()
        || polarizations.len() != vectors.len()
        || polarizations.iter().any(|&p| p > 1)
    {
        return Err(Error::InvalidInput(
            "require matching nonempty vectors and polarizations".into(),
        ));
    }
    let turns = turns % 3;
    let mut value = DMatrix::zeros(2, vectors.len());
    let fill = |(group, output): (usize, &mut [Complex])| -> Result<()> {
        let first = 2 * group;
        let pair = permutation_pair::<0>(vectors[first], turns)?;
        for (offset, column) in output.chunks_mut(2).enumerate() {
            let i = first + offset;
            let pair = if offset == 0 || vectors[i] == vectors[first] {
                pair
            } else {
                permutation_pair::<0>(vectors[i], turns)?
            };
            for (out, coefficient) in
                column
                    .iter_mut()
                    .zip(permutation_polarization(pair, polarizations[i], helicity))
            {
                *out = coefficient.value;
            }
        }
        Ok(())
    };
    if vectors.len() >= 1024 {
        value
            .as_mut_slice()
            .par_chunks_mut(4)
            .enumerate()
            .try_for_each(fill)?;
    } else {
        value
            .as_mut_slice()
            .chunks_mut(4)
            .enumerate()
            .try_for_each(fill)?;
    }
    if value.iter().any(|&z| !finite(z)) {
        return Err(Error::InvalidInput("plane permutation overflow".into()));
    }
    Ok((
        value,
        PermutationResidual {
            vectors,
            polarizations,
            turns,
            helicity,
        },
    ))
}

impl PermutationResidual {
    /// Number of input plane modes.
    #[must_use]
    pub fn modes(&self) -> usize {
        self.vectors.len()
    }

    /// Contract the complex-wavevector derivatives of both polarization outputs.
    /// Direction derivatives at an axial polarization gauge are undefined.
    pub fn pullback(self, g: &DMatrix<Complex>) -> Result<Vec<[Complex; 3]>> {
        if g.shape() != (2, self.modes()) || g.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput(
                "invalid plane-permutation cotangent".into(),
            ));
        }
        if self.turns == 0 {
            return Ok(vec![[Complex::default(); 3]; self.modes()]);
        }
        let gradient = |i: usize| -> Result<[Complex; 3]> {
            let coefficients = permutation_polarization(
                permutation_pair::<3>(self.vectors[i], self.turns)?,
                self.polarizations[i],
                self.helicity,
            );
            Ok(std::array::from_fn(|axis| {
                (0..2)
                    .map(|p| g[(p, i)] * coefficients[p].derivative[axis].conj())
                    .sum()
            }))
        };
        if self.modes() >= 1024 {
            (0..self.modes()).into_par_iter().map(gradient).collect()
        } else {
            (0..self.modes()).map(gradient).collect()
        }
    }
}

/// Direct scalar plane translation phase.
#[inline]
pub fn translation(vector: [Complex; 3], point: [f64; 3]) -> Result<Complex> {
    if vector.iter().any(|&v| !finite(v)) || point.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "plane translation requires finite arguments".into(),
        ));
    }
    let value = phase(vector, point);
    if !finite(value) {
        return Err(Error::SpecialFunction("nonfinite plane translation".into()));
    }
    Ok(value)
}
/// One cyclic-coordinate polarization coefficient without allocating a matrix.
pub fn permutation_coefficient(
    vector: [Complex; 3],
    destination: u8,
    source: u8,
    turns: usize,
    helicity: bool,
) -> Result<Complex> {
    if destination > 1 || source > 1 {
        return Err(Error::InvalidInput("polarization must be 0 or 1".into()));
    }
    let value =
        permutation_polarization(permutation_pair::<0>(vector, turns % 3)?, source, helicity)
            [usize::from(destination)]
        .value;
    if !finite(value) {
        return Err(Error::SpecialFunction("nonfinite plane permutation".into()));
    }
    Ok(value)
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

/// Normal wavenumber on the outgoing branch, including the zero cutoff value.
#[must_use]
pub fn wave_vector_z(kx: Complex, ky: Complex, k: Complex) -> Complex {
    let root = complex_sqrt(k * k - kx * kx - ky * ky);
    if root.im < 0.0 { -root } else { root }
}
