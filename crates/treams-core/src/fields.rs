//! Cartesian vector spherical waves without polar-coordinate singularities.
#![allow(clippy::indexing_slicing)] // Fixed Cartesian vectors and Hessians.

use crate::{
    Complex, Error, Result,
    angular::solid,
    finite,
    special::{Radial, spherical},
    waves::Mode,
};

/// Multipole basis for Cartesian field evaluation.
#[derive(Clone, Debug)]
pub enum FieldBasis {
    /// Spherical vector waves.
    Spherical(crate::basis::Basis),
    /// Cylindrical vector waves; axial wavenumbers are fixed mode labels.
    Cylindrical(crate::cylwaves::Basis),
}
impl From<crate::basis::Basis> for FieldBasis {
    fn from(value: crate::basis::Basis) -> Self {
        Self::Spherical(value)
    }
}
impl From<crate::cylwaves::Basis> for FieldBasis {
    fn from(value: crate::cylwaves::Basis) -> Self {
        Self::Cylindrical(value)
    }
}
impl FieldBasis {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Spherical(b) => b.validate(),
            Self::Cylindrical(b) => b.validate(),
        }
    }
    fn origins(&self) -> &[[f64; 3]] {
        match self {
            Self::Spherical(b) => &b.positions,
            Self::Cylindrical(b) => &b.positions,
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Spherical(b) => b.modes.len(),
            Self::Cylindrical(b) => b.modes.len(),
        }
    }
    fn origin_pol(&self, i: usize) -> (usize, usize) {
        match self {
            Self::Spherical(b) => (b.modes[i].0, usize::from(b.modes[i].1.pol)),
            Self::Cylindrical(b) => (b.modes[i].0, usize::from(b.modes[i].1.pol)),
        }
    }
    fn wave<const DERIVATIVES: bool>(
        &self,
        i: usize,
        k: Complex,
        r: [f64; 3],
        helicity: bool,
        radial: Radial,
    ) -> Result<VectorWave> {
        match self {
            Self::Spherical(b) => {
                spherical_wave_impl::<DERIVATIVES>(b.modes[i].1, k, r, helicity, radial)
            }
            Self::Cylindrical(b) => {
                if DERIVATIVES {
                    cylindrical_wave_impl::<4>(b.modes[i].1, k, r, helicity, radial)
                } else {
                    cylindrical_wave_impl::<0>(b.modes[i].1, k, r, helicity, radial)
                }
            }
        }
    }
}

/// Electric vector spherical wave and its analytic Cartesian derivatives.
#[derive(Clone, Copy, Debug)]
pub struct VectorWave {
    /// Cartesian electric field.
    pub value: [Complex; 3],
    /// `position[component][axis]` is the field's Cartesian Jacobian.
    pub position: [[Complex; 3]; 3],
    /// Complex wave number derivative.
    pub k: [Complex; 3],
}

/// Cartesian cylindrical vector wave and its spatial/medium-wavenumber derivatives.
/// Adjacent scalar harmonics remove all polar-coordinate divisions at the axis.
pub fn cylindrical_wave(
    mode: crate::cylwaves::Mode,
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    radial: Radial,
) -> Result<VectorWave> {
    cylindrical_wave_impl::<4>(mode, k, position, helicity, radial)
}

fn cylindrical_wave_impl<const N: usize>(
    mode: crate::cylwaves::Mode,
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    radial: Radial,
) -> Result<VectorWave> {
    use crate::{cylwaves, jet::Jet};
    mode.validate()?;
    if !finite(k) || k == Complex::default() || position.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidInput(
            "require a finite nonzero wavenumber and finite field position".into(),
        ));
    }
    let r: [Jet<N>; 3] = std::array::from_fn(|i| Jet::variable(position[i], i));
    let k = Jet::<N>::variable(k, 3);
    let mut transverse = (k * k - mode.kz * mode.kz).sqrt();
    if transverse.value.im < 0.0 {
        transverse = -transverse;
    }
    if transverse.value == Complex::default() {
        return Err(Error::InvalidInput(
            "cylindrical cutoff requires a limiting formulation".into(),
        ));
    }
    let rho = position[0].hypot(position[1]);
    let [lower, center, upper] =
        if radial == Radial::Regular && (transverse.value * rho).norm() < 0.5 {
            [mode.m - 1, mode.m, mode.m + 1]
                .map(|m| cylwaves::regular_harmonic(m, transverse, r, Jet::constant(mode.kz)))
        } else {
            if rho == 0.0 {
                return Err(Error::SpecialFunction(
                    "outgoing cylindrical wave is singular on the axis".into(),
                ));
            }
            let radius = (r[0] * r[0] + r[1] * r[1]).sqrt();
            let argument = transverse * radius;
            let radial = crate::special::cylindrical(mode.m, argument.value, radial)?;
            let value = argument.map(radial.value, radial.first);
            let derivative = argument.map(radial.first, radial.second);
            let azimuth = (r[0] + Complex::i() * r[1]) / radius;
            let phase = (Complex::i() * mode.kz * r[2]).exp() * azimuth.powi(mode.m);
            let adjacent = f64::from(mode.m) * value / argument;
            [
                (adjacent + derivative) * phase / azimuth,
                value * phase,
                (adjacent - derivative) * phase * azimuth,
            ]
        };
    let m = [
        0.5 * Complex::i() * (lower + upper),
        0.5 * (upper - lower),
        Jet::default(),
    ];
    let n = [
        0.5 * Complex::i() * mode.kz / k * (lower - upper),
        -0.5 * mode.kz / k * (lower + upper),
        transverse / k * center,
    ];
    let fields: [Jet<N>; 3] = std::array::from_fn(|i| {
        if helicity {
            (n[i] + (2.0 * f64::from(mode.pol) - 1.0) * m[i]) * std::f64::consts::FRAC_1_SQRT_2
        } else if mode.pol == 0 {
            m[i]
        } else {
            n[i]
        }
    });
    Ok(VectorWave {
        value: fields.map(|v| v.value),
        position: fields
            .map(|v| std::array::from_fn(|i| v.derivative.get(i).copied().unwrap_or_default())),
        k: fields.map(|v| v.derivative.get(3).copied().unwrap_or_default()),
    })
}

fn cross(a: [Complex; 3], b: [Complex; 3]) -> [Complex; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// z_l(kr)/r^l has a finite regular limit. Evaluating its series directly
// avoids divisions by zero and cancellation in the near-origin field gradients.
fn scaled_radial(l: u32, k: Complex, r: f64, radial: Radial) -> Result<Complex> {
    let x = k * r;
    if radial == Radial::Regular && x.norm() < 0.5 {
        let mut denominator = 1.0;
        for i in 0..=l {
            denominator *= f64::from(2 * i + 1);
        }
        let mut coefficient = k.powu(l) / denominator;
        let mut sum = Complex::default();
        for q in 0..32 {
            sum += coefficient * x.powu(2 * q);
            coefficient /= -2.0 * f64::from(q + 1) * f64::from(2 * l + 2 * q + 3);
        }
        return Ok(sum);
    }
    Ok(spherical(l, x, radial)?.value / r.powf(f64::from(l)))
}

/// Evaluate a vector spherical wave in treams normalization.
/// Parity polarization 0 is M, 1 is N; helicity is `(N + (2pol-1) M)/sqrt(2)`.
pub fn spherical_wave(
    mode: Mode,
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    radial: Radial,
) -> Result<VectorWave> {
    spherical_wave_impl::<true>(mode, k, position, helicity, radial)
}

fn spherical_wave_impl<const DERIVATIVES: bool>(
    mode: Mode,
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    radial: Radial,
) -> Result<VectorWave> {
    mode.validate()?;
    if !finite(k) || k.norm_sqr() == 0.0 || position.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "require a finite nonzero wavenumber and finite position".into(),
        ));
    }
    let r2 = position.iter().map(|v| v * v).sum::<f64>();
    let l = u32::try_from(mode.l).map_err(|_| Error::InvalidInput("invalid degree".into()))?;
    let degree = f64::from(l);
    let c = scaled_radial(l, k, r2.sqrt(), radial)?;
    let e = scaled_radial(l + 1, k, r2.sqrt(), radial)?;
    let f = if DERIVATIVES {
        scaled_radial(l + 2, k, r2.sqrt(), radial)?
    } else {
        Complex::default()
    };
    let ck = degree / k * c - r2 * e;
    let ek = (degree + 1.0) / k * e - r2 * f;
    let d = (degree + 1.0) / k * c - r2 * e;
    let dk = (degree + 1.0) / k * ck - (degree + 1.0) / k.powu(2) * c - r2 * ek;
    let solid = solid::<DERIVATIVES>(mode.l, mode.m, position);
    let vector = position.map(|v| Complex::new(v, 0.0));
    let rotation = cross(vector, solid.gradient);
    let normalization = Complex::i()
        * ((2.0 * degree + 1.0) / (4.0 * std::f64::consts::PI * degree * (degree + 1.0))).sqrt()
        * (0.5
            * (libm::lgamma(f64::from(mode.l - mode.m + 1))
                - libm::lgamma(f64::from(mode.l + mode.m + 1))))
        .exp();
    let (weight_n, weight_m) = if helicity {
        (
            std::f64::consts::FRAC_1_SQRT_2,
            (2.0 * f64::from(mode.pol) - 1.0) * std::f64::consts::FRAC_1_SQRT_2,
        )
    } else if mode.pol == 1 {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let combine = |n, m| normalization * (weight_n * n - weight_m * m);
    let value = std::array::from_fn(|i| {
        combine(
            d * solid.gradient[i] + degree * e * solid.value * position[i],
            c * rotation[i],
        )
    });
    let wave_k = std::array::from_fn(|i| {
        combine(
            dk * solid.gradient[i] + degree * ek * solid.value * position[i],
            ck * rotation[i],
        )
    });
    let mut jacobian = [[Complex::default(); 3]; 3];
    for axis in 0..if DERIVATIVES { 3 } else { 0 } {
        let ca = -k * e * position[axis];
        let ea = -k * f * position[axis];
        let da = (degree + 1.0) / k * ca - 2.0 * position[axis] * e - r2 * ea;
        let mut unit = [Complex::default(); 3];
        unit[axis] = Complex::new(1.0, 0.0);
        let first = cross(unit, solid.gradient);
        let second = cross(vector, std::array::from_fn(|i| solid.hessian[i][axis]));
        for i in 0..3 {
            let dn = da * solid.gradient[i]
                + d * solid.hessian[i][axis]
                + degree
                    * (ea * solid.value * position[i]
                        + e * solid.gradient[axis] * position[i]
                        + if i == axis {
                            e * solid.value
                        } else {
                            Complex::default()
                        });
            let dm = ca * rotation[i] + c * (first[i] + second[i]);
            jacobian[i][axis] = combine(dn, dm);
        }
    }
    Ok(VectorWave {
        value,
        position: jacobian,
        k: wave_k,
    })
}

/// Field samples and the data needed for an analytic reverse pass.
/// Storage is linear in sample and mode count; no sample-by-mode Jacobian is retained.
#[derive(Debug)]
pub struct FieldResidual {
    geometry: FieldGeometry,
    coefficients: Vec<Complex>,
    /// Electric field at each Cartesian sample.
    pub value: Vec<[Complex; 3]>,
}

#[derive(Debug)]
struct FieldGeometry {
    basis: FieldBasis,
    points: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
    radial: Radial,
}
impl FieldGeometry {
    fn new(
        basis: FieldBasis,
        points: Vec<[f64; 3]>,
        ks: [Complex; 2],
        helicity: bool,
        radial: Radial,
    ) -> Result<Self> {
        basis.validate()?;
        if points.iter().flatten().any(|v| !v.is_finite())
            || ks.iter().any(|&v| !finite(v) || v == Complex::default())
            || (!helicity && ks[0] != ks[1])
        {
            return Err(Error::InvalidInput("require finite field points and nonzero wavenumbers; parity requires an achiral medium".into()));
        }
        Ok(Self {
            basis,
            points,
            ks,
            helicity,
            radial,
        })
    }
    fn wave<const DERIVATIVES: bool>(&self, i: usize, point: [f64; 3]) -> Result<VectorWave> {
        let (particle, pol) = self.basis.origin_pol(i);
        let position = std::array::from_fn(|a| point[a] - self.basis.origins()[particle][a]);
        self.basis
            .wave::<DERIVATIVES>(i, self.ks[pol], position, self.helicity, self.radial)
    }
}

/// Field cotangents under the real Hermitian pairing.
#[derive(Debug)]
pub struct FieldGradient {
    /// Complex multipole amplitude cotangents.
    pub coefficients: Vec<Complex>,
    /// Real sample-coordinate cotangents.
    pub points: Vec<[f64; 3]>,
    /// Real expansion-origin cotangents.
    pub origins: Vec<[f64; 3]>,
    /// Complex negative/positive helicity wavenumber cotangents.
    pub ks: [Complex; 2],
}

/// Evaluate weighted electric fields in parallel over samples.
pub fn field(
    basis: impl Into<FieldBasis>,
    coefficients: Vec<Complex>,
    points: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
    radial: Radial,
) -> Result<FieldResidual> {
    use rayon::prelude::*;
    let basis = basis.into();
    if coefficients.len() != basis.len() || coefficients.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput(
            "require one finite field coefficient per mode".into(),
        ));
    }
    let geometry = FieldGeometry::new(basis, points, ks, helicity, radial)?;
    let value = geometry
        .points
        .par_iter()
        .map(|point| {
            let mut value = [Complex::default(); 3];
            for (i, &amplitude) in coefficients.iter().enumerate() {
                let wave = geometry.wave::<false>(i, *point)?;
                for (v, f) in value.iter_mut().zip(wave.value) {
                    *v += amplitude * f;
                }
            }
            Ok(value)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(FieldResidual {
        geometry,
        coefficients,
        value,
    })
}

impl FieldResidual {
    /// Contract analytic field derivatives without storing a dense Jacobian.
    pub fn pullback(self, cotangent: &[[Complex; 3]]) -> Result<FieldGradient> {
        if cotangent.len() != self.geometry.points.len()
            || cotangent.iter().flatten().any(|&v| !finite(v))
        {
            return Err(Error::InvalidInput("invalid field cotangent".into()));
        }
        self.geometry
            .pullback(Some(&self.coefficients), |sample, _| cotangent[sample])
    }
}

impl FieldGeometry {
    fn pullback(
        &self,
        coefficients: Option<&[Complex]>,
        cotangent: impl Fn(usize, usize) -> [Complex; 3] + Sync,
    ) -> Result<FieldGradient> {
        use rayon::prelude::*;
        let mut points = vec![[0.0; 3]; self.points.len()];
        let zero = || FieldGradient {
            coefficients: vec![Complex::default(); coefficients.map_or(0, <[Complex]>::len)],
            points: Vec::new(),
            origins: vec![[0.0; 3]; self.basis.origins().len()],
            ks: [Complex::default(); 2],
        };
        let mut result = points
            .par_iter_mut()
            .zip(self.points.par_iter())
            .enumerate()
            .try_fold(zero, |mut sum, (sample, (point_gradient, point))| {
                for i in 0..self.basis.len() {
                    let (particle, pol) = self.basis.origin_pol(i);
                    let wave = self.wave::<true>(i, *point)?;
                    let amplitude = coefficients.map_or(Complex::new(1.0, 0.0), |c| c[i]);
                    let g = cotangent(sample, i);
                    for (component, &cot) in g.iter().enumerate() {
                        if coefficients.is_some() {
                            sum.coefficients[i] += wave.value[component].conj() * cot;
                        }
                        sum.ks[pol] += (amplitude * wave.k[component]).conj() * cot;
                        for (axis, point_derivative) in point_gradient.iter_mut().enumerate() {
                            let derivative =
                                (cot.conj() * amplitude * wave.position[component][axis]).re;
                            *point_derivative += derivative;
                            sum.origins[particle][axis] -= derivative;
                        }
                    }
                }
                Ok(sum)
            })
            .try_reduce(zero, |mut a, b| {
                for (x, y) in a.coefficients.iter_mut().zip(b.coefficients) {
                    *x += y;
                }
                for (x, y) in a
                    .origins
                    .iter_mut()
                    .flatten()
                    .zip(b.origins.iter().flatten())
                {
                    *x += y;
                }
                for (x, y) in a.ks.iter_mut().zip(b.ks) {
                    *x += y;
                }
                Ok(a)
            })?;
        result.points = points;
        Ok(result)
    }
}

/// Linear-storage residual of a full field evaluation matrix.
#[derive(Debug)]
pub struct OperatorResidual {
    geometry: FieldGeometry,
}

/// Matrix mapping multipole amplitudes to Cartesian samples. Rows pack (sample, component).
/// The returned context retains geometry only, without the output or any derivative matrix.
pub fn operator(
    basis: impl Into<FieldBasis>,
    points: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
    radial: Radial,
) -> Result<(nalgebra::DMatrix<Complex>, OperatorResidual)> {
    use rayon::prelude::*;
    let geometry = FieldGeometry::new(basis.into(), points, ks, helicity, radial)?;
    let n = geometry.points.len();
    let mut value = nalgebra::DMatrix::zeros(3 * n, geometry.basis.len());
    if n > 0 {
        value
            .as_mut_slice()
            .par_chunks_mut(3 * n)
            .enumerate()
            .try_for_each(|(i, column)| -> Result<()> {
                for (sample, &point) in geometry.points.iter().enumerate() {
                    let wave = geometry.wave::<false>(i, point)?;
                    column[3 * sample..3 * sample + 3].copy_from_slice(&wave.value);
                }
                Ok(())
            })?;
    }
    Ok((value, OperatorResidual { geometry }))
}
impl OperatorResidual {
    /// Flattened operator shape (three times samples, modes).
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (3 * self.geometry.points.len(), self.geometry.basis.len())
    }
    /// Sample, origin and wavenumber gradients; the coefficient gradient is empty.
    pub fn pullback(self, cotangent: &nalgebra::DMatrix<Complex>) -> Result<FieldGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "invalid field operator cotangent".into(),
            ));
        }
        self.geometry.pullback(None, |sample, mode| {
            std::array::from_fn(|i| cotangent[(3 * sample + i, mode)])
        })
    }
}
