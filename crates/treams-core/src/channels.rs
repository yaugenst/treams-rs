//! Spherical multipole / periodic plane-wave channels and analytic pullbacks.
#![allow(clippy::indexing_slicing)] // Validated modes and fixed seven-parameter local jets.

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::{
    Complex, Error, Result, basis::Basis, finite, jet::Jet, special::angular_jets, waves::Mode,
};

struct Geometry<const N: usize> {
    vector: [Jet<N>; 3],
    k: Jet<N>,
    kz: Jet<N>,
    sine: Jet<N>,
    cosine: Jet<N>,
    azimuth: Jet<N>,
    area: Jet<N>,
}
impl<const N: usize> Geometry<N> {
    fn new(k: Complex, q: [f64; 2], side: usize, area: f64, fixed_q: bool) -> Result<Self> {
        let input = Jet::variable(k, 3);
        let xy: [Jet<N>; 2] = std::array::from_fn(|i| {
            if fixed_q {
                Jet::constant(q[i])
            } else {
                Jet::variable(q[i], 4 + i)
            }
        });
        let k = (input * input).sqrt();
        let mut kz = (input * input - xy[0] * xy[0] - xy[1] * xy[1]).sqrt();
        if kz.value == Complex::default() {
            return Err(Error::InvalidInput(
                "plane-wave channel is at a diffraction threshold".into(),
            ));
        }
        if kz.value.im < 0.0 || (kz.value.im == 0.0 && kz.value.re < 0.0) {
            kz = -kz;
        }
        let scale = q[0].abs().max(q[1].abs());
        let (mut sine, azimuth) = if scale == 0.0 {
            if N > 0 && !fixed_q {
                return Err(Error::InvalidInput("normal-incidence azimuth is undefined; use fixed_q=True for gradients at fixed incidence".into()));
            }
            (Jet::default(), Jet::constant(1.0))
        } else {
            let x = xy[0] / scale;
            let y = xy[1] / scale;
            let norm = (x * x + y * y).sqrt();
            (norm * scale / k, (x + Complex::i() * y) / norm)
        };
        if sine.value.re < 0.0 {
            sine = -sine;
        }
        let axial = if side == 0 { kz } else { -kz };
        Ok(Self {
            vector: [xy[0], xy[1], axial],
            k,
            kz,
            sine,
            cosine: axial / k,
            azimuth,
            area: Jet::variable(area, 6),
        })
    }

    fn entry(&self, mode: Mode, pol: u8, position: [f64; 3], helicity: bool) -> [Jet<N>; 2] {
        if helicity && mode.pol != pol {
            return [Jet::default(); 2];
        }
        let (l, m) = (mode.l, mode.m);
        let [pi, tau] = angular_jets(l, m, self.cosine, self.sine);
        let angular = if helicity {
            tau + (2.0 * f64::from(pol) - 1.0) * pi
        } else if mode.pol == pol {
            tau
        } else {
            pi
        };
        let normalization = (std::f64::consts::PI * f64::from(2 * l + 1) / f64::from(l * (l + 1)))
            .sqrt()
            * (0.5 * (libm::lgamma(f64::from(l - m + 1)) - libm::lgamma(f64::from(l + m + 1))))
                .exp();
        let phase = (Complex::i()
            * self
                .vector
                .into_iter()
                .enumerate()
                .map(|(i, k)| k * Jet::variable(position[i], i))
                .sum::<Jet<N>>())
        .exp();
        let incident =
            2.0 * Complex::i().powi(l) * normalization * self.azimuth.powi(-m) * angular * phase;
        let outgoing = (-Complex::i()).powi(l) * normalization * self.azimuth.powi(m) * angular
            / (self.area * self.k * self.kz * phase);
        [incident, outgoing]
    }
}

/// Direct spherical-to-plane radiation coefficient for a plane wavevector.
pub fn spherical_to_plane(
    mode: Mode,
    vector: [Complex; 3],
    pol: u8,
    area: f64,
    helicity: bool,
) -> Result<Complex> {
    mode.validate()?;
    if pol > 1
        || vector.iter().any(|&v| !finite(v))
        || vector[0].im != 0.0
        || vector[1].im != 0.0
        || !area.is_finite()
        || area == 0.0
    {
        return Err(Error::InvalidInput("real transverse wavevector, finite axial component, polarization 0/1 and nonzero area required".into()));
    }
    let side = usize::from(vector[2].im < 0.0 || (vector[2].im == 0.0 && vector[2].re < 0.0));
    let k = crate::complex_sqrt(vector.iter().map(|v| v * v).sum());
    let geometry = Geometry::<0>::new(k, [vector[0].re, vector[1].re], side, area.abs(), true)?;
    let value = geometry.entry(mode, pol, [0.0; 3], helicity)[1].value;
    if !finite(value) {
        return Err(Error::SpecialFunction(
            "nonfinite plane radiation coefficient".into(),
        ));
    }
    Ok(value)
}

/// Retained physical channel inputs; local seven-parameter derivatives are recomputed.
#[derive(Clone, Debug)]
pub struct ChannelsResidual {
    basis: Basis,
    ks: [Complex; 2],
    q: Vec<[f64; 2]>,
    polarizations: Vec<u8>,
    area: f64,
    helicity: bool,
    /// Rows pack (incident/emitted, up/down, multipole); columns are plane modes.
    pub value: DMatrix<Complex>,
}

fn validate_channels(
    ks: [Complex; 2],
    q: &[[f64; 2]],
    polarizations: &[u8],
    area: f64,
    helicity: bool,
) -> Result<()> {
    if q.is_empty()
        || q.len() != polarizations.len()
        || q.iter().flatten().any(|v| !v.is_finite())
        || polarizations.iter().any(|&p| p > 1)
        || ks.iter().any(|&k| !finite(k) || k == Complex::default())
        || !area.is_finite()
        || area <= 0.0
    {
        return Err(Error::InvalidInput("channels require finite transverse vectors, polarizations 0/1, nonzero wavenumbers and positive unit-cell measure".into()));
    }
    if !helicity && ks[0] != ks[1] {
        return Err(Error::InvalidInput(
            "parity channels require an achiral medium".into(),
        ));
    }
    Ok(())
}

/// Both incidence and radiation channels of a 2D periodic spherical basis.
pub fn spherical(
    basis: Basis,
    ks: [Complex; 2],
    q: Vec<[f64; 2]>,
    polarizations: Vec<u8>,
    area: f64,
    helicity: bool,
) -> Result<ChannelsResidual> {
    basis.validate()?;
    validate_channels(ks, &q, &polarizations, area, helicity)?;
    let d = basis.modes.len();
    let mut value = DMatrix::zeros(4 * d, q.len());
    value
        .as_mut_slice()
        .par_chunks_mut(4 * d)
        .enumerate()
        .try_for_each(|(j, column)| -> Result<()> {
            for side in 0..2 {
                let geometry =
                    Geometry::<0>::new(ks[usize::from(polarizations[j])], q[j], side, area, true)?;
                for (i, &(p, mode)) in basis.modes.iter().enumerate() {
                    let pair = geometry.entry(mode, polarizations[j], basis.positions[p], helicity);
                    column[side * d + i] = pair[0].value;
                    column[(2 + side) * d + i] = pair[1].value;
                }
            }
            Ok(())
        })?;
    if value.iter().any(|&v| !finite(v)) {
        return Err(Error::SpecialFunction(
            "non-finite plane-wave channel".into(),
        ));
    }
    Ok(ChannelsResidual {
        basis,
        ks,
        q,
        polarizations,
        area,
        helicity,
        value,
    })
}

/// Cotangents of origins, medium wavenumbers, transverse plane vectors and cell area.
#[derive(Clone, Debug)]
pub struct ChannelGradient {
    /// Multipole origin cotangents.
    pub positions: Vec<[f64; 3]>,
    /// Complex medium wavenumber cotangents.
    pub ks: [Complex; 2],
    /// Transverse plane-wave cotangents (zero when `fixed_q` is requested).
    pub q: Vec<[f64; 2]>,
    /// Unit-cell area cotangent.
    pub area: f64,
}
impl ChannelGradient {
    fn zeros(positions: usize, planes: usize) -> Self {
        Self {
            positions: vec![[0.0; 3]; positions],
            ks: [Complex::default(); 2],
            q: vec![[0.0; 2]; planes],
            area: 0.0,
        }
    }
    fn add(&mut self, other: Self) {
        for (a, b) in self.positions.iter_mut().zip(other.positions) {
            for (a, b) in a.iter_mut().zip(b) {
                *a += b;
            }
        }
        for (a, b) in self.q.iter_mut().zip(other.q) {
            for (a, b) in a.iter_mut().zip(b) {
                *a += b;
            }
        }
        for (a, b) in self.ks.iter_mut().zip(other.ks) {
            *a += b;
        }
        self.area += other.area;
    }
}
impl ChannelsResidual {
    /// Contract both channel arrays; `fixed_q` holds the incident/diffraction directions fixed.
    pub fn pullback(self, g: &DMatrix<Complex>, fixed_q: bool) -> Result<ChannelGradient> {
        if g.shape() != self.value.shape() || g.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid channel cotangent".into()));
        }
        let d = self.basis.modes.len();
        let zero = || ChannelGradient::zeros(self.basis.positions.len(), self.q.len());
        self.q
            .par_iter()
            .enumerate()
            .try_fold(zero, |mut result, (j, &q)| -> Result<_> {
                let pol = self.polarizations[j];
                for side in 0..2 {
                    let geometry =
                        Geometry::<7>::new(self.ks[usize::from(pol)], q, side, self.area, fixed_q)?;
                    for (i, &(p, mode)) in self.basis.modes.iter().enumerate() {
                        let weights = [g[(side * d + i, j)], g[((2 + side) * d + i, j)]];
                        if weights.iter().all(|&z| z == Complex::default()) {
                            continue;
                        }
                        let pair =
                            geometry.entry(mode, pol, self.basis.positions[p], self.helicity);
                        let gradient: [Complex; 7] = std::array::from_fn(|a| {
                            weights[0] * pair[0].derivative[a].conj()
                                + weights[1] * pair[1].derivative[a].conj()
                        });
                        for (a, g) in result.positions[p].iter_mut().zip(&gradient[..3]) {
                            *a += g.re;
                        }
                        result.ks[usize::from(pol)] += gradient[3];
                        if !fixed_q {
                            for (a, g) in result.q[j].iter_mut().zip(&gradient[4..6]) {
                                *a += g.re;
                            }
                        }
                        result.area += gradient[6].re;
                    }
                }
                Ok(result)
            })
            .try_reduce(zero, |mut a, b| {
                a.add(b);
                Ok(a)
            })
    }
}

struct CylGeometry<const N: usize> {
    vector: [Jet<N>; 3],
    transverse: Jet<N>,
    normal: Jet<N>,
    period: Jet<N>,
}
impl<const N: usize> CylGeometry<N> {
    fn new(k: Complex, q: [f64; 2], side: usize, period: f64, fixed_q: bool) -> Result<Self> {
        let k = Jet::variable(k, 3);
        let kz = Jet::constant(q[0]);
        let kx = if fixed_q {
            Jet::constant(q[1])
        } else {
            Jet::variable(q[1], 4)
        };
        let mut transverse = (k * k - kz * kz).sqrt();
        let mut normal = (k * k - kz * kz - kx * kx).sqrt();
        if transverse.value == Complex::default() || normal.value == Complex::default() {
            return Err(Error::InvalidInput(
                "cylindrical plane channel is at a cutoff or diffraction threshold".into(),
            ));
        }
        if transverse.value.im < 0.0 {
            transverse = -transverse;
        }
        if normal.value.im < 0.0 || (normal.value.im == 0.0 && normal.value.re < 0.0) {
            normal = -normal;
        }
        Ok(Self {
            vector: [kx, if side == 0 { normal } else { -normal }, kz],
            transverse,
            normal,
            period: Jet::variable(period, 5),
        })
    }
    fn entry(&self, mode: crate::cylwaves::Mode, pol: u8, position: [f64; 3]) -> [Jet<N>; 2] {
        if !crate::plane::cylindrical_mode_matches(mode, self.vector[2].value, pol) {
            return [Jet::default(); 2];
        }
        let phase = (Complex::i()
            * self
                .vector
                .iter()
                .enumerate()
                .map(|(a, &k)| k * Jet::variable(position[a], a))
                .sum::<Jet<N>>())
        .exp();
        let (incoming_angle, outgoing_angle) = if self.transverse.value == Complex::default() {
            (
                Jet::constant(Complex::i().powi(mode.m)),
                Jet::constant((-Complex::i()).powi(mode.m)),
            )
        } else {
            (
                ((Complex::i() * self.vector[0] + self.vector[1]) / self.transverse).powi(mode.m),
                ((-Complex::i() * self.vector[0] + self.vector[1]) / self.transverse).powi(mode.m),
            )
        };
        let incident = incoming_angle * phase;
        let outgoing = 2.0 * outgoing_angle / (self.period * self.normal * phase);
        [incident, outgoing]
    }
}

/// Direct cylindrical-to-plane radiation, using the supplied normal wavevector.
#[allow(clippy::float_cmp)] // Direct coefficients preserve exact axial labels.
pub fn cylindrical_to_plane(
    mode: crate::cylwaves::Mode,
    vector: [Complex; 3],
    pol: u8,
    period: f64,
) -> Result<Complex> {
    mode.validate()?;
    if pol > 1
        || vector.iter().any(|&v| !finite(v))
        || vector[0].im != 0.0
        || vector[2].im != 0.0
        || !period.is_finite()
        || period == 0.0
    {
        return Err(Error::InvalidInput(
            "finite vector with real kx/kz, polarization 0/1 and nonzero period required".into(),
        ));
    }
    if mode.kz != vector[2].re || mode.pol != pol {
        return Ok(Complex::default());
    }
    let mut normal = vector[1];
    if normal == Complex::default() {
        return Err(Error::InvalidInput(
            "plane channel is at a diffraction threshold".into(),
        ));
    }
    if normal.im < 0.0 || (normal.im == 0.0 && normal.re < 0.0) {
        normal = -normal;
    }
    let geometry = CylGeometry::<0> {
        vector: vector.map(Jet::constant),
        transverse: Jet::constant(crate::complex_sqrt(
            vector[0] * vector[0] + vector[1] * vector[1],
        )),
        normal: Jet::constant(normal),
        period: Jet::constant(period.abs()),
    };
    let value = geometry.entry(mode, pol, [0.0; 3])[1].value;
    if !finite(value) {
        return Err(Error::SpecialFunction(
            "nonfinite plane radiation coefficient".into(),
        ));
    }
    Ok(value)
}

/// Cylindrical channels for a lattice along x, radiating toward positive/negative y.
#[derive(Clone, Debug)]
pub struct CylChannelsResidual {
    basis: crate::cylwaves::Basis,
    ks: [Complex; 2],
    q: Vec<[f64; 2]>,
    polarizations: Vec<u8>,
    period: f64,
    /// Rows pack (incident/emitted, up/down, cylindrical mode); columns are plane modes.
    pub value: DMatrix<Complex>,
}
/// Incidence/radiation channels; q=(kz,kx) matches a zx-aligned plane basis.
/// Axial kz values are fixed labels. The polarization convention is shared by both families.
pub fn cylindrical(
    basis: crate::cylwaves::Basis,
    ks: [Complex; 2],
    q: Vec<[f64; 2]>,
    polarizations: Vec<u8>,
    period: f64,
    helicity: bool,
) -> Result<CylChannelsResidual> {
    basis.validate()?;
    validate_channels(ks, &q, &polarizations, period, helicity)?;
    let d = basis.modes.len();
    let mut value = DMatrix::zeros(4 * d, q.len());
    value
        .as_mut_slice()
        .par_chunks_mut(4 * d)
        .enumerate()
        .try_for_each(|(j, column)| -> Result<()> {
            let pol = polarizations[j];
            for side in 0..2 {
                let geometry =
                    CylGeometry::<0>::new(ks[usize::from(pol)], q[j], side, period, false)?;
                for (i, &(p, mode)) in basis.modes.iter().enumerate() {
                    let [incident, outgoing] = geometry.entry(mode, pol, basis.positions[p]);
                    column[side * d + i] = incident.value;
                    column[(2 + side) * d + i] = outgoing.value;
                }
            }
            Ok(())
        })?;
    Ok(CylChannelsResidual {
        basis,
        ks,
        q,
        polarizations,
        period,
        value,
    })
}
impl CylChannelsResidual {
    /// Origin, medium and kx cotangents; `q[:,0]` is fixed and area denotes the period.
    pub fn pullback(self, g: &DMatrix<Complex>, fixed_q: bool) -> Result<ChannelGradient> {
        if g.shape() != self.value.shape() || g.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput(
                "invalid cylindrical channel cotangent".into(),
            ));
        }
        let d = self.basis.modes.len();
        let zero = || ChannelGradient::zeros(self.basis.positions.len(), self.q.len());
        self.q
            .par_iter()
            .enumerate()
            .try_fold(zero, |mut result, (j, &q)| -> Result<_> {
                let pol = self.polarizations[j];
                for side in 0..2 {
                    let geometry = CylGeometry::<6>::new(
                        self.ks[usize::from(pol)],
                        q,
                        side,
                        self.period,
                        fixed_q,
                    )?;
                    for (i, &(p, mode)) in self.basis.modes.iter().enumerate() {
                        let weights = [g[(side * d + i, j)], g[((2 + side) * d + i, j)]];
                        if weights.iter().all(|&v| v == Complex::default()) {
                            continue;
                        }
                        let pair = geometry.entry(mode, pol, self.basis.positions[p]);
                        let gradient: [Complex; 6] = std::array::from_fn(|a| {
                            weights[0] * pair[0].derivative[a].conj()
                                + weights[1] * pair[1].derivative[a].conj()
                        });
                        for (g, v) in result.positions[p].iter_mut().zip(&gradient[..3]) {
                            *g += v.re;
                        }
                        result.ks[usize::from(pol)] += gradient[3];
                        result.q[j][1] += gradient[4].re;
                        result.area += gradient[5].re;
                    }
                }
                Ok(result)
            })
            .try_reduce(zero, |mut a, b| {
                a.add(b);
                Ok(a)
            })
    }
}
