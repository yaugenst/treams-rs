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
    if q.is_empty()
        || q.len() != polarizations.len()
        || q.iter().flatten().any(|v| !v.is_finite())
        || polarizations.iter().any(|&p| p > 1)
        || ks.iter().any(|&k| !finite(k) || k == Complex::default())
        || !area.is_finite()
        || area <= 0.0
    {
        return Err(Error::InvalidInput("channels require finite transverse vectors, polarizations 0/1, nonzero wavenumbers and positive unit-cell area".into()));
    }
    if !helicity && ks[0] != ks[1] {
        return Err(Error::InvalidInput(
            "parity channels require an achiral medium".into(),
        ));
    }
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
