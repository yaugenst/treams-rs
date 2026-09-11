//! Cartesian cylindrical-wave translation and analytic derivatives.
#![allow(clippy::indexing_slicing)] // Validated modes, basis indices and matrix shapes.

use crate::{
    Complex, Error, Result, finite,
    jet::Jet,
    special::{Radial, cylindrical},
};
use nalgebra::DMatrix;

/// Cylindrical mode at one axial wavenumber.
#[derive(Clone, Copy, Debug)]
pub struct Mode {
    /// Axial wavenumber (a fixed mode label for basis expansion).
    pub kz: f64,
    /// Integer azimuthal order.
    pub m: i32,
    /// Negative/positive helicity, or M/N parity.
    pub pol: u8,
}
impl Mode {
    pub(crate) fn validate(self) -> Result<()> {
        if !self.kz.is_finite() || self.m.unsigned_abs() > 128 || self.pol > 1 {
            return Err(Error::InvalidInput(
                "require finite kz, |m| <= 128 and polarization 0 or 1".into(),
            ));
        }
        Ok(())
    }
}

/// Translation coefficient and its continuous derivatives with common kz held equal.
#[derive(Clone, Copy, Debug, Default)]
pub struct Translation {
    /// Complex coefficient.
    pub value: Complex,
    /// Cartesian displacement derivatives.
    pub position: [Complex; 3],
    /// Medium wavenumber derivative.
    pub k: Complex,
    /// Common axial wavenumber derivative.
    pub kz: Complex,
}

/// Translate a cylindrical wave. Distinct axial wavenumbers and polarizations decouple.
#[allow(clippy::float_cmp)] // kz labels and the self origin are exact mode identities.
pub fn translate(
    to: Mode,
    from: Mode,
    k: Complex,
    position: [f64; 3],
    radial: Radial,
) -> Result<Translation> {
    to.validate()?;
    from.validate()?;
    if !finite(k) || k.norm_sqr() == 0.0 || position.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "require a finite nonzero wavenumber and finite displacement".into(),
        ));
    }
    if to.kz != from.kz
        || to.pol != from.pol
        || (radial == Radial::Outgoing && position == [0.0; 3])
    {
        return Ok(Translation::default());
    }
    let order = from.m - to.m;
    let kz = from.kz;
    let [x, y, z] = position;
    let rho = x.hypot(y);
    let mut krho = (k * k - kz * kz).sqrt();
    if krho.im < 0.0 {
        krho = -krho;
    }
    if krho.norm_sqr() == 0.0 {
        return Err(Error::InvalidInput(
            "cylindrical cutoff requires a limiting formulation".into(),
        ));
    }
    if radial == Radial::Regular && (krho * rho).norm() < 0.5 {
        let k = Jet::<5>::variable(k, 3);
        let kz = Jet::variable(kz, 4);
        let mut transverse = (k * k - kz * kz).sqrt();
        if transverse.value.im < 0.0 {
            transverse = -transverse;
        }
        let wave = regular_harmonic(
            order,
            transverse,
            std::array::from_fn(|i| Jet::variable(position[i], i)),
            kz,
        );
        return Ok(Translation {
            value: wave.value,
            position: std::array::from_fn(|i| wave.derivative[i]),
            k: wave.derivative[3],
            kz: wave.derivative[4],
        });
    }
    let phi = y.atan2(x);
    let phase = (Complex::i() * (f64::from(order) * phi + kz * z)).exp();
    let radial_jet = cylindrical(order, krho * rho, radial)?;
    let value = radial_jet.value * phase;
    let transverse = if rho == 0.0 {
        let phase_z = (Complex::i() * kz * z).exp();
        match order {
            1 => [krho * phase_z * 0.5, Complex::i() * krho * phase_z * 0.5],
            -1 => [-krho * phase_z * 0.5, Complex::i() * krho * phase_z * 0.5],
            _ => [Complex::default(); 2],
        }
    } else {
        let dr = radial_jet.first * krho * phase;
        let azimuthal = Complex::i() * f64::from(order) * (value / rho);
        [
            dr * (x / rho) - azimuthal * (y / rho),
            dr * (y / rho) + azimuthal * (x / rho),
        ]
    };
    Ok(Translation {
        value,
        position: [transverse[0], transverse[1], Complex::i() * kz * value],
        k: radial_jet.first * rho * k / krho * phase,
        kz: -radial_jet.first * rho * kz / krho * phase + Complex::i() * z * value,
    })
}

/// Regular `J_m` harmonic as a Cartesian power series, for `|k_rho rho| < 0.5`.
/// Factoring the azimuthal polynomial keeps derivatives regular on the cylinder axis.
pub(crate) fn regular_harmonic<const N: usize>(
    order: i32,
    transverse: Jet<N>,
    r: [Jet<N>; 3],
    kz: Jet<N>,
) -> Jet<N> {
    let m = order.abs();
    let azimuth = r[0] + Complex::i() * if order < 0 { -r[1] } else { r[1] };
    let sign = if order < 0 && m % 2 == 1 { -1.0 } else { 1.0 };
    let step = -0.25 * transverse * transverse * (r[0] * r[0] + r[1] * r[1]);
    let mut term = Jet::constant((-libm::lgamma(f64::from(m + 1))).exp());
    let mut sum = term;
    for q in 1..32 {
        term = term * step / f64::from(q * (m + q));
        sum += term;
        if term.norm() <= f64::EPSILON * sum.norm() {
            break;
        }
    }
    sign * (0.5 * transverse * azimuth).powi(m) * sum * (Complex::i() * kz * r[2]).exp()
}

/// Cylindrical modes at Cartesian origins.
#[derive(Clone, Debug)]
pub struct Basis {
    /// Origin index and mode.
    pub modes: Vec<(usize, Mode)>,
    /// Expansion origins.
    pub positions: Vec<[f64; 3]>,
}
impl Basis {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.modes.is_empty() || self.positions.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::InvalidInput(
                "basis must be nonempty with finite origins".into(),
            ));
        }
        for &(index, mode) in &self.modes {
            mode.validate()?;
            if index >= self.positions.len() {
                return Err(Error::InvalidInput(
                    "basis origin index outside positions".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Cylindrical expansion residual; kz matching is a fixed basis selection.
#[derive(Debug)]
pub struct ExpansionResidual {
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    radial: Radial,
    /// Translation matrix in the supplied mode order.
    pub value: DMatrix<Complex>,
}

/// Construct a cylindrical translation matrix.
pub fn expansion(
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    radial: Radial,
) -> Result<ExpansionResidual> {
    use rayon::prelude::*;
    destination.validate()?;
    source.validate()?;
    if ks.iter().any(|&k| !finite(k) || k.norm_sqr() == 0.0) {
        return Err(Error::InvalidInput(
            "finite nonzero wavenumbers required".into(),
        ));
    }
    let mut value = DMatrix::zeros(destination.modes.len(), source.modes.len());
    value
        .as_mut_slice()
        .par_chunks_mut(destination.modes.len())
        .zip(source.modes.par_iter())
        .try_for_each(|(column, &(q, from))| {
            for (out, &(p, to)) in column.iter_mut().zip(&destination.modes) {
                let position =
                    std::array::from_fn(|a| destination.positions[p][a] - source.positions[q][a]);
                *out = translate(to, from, ks[usize::from(from.pol)], position, radial)?.value;
            }
            Ok(())
        })?;
    Ok(ExpansionResidual {
        destination,
        source,
        ks,
        radial,
        value,
    })
}
impl ExpansionResidual {
    /// Contract origin and complex wavenumber derivatives. Axial mode labels stay fixed.
    pub fn pullback(
        self,
        cotangent: &DMatrix<Complex>,
    ) -> Result<crate::basis::TranslationGradient> {
        if cotangent.shape() != (self.destination.modes.len(), self.source.modes.len())
            || cotangent.iter().any(|&g| !finite(g))
        {
            return Err(Error::InvalidInput(
                "invalid cylindrical expansion cotangent".into(),
            ));
        }
        use rayon::prelude::*;
        let empty = || crate::basis::TranslationGradient {
            destination: vec![[0.0; 3]; self.destination.positions.len()],
            source: vec![[0.0; 3]; self.source.positions.len()],
            ks: [Complex::default(); 2],
        };
        let accumulate = |mut result: crate::basis::TranslationGradient,
                          (j, &(q, from)): (usize, &(usize, Mode))|
         -> Result<_> {
            for (i, &(p, to)) in self.destination.modes.iter().enumerate() {
                let g = cotangent[(i, j)];
                if g == Complex::default() {
                    continue;
                }
                let position = std::array::from_fn(|a| {
                    self.destination.positions[p][a] - self.source.positions[q][a]
                });
                let jet = translate(
                    to,
                    from,
                    self.ks[usize::from(from.pol)],
                    position,
                    self.radial,
                )?;
                result.ks[usize::from(from.pol)] += jet.k.conj() * g;
                for axis in 0..3 {
                    let derivative = (g.conj() * jet.position[axis]).re;
                    result.destination[p][axis] += derivative;
                    result.source[q][axis] -= derivative;
                }
            }
            Ok(result)
        };
        if self.source.modes.len() < 64 {
            return self
                .source
                .modes
                .iter()
                .enumerate()
                .try_fold(empty(), accumulate);
        }
        self.source
            .modes
            .par_iter()
            .enumerate()
            .try_fold(empty, accumulate)
            .try_reduce(empty, |mut left, right| {
                for (a, b) in left
                    .destination
                    .iter_mut()
                    .chain(&mut left.source)
                    .zip(right.destination.into_iter().chain(right.source))
                {
                    for (out, value) in a.iter_mut().zip(b) {
                        *out += value;
                    }
                }
                for (out, value) in left.ks.iter_mut().zip(right.ks) {
                    *out += value;
                }
                Ok(left)
            })
    }
}

/// Periodic cylindrical coupling with exact axial/polarization selection and cached orders.
#[allow(clippy::float_cmp)] // Axial wavenumbers are exact discrete basis labels.
pub fn periodic(
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    lattice: crate::lattice::Lattice,
    eta: Complex,
) -> Result<PeriodicResidual> {
    use rayon::prelude::*;
    use std::collections::HashMap;
    destination.validate()?;
    source.validate()?;
    if ks.iter().any(|&k| !finite(k) || k == Complex::default()) {
        return Err(Error::InvalidInput(
            "finite nonzero medium wavenumbers required".into(),
        ));
    }
    let shared = ks[0] == ks[1];
    let mut indices = HashMap::new();
    let mut requests = Vec::new();
    let mut entries = Vec::new();
    for (j, &(q, from)) in source.modes.iter().enumerate() {
        for (i, &(p, to)) in destination.modes.iter().enumerate() {
            if to.kz != from.kz || to.pol != from.pol {
                continue;
            }
            let key = (
                p,
                q,
                from.kz.to_bits(),
                if shared { 0 } else { from.pol },
                from.m - to.m,
            );
            let next = requests.len();
            let index = *indices.entry(key).or_insert_with(|| {
                requests.push(key);
                next
            });
            entries.push((i, j, index));
        }
    }
    let values = requests
        .par_iter()
        .map(|&(p, q, kz, pol, m)| {
            let kz = f64::from_bits(kz);
            let k = ks[usize::from(pol)];
            let mut krho = (k * k - kz * kz).sqrt();
            if krho.im < 0.0 {
                krho = -krho;
            }
            let r: [f64; 3] =
                std::array::from_fn(|a| source.positions[q][a] - destination.positions[p][a]);
            Ok(crate::lattice::sum(
                crate::lattice::Wave::Cylindrical { m },
                krho,
                &lattice,
                [r[0], r[1], 0.0],
                eta,
            )? * (-Complex::i() * kz * r[2]).exp())
        })
        .collect::<Result<Vec<_>>>()?;
    let mut matrix = DMatrix::zeros(destination.modes.len(), source.modes.len());
    for (i, j, index) in entries {
        matrix[(i, j)] = values[index];
    }
    Ok(PeriodicResidual {
        destination,
        source,
        ks,
        lattice,
        eta,
        requests,
        value: matrix,
    })
}

/// Cylindrical periodic coupling context; equal axial labels remain fixed selections.
#[derive(Debug)]
pub struct PeriodicResidual {
    destination: Basis,
    source: Basis,
    ks: [Complex; 2],
    lattice: crate::lattice::Lattice,
    eta: Complex,
    requests: Vec<(usize, usize, u64, u8, i32)>,
    /// Periodic outgoing-to-regular coupling.
    pub value: DMatrix<Complex>,
}
impl PeriodicResidual {
    /// Consume the context and differentiate origins, medium wavenumbers and lattice geometry.
    #[allow(clippy::float_cmp)] // Fixed axial labels.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<crate::basis::PeriodicGradient> {
        use rayon::prelude::*;
        use std::collections::HashMap;
        if cotangent.shape() != self.value.shape() || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "invalid periodic expansion cotangent".into(),
            ));
        }
        let indices: HashMap<_, _> = self
            .requests
            .iter()
            .enumerate()
            .map(|(i, &key)| (key, i))
            .collect();
        let shared = self.ks[0] == self.ks[1];
        let mut g = vec![[Complex::default(); 2]; self.requests.len()];
        for (j, &(q, from)) in self.source.modes.iter().enumerate() {
            for (i, &(p, to)) in self.destination.modes.iter().enumerate() {
                if to.kz != from.kz || to.pol != from.pol {
                    continue;
                }
                g[indices[&(
                    p,
                    q,
                    from.kz.to_bits(),
                    if shared { 0 } else { from.pol },
                    from.m - to.m,
                )]][usize::from(from.pol)] += cotangent[(i, j)];
            }
        }
        let gradients = self
            .requests
            .par_iter()
            .zip(g)
            .map(|(&(p, q, kz, pol, m), g)| {
                let kz = f64::from_bits(kz);
                let k = self.ks[usize::from(pol)];
                let mut krho = (k * k - kz * kz).sqrt();
                if krho.im < 0.0 {
                    krho = -krho;
                }
                let r: [f64; 3] = std::array::from_fn(|a| {
                    self.source.positions[q][a] - self.destination.positions[p][a]
                });
                let phase = (-Complex::i() * kz * r[2]).exp();
                let jet = crate::lattice::derivatives(
                    crate::lattice::Wave::Cylindrical { m },
                    krho,
                    &self.lattice,
                    [r[0], r[1], 0.0],
                    self.eta,
                )?;
                // Equal wavenumbers share the Ewald jet, but keep independent k cotangents.
                let total: Complex = g.iter().sum();
                let scalar_g = total * phase.conj();
                let spectral = g.map(|g| (jet.k * k / krho).conj() * g * phase.conj());
                let mut gradient = crate::lattice::Gradient {
                    k: Complex::default(),
                    position: jet.position.map(|d| (scalar_g.conj() * d).re),
                    bloch: jet.bloch.map(|d| (scalar_g.conj() * d).re),
                    vectors: jet.vectors.map(|row| row.map(|d| (scalar_g.conj() * d).re)),
                };
                gradient.position[2] = (total.conj() * (-Complex::i() * kz) * phase * jet.value).re;
                Ok((p, q, spectral, gradient))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut result = crate::basis::PeriodicGradient::new(
            self.destination.positions.len(),
            self.source.positions.len(),
            self.lattice.dimension(),
        );
        for (p, q, spectral, g) in gradients {
            result.lattice(&g);
            for (g, k) in result.expansion.ks.iter_mut().zip(spectral) {
                *g += k;
            }
            for (axis, value) in g.position.into_iter().enumerate() {
                result.expansion.destination[p][axis] -= value;
                result.expansion.source[q][axis] += value;
            }
        }
        Ok(result)
    }
}
