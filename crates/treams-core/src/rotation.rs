//! Multipole rotations in the z-y-z convention, with analytic Euler-angle pullbacks.
#![allow(clippy::indexing_slicing)] // Validated mode labels and complete angular blocks.
#![allow(clippy::float_cmp)] // Exact zero rotations and fixed cylindrical mode labels.

use crate::{Complex, Error, Result, basis::Basis, cylwaves, finite, interaction::product};
use nalgebra::{DMatrix, SymmetricEigen};
use rayon::prelude::*;
use std::collections::BTreeMap;

fn ladder(l: i32, m: i32) -> f64 {
    0.5 * f64::from(l * (l + 1) - m * (m + 1)).max(0.0).sqrt()
}
fn index(l: i32, m: i32) -> usize {
    usize::try_from(l + m).unwrap_or_default()
}

/// Complete real Wigner small-d matrix, ordered -l..l.
/// A symmetric angular-momentum eigensystem avoids factorial-sum cancellation.
pub fn wigner_d(l: i32, theta: f64) -> Result<DMatrix<f64>> {
    if !(0..=128).contains(&l) || !theta.is_finite() {
        return Err(Error::InvalidInput(
            "require 0 <= l <= 128 and a finite rotation angle".into(),
        ));
    }
    let n = index(l, l) + 1;
    if theta == 0.0 {
        return Ok(DMatrix::identity(n, n));
    }
    let mut generator = DMatrix::zeros(n, n);
    for m in -l..l {
        let i = index(l, m);
        generator[(i, i + 1)] = ladder(l, m);
        generator[(i + 1, i)] = generator[(i, i + 1)];
    }
    let eigen = SymmetricEigen::new(generator);
    let vectors = eigen.eigenvectors.map(|x| Complex::new(x, 0.0));
    let mut weighted = vectors.clone();
    for j in 0..n {
        // The exact eigenvalues are integers; rounding prevents accumulated phase error.
        let phase = (-Complex::i() * theta * eigen.eigenvalues[j].round()).exp();
        for i in 0..n {
            weighted[(i, j)] *= phase;
        }
    }
    let exponential = product(&weighted, &vectors.transpose());
    Ok(DMatrix::from_fn(n, n, |i, j| {
        let power = i32::try_from(i).unwrap_or_default() - i32::try_from(j).unwrap_or_default();
        ((-Complex::i()).powi(power) * exponential[(i, j)]).re
    }))
}

#[derive(Debug)]
struct Spherical {
    destination: Basis,
    source: Basis,
    tables: BTreeMap<i32, DMatrix<f64>>,
    angles: [f64; 3],
}

/// A rotation and the angular blocks needed by its reverse pass.
#[derive(Debug)]
pub struct RotationResidual {
    rows: Vec<i32>,
    columns: Vec<i32>,
    spherical: Option<Spherical>,
    /// Rotation from the source to destination multipole coefficients.
    pub value: DMatrix<Complex>,
}

/// Spherical Euler rotation; distinct origins, degrees and polarizations decouple.
pub fn spherical(destination: Basis, source: Basis, angles: [f64; 3]) -> Result<RotationResidual> {
    destination.validate()?;
    source.validate()?;
    if angles.iter().any(|a| !a.is_finite()) {
        return Err(Error::InvalidInput("rotation angles must be finite".into()));
    }
    let mut tables = BTreeMap::new();
    for &(_, mode) in &destination.modes {
        if let std::collections::btree_map::Entry::Vacant(e) = tables.entry(mode.l) {
            e.insert(wigner_d(mode.l, angles[1])?);
        }
    }
    let value = DMatrix::from_fn(destination.modes.len(), source.modes.len(), |i, j| {
        let (p, to) = destination.modes[i];
        let (q, from) = source.modes[j];
        if p != q || to.l != from.l || to.pol != from.pol {
            return Complex::default();
        }
        let d = &tables[&to.l];
        (-Complex::i() * (f64::from(to.m) * angles[0] + f64::from(from.m) * angles[2])).exp()
            * d[(index(to.l, to.m), index(from.l, from.m))]
    });
    Ok(RotationResidual {
        rows: destination.modes.iter().map(|(_, m)| m.m).collect(),
        columns: source.modes.iter().map(|(_, m)| m.m).collect(),
        spherical: Some(Spherical {
            destination,
            source,
            tables,
            angles,
        }),
        value,
    })
}

/// Cylindrical rotation about z; axial labels and the transverse plane stay fixed.
pub fn cylindrical(
    destination: &cylwaves::Basis,
    source: &cylwaves::Basis,
    angles: [f64; 3],
) -> Result<RotationResidual> {
    destination.validate()?;
    source.validate()?;
    if angles.iter().any(|a| !a.is_finite()) || angles[1] != 0.0 {
        return Err(Error::InvalidInput(
            "cylindrical rotation requires finite angles and theta=0".into(),
        ));
    }
    let value = DMatrix::from_fn(destination.modes.len(), source.modes.len(), |i, j| {
        let (p, to) = destination.modes[i];
        let (q, from) = source.modes[j];
        if p != q || to.kz != from.kz || to.m != from.m || to.pol != from.pol {
            Complex::default()
        } else {
            (-Complex::i() * f64::from(to.m) * (angles[0] + angles[2])).exp()
        }
    });
    Ok(RotationResidual {
        rows: destination.modes.iter().map(|(_, m)| m.m).collect(),
        columns: source.modes.iter().map(|(_, m)| m.m).collect(),
        spherical: None,
        value,
    })
}

impl RotationResidual {
    /// Euler-angle cotangents. Cylindrical theta is a fixed zero constraint.
    pub fn pullback(self, g: &DMatrix<Complex>) -> Result<[f64; 3]> {
        if g.shape() != self.value.shape() || g.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid rotation cotangent".into()));
        }
        let mut result = [0.0; 3];
        for (j, &m) in self.columns.iter().enumerate() {
            for (i, &mu) in self.rows.iter().enumerate() {
                let paired = g[(i, j)].conj() * (-Complex::i()) * self.value[(i, j)];
                result[0] += f64::from(mu) * paired.re;
                result[2] += f64::from(m) * paired.re;
                if let Some(s) = &self.spherical {
                    let (p, to) = s.destination.modes[i];
                    let (q, from) = s.source.modes[j];
                    if p != q || to.l != from.l || to.pol != from.pol {
                        continue;
                    }
                    let l = to.l;
                    let d = &s.tables[&l];
                    let column = index(l, m);
                    let derivative = if mu < l {
                        ladder(l, mu) * d[(index(l, mu + 1), column)]
                    } else {
                        0.0
                    } - if mu > -l {
                        ladder(l, mu - 1) * d[(index(l, mu - 1), column)]
                    } else {
                        0.0
                    };
                    let phase = (-Complex::i()
                        * (f64::from(mu) * s.angles[0] + f64::from(m) * s.angles[2]))
                        .exp();
                    result[1] += (g[(i, j)].conj() * phase * derivative).re;
                }
            }
        }
        Ok(result)
    }
}

// Jacobi recurrence (DLMF 18.9.1-2) evaluates one Wigner element in O(l) time and O(1)
// storage. Half-angle powers preserve off-diagonal limits when cos(theta) rounds
// to one. Symmetries keep both Jacobi parameters nonnegative.
fn wigner_element<const N: usize>(
    l: i32,
    mut m: i32,
    mut k: i32,
    theta: crate::jet::Jet<N>,
) -> crate::jet::Jet<N> {
    use crate::jet::Jet;
    let parity = |n: i32| if n % 2 == 0 { 1.0 } else { -1.0 };
    let mut sign = 1.0;
    if k.abs() > m.abs() {
        sign *= parity(m - k);
        std::mem::swap(&mut m, &mut k);
    }
    if m < 0 {
        sign *= parity(m - k);
        m = -m;
        k = -k;
    }
    let (a, b) = (f64::from(m - k), f64::from(m + k));
    let x = theta.map(theta.value.cos(), -theta.value.sin());
    let half = theta * 0.5;
    let sine = half.map(half.value.sin(), half.value.cos());
    let cosine = half.map(half.value.cos(), -half.value.sin());
    let normalization = (0.5
        * (libm::lgamma(f64::from(l + m + 1)) + libm::lgamma(f64::from(l - m + 1))
            - libm::lgamma(f64::from(l + k + 1))
            - libm::lgamma(f64::from(l - k + 1))))
    .exp();
    let n = l - m;
    let mut prev = Jet::constant(1.0);
    let mut polynomial = if n == 0 {
        prev
    } else {
        0.5 * ((a - b) + (a + b + 2.0) * x)
    };
    for j in 2..=n {
        let j = f64::from(j);
        let t = 2.0 * j + a + b;
        let next = ((t - 1.0) * (t * (t - 2.0) * x + a * a - b * b) * polynomial
            - 2.0 * (j + a - 1.0) * (j + b - 1.0) * t * prev)
            * (1.0 / (2.0 * j * (j + a + b) * (t - 2.0)));
        prev = polynomial;
        polynomial = next;
    }
    sign * parity(m - k) * normalization * sine.powi(m - k) * cosine.powi(m + k) * polynomial
}

/// Individual Wigner small-d element at a real or complex angle, without a matrix.
pub fn wigner_small(l: i32, m: i32, k: i32, theta: Complex) -> Result<Complex> {
    if !(0..=128).contains(&l) || !finite(theta) {
        return Err(Error::InvalidInput(
            "require 0 <= l <= 128 and finite Wigner angle".into(),
        ));
    }
    if m.unsigned_abs() > l.unsigned_abs() || k.unsigned_abs() > l.unsigned_abs() {
        return Ok(Complex::default());
    }
    let value = wigner_element(l, m, k, crate::jet::Jet::<0>::constant(theta)).value;
    if !finite(value) {
        return Err(Error::SpecialFunction("nonfinite Wigner result".into()));
    }
    Ok(value)
}

/// Wigner D element in the z-y-z convention, allowing complex Euler angles.
pub fn wigner(l: i32, m: i32, k: i32, angles: [Complex; 3]) -> Result<Complex> {
    if angles.iter().any(|&z| !finite(z)) {
        return Err(Error::InvalidInput("Wigner angles must be finite".into()));
    }
    let value = (-Complex::i() * (f64::from(m) * angles[0] + f64::from(k) * angles[2])).exp()
        * wigner_small(l, m, k, angles[1])?;
    if !finite(value) {
        return Err(Error::SpecialFunction("nonfinite Wigner result".into()));
    }
    Ok(value)
}

/// Owned Wigner labels and broadcast Euler angles; reverse uses local chain rules.
#[derive(Debug)]
pub struct WignerResidual {
    labels: Vec<[i32; 3]>,
    angles: [Vec<Complex>; 3],
    size: usize,
}

impl WignerResidual {
    fn element(&self, i: usize) -> ([i32; 3], [Complex; 3]) {
        (
            self.labels[if self.labels.len() == 1 { 0 } else { i }],
            std::array::from_fn(|axis| {
                self.angles[axis][if self.angles[axis].len() == 1 { 0 } else { i }]
            }),
        )
    }

    /// Real-Hermitian pullback to each complex Euler angle, reducing scalar inputs.
    pub fn pullback(self, cotangent: &[Complex]) -> Result<[Vec<Complex>; 3]> {
        if cotangent.len() != self.size || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "Wigner cotangent must be finite and match output".into(),
            ));
        }
        let evaluate = |(i, &g): (usize, &Complex)| {
            let ([l, m, k], angles) = self.element(i);
            if g == Complex::default()
                || m.unsigned_abs() > l.unsigned_abs()
                || k.unsigned_abs() > l.unsigned_abs()
            {
                return Ok([Complex::default(); 3]);
            }
            let angles: [crate::jet::Jet<3>; 3] =
                std::array::from_fn(|axis| crate::jet::Jet::variable(angles[axis], axis));
            let value = (-Complex::i() * (f64::from(m) * angles[0] + f64::from(k) * angles[2]))
                .exp()
                * wigner_element(l, m, k, angles[1]);
            if !value.finite() {
                return Err(Error::SpecialFunction("nonfinite Wigner derivative".into()));
            }
            Ok(value.derivative.map(|derivative| g * derivative.conj()))
        };
        let gradients: Vec<[Complex; 3]> = if self.size >= 1024 {
            cotangent
                .par_iter()
                .enumerate()
                .map(evaluate)
                .collect::<Result<_>>()?
        } else {
            cotangent
                .iter()
                .enumerate()
                .map(evaluate)
                .collect::<Result<_>>()?
        };
        Ok(std::array::from_fn(|axis| {
            if self.angles[axis].len() == 1 {
                vec![gradients.iter().map(|g| g[axis]).sum()]
            } else {
                gradients.iter().map(|g| g[axis]).collect()
            }
        }))
    }
}

/// Broadcast Wigner values with all three Euler angles differentiable.
pub fn wigner_array(
    labels: Vec<[i32; 3]>,
    angles: [Vec<Complex>; 3],
) -> Result<(Vec<Complex>, WignerResidual)> {
    let sizes = [
        labels.len(),
        angles[0].len(),
        angles[1].len(),
        angles[2].len(),
    ];
    let size = if sizes.contains(&0) {
        0
    } else {
        sizes.into_iter().max().unwrap_or_default()
    };
    if sizes.iter().any(|&n| n != 1 && n != size) {
        return Err(Error::InvalidInput(
            "Wigner arrays must have equal lengths or scalar inputs".into(),
        ));
    }
    let residual = WignerResidual {
        labels,
        angles,
        size,
    };
    let evaluate = |i| {
        let ([l, m, k], a) = residual.element(i);
        wigner(l, m, k, a)
    };
    let values = if size >= 1024 {
        (0..size)
            .into_par_iter()
            .map(evaluate)
            .collect::<Result<_>>()?
    } else {
        (0..size).map(evaluate).collect::<Result<_>>()?
    };
    Ok((values, residual))
}
