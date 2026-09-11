//! Multipole rotations in the z-y-z convention, with analytic Euler-angle pullbacks.
#![allow(clippy::indexing_slicing)] // Validated mode labels and complete angular blocks.
#![allow(clippy::float_cmp)] // Exact zero rotations and fixed cylindrical mode labels.

use crate::{Complex, Error, Result, basis::Basis, cylwaves, finite, interaction::product};
use nalgebra::{DMatrix, SymmetricEigen};
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
