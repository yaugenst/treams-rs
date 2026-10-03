//! Multipole rotations in the z-y-z convention, with analytic Euler-angle pullbacks.
//!
//! [`sw_rotate`] and [`cw_rotate`] give the coefficient of one mode pair; [`sw_rotation`]
//! and [`cw_rotation`] give the matrix between two bases. Both matrices return one
//! [`RotationResidual`], whose pullback gives the cotangents of the Euler angles
//! `[phi, theta, psi]`, so the two families share this module instead of living in
//! [`crate::sw`] and [`crate::cw`].
//!
//! Upstream: `treams.sw.rotate`, `treams.cw.rotate` and the `treams.Rotate` operator.
#![allow(clippy::indexing_slicing)] // Validated mode labels and complete angular blocks.
#![allow(clippy::float_cmp)] // Exact zero rotations and exact axial labels.

use crate::{
    Complex, Error, MAX_DEGREE, Result,
    basis::ModeLabel,
    cw,
    numerics::{self, finite, label_bits},
    special::{index, ladder, pol_index, wigner_d, wigner_small_d_matrix},
    sw,
};
use nalgebra::DMatrix;
use std::collections::{BTreeMap, HashMap};

/// Entries `(row, column)` whose coupling keys agree, in column-major order; all
/// other entries of a rotation vanish.
fn coupled<K: Eq + std::hash::Hash>(rows: &[K], columns: &[K]) -> Vec<(usize, usize)> {
    let mut groups: HashMap<&K, Vec<usize>> = HashMap::new();
    for (i, key) in rows.iter().enumerate() {
        groups.entry(key).or_default().push(i);
    }
    columns
        .iter()
        .enumerate()
        .flat_map(|(j, key)| groups.get(key).into_iter().flatten().map(move |&i| (i, j)))
        .collect()
}

/// The Wigner small-d blocks and Euler phases of a spherical rotation.
#[derive(Debug)]
struct Spherical {
    /// Degree of each row.
    degrees: Vec<i32>,
    tables: BTreeMap<i32, DMatrix<f64>>,
    /// Row phases `exp(-i mu phi)` and column phases `exp(-i m psi)`.
    phases: [Vec<Complex>; 2],
}

/// What [`sw_rotation`] and [`cw_rotation`] save for their pullback: the rotation
/// matrix, its coupled entries and, for spherical bases, the angular blocks.
///
/// The pullback adds the entries in order on the calling thread, so its result does not
/// depend on the thread count.
#[derive(Debug)]
pub struct RotationResidual {
    rows: Vec<i32>,
    columns: Vec<i32>,
    /// The coupled entries; all others vanish.
    entries: Vec<(usize, usize)>,
    spherical: Option<Spherical>,
    value: DMatrix<Complex>,
}

/// Spherical Euler rotation; distinct positions, degrees and polarizations decouple.
///
/// Upstream: the `treams.Rotate` operator (`treams.operators.rotate`) between spherical
/// bases.
pub fn sw_rotation(
    destination: &sw::Basis,
    source: &sw::Basis,
    angles: [f64; 3],
) -> Result<RotationResidual> {
    destination.validate()?;
    source.validate()?;
    if angles.iter().any(|a| !a.is_finite()) {
        return Err(Error::InvalidInput("rotation angles must be finite".into()));
    }
    let mut tables = BTreeMap::new();
    for &(_, mode) in &destination.modes {
        if let std::collections::btree_map::Entry::Vacant(e) = tables.entry(mode.l) {
            e.insert(wigner_small_d_matrix(mode.l, angles[1])?);
        }
    }
    let key = |basis: &sw::Basis| -> Vec<_> {
        basis
            .modes
            .iter()
            .map(|&(pidx, mode)| (pidx, mode.l, mode.pol))
            .collect()
    };
    let entries = coupled(&key(destination), &key(source));
    let orders = |basis: &sw::Basis| -> Vec<_> { basis.modes.iter().map(|(_, m)| m.m).collect() };
    let (rows, columns) = (orders(destination), orders(source));
    let phase = |m: i32, angle: f64| (-Complex::i() * f64::from(m) * angle).exp();
    let spherical = Spherical {
        degrees: destination.modes.iter().map(|(_, m)| m.l).collect(),
        tables,
        phases: [
            rows.iter().map(|&mu| phase(mu, angles[0])).collect(),
            columns.iter().map(|&m| phase(m, angles[2])).collect(),
        ],
    };
    let mut value = numerics::zeros(rows.len(), columns.len())?;
    for &(i, j) in &entries {
        let l = spherical.degrees[i];
        let d = spherical.tables[&l][(index(l, rows[i]), index(l, columns[j]))];
        value[(i, j)] = spherical.phases[0][i] * spherical.phases[1][j] * d;
    }
    Ok(RotationResidual {
        rows,
        columns,
        entries,
        spherical: Some(spherical),
        value,
    })
}

/// Cylindrical rotation about z; axial labels and the transverse plane stay fixed.
///
/// Upstream: the `treams.Rotate` operator (`treams.operators.rotate`) between cylindrical
/// bases.
pub fn cw_rotation(
    destination: &cw::Basis,
    source: &cw::Basis,
    angles: [f64; 3],
) -> Result<RotationResidual> {
    destination.validate()?;
    source.validate()?;
    if angles.iter().any(|a| !a.is_finite()) || angles[1] != 0.0 {
        return Err(Error::InvalidInput(
            "cylindrical rotation requires finite angles and theta=0".into(),
        ));
    }
    let key = |basis: &cw::Basis| -> Vec<_> {
        basis
            .modes
            .iter()
            .map(|&(pidx, mode)| (pidx, label_bits(mode.kz), mode.m, mode.pol))
            .collect()
    };
    let entries = coupled(&key(destination), &key(source));
    let orders = |basis: &cw::Basis| -> Vec<_> { basis.modes.iter().map(|(_, m)| m.m).collect() };
    let (rows, columns) = (orders(destination), orders(source));
    let mut value = numerics::zeros(rows.len(), columns.len())?;
    for &(i, j) in &entries {
        value[(i, j)] = (-Complex::i() * f64::from(rows[i]) * (angles[0] + angles[2])).exp();
    }
    Ok(RotationResidual {
        rows,
        columns,
        entries,
        spherical: None,
        value,
    })
}

impl RotationResidual {
    /// The rotation from the source to the destination multipole coefficients, which
    /// the pullback reads.
    #[must_use]
    pub const fn value(&self) -> &DMatrix<Complex> {
        &self.value
    }

    /// The shape of the rotation: destination and source mode counts.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.value.shape()
    }

    /// Euler-angle cotangents. Cylindrical theta is a fixed zero constraint.
    pub fn pullback(self, cotangent: &DMatrix<Complex>) -> Result<[f64; 3]> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid rotation cotangent".into()));
        }
        let mut result = [0.0; 3];
        for &(i, j) in &self.entries {
            let (mu, m) = (self.rows[i], self.columns[j]);
            let g = cotangent[(i, j)].conj();
            // `d/dphi D = -i mu D` and `d/dpsi D = -i m D`.
            let paired = (g * (-Complex::i()) * self.value[(i, j)]).re;
            result[0] += f64::from(mu) * paired;
            result[2] += f64::from(m) * paired;
            if let Some(s) = &self.spherical {
                // `d/dtheta d_(mu m) = c(mu) d_(mu+1, m) - c(mu-1) d_(mu-1, m)` with the
                // ladder coefficients `c`.
                let l = s.degrees[i];
                let d = &s.tables[&l];
                let column = index(l, m);
                let raised = if mu < l {
                    ladder(l, mu) * d[(index(l, mu + 1), column)]
                } else {
                    0.0
                };
                let lowered = if mu > -l {
                    ladder(l, mu - 1) * d[(index(l, mu - 1), column)]
                } else {
                    0.0
                };
                let phase = s.phases[0][i] * s.phases[1][j];
                result[1] += (g * phase * (raised - lowered)).re;
            }
        }
        Ok(result)
    }
}

/// Rotation coefficient of one spherical mode pair.
///
/// For the Euler angles `[phi, theta, psi]`, the coefficient from `source` to
/// `destination` is the Wigner D-matrix element `D^l_(m' m)(phi, theta, psi)`
/// ([`wigner_d`]) for equal degrees and polarizations, and zero otherwise; `m'` is the
/// order of `destination` and `m` that of `source`.
///
/// Upstream: `treams.sw.rotate`.
/// Differences: modes with `l = 0`, `l > 128` or `|m| > l` give an error, and so do
/// non-finite angles at equal degrees and polarizations; treams returns a value or NaN for
/// them, for example 1 at `l = 0` and zero for `|m| > l`.
#[inline]
pub fn sw_rotate(
    destination: sw::Mode,
    source: sw::Mode,
    [phi, theta, psi]: [f64; 3],
) -> Result<Complex> {
    destination.validate()?;
    source.validate()?;
    if destination.l != source.l || destination.pol != source.pol {
        Ok(Complex::default())
    } else {
        wigner_d(
            destination.l,
            destination.m,
            source.m,
            [phi.into(), theta.into(), psi.into()],
        )
    }
}

/// Rotation coefficient of one cylindrical mode pair about the z axis.
///
/// For the angle `phi`, the coefficient from `(qz, m, q)` to `(kz, mu, p)` is
/// `exp(-i m phi)` for equal modes and zero otherwise. The polarizations `p` and `q`
/// must be 0 or 1, the axial wavenumbers and `phi` finite, and the orders satisfy
/// `|mu|, |m| <= 128` ([`MAX_DEGREE`]). The labels are `i64` so that every integer label
/// gets these checks before its conversion.
///
/// Upstream: `treams.cw.rotate`.
/// Differences: treams accepts orders above 128 and non-finite arguments, where this
/// function gives an error.
#[inline]
pub fn cw_rotate(kz: f64, mu: i64, p: i64, qz: f64, m: i64, q: i64, phi: f64) -> Result<Complex> {
    let p = pol_index(p)?;
    let q = pol_index(q)?;
    let bound = u64::from(MAX_DEGREE.unsigned_abs());
    if !kz.is_finite()
        || !qz.is_finite()
        || !phi.is_finite()
        || mu.unsigned_abs() > bound
        || m.unsigned_abs() > bound
    {
        return Err(Error::InvalidInput(
            // 128 is MAX_DEGREE.
            "finite axial/angle arguments and |orders| <= 128 required".into(),
        ));
    }
    if kz != qz || mu != m || p != q {
        Ok(Complex::default())
    } else {
        let order =
            f64::from(i32::try_from(m).map_err(|_| Error::InvalidInput("invalid order".into()))?);
        let (sin, cos) = (-order * phi).sin_cos();
        Ok(Complex::new(cos, sin))
    }
}
