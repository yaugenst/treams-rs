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
    saved::{Reader, SavedState, Writer, invalid},
    special::{index, ladder, pol_index, wigner_d, wigner_small_d_matrix},
    sw,
};
use nalgebra::DMatrix;
use std::collections::{BTreeMap, BTreeSet, HashMap};

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

fn sw_keys(modes: &[(usize, sw::Mode)]) -> Vec<(usize, i32, u8)> {
    modes
        .iter()
        .map(|&(pidx, mode)| (pidx, mode.l, mode.pol))
        .collect()
}

fn cw_keys(modes: &[(usize, cw::Mode)]) -> Vec<(usize, u64, i32, u8)> {
    modes
        .iter()
        .map(|&(pidx, mode)| (pidx, label_bits(mode.kz), mode.m, mode.pol))
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
    let entries = coupled(&sw_keys(&destination.modes), &sw_keys(&source.modes));
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
    let entries = coupled(&cw_keys(&destination.modes), &cw_keys(&source.modes));
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
    /// Saved-state size from spherical mode labels, without evaluating a rotation.
    pub fn sw_state_size(
        destination: &[(usize, sw::Mode)],
        source: &[(usize, sw::Mode)],
    ) -> Result<usize> {
        for &(_, mode) in destination.iter().chain(source) {
            mode.validate()?;
        }
        let degrees = destination.iter().map(|(_, mode)| mode.l).collect();
        state_size(
            destination.len(),
            source.len(),
            coupled_count(&sw_keys(destination), &sw_keys(source))?,
            Some(&degrees),
        )
    }

    /// Saved-state size from cylindrical mode labels, without evaluating a rotation.
    pub fn cw_state_size(
        destination: &[(usize, cw::Mode)],
        source: &[(usize, cw::Mode)],
    ) -> Result<usize> {
        for &(_, mode) in destination.iter().chain(source) {
            mode.validate()?;
        }
        state_size(
            destination.len(),
            source.len(),
            coupled_count(&cw_keys(destination), &cw_keys(source))?,
            None,
        )
    }

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

    /// Contract the Euler-angle derivatives with one real direction. Cylindrical
    /// theta remains a fixed zero constraint and contributes no derivative.
    pub fn pushforward(&self, angles: [f64; 3]) -> Result<DMatrix<Complex>> {
        if angles.iter().any(|a| !a.is_finite()) {
            return Err(Error::InvalidInput(
                "rotation tangents must be finite".into(),
            ));
        }
        let mut result = numerics::zeros(self.shape().0, self.shape().1)?;
        for &(i, j) in &self.entries {
            let (mu, m) = (self.rows[i], self.columns[j]);
            result[(i, j)] = -Complex::i()
                * self.value[(i, j)]
                * (f64::from(mu) * angles[0] + f64::from(m) * angles[2])
                + self.theta_derivative(i, j) * angles[1];
        }
        Ok(result)
    }

    /// The shared ladder rule used by the JVP and VJP.
    fn theta_derivative(&self, i: usize, j: usize) -> Complex {
        let Some(s) = &self.spherical else {
            return Complex::default();
        };
        let (mu, m) = (self.rows[i], self.columns[j]);
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
        s.phases[0][i] * s.phases[1][j] * (raised - lowered)
    }

    /// Euler-angle cotangents. Cylindrical theta is a fixed zero constraint.
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<[f64; 3]> {
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
            result[1] += (g * self.theta_derivative(i, j)).re;
        }
        Ok(result)
    }
}

fn coupled_count<K: Eq + std::hash::Hash>(rows: &[K], columns: &[K]) -> Result<usize> {
    let mut counts = HashMap::new();
    for key in rows {
        *counts.entry(key).or_insert(0_usize) += 1;
    }
    columns.iter().try_fold(0_usize, |total, key| {
        total
            .checked_add(*counts.get(key).unwrap_or(&0))
            .ok_or_else(invalid)
    })
}

fn bytes(count: usize, stride: usize) -> Result<usize> {
    count.checked_mul(stride).ok_or_else(invalid)
}

fn table_dimension(degree: i32) -> Result<usize> {
    if !(1..=MAX_DEGREE).contains(&degree) {
        return Err(invalid());
    }
    usize::try_from(2 * degree + 1).map_err(|_| invalid())
}

fn state_size(
    rows: usize,
    columns: usize,
    entries: usize,
    degrees: Option<&BTreeSet<i32>>,
) -> Result<usize> {
    let labels = rows.checked_add(columns).ok_or_else(invalid)?;
    let matrix = bytes(bytes(rows, columns)?, 16)?;
    let mut parts = vec![25, bytes(labels, 4)?, bytes(entries, 16)?, matrix];
    if let Some(degrees) = degrees {
        parts.extend([bytes(rows, 4)?, bytes(labels, 16)?, 8]);
        for &degree in degrees {
            let dimension = table_dimension(degree)?;
            parts.extend([4, bytes(bytes(dimension, dimension)?, 8)?]);
        }
    }
    parts.into_iter().try_fold(0_usize, |total, part| {
        total.checked_add(part).ok_or_else(invalid)
    })
}

/// Check the entire byte layout before any residual allocation. Wigner block sizes
/// are determined by their degree labels, so sparse blocks retain their native shape.
fn check_state_layout(state: &[u8]) -> Result<()> {
    let mut reader = Reader::new(state);
    let spherical = match reader.byte()? {
        0 => false,
        1 => true,
        _ => return Err(invalid()),
    };
    let (rows, columns, entries) = (reader.usize()?, reader.usize()?, reader.usize()?);
    let labels = rows.checked_add(columns).ok_or_else(invalid)?;
    reader.raw(bytes(labels, 4)?)?;
    reader.raw(bytes(entries, 16)?)?;
    reader.raw(bytes(bytes(rows, columns)?, 16)?)?;
    if spherical {
        reader.raw(bytes(rows, 4)?)?;
        reader.raw(bytes(labels, 16)?)?;
        for _ in 0..reader.count(4)? {
            let dimension = table_dimension(reader.i32()?)?;
            reader.raw(bytes(bytes(dimension, dimension)?, 8)?)?;
        }
    }
    reader.finish()
}

impl SavedState for RotationResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let degrees = self
            .spherical
            .as_ref()
            .map(|s| s.tables.keys().copied().collect());
        let mut writer = Writer::new(state_size(
            self.rows.len(),
            self.columns.len(),
            self.entries.len(),
            degrees.as_ref(),
        )?);
        writer.byte(u8::from(self.spherical.is_some()));
        writer.usize(self.rows.len());
        writer.usize(self.columns.len());
        writer.usize(self.entries.len());
        for &order in self.rows.iter().chain(&self.columns) {
            writer.i32(order);
        }
        for &(row, column) in &self.entries {
            writer.usize(row);
            writer.usize(column);
        }
        for &value in &self.value {
            writer.complex(value);
        }
        if let Some(s) = &self.spherical {
            for &degree in &s.degrees {
                writer.i32(degree);
            }
            for &phase in s.phases.iter().flatten() {
                writer.complex(phase);
            }
            writer.usize(s.tables.len());
            for (&degree, table) in &s.tables {
                writer.i32(degree);
                for &value in table {
                    writer.f64(value);
                }
            }
        }
        Ok(writer.finish())
    }

    fn from_state(state: &[u8]) -> Result<Self> {
        check_state_layout(state)?;
        let mut reader = Reader::new(state);
        let spherical = reader.byte()? == 1;
        let (nrows, ncolumns, nentries) = (reader.usize()?, reader.usize()?, reader.usize()?);
        let rows = read_orders(&mut reader, nrows)?;
        let columns = read_orders(&mut reader, ncolumns)?;
        let entries = (0..nentries)
            .map(|_| {
                let (i, j) = (reader.usize()?, reader.usize()?);
                if i >= nrows || j >= ncolumns {
                    return Err(invalid());
                }
                Ok((i, j))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut value = numerics::zeros(nrows, ncolumns)?;
        for z in &mut value {
            *z = reader.complex()?;
        }
        let spherical = if spherical {
            let degrees = (0..nrows)
                .map(|_| reader.i32())
                .collect::<Result<Vec<_>>>()?;
            let mut phases = [Vec::with_capacity(nrows), Vec::with_capacity(ncolumns)];
            for (phase, length) in phases.iter_mut().zip([nrows, ncolumns]) {
                for _ in 0..length {
                    phase.push(reader.complex()?);
                }
            }
            let mut tables = BTreeMap::new();
            for _ in 0..reader.usize()? {
                let degree = reader.i32()?;
                let dimension = table_dimension(degree)?;
                let values = (0..dimension * dimension)
                    .map(|_| reader.f64())
                    .collect::<Result<Vec<_>>>()?;
                let table = DMatrix::from_vec(dimension, dimension, values);
                if tables.insert(degree, table).is_some() {
                    return Err(invalid());
                }
            }
            if degrees
                .iter()
                .zip(&rows)
                .any(|(l, m)| !tables.contains_key(l) || m.abs() > *l)
                || entries.iter().any(|&(i, j)| columns[j].abs() > degrees[i])
            {
                return Err(invalid());
            }
            Some(Spherical {
                degrees,
                tables,
                phases,
            })
        } else {
            None
        };
        reader.finish()?;
        Ok(Self {
            rows,
            columns,
            entries,
            spherical,
            value,
        })
    }
}

fn read_orders(reader: &mut Reader<'_>, count: usize) -> Result<Vec<i32>> {
    (0..count)
        .map(|_| {
            let order = reader.i32()?;
            if order.unsigned_abs() > MAX_DEGREE.unsigned_abs() {
                return Err(invalid());
            }
            Ok(order)
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn check_saved_rotation(residual: &RotationResidual, size: usize) {
        let state = residual.save_state().unwrap();
        assert_eq!(state.len(), size);
        let restored = RotationResidual::from_state(&state).unwrap();
        assert_eq!(restored.save_state().unwrap(), state);
        assert_eq!(restored.value(), residual.value());
        let direction = [0.3, -0.4, 0.2];
        assert_eq!(
            restored.pushforward(direction).unwrap(),
            residual.pushforward(direction).unwrap()
        );
        let cotangent = crate::test_support::patterned(residual.shape().0, residual.shape().1, 0.3);
        assert_eq!(
            restored.pullback(&cotangent).unwrap(),
            residual.pullback(&cotangent).unwrap()
        );
        for length in 0..state.len() {
            assert!(RotationResidual::from_state(&state[..length]).is_err());
        }
        let mut malformed = state.clone();
        malformed[1..9].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(RotationResidual::from_state(&malformed).is_err());
        malformed = state;
        malformed.push(0);
        assert!(RotationResidual::from_state(&malformed).is_err());
    }

    #[test]
    fn saved_spherical_rotation_preserves_sparse_blocks_and_derivatives() {
        let basis = |labels: &[(usize, i32, i32, u8)]| sw::Basis {
            modes: labels
                .iter()
                .map(|&(position, l, m, pol)| (position, sw::Mode { l, m, pol }))
                .collect(),
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0]],
        };
        let destination = basis(&[(0, 1, 0, 0), (1, 2, 2, 1), (0, 3, -2, 0)]);
        let source = basis(&[(0, 1, 1, 0), (1, 2, -1, 1), (0, 3, 1, 1), (0, 1, -1, 0)]);
        let residual = sw_rotation(&destination, &source, [0.3, 0.8, -0.4]).unwrap();
        let size = RotationResidual::sw_state_size(&destination.modes, &source.modes).unwrap();
        check_saved_rotation(&residual, size);
        let state = residual.save_state().unwrap();
        let restored = RotationResidual::from_state(&state).unwrap();
        assert_eq!(restored.entries, residual.entries);
        let spherical = restored.spherical.unwrap();
        assert_eq!(
            spherical.tables.keys().copied().collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(spherical.tables[&3].shape(), (7, 7));
    }

    #[test]
    fn saved_cylindrical_rotation_preserves_sparse_couplings_and_derivatives() {
        let basis = |labels: &[(usize, f64, i32, u8)]| cw::Basis {
            modes: labels
                .iter()
                .map(|&(position, kz, m, pol)| (position, cw::Mode { kz, m, pol }))
                .collect(),
            positions: vec![[0.0; 3]],
        };
        let destination = basis(&[(0, 0.0, -1, 0), (0, 0.2, 2, 1)]);
        let source = basis(&[(0, -0.0, -1, 0), (0, 0.2, 1, 1), (0, 0.2, 2, 1)]);
        let residual = cw_rotation(&destination, &source, [0.3, 0.0, -0.4]).unwrap();
        let size = RotationResidual::cw_state_size(&destination.modes, &source.modes).unwrap();
        check_saved_rotation(&residual, size);
        let restored = RotationResidual::from_state(&residual.save_state().unwrap()).unwrap();
        assert_eq!(restored.entries, [(0, 0), (1, 2)]);
        assert!(restored.spherical.is_none());
    }
}
