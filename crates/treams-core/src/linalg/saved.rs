//! Fixed numerical layouts for invocation-owned JAX state. Dimensions are stored
//! as u64; all other data are IEEE f64 components in column-major order.
//! treams-rs extension.

use std::sync::OnceLock;

use faer::{Mat, MatMut, MatRef, perm::Perm};

use super::{EigResidual, Equilibration, Lu, SolveResidual, SvdvalsResidual, faer_zeros, view};
use crate::{
    Complex, Error, Result, numerics,
    saved::{Reader, SavedState, Writer, invalid, read_complex, read_matrix, total},
};

fn matrix_size(rows: usize, columns: usize) -> Result<usize> {
    if rows == 0 || columns == 0 {
        return Err(invalid());
    }
    rows.checked_mul(columns)
        .and_then(|count| count.checked_mul(16))
        .ok_or_else(invalid)
}

impl Lu {
    /// Bytes for factors, forward permutation, and canonical row/column scales.
    pub(crate) fn state_size(n: usize) -> Result<usize> {
        total(&[matrix_size(n, n)?, n.checked_mul(24).ok_or_else(invalid)?])
    }

    /// Write numerical factors without refactorizing their operator.
    pub(crate) fn write_state(&self, writer: &mut Writer) {
        write_lu(writer, self);
    }

    /// Restore numerical factors; the containing record provides the dimension.
    pub(crate) fn read_state(reader: &mut Reader<'_>, n: usize) -> Result<Self> {
        read_lu(reader, n)
    }
}

fn write_matrix(writer: &mut Writer, matrix: MatRef<'_, Complex>) {
    for j in 0..matrix.ncols() {
        for i in 0..matrix.nrows() {
            writer.complex(matrix[(i, j)]);
        }
    }
}

fn read_values(reader: &mut Reader<'_>, mut matrix: MatMut<'_, Complex>) -> Result<()> {
    for j in 0..matrix.ncols() {
        for i in 0..matrix.nrows() {
            matrix[(i, j)] = read_complex(reader)?;
        }
    }
    Ok(())
}

fn read_faer(reader: &mut Reader<'_>, rows: usize, columns: usize) -> Result<Mat<Complex>> {
    let mut matrix = faer_zeros(rows, columns)?;
    read_values(reader, matrix.as_mut())?;
    Ok(matrix)
}

/// Canonical LU state uses identity scales when equilibration was unnecessary.
/// The inverse permutation is derived on restore, rather than stored twice.
fn write_lu(writer: &mut Writer, lu: &Lu) {
    write_matrix(writer, view(&lu.factors));
    for &index in lu.permutation.arrays().0 {
        writer.usize(index);
    }
    let n = lu.factors.nrows();
    for column in [false, true] {
        for i in 0..n {
            writer.f64(lu.equilibration.as_ref().map_or(1.0, |scales| {
                if column {
                    scales.column[i]
                } else {
                    scales.row[i]
                }
            }));
        }
    }
}

fn read_lu(reader: &mut Reader<'_>, n: usize) -> Result<Lu> {
    let factors = read_matrix(reader, n, n)?;
    if factors
        .diagonal()
        .iter()
        .any(|&value| value == Complex::default())
    {
        return Err(invalid());
    }
    let mut forward = numerics::filled(n, 0_usize)?;
    let mut inverse = numerics::filled(n, usize::MAX)?;
    for (i, index) in forward.iter_mut().enumerate() {
        *index = reader.usize()?;
        let entry = inverse.get_mut(*index).ok_or_else(invalid)?;
        if *entry != usize::MAX {
            return Err(invalid());
        }
        *entry = i;
    }
    let mut row = numerics::filled(n, 1.0)?;
    let mut column = numerics::filled(n, 1.0)?;
    for value in row.iter_mut().chain(column.iter_mut()) {
        *value = reader.f64()?;
        if !value.is_finite() || *value <= 0.0 {
            return Err(invalid());
        }
    }
    let equilibration = if row
        .iter()
        .chain(&column)
        .all(|&value| value.to_bits() == 1.0_f64.to_bits())
    {
        None
    } else {
        Some(Equilibration { row, column })
    };
    Ok(Lu {
        factors,
        permutation: Perm::new_checked(forward.into_boxed_slice(), inverse.into_boxed_slice(), n),
        equilibration,
    })
}

impl SolveResidual {
    /// Byte count for the native saved state of an `n × rhs` solution.
    pub fn state_size(n: usize, rhs: usize) -> Result<usize> {
        total(&[16, Lu::state_size(n)?, matrix_size(n, rhs)?])
    }
}

impl SavedState for SolveResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let (n, rhs) = self.shape();
        let mut writer = Writer::new(Self::state_size(n, rhs)?);
        writer.usize(n);
        writer.usize(rhs);
        self.lu.write_state(&mut writer);
        write_matrix(&mut writer, view(&self.value));
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let (n, rhs) = (reader.usize()?, reader.usize()?);
        if bytes.len() != Self::state_size(n, rhs)? {
            return Err(invalid());
        }
        let lu = Lu::read_state(&mut reader, n)?;
        let value = read_matrix(&mut reader, n, rhs)?;
        reader.finish()?;
        Ok(Self { lu, value })
    }
}

impl SvdvalsResidual {
    /// Byte count for the native saved state of an `m × n` operator.
    pub fn state_size(m: usize, n: usize) -> Result<usize> {
        let k = m.min(n);
        total(&[
            16,
            matrix_size(m, k)?,
            matrix_size(n, k)?,
            k.checked_mul(8).ok_or_else(invalid)?,
        ])
    }
}

impl SavedState for SvdvalsResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let (m, n) = self.shape();
        let mut writer = Writer::new(Self::state_size(m, n)?);
        writer.usize(m);
        writer.usize(n);
        write_matrix(&mut writer, self.u.as_ref());
        write_matrix(&mut writer, self.v.as_ref());
        for &value in &self.values {
            writer.f64(value);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let (m, n) = (reader.usize()?, reader.usize()?);
        if bytes.len() != Self::state_size(m, n)? {
            return Err(invalid());
        }
        let k = m.min(n);
        let u = read_faer(&mut reader, m, k)?;
        let v = read_faer(&mut reader, n, k)?;
        let mut values = numerics::filled(k, 0.0)?;
        for value in &mut values {
            *value = reader.f64()?;
            if !value.is_finite() || *value < 0.0 {
                return Err(invalid());
            }
        }
        reader.finish()?;
        Ok(Self { u, v, values })
    }
}

impl EigResidual {
    /// Byte count for the native saved state of an `n × n` eigensystem.
    pub fn state_size(n: usize) -> Result<usize> {
        total(&[
            17,
            matrix_size(n, n)?,
            n.checked_mul(24).ok_or_else(invalid)?,
            Lu::state_size(n)?,
        ])
    }
}

impl SavedState for EigResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let n = self.values.len();
        let mut writer = Writer::new(Self::state_size(n)?);
        writer.usize(n);
        for &value in &self.values {
            writer.complex(value);
        }
        write_matrix(&mut writer, view(&self.vectors));
        for &pivot in &self.pivots {
            writer.usize(pivot);
        }
        writer.f64(self.scale);
        match self.vectors_lu() {
            Ok(lu) => {
                writer.byte(0);
                lu.write_state(&mut writer);
            }
            Err(Error::Singular) => {
                // Singular eigenbases still support a trace pullback. This branch
                // has the same shape-defined LU slots, with canonical identities.
                writer.byte(1);
                for j in 0..n {
                    for i in 0..n {
                        writer.complex(if i == j {
                            Complex::from(1.0)
                        } else {
                            Complex::default()
                        });
                    }
                }
                for i in 0..n {
                    writer.usize(i);
                }
                for _ in 0..2 * n {
                    writer.f64(1.0);
                }
            }
            Err(error) => return Err(error),
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let n = reader.usize()?;
        if bytes.len() != Self::state_size(n)? {
            return Err(invalid());
        }
        let mut values = numerics::filled(n, Complex::default())?;
        for value in &mut values {
            *value = read_complex(&mut reader)?;
        }
        let vectors = read_matrix(&mut reader, n, n)?;
        let mut pivots = numerics::filled(n, 0_usize)?;
        for pivot in &mut pivots {
            *pivot = reader.usize()?;
            if *pivot >= n {
                return Err(invalid());
            }
        }
        let scale = reader.f64()?;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(invalid());
        }
        let status = reader.byte()?;
        let lu = Lu::read_state(&mut reader, n)?;
        let lu = match status {
            0 => Ok(lu),
            1 => Err(Error::Singular),
            _ => return Err(invalid()),
        };
        reader.finish()?;
        Ok(Self {
            values,
            vectors,
            pivots,
            scale,
            vectors_lu: OnceLock::from(lu),
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;

    use super::*;
    use crate::linalg::{eig, solve_owned, svdvals};

    #[test]
    fn solve_state_preserves_factors_scales_and_both_derivatives() {
        for scaled in [false, true] {
            let operator = DMatrix::from_fn(3, 3, |i, j| {
                let value = Complex::new(if (i + 1) % 3 == j { 3.0 } else { 0.2 }, 0.1);
                let exponent = if scaled {
                    [-64, 0, 64][i] + [-40, 0, 40][j]
                } else {
                    0
                };
                value * 2.0_f64.powi(exponent)
            });
            let rhs = &operator * DMatrix::from_element(3, 2, Complex::new(0.3, -0.1));
            let residual = solve_owned(operator.clone(), rhs.clone()).unwrap();
            assert_eq!(residual.lu.equilibration.is_some(), scaled);
            assert_ne!(residual.lu.permutation.arrays().0, &[0, 1, 2]);
            if let Some(scales) = &residual.lu.equilibration {
                assert_ne!(scales.row, vec![1.0; 3]);
                assert_ne!(scales.column, vec![1.0; 3]);
            }
            let bytes = residual.save_state().unwrap();
            assert_eq!(bytes.len(), SolveResidual::state_size(3, 2).unwrap());
            let restored = SolveResidual::from_state(&bytes).unwrap();
            assert_eq!(restored.save_state().unwrap(), bytes);
            assert_eq!(restored.value(), residual.value());
            assert_eq!(restored.lu.factors, residual.lu.factors);
            assert_eq!(
                restored.lu.permutation.arrays(),
                residual.lu.permutation.arrays()
            );
            let operator_direction = operator * Complex::from(0.2);
            let rhs_direction = rhs * Complex::from(0.3);
            assert_eq!(
                restored
                    .pushforward(&operator_direction, rhs_direction.clone())
                    .unwrap(),
                residual
                    .pushforward(&operator_direction, rhs_direction)
                    .unwrap()
            );
            let cotangent = DMatrix::from_element(3, 2, Complex::new(0.5, 0.2));
            let expected = residual.pullback(cotangent.clone()).unwrap();
            let actual = restored.pullback(cotangent).unwrap();
            assert_eq!(actual.operator, expected.operator);
            assert_eq!(actual.rhs, expected.rhs);
        }
    }

    #[test]
    fn eig_state_carries_the_cached_derivative_factorization() {
        let operator = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex::new(2.0, 0.1),
                Complex::new(0.2, -0.1),
                Complex::default(),
                Complex::new(3.0, -0.2),
            ],
        );
        let residual = eig(&operator).unwrap();
        assert!(residual.vectors_lu.get().is_none());
        let bytes = residual.save_state().unwrap();
        assert_eq!(bytes.len(), EigResidual::state_size(2).unwrap());
        let restored = EigResidual::from_state(&bytes).unwrap();
        assert!(restored.vectors_lu.get().unwrap().is_ok());
        assert_eq!(restored.save_state().unwrap(), bytes);
        assert_eq!(
            restored.vectors_lu().unwrap().factors,
            residual.vectors_lu().unwrap().factors
        );
        let direction = DMatrix::from_element(2, 2, Complex::new(0.2, 0.1));
        assert_eq!(
            restored.pushforward(&direction).unwrap(),
            residual.pushforward(&direction).unwrap()
        );
        let values = [Complex::new(0.2, 0.3), Complex::new(-0.4, 0.1)];
        let vectors = DMatrix::from_element(2, 2, Complex::new(-0.2, 0.3));
        assert_eq!(
            restored.pullback(&values, vectors.clone()).unwrap(),
            residual.pullback(&values, vectors).unwrap()
        );
    }

    #[test]
    fn eig_state_preserves_trace_pullbacks_with_a_singular_eigenbasis() {
        // An exact Jordan eigenbasis, with the undefined inverse recorded as a
        // numerical status rather than preventing the valid trace derivative.
        let residual = EigResidual {
            values: vec![Complex::from(1.0); 2],
            vectors: DMatrix::from_row_slice(
                2,
                2,
                &[
                    Complex::from(1.0),
                    Complex::from(1.0),
                    Complex::default(),
                    Complex::default(),
                ],
            ),
            pivots: vec![0; 2],
            scale: 1.0,
            vectors_lu: OnceLock::new(),
        };
        let bytes = residual.save_state().unwrap();
        assert_eq!(bytes.len(), EigResidual::state_size(2).unwrap());
        let restored = EigResidual::from_state(&bytes).unwrap();
        assert!(matches!(
            restored.vectors_lu.get(),
            Some(Err(Error::Singular))
        ));
        assert_eq!(restored.save_state().unwrap(), bytes);
        assert_eq!(
            restored
                .pullback(&[Complex::from(1.0); 2], DMatrix::zeros(2, 2))
                .unwrap(),
            DMatrix::identity(2, 2)
        );
        assert!(restored.pushforward(&DMatrix::identity(2, 2)).is_err());
    }

    #[test]
    fn rectangular_svd_state_preserves_both_derivatives() {
        let tall = DMatrix::from_fn(3, 2, |i, j| {
            Complex::new(if i == j { [2.0, 3.0, 4.0][i] } else { 0.2 }, 0.1)
        });
        for operator in [tall.clone(), tall.transpose()] {
            let residual = svdvals(&operator).unwrap();
            let bytes = residual.save_state().unwrap();
            assert_eq!(
                bytes.len(),
                SvdvalsResidual::state_size(operator.nrows(), operator.ncols()).unwrap()
            );
            let restored = SvdvalsResidual::from_state(&bytes).unwrap();
            assert_eq!(restored.save_state().unwrap(), bytes);
            assert_eq!(restored.values(), residual.values());
            let direction =
                DMatrix::from_element(operator.nrows(), operator.ncols(), Complex::new(-0.2, 0.1));
            assert_eq!(
                restored.pushforward(&direction).unwrap(),
                residual.pushforward(&direction).unwrap()
            );
            assert_eq!(
                restored.pullback(&[0.2, -0.4]).unwrap(),
                residual.pullback(&[0.2, -0.4]).unwrap()
            );
        }
    }

    #[test]
    fn solve_restore_rejects_wrong_lengths_and_invalid_permutations() {
        let residual = solve_owned(DMatrix::identity(2, 2), DMatrix::identity(2, 2)).unwrap();
        let bytes = residual.save_state().unwrap();
        assert!(SolveResidual::from_state(&bytes[..bytes.len() - 1]).is_err());
        let mut extended = bytes.clone();
        extended.push(0);
        assert!(SolveResidual::from_state(&extended).is_err());
        let mut invalid_permutation = bytes.clone();
        let permutation_start = 16 + 16 * 2 * 2;
        invalid_permutation[permutation_start + 8..permutation_start + 16]
            .copy_from_slice(&0_u64.to_le_bytes());
        assert!(SolveResidual::from_state(&invalid_permutation).is_err());
        let mut invalid_size = bytes;
        invalid_size[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(SolveResidual::from_state(&invalid_size).is_err());
    }
}
