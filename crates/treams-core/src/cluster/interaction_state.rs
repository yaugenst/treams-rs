//! Array-owned interaction factors and fields, a treams-rs extension.

use std::sync::Arc;

use nalgebra::DMatrix;

use super::{IlluminateResidual, InteractionFactor, InteractionResidual, LocalMatrix};
use crate::{
    Complex, Error, Result,
    linalg::Lu,
    numerics,
    saved::{Reader, SavedState, Writer, write_matrix},
};

fn invalid() -> Error {
    Error::InvalidInput("invalid cluster derivative state".into())
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(invalid)
}

fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(invalid)
}

fn matrix_size(rows: usize, columns: usize) -> Result<usize> {
    mul(16, mul(rows, columns)?)
}

fn dimensions(sizes: &[usize]) -> Result<(usize, usize)> {
    if sizes.is_empty() || sizes.contains(&0) {
        return Err(invalid());
    }
    sizes
        .iter()
        .try_fold((0, 0), |(dimension, entries), &size| {
            Ok((add(dimension, size)?, add(entries, mul(size, size)?)?))
        })
}

fn factor_size(blocks: usize, dimension: usize, entries: usize) -> Result<usize> {
    let header = add(8, mul(8, blocks)?)?;
    let matrices = add(mul(16, entries)?, matrix_size(dimension, dimension)?)?;
    add(add(header, matrices)?, Lu::state_size(dimension)?)
}

fn read_matrix(reader: &mut Reader<'_>, rows: usize, columns: usize) -> Result<DMatrix<Complex>> {
    let mut matrix = numerics::zeros(rows, columns)?;
    for value in matrix.as_mut_slice() {
        *value = reader.complex()?;
    }
    Ok(matrix)
}

/// Read only the dimension header. The count is bounded before allocating the
/// short block-size vector; no matrix or factor allocation precedes the size check.
fn read_sizes(reader: &mut Reader<'_>) -> Result<Vec<usize>> {
    let count = reader.usize()?;
    if count == 0 || count > reader.remaining_len() / 8 {
        return Err(invalid());
    }
    (0..count).map(|_| reader.usize()).collect()
}

impl InteractionFactor {
    fn write_state(&self, writer: &mut Writer) {
        writer.usize(self.local.blocks.len());
        for block in &self.local.blocks {
            writer.usize(block.nrows());
        }
        for block in &self.local.blocks {
            write_matrix(writer, block);
        }
        write_matrix(writer, &self.coupling);
        self.lu.write_state(writer);
    }

    fn read_body(reader: &mut Reader<'_>, sizes: &[usize], dimension: usize) -> Result<Self> {
        let blocks = sizes
            .iter()
            .map(|&size| read_matrix(reader, size, size))
            .collect::<Result<Vec<_>>>()?;
        let coupling = read_matrix(reader, dimension, dimension)?;
        let lu = Lu::read_state(reader, dimension)?;
        Ok(Self {
            local: LocalMatrix { blocks },
            coupling,
            lu,
        })
    }
}

impl InteractionResidual {
    /// Bytes needed to save local blocks, coupling, LU and interacting T-matrix.
    pub fn state_size(local_sizes: &[usize]) -> Result<usize> {
        let (dimension, entries) = dimensions(local_sizes)?;
        add(
            factor_size(local_sizes.len(), dimension, entries)?,
            matrix_size(dimension, dimension)?,
        )
    }

    /// State size for equally sized particle blocks, without a temporary size list.
    pub(crate) fn state_size_uniform(block_size: usize, particles: usize) -> Result<usize> {
        if block_size == 0 || particles == 0 {
            return Err(invalid());
        }
        let dimension = mul(block_size, particles)?;
        let entries = mul(mul(block_size, block_size)?, particles)?;
        add(
            factor_size(particles, dimension, entries)?,
            matrix_size(dimension, dimension)?,
        )
    }

    pub(crate) fn write_state(&self, writer: &mut Writer) {
        self.factor.write_state(writer);
        write_matrix(writer, &self.value);
    }

    pub(crate) fn read_state(reader: &mut Reader<'_>) -> Result<Self> {
        let sizes = read_sizes(reader)?;
        let header = add(8, mul(8, sizes.len())?)?;
        let expected = Self::state_size(&sizes)?;
        if expected - header > reader.remaining_len() {
            return Err(invalid());
        }
        let (dimension, _) = dimensions(&sizes)?;
        let factor = InteractionFactor::read_body(reader, &sizes, dimension)?;
        let value = read_matrix(reader, dimension, dimension)?;
        Ok(Self { factor, value })
    }
}

impl SavedState for InteractionResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let sizes: Vec<_> = self.local_shapes().map(|(rows, _)| rows).collect();
        let mut writer = Writer::new(Self::state_size(&sizes)?);
        self.write_state(&mut writer);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let residual = Self::read_state(&mut reader)?;
        reader.finish()?;
        Ok(residual)
    }
}

impl IlluminateResidual {
    /// Bytes needed to save local blocks, coupling, LU, incident and scattered fields.
    pub fn state_size(local_sizes: &[usize], columns: usize) -> Result<usize> {
        if columns == 0 {
            return Err(invalid());
        }
        let (dimension, entries) = dimensions(local_sizes)?;
        add(
            add(8, factor_size(local_sizes.len(), dimension, entries)?)?,
            mul(2, matrix_size(dimension, columns)?)?,
        )
    }
}

impl SavedState for IlluminateResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let sizes: Vec<_> = self.local_shapes().map(|(rows, _)| rows).collect();
        let mut writer = Writer::new(Self::state_size(&sizes, self.value.ncols())?);
        writer.usize(self.value.ncols());
        self.factor.write_state(&mut writer);
        write_matrix(&mut writer, &self.incident);
        write_matrix(&mut writer, &self.value);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let columns = reader.usize()?;
        let sizes = read_sizes(&mut reader)?;
        if Self::state_size(&sizes, columns)? != bytes.len() {
            return Err(invalid());
        }
        let (dimension, _) = dimensions(&sizes)?;
        let factor = Arc::new(InteractionFactor::read_body(
            &mut reader,
            &sizes,
            dimension,
        )?);
        let incident = read_matrix(&mut reader, dimension, columns)?;
        let value = read_matrix(&mut reader, dimension, columns)?;
        reader.finish()?;
        Ok(Self {
            factor,
            incident,
            value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cluster::interaction_blocks, test_support::patterned};

    #[test]
    fn saved_interactions_retain_blocks_factors_and_both_derivatives() {
        let blocks = vec![patterned(1, 1, 0.2), patterned(2, 2, 0.4)];
        let coupling = patterned(3, 3, 0.6) * Complex::from(0.02);
        let direction = patterned(3, 3, 0.8);
        let residual = interaction_blocks(blocks.clone(), coupling.clone()).unwrap();
        let bytes = residual.save_state().unwrap();
        assert_eq!(
            bytes.len(),
            InteractionResidual::state_size(&[1, 2]).unwrap()
        );
        let restored = InteractionResidual::from_state(&bytes).unwrap();
        assert_eq!(restored.save_state().unwrap(), bytes);
        assert_eq!(restored.value(), residual.value());
        assert_eq!(
            restored.pushforward_blocks(&blocks, &direction).unwrap(),
            residual.pushforward_blocks(&blocks, &direction).unwrap()
        );
        let actual = restored.pullback_blocks(&direction).unwrap();
        let expected = residual.pullback_blocks(&direction).unwrap();
        assert_eq!(actual.local, expected.local);
        assert_eq!(actual.coupling, expected.coupling);
        assert!(InteractionResidual::from_state(&bytes[..bytes.len() - 1]).is_err());
        let mut malformed = bytes;
        malformed.extend_from_slice(&[0]);
        assert!(InteractionResidual::from_state(&malformed).is_err());
        malformed[..8].fill(255);
        assert!(InteractionResidual::from_state(&malformed).is_err());

        let factor = Arc::new(InteractionFactor::from_blocks(blocks.clone(), coupling).unwrap());
        let incident = patterned(3, 2, 1.0);
        let residual = factor.record(incident.clone()).unwrap();
        let bytes = residual.save_state().unwrap();
        assert_eq!(
            bytes.len(),
            IlluminateResidual::state_size(&[1, 2], 2).unwrap()
        );
        let restored = IlluminateResidual::from_state(&bytes).unwrap();
        assert_eq!(restored.save_state().unwrap(), bytes);
        assert_eq!(restored.value(), residual.value());
        assert_eq!(
            restored
                .pushforward(&blocks, &direction, &incident)
                .unwrap(),
            residual
                .pushforward(&blocks, &direction, &incident)
                .unwrap()
        );
        let actual = restored.pullback(&incident).unwrap();
        let expected = residual.pullback(&incident).unwrap();
        assert_eq!(actual.local, expected.local);
        assert_eq!(actual.coupling, expected.coupling);
        assert_eq!(actual.incident, expected.incident);
        assert!(IlluminateResidual::from_state(&bytes[..bytes.len() - 1]).is_err());
        let mut malformed = bytes;
        malformed[..8].fill(255);
        assert!(IlluminateResidual::from_state(&malformed).is_err());
    }
}
