//! Numeric derivative state carried by array-only framework callbacks.
//!
//! These bytes belong to one invocation of the current native extension. They
//! are not a persistent file format. Each residual defines a fixed layout from
//! its input shapes, including canonical storage for optional numerical data.
//! treams-rs extension.

use nalgebra::DMatrix;

use crate::{Complex, Error, Result, numerics};

/// A residual that can cross an array-only callback boundary without rerunning
/// its forward calculation.
pub trait SavedState: Sized {
    /// Copy the saved numerical state into its fixed byte layout.
    fn save_state(&self) -> Result<Vec<u8>>;

    /// Restore numerical state from bytes produced by this extension.
    fn from_state(bytes: &[u8]) -> Result<Self>;
}

/// Write fixed-width numbers without alignment assumptions or unsafe casts.
#[derive(Debug)]
pub struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    /// Allocate storage for a layout whose size has already been checked.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
        }
    }

    /// Append a length or index as an unsigned 64-bit integer.
    pub fn usize(&mut self, value: usize) {
        self.bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }

    /// Append one real scalar.
    pub fn f64(&mut self, value: f64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Append a complex scalar, real part followed by imaginary part.
    pub fn complex(&mut self, value: Complex) {
        self.f64(value.re);
        self.f64(value.im);
    }

    /// Append a tag or boolean.
    pub fn byte(&mut self, value: u8) {
        self.bytes.push(value);
    }

    /// Append one signed mode label.
    pub fn i32(&mut self, value: i32) {
        self.raw(&value.to_le_bytes());
    }

    /// Append an already encoded section.
    pub fn raw(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    /// Return the complete byte buffer.
    #[must_use]
    pub fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// Read a private invocation state; malformed or truncated buffers are errors.
#[derive(Debug)]
pub struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    /// Borrow an encoded invocation state.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let Some((head, tail)) = self.bytes.split_at_checked(N) else {
            return Err(invalid());
        };
        self.bytes = tail;
        head.try_into().map_err(|_| invalid())
    }

    /// Read one length or index.
    pub fn usize(&mut self) -> Result<usize> {
        usize::try_from(u64::from_le_bytes(self.take()?)).map_err(|_| invalid())
    }

    /// Read one real scalar.
    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take()?))
    }

    /// Read one complex scalar.
    pub fn complex(&mut self) -> Result<Complex> {
        Ok(Complex::new(self.f64()?, self.f64()?))
    }

    /// Read one tag or boolean.
    pub fn byte(&mut self) -> Result<u8> {
        Ok(u8::from_le_bytes(self.take()?))
    }

    /// Read one signed mode label.
    pub fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take()?))
    }

    /// Read a section without allocating or copying it.
    pub fn raw(&mut self, len: usize) -> Result<&'a [u8]> {
        let Some((head, tail)) = self.bytes.split_at_checked(len) else {
            return Err(invalid());
        };
        self.bytes = tail;
        Ok(head)
    }

    /// The number of bytes not yet read.
    #[must_use]
    pub const fn remaining_len(&self) -> usize {
        self.bytes.len()
    }

    /// Validate an encoded vector count against the remaining storage before allocating.
    pub fn count(&mut self, stride: usize) -> Result<usize> {
        let count = self.usize()?;
        if stride == 0 || count > self.remaining_len() / stride {
            return Err(invalid());
        }
        Ok(count)
    }

    /// Reject trailing data after the complete expected layout.
    pub fn finish(self) -> Result<()> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(invalid())
        }
    }
}

pub(crate) fn invalid() -> Error {
    Error::InvalidInput("invalid native derivative state".into())
}

pub(crate) fn total(parts: &[usize]) -> Result<usize> {
    parts.iter().try_fold(0_usize, |size, &part| {
        size.checked_add(part).ok_or_else(invalid)
    })
}

pub(crate) fn product(parts: &[usize]) -> Result<usize> {
    parts.iter().try_fold(1_usize, |size, &part| {
        size.checked_mul(part).ok_or_else(invalid)
    })
}

pub(crate) fn read_complex(reader: &mut Reader<'_>) -> Result<Complex> {
    let value = reader.complex()?;
    if !numerics::finite(value) {
        return Err(invalid());
    }
    Ok(value)
}

pub(crate) fn read_f64(reader: &mut Reader<'_>) -> Result<f64> {
    let value = reader.f64()?;
    if !value.is_finite() {
        return Err(invalid());
    }
    Ok(value)
}

pub(crate) fn write_matrix(writer: &mut Writer, matrix: &DMatrix<Complex>) {
    for &value in matrix.as_slice() {
        writer.complex(value);
    }
}

pub(crate) fn read_matrix(
    reader: &mut Reader<'_>,
    rows: usize,
    columns: usize,
) -> Result<DMatrix<Complex>> {
    let mut matrix = numerics::zeros(rows, columns)?;
    for value in matrix.as_mut_slice() {
        *value = read_complex(reader)?;
    }
    Ok(matrix)
}

/// Byte layout of mode labels followed by Cartesian positions, both length prefixed.
fn basis_bytes(modes: usize, positions: usize, mode_bytes: usize) -> Result<usize> {
    modes
        .checked_mul(mode_bytes)
        .and_then(|n| positions.checked_mul(24).and_then(|p| n.checked_add(p)))
        .and_then(|n| n.checked_add(16))
        .ok_or_else(invalid)
}

pub(crate) fn sw_basis_size(modes: usize, positions: usize) -> Result<usize> {
    basis_bytes(modes, positions, 17)
}

pub(crate) fn cw_basis_size(modes: usize, positions: usize) -> Result<usize> {
    basis_bytes(modes, positions, 21)
}

fn read_basis_counts(reader: &mut Reader<'_>, mode_bytes: usize) -> Result<(usize, usize)> {
    let modes = reader.count(mode_bytes)?;
    reader.raw(modes * mode_bytes)?;
    let positions = reader.count(24)?;
    reader.raw(positions * 24)?;
    Ok((modes, positions))
}

pub(crate) fn read_sw_basis_dimensions(reader: &mut Reader<'_>) -> Result<(usize, usize)> {
    read_basis_counts(reader, 17)
}

pub(crate) fn read_cw_basis_dimensions(reader: &mut Reader<'_>) -> Result<(usize, usize)> {
    read_basis_counts(reader, 21)
}

/// Inspect and skip the complete encoded basis without allocating its arrays.
pub(crate) fn read_basis_dimensions(reader: &mut Reader<'_>) -> Result<(usize, usize, bool)> {
    let cylindrical = match reader.byte()? {
        0 => false,
        1 => true,
        _ => return Err(invalid()),
    };
    let (modes, positions) = read_basis_counts(reader, if cylindrical { 21 } else { 17 })?;
    Ok((modes, positions, cylindrical))
}

pub(crate) fn write_positions(writer: &mut Writer, positions: &[[f64; 3]]) {
    writer.usize(positions.len());
    for point in positions {
        for &coordinate in point {
            writer.f64(coordinate);
        }
    }
}

pub(crate) fn read_positions(reader: &mut Reader<'_>) -> Result<Vec<[f64; 3]>> {
    (0..reader.count(24)?)
        .map(|_| Ok([reader.f64()?, reader.f64()?, reader.f64()?]))
        .collect()
}

pub(crate) fn write_sw_basis(writer: &mut Writer, basis: &crate::sw::Basis) {
    writer.usize(basis.modes.len());
    for &(position, mode) in &basis.modes {
        writer.usize(position);
        writer.i32(mode.l);
        writer.i32(mode.m);
        writer.byte(mode.pol);
    }
    write_positions(writer, &basis.positions);
}

pub(crate) fn read_sw_basis(reader: &mut Reader<'_>) -> Result<crate::sw::Basis> {
    let modes = (0..reader.count(17)?)
        .map(|_| {
            Ok((
                reader.usize()?,
                crate::sw::Mode {
                    l: reader.i32()?,
                    m: reader.i32()?,
                    pol: reader.byte()?,
                },
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let basis = crate::sw::Basis {
        modes,
        positions: read_positions(reader)?,
    };
    basis.validate()?;
    Ok(basis)
}

pub(crate) fn write_cw_basis(writer: &mut Writer, basis: &crate::cw::Basis) {
    writer.usize(basis.modes.len());
    for &(position, mode) in &basis.modes {
        writer.usize(position);
        writer.f64(mode.kz);
        writer.i32(mode.m);
        writer.byte(mode.pol);
    }
    write_positions(writer, &basis.positions);
}

pub(crate) fn read_cw_basis(reader: &mut Reader<'_>) -> Result<crate::cw::Basis> {
    let modes = (0..reader.count(21)?)
        .map(|_| {
            Ok((
                reader.usize()?,
                crate::cw::Mode {
                    kz: reader.f64()?,
                    m: reader.i32()?,
                    pol: reader.byte()?,
                },
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let basis = crate::cw::Basis {
        modes,
        positions: read_positions(reader)?,
    };
    basis.validate()?;
    Ok(basis)
}

pub(crate) fn write_basis(writer: &mut Writer, basis: &crate::basis::MultipoleBasis) {
    match basis {
        crate::basis::MultipoleBasis::Spherical(basis) => {
            writer.byte(0);
            write_sw_basis(writer, basis);
        }
        crate::basis::MultipoleBasis::Cylindrical(basis) => {
            writer.byte(1);
            write_cw_basis(writer, basis);
        }
    }
}

pub(crate) fn read_basis(reader: &mut Reader<'_>) -> Result<crate::basis::MultipoleBasis> {
    match reader.byte()? {
        0 => read_sw_basis(reader).map(Into::into),
        1 => read_cw_basis(reader).map(Into::into),
        _ => Err(invalid()),
    }
}

pub(crate) fn write_radial(writer: &mut Writer, radial: crate::special::Radial) {
    writer.byte(u8::from(radial == crate::special::Radial::Singular));
}

pub(crate) fn read_radial(reader: &mut Reader<'_>) -> Result<crate::special::Radial> {
    match reader.byte()? {
        0 => Ok(crate::special::Radial::Regular),
        1 => Ok(crate::special::Radial::Singular),
        _ => Err(invalid()),
    }
}
