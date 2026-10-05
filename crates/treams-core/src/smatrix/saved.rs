//! Shared fixed-layout numerical state helpers for planar residuals.
//!
//! treams-rs extension. Each enclosing residual checks its complete byte count
//! before these helpers allocate matrices; dimensions are part of that header.

use super::StoredBlock;
pub(super) use crate::saved::{
    invalid, product, read_complex, read_f64, read_matrix, total, write_matrix,
};
use crate::{
    Result,
    saved::{Reader, Writer},
};

pub(super) fn write_stored(writer: &mut Writer, block: &StoredBlock) {
    writer.byte(u8::from(block.row_major));
    write_matrix(writer, &block.values);
}

pub(super) fn read_stored(reader: &mut Reader<'_>, n: usize) -> Result<StoredBlock> {
    let row_major = match reader.byte()? {
        0 => false,
        1 => true,
        _ => return Err(invalid()),
    };
    Ok(StoredBlock {
        values: read_matrix(reader, n, n)?,
        row_major,
    })
}
