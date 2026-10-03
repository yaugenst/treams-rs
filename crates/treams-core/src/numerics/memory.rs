//! Allocations that return [`Error::OutOfMemory`] when the system refuses them, for
//! outputs and workspaces that grow with the problem. treams-rs extension.
//!
//! `vec!`, `Vec::with_capacity` and `DMatrix::zeros` call `handle_alloc_error` on a
//! refused request, which aborts the process, and panic when the byte count overflows.
//! These helpers reserve with `Vec::try_reserve_exact` instead and report the request
//! in bytes. On Linux with memory overcommit a reservation larger than the free memory
//! can still succeed, and the kernel may kill the process later when the pages are
//! written; no allocation API reports that.

use nalgebra::DMatrix;

use crate::{Complex, Error, Result};

/// A zero `rows × cols` matrix.
pub(crate) fn zeros(rows: usize, cols: usize) -> Result<DMatrix<Complex>> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| Error::out_of_memory(usize::MAX, size_of::<Complex>()))?;
    Ok(DMatrix::from_vec(
        rows,
        cols,
        filled(len, Complex::default())?,
    ))
}

/// `len` copies of `value`.
pub(crate) fn filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut values = Vec::new();
    reserve(&mut values, len)?;
    values.resize(len, value);
    Ok(values)
}

/// Reserve room for exactly `additional` more elements in `values`.
pub(crate) fn reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<()> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| Error::out_of_memory(values.len().saturating_add(additional), size_of::<T>()))
}

/// Refused, overflowing and ordinary requests.
#[cfg(test)]
mod tests {
    use super::{filled, reserve, zeros};
    use crate::Error;

    #[test]
    fn requests_beyond_the_address_space_return_out_of_memory() {
        // 2^48 elements of 16 bytes: 2^52 bytes, beyond the user address space of
        // every supported operating system.
        let error = zeros(1 << 24, 1 << 24).unwrap_err();
        assert!(
            matches!(&error, Error::OutOfMemory(message) if message == "cannot allocate 4503599627370496 bytes"),
            "{error:?}"
        );
        assert!(!error.is_numerical());
        let mut values = vec![0_u8; 3];
        assert!(matches!(
            reserve(&mut values, 1 << 52),
            Err(Error::OutOfMemory(_))
        ));
        assert_eq!(values, [0, 0, 0]);
    }

    #[test]
    fn overflowing_requests_return_out_of_memory_without_panicking() {
        let more = format!("cannot allocate more than {} bytes", usize::MAX);
        for (rows, cols) in [(usize::MAX, 1), (usize::MAX, 2), (1 << 33, 1 << 33)] {
            let error = zeros(rows, cols).unwrap_err();
            assert!(
                matches!(&error, Error::OutOfMemory(message) if *message == more),
                "{rows} x {cols}: {error:?}"
            );
        }
        assert!(matches!(
            filled(usize::MAX, 0.0_f64),
            Err(Error::OutOfMemory(message)) if message == more
        ));
        let mut values = vec![0_u64; 2];
        assert!(matches!(
            reserve(&mut values, usize::MAX),
            Err(Error::OutOfMemory(message)) if message == more
        ));
    }

    #[test]
    fn granted_requests_hold_the_requested_values() {
        let matrix = zeros(3, 2).unwrap();
        assert_eq!(matrix.shape(), (3, 2));
        assert!(matrix.iter().all(|&z| z == crate::Complex::default()));
        assert_eq!(zeros(0, 7).unwrap().shape(), (0, 7));
        assert_eq!(filled(4, 1.5).unwrap(), [1.5; 4]);
        let mut values = vec![1];
        reserve(&mut values, 5).unwrap();
        assert!(values.capacity() >= 6);
    }
}
