//! What every context shares: the `context!` macro, the one-use residual, the
//! cotangent error, core errors as `ValueError`, the radial flag and GIL release.
use nalgebra::DMatrix;
use num_complex::Complex64;
use pyo3::{exceptions::PyValueError, marker::Ungil, prelude::*};
use treams_core::special::Radial;

use crate::convert::{Cotangent, matrix_cotangent};

/// Define a `#[pyclass]` pullback context that owns a one-use residual and
/// extra fields; `new` takes the residual and then the fields in order.
///
/// `#[pymodule]` sets the module only of classes defined inside it, so each
/// class sets it here: reprs, pickling errors and the reference then show
/// `treams_rs._native.<Name>`.
macro_rules! context {
    ($(#[$doc:meta])* $name:ident($residual:ty $(, $(#[$meta:meta])* $field:ident: $type:ty)* $(,)?)) => {
        $(#[$doc])*
        #[pyclass(module = "treams_rs._native")]
        #[derive(Debug)]
        pub(crate) struct $name {
            residual: $crate::context::OneUse<$residual>,
            $($(#[$meta])* $field: $type,)*
        }
        impl $name {
            const fn new(residual: $residual $(, $field: $type)*) -> Self {
                Self { residual: $crate::context::OneUse::new(residual) $(, $field)* }
            }
        }
    };
}
pub(crate) use context;

/// A residual that one pullback consumes.
///
/// A pullback checks its cotangent before it takes the residual, so a rejected
/// cotangent leaves the residual for a corrected retry.
#[derive(Debug)]
pub(crate) struct OneUse<R>(Option<R>);

impl<R> OneUse<R> {
    pub(crate) const fn new(residual: R) -> Self {
        Self(Some(residual))
    }

    /// The residual, unless a pullback has consumed it.
    pub(crate) fn peek(&self) -> PyResult<&R> {
        self.0.as_ref().ok_or_else(consumed)
    }

    /// Consume the residual.
    pub(crate) fn take(&mut self) -> PyResult<R> {
        self.0.take().ok_or_else(consumed)
    }

    /// Consume the residual once `check` accepts the cotangent for it, and return
    /// both; when `check` fails, the residual stays for a corrected retry.
    pub(crate) fn take_if<T>(&mut self, check: impl FnOnce(&R) -> PyResult<T>) -> PyResult<(R, T)> {
        let cotangent = check(self.peek()?)?;
        Ok((self.take()?, cotangent))
    }

    /// Consume the residual with a finite matrix cotangent of the residual's
    /// output `shape`, copied into column-major storage.
    pub(crate) fn take_with_matrix(
        &mut self,
        cotangent: &Cotangent<'_>,
        shape: impl FnOnce(&R) -> (usize, usize),
    ) -> PyResult<(R, DMatrix<Complex64>)> {
        self.take_if(|residual| matrix_cotangent(cotangent, shape(residual)))
    }
}

/// The error of a cotangent that is not finite or not of the `expected` shape,
/// for example "cotangent must be finite with shape (2, 2, 3, 3)".
pub(crate) fn cotangent_error(expected: &[usize]) -> PyErr {
    let shape = match expected {
        [length] => format!("({length},)"),
        _ => format!(
            "({})",
            expected
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    PyValueError::new_err(format!("cotangent must be finite with shape {shape}"))
}

/// The error of a pullback whose one-use residual is already consumed.
fn consumed() -> PyErr {
    PyValueError::new_err("pullback residual has already been consumed")
}

/// Raise every core error variant as `ValueError` with the error's message.
pub(crate) fn error(error: treams_core::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Singular (outgoing Hankel H1) radial functions for `true`, regular ones for `false`.
pub(crate) const fn radial(singular: bool) -> Radial {
    if singular {
        Radial::Singular
    } else {
        Radial::Regular
    }
}

/// Run fallible native work with the GIL released; core errors raise `ValueError`.
pub(crate) fn detached<T>(
    py: Python<'_>,
    work: impl Ungil + FnOnce() -> treams_core::Result<T>,
) -> PyResult<T>
where
    treams_core::Result<T>: Ungil,
{
    py.detach(work).map_err(error)
}
