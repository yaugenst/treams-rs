//! What every context shares: the `context!` macro, saved-state conversion, the
//! cotangent error, core errors as `ValueError` or `MemoryError`, the radial flag and
//! GIL release.
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::{
    exceptions::{PyMemoryError, PyValueError},
    marker::Ungil,
    prelude::*,
};
use treams_core::saved::SavedState;
use treams_core::{Error, special::Radial};

/// Define a `#[pyclass]` derivative context that owns a reusable residual and
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
            residual: $residual,
            $($(#[$meta])* $field: $type,)*
        }
        impl $name {
            const fn new(residual: $residual $(, $field: $type)*) -> Self {
                Self { residual $(, $field)* }
            }
        }
    };
}
pub(crate) use context;

/// Export numeric state for a framework callback; `NumPy` owns the copied bytes.
pub(crate) fn state_array<'py, R: SavedState + Sync>(
    py: Python<'py>,
    residual: &R,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    Ok(detached(py, || residual.save_state())?.into_pyarray(py))
}

/// Restore a private state buffer, checking its layout at the native boundary.
pub(crate) fn restore_state<R: SavedState>(state: &PyReadonlyArray1<'_, u8>) -> PyResult<R> {
    R::from_state(state.as_slice()?).map_err(error)
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

/// Raise a core error with its message: `OutOfMemory` as `MemoryError`, every other
/// variant as `ValueError`.
pub(crate) fn error(error: Error) -> PyErr {
    match error {
        Error::OutOfMemory(message) => PyMemoryError::new_err(message),
        error => PyValueError::new_err(error.to_string()),
    }
}

/// Singular (outgoing Hankel H1) radial functions for `true`, regular ones for `false`.
pub(crate) const fn radial(singular: bool) -> Radial {
    if singular {
        Radial::Singular
    } else {
        Radial::Regular
    }
}

/// Run fallible native work with the GIL released; core errors raise as [`error`]
/// maps them.
pub(crate) fn detached<T>(
    py: Python<'_>,
    work: impl Ungil + FnOnce() -> treams_core::Result<T>,
) -> PyResult<T>
where
    treams_core::Result<T>: Ungil,
{
    py.detach(work).map_err(error)
}
