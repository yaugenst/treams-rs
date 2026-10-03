//! `rotation` and `cylindrical_rotation` with `RotationContext`: rotation matrices
//! of spherical and cylindrical bases (`treams_core::rotation`).
use pyo3::prelude::*;
use treams_core::fpenv::ieee;

use crate::{
    args::{make_basis, make_cyl_basis},
    context::{context, detached},
    convert::{C2, Cotangent, matrix},
};

context!(RotationContext(treams_core::rotation::RotationResidual));
#[pymethods]
impl RotationContext {
    fn pullback(&mut self, py: Python<'_>, cotangent: Cotangent<'_>) -> PyResult<[f64; 3]> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, treams_core::rotation::RotationResidual::shape)?;
            detached(py, move || residual.pullback(&g))
        })
    }
}

/// A rotation matrix, copied into a C-ordered array because the residual keeps it, and
/// its context.
fn finish_rotation(
    py: Python<'_>,
    residual: treams_core::rotation::RotationResidual,
) -> PyResult<(C2<'_>, RotationContext)> {
    Ok((
        matrix(py, residual.value())?,
        RotationContext::new(residual),
    ))
}

/// Record the rotation matrix of a spherical basis: `rotation::sw_rotation`.
#[pyfunction]
pub(crate) fn rotation(
    py: Python<'_>,
    destination: Vec<(usize, i32, i32, u8)>,
    source: Vec<(usize, i32, i32, u8)>,
    destination_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    angles: [f64; 3],
) -> PyResult<(C2<'_>, RotationContext)> {
    ieee(|| {
        let destination = make_basis(destination, destination_positions);
        let source = make_basis(source, source_positions);
        let residual = detached(py, move || {
            treams_core::rotation::sw_rotation(&destination, &source, angles)
        })?;
        finish_rotation(py, residual)
    })
}

/// Record the rotation matrix of a cylindrical basis: `rotation::cw_rotation`.
#[pyfunction]
pub(crate) fn cylindrical_rotation(
    py: Python<'_>,
    destination: Vec<(usize, f64, i32, u8)>,
    source: Vec<(usize, f64, i32, u8)>,
    destination_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    angles: [f64; 3],
) -> PyResult<(C2<'_>, RotationContext)> {
    ieee(|| {
        let destination = make_cyl_basis(destination, destination_positions);
        let source = make_cyl_basis(source, source_positions);
        let residual = detached(py, move || {
            treams_core::rotation::cw_rotation(&destination, &source, angles)
        })?;
        finish_rotation(py, residual)
    })
}
