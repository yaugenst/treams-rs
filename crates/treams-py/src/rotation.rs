//! `rotation` and `cylindrical_rotation` with `RotationContext`: rotation matrices
//! of spherical and cylindrical bases (`treams_core::rotation`).
#![allow(clippy::indexing_slicing)] // The Euler tangent length is validated before indexing.
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use treams_core::{fpenv::ieee, rotation::RotationResidual};

use crate::{
    args::{make_basis, make_cyl_basis},
    context::{context, detached, error, restore_state, state_array},
    convert::{C2, Cotangent, RealTangent, matrix, matrix_cotangent, owned_matrix, vector_tangent},
};

context!(RotationContext(RotationResidual));
#[pymethods]
impl RotationContext {
    #[staticmethod]
    #[pyo3(signature = (destination, source, cylindrical=false))]
    fn _state_spec(
        destination: &Bound<'_, PyAny>,
        source: &Bound<'_, PyAny>,
        cylindrical: bool,
    ) -> PyResult<usize> {
        ieee(|| {
            if cylindrical {
                RotationResidual::cw_state_size(
                    &make_cyl_basis(destination.extract()?, Vec::new()).modes,
                    &make_cyl_basis(source.extract()?, Vec::new()).modes,
                )
            } else {
                RotationResidual::sw_state_size(
                    &make_basis(destination.extract()?, Vec::new()).modes,
                    &make_basis(source.extract()?, Vec::new()).modes,
                )
            }
            .map_err(error)
        })
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(&self, py: Python<'py>, angles: RealTangent<'py>) -> PyResult<C2<'py>> {
        ieee(|| {
            let direction = vector_tangent(&angles, 3)?;
            let angles = [direction[0], direction[1], direction[2]];
            let residual = &self.residual;
            let result = detached(py, || residual.pushforward(angles))?;
            owned_matrix(py, result)
        })
    }

    fn pullback(&self, py: Python<'_>, cotangent: Cotangent<'_>) -> PyResult<[f64; 3]> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
            detached(py, move || residual.pullback(&g))
        })
    }
}

/// A rotation matrix, copied into a C-ordered array because the residual keeps it, and
/// its context.
fn finish_rotation(
    py: Python<'_>,
    residual: RotationResidual,
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
