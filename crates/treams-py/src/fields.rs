//! `field`, `field_operator` and their cylindrical twins with `FieldContext` and
//! `FieldOperatorContext`: electric fields of multipole bases at sample points
//! (`treams_core::fields`).

use crate::{
    args::{make_basis, make_cyl_basis},
    context::{context, cotangent_error, detached, radial},
    convert::{
        C1, C2, C3, Cotangent, R1, R2, all_finite, cotangent_view, layout_error, merged_cotangent,
        rows, rows_array,
    },
};
use nalgebra::DMatrix;
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::{Array3, Ix2, Ix3},
};
use pyo3::prelude::*;
use treams_core::{
    Complex,
    basis::MultipoleBasis,
    fields::{self, FieldResidual},
    fpenv::ieee,
};

/// A column-major `(3 N, modes)` operator, whose rows run over (sample, Cartesian
/// component), as an `(N, 3, modes)` array that keeps its storage.
pub(crate) fn operator_array(value: DMatrix<Complex>) -> PyResult<Array3<Complex>> {
    let shape = (value.ncols(), value.nrows() / 3, 3);
    Array3::from_shape_vec(shape, Vec::from(value.data))
        .map(|a| a.permuted_axes([1, 2, 0]))
        .map_err(layout_error)
}

context!(FieldContext(FieldResidual));

impl FieldContext {
    fn take(&mut self, cotangent: &Cotangent<'_>) -> PyResult<(FieldResidual, Vec<[Complex; 3]>)> {
        self.residual.take_if(|residual| {
            let expected: [usize; 2] = residual.shape().into();
            let g = rows(cotangent_view::<_, Ix2>(cotangent, &expected)?, "cotangent")?;
            if all_finite(g.as_flattened()) {
                Ok(g)
            } else {
                Err(cotangent_error(&expected))
            }
        })
    }
}
#[pymethods]
impl FieldContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C1<'py>, R2<'py>, R2<'py>, C1<'py>)> {
        ieee(|| {
            let (residual, g) = self.take(&cotangent)?;
            let result = detached(py, move || residual.pullback(&g))?;
            Ok((
                result.coefficients.into_pyarray(py),
                rows_array(py, result.points)?,
                rows_array(py, result.positions)?,
                result.ks.to_vec().into_pyarray(py),
            ))
        })
    }
    fn pullback_axial<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C1<'py>, R2<'py>, R2<'py>, C1<'py>, R1<'py>)> {
        ieee(|| {
            let (residual, g) = self.take(&cotangent)?;
            let (result, kz) = detached(py, move || residual.pullback_axial(&g))?;
            Ok((
                result.coefficients.into_pyarray(py),
                rows_array(py, result.points)?,
                rows_array(py, result.positions)?,
                result.ks.to_vec().into_pyarray(py),
                kz.into_pyarray(py),
            ))
        })
    }
}

/// Record the electric field of spherical-wave coefficients at points: `fields::field`.
#[pyfunction]
pub(crate) fn field<'py>(
    py: Python<'py>,
    modes: Vec<(usize, i32, i32, u8)>,
    positions: Vec<[f64; 3]>,
    coefficients: PyReadonlyArray1<'py, Complex>,
    points: PyReadonlyArray2<'py, f64>,
    ks: [Complex; 2],
    helicity: bool,
    singular: bool,
) -> PyResult<(C2<'py>, FieldContext)> {
    ieee(|| {
        evaluate(
            py,
            make_basis(modes, positions).into(),
            coefficients,
            points,
            ks,
            helicity,
            singular,
        )
    })
}

/// Record the electric field of cylindrical-wave coefficients at points: `fields::field`.
#[pyfunction]
pub(crate) fn cylindrical_field<'py>(
    py: Python<'py>,
    modes: Vec<(usize, f64, i32, u8)>,
    positions: Vec<[f64; 3]>,
    coefficients: PyReadonlyArray1<'py, Complex>,
    points: PyReadonlyArray2<'py, f64>,
    ks: [Complex; 2],
    helicity: bool,
    singular: bool,
) -> PyResult<(C2<'py>, FieldContext)> {
    ieee(|| {
        evaluate(
            py,
            make_cyl_basis(modes, positions).into(),
            coefficients,
            points,
            ks,
            helicity,
            singular,
        )
    })
}

/// The field samples, moved into a C-ordered `(N, 3)` array, and their context.
fn evaluate<'py>(
    py: Python<'py>,
    basis: MultipoleBasis,
    coefficients: PyReadonlyArray1<'py, Complex>,
    points: PyReadonlyArray2<'py, f64>,
    ks: [Complex; 2],
    helicity: bool,
    singular: bool,
) -> PyResult<(C2<'py>, FieldContext)> {
    let coefficients = coefficients.as_array().to_vec();
    let points = rows(points.as_array(), "points")?;
    let (value, residual) = detached(py, move || {
        fields::field(basis, coefficients, points, ks, helicity, radial(singular))
    })?;
    Ok((rows_array(py, value)?, FieldContext::new(residual)))
}

context!(FieldOperatorContext(fields::OperatorResidual));
impl FieldOperatorContext {
    fn take(
        &mut self,
        cotangent: &Cotangent<'_>,
    ) -> PyResult<(fields::OperatorResidual, DMatrix<Complex>)> {
        self.residual.take_if(|residual| {
            // The native rows run over (sample, Cartesian component).
            let (rows, modes) = residual.shape();
            merged_cotangent::<Ix3>(cotangent, &[rows / 3, 3, modes], (rows, modes))
        })
    }
}
#[pymethods]
impl FieldOperatorContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, R2<'py>, C1<'py>)> {
        ieee(|| {
            let (residual, g) = self.take(&cotangent)?;
            let result = detached(py, move || residual.pullback(&g))?;
            Ok((
                rows_array(py, result.points)?,
                rows_array(py, result.positions)?,
                result.ks.to_vec().into_pyarray(py),
            ))
        })
    }
    fn pullback_axial<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, R2<'py>, C1<'py>, R1<'py>)> {
        ieee(|| {
            let (residual, g) = self.take(&cotangent)?;
            let (result, kz) = detached(py, move || residual.pullback_axial(&g))?;
            Ok((
                rows_array(py, result.points)?,
                rows_array(py, result.positions)?,
                result.ks.to_vec().into_pyarray(py),
                kz.into_pyarray(py),
            ))
        })
    }
}

/// Record the matrix from spherical-wave coefficients to field samples: `fields::operator`.
#[pyfunction]
pub(crate) fn field_operator<'py>(
    py: Python<'py>,
    modes: Vec<(usize, i32, i32, u8)>,
    positions: Vec<[f64; 3]>,
    points: PyReadonlyArray2<'py, f64>,
    ks: [Complex; 2],
    helicity: bool,
    singular: bool,
) -> PyResult<(C3<'py>, FieldOperatorContext)> {
    ieee(|| {
        evaluate_operator(
            py,
            make_basis(modes, positions).into(),
            points,
            ks,
            helicity,
            singular,
        )
    })
}
/// Record the matrix from cylindrical-wave coefficients to field samples: `fields::operator`.
#[pyfunction]
pub(crate) fn cylindrical_field_operator<'py>(
    py: Python<'py>,
    modes: Vec<(usize, f64, i32, u8)>,
    positions: Vec<[f64; 3]>,
    points: PyReadonlyArray2<'py, f64>,
    ks: [Complex; 2],
    helicity: bool,
    singular: bool,
) -> PyResult<(C3<'py>, FieldOperatorContext)> {
    ieee(|| {
        evaluate_operator(
            py,
            make_cyl_basis(modes, positions).into(),
            points,
            ks,
            helicity,
            singular,
        )
    })
}

/// The field operator, moved into an `(N, 3, modes)` array whose strides follow the
/// column-major matrix, and its context.
fn evaluate_operator<'py>(
    py: Python<'py>,
    basis: MultipoleBasis,
    points: PyReadonlyArray2<'py, f64>,
    ks: [Complex; 2],
    helicity: bool,
    singular: bool,
) -> PyResult<(C3<'py>, FieldOperatorContext)> {
    let points = rows(points.as_array(), "points")?;
    let (value, residual) = detached(py, move || {
        fields::operator(basis, points, ks, helicity, radial(singular))
    })?;
    Ok((
        operator_array(value)?.into_pyarray(py),
        FieldOperatorContext::new(residual),
    ))
}
