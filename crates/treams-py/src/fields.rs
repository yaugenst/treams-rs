//! `field`, `field_operator` and their cylindrical twins with `FieldContext` and
//! `FieldOperatorContext`: electric fields of multipole bases at sample points
//! (`treams_core::fields`).

use crate::{
    args::{make_basis, make_cyl_basis},
    context::{context, cotangent_error, detached, error, radial},
    convert::{
        C1, C2, C3, Cotangent, R1, R2, RealTangent, Tangent, all_finite, cotangent_view,
        finite_tangent, layout_error, merged_cotangent, rows, rows_array, vector_tangent,
    },
};
use nalgebra::DMatrix;
use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::{Array3, Ix2, Ix3},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    basis::MultipoleBasis,
    fields::{self, FieldResidual},
    fpenv::ieee,
    saved::SavedState,
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

/// Owned and validated geometry directions, ready for work without the GIL.
struct GeometryTangent {
    points: Vec<[f64; 3]>,
    positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
}

impl GeometryTangent {
    fn new(
        points: &RealTangent<'_>,
        positions: &RealTangent<'_>,
        ks: &Tangent<'_>,
        samples: usize,
        centres: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            points: rows(
                finite_tangent::<_, Ix2>(points, &[samples, 3])?,
                "points tangent",
            )?,
            positions: rows(
                finite_tangent::<_, Ix2>(positions, &[centres, 3])?,
                "positions tangent",
            )?,
            ks: vector_tangent(ks, 2)?
                .try_into()
                .map_err(|_| PyValueError::new_err("wavenumber tangent must have shape (2,)"))?,
        })
    }
}

impl FieldContext {
    fn cotangent(
        &self,
        cotangent: &Cotangent<'_>,
    ) -> PyResult<(&FieldResidual, Vec<[Complex; 3]>)> {
        let residual = &self.residual;
        let expected: [usize; 2] = residual.shape().into();
        let g = rows(cotangent_view::<_, Ix2>(cotangent, &expected)?, "cotangent")?;
        if all_finite(g.as_flattened()) {
            Ok((residual, g))
        } else {
            Err(cotangent_error(&expected))
        }
    }

    fn push<'py>(
        &self,
        py: Python<'py>,
        coefficients: &Tangent<'py>,
        points: &RealTangent<'py>,
        positions: &RealTangent<'py>,
        ks: &Tangent<'py>,
        kz: Option<&RealTangent<'py>>,
    ) -> PyResult<C2<'py>> {
        let residual = &self.residual;
        let (modes, centres) = residual.input_sizes();
        let coefficients = vector_tangent(coefficients, modes)?;
        let tangent = GeometryTangent::new(points, positions, ks, residual.shape().0, centres)?;
        let kz = kz.map(|kz| vector_tangent(kz, modes)).transpose()?;
        let value = detached(py, move || match kz {
            Some(kz) => residual.pushforward_axial(
                &coefficients,
                &tangent.points,
                &tangent.positions,
                tangent.ks,
                &kz,
            ),
            None => residual.pushforward(
                &coefficients,
                &tangent.points,
                &tangent.positions,
                tangent.ks,
            ),
        })?;
        rows_array(py, value)
    }
}
#[pymethods]
impl FieldContext {
    #[staticmethod]
    fn _state_spec(
        modes: usize,
        positions: usize,
        points: usize,
        cylindrical: bool,
    ) -> PyResult<usize> {
        ieee(|| FieldResidual::state_size(modes, positions, points, cylindrical).map_err(error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| Ok(detached(py, || self.residual.save_state())?.into_pyarray(py)))
    }

    #[staticmethod]
    fn _from_state(py: Python<'_>, state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| {
            let bytes = state.as_slice()?;
            detached(py, || FieldResidual::from_state(bytes)).map(Self::new)
        })
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        coefficients: Tangent<'py>,
        points: RealTangent<'py>,
        positions: RealTangent<'py>,
        ks: Tangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| self.push(py, &coefficients, &points, &positions, &ks, None))
    }

    fn pushforward_axial<'py>(
        &self,
        py: Python<'py>,
        coefficients: Tangent<'py>,
        points: RealTangent<'py>,
        positions: RealTangent<'py>,
        ks: Tangent<'py>,
        kz: RealTangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| self.push(py, &coefficients, &points, &positions, &ks, Some(&kz)))
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C1<'py>, R2<'py>, R2<'py>, C1<'py>)> {
        ieee(|| {
            let (residual, g) = self.cotangent(&cotangent)?;
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
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C1<'py>, R2<'py>, R2<'py>, C1<'py>, R1<'py>)> {
        ieee(|| {
            let (residual, g) = self.cotangent(&cotangent)?;
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
    fn cotangent(
        &self,
        cotangent: &Cotangent<'_>,
    ) -> PyResult<(&fields::OperatorResidual, DMatrix<Complex>)> {
        let residual = &self.residual;
        // The native rows run over (sample, Cartesian component).
        let (rows, modes) = residual.shape();
        let g = merged_cotangent::<Ix3>(cotangent, &[rows / 3, 3, modes], (rows, modes))?;
        Ok((residual, g))
    }

    fn push<'py>(
        &self,
        py: Python<'py>,
        points: &RealTangent<'py>,
        positions: &RealTangent<'py>,
        ks: &Tangent<'py>,
        kz: Option<&RealTangent<'py>>,
    ) -> PyResult<C3<'py>> {
        let residual = &self.residual;
        let (components, modes) = residual.shape();
        let tangent = GeometryTangent::new(
            points,
            positions,
            ks,
            components / 3,
            residual.position_count(),
        )?;
        let kz = kz.map(|kz| vector_tangent(kz, modes)).transpose()?;
        let value = detached(py, move || match kz {
            Some(kz) => {
                residual.pushforward_axial(&tangent.points, &tangent.positions, tangent.ks, &kz)
            }
            None => residual.pushforward(&tangent.points, &tangent.positions, tangent.ks),
        })?;
        Ok(operator_array(value)?.into_pyarray(py))
    }
}
#[pymethods]
impl FieldOperatorContext {
    #[staticmethod]
    fn _state_spec(
        modes: usize,
        positions: usize,
        points: usize,
        cylindrical: bool,
    ) -> PyResult<usize> {
        ieee(|| {
            fields::OperatorResidual::state_size(modes, positions, points, cylindrical)
                .map_err(error)
        })
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| Ok(detached(py, || self.residual.save_state())?.into_pyarray(py)))
    }

    #[staticmethod]
    fn _from_state(py: Python<'_>, state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| {
            let bytes = state.as_slice()?;
            detached(py, || fields::OperatorResidual::from_state(bytes)).map(Self::new)
        })
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        points: RealTangent<'py>,
        positions: RealTangent<'py>,
        ks: Tangent<'py>,
    ) -> PyResult<C3<'py>> {
        ieee(|| self.push(py, &points, &positions, &ks, None))
    }

    fn pushforward_axial<'py>(
        &self,
        py: Python<'py>,
        points: RealTangent<'py>,
        positions: RealTangent<'py>,
        ks: Tangent<'py>,
        kz: RealTangent<'py>,
    ) -> PyResult<C3<'py>> {
        ieee(|| self.push(py, &points, &positions, &ks, Some(&kz)))
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, R2<'py>, C1<'py>)> {
        ieee(|| {
            let (residual, g) = self.cotangent(&cotangent)?;
            let result = detached(py, move || residual.pullback(&g))?;
            Ok((
                rows_array(py, result.points)?,
                rows_array(py, result.positions)?,
                result.ks.to_vec().into_pyarray(py),
            ))
        })
    }
    fn pullback_axial<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, R2<'py>, C1<'py>, R1<'py>)> {
        ieee(|| {
            let (residual, g) = self.cotangent(&cotangent)?;
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
