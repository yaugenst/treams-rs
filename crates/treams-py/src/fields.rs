//! Field samples and one-use native pullback contexts.
#![allow(clippy::indexing_slicing)] // Fixed triples and validated array dimensions.

use crate::{basis::make_basis, error};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::Array2,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    fields::{self, FieldResidual},
    special::Radial,
};

fn triples<T: numpy::Element + Copy>(value: PyReadonlyArray2<'_, T>) -> PyResult<Vec<[T; 3]>> {
    let array = value.as_array();
    if array.ncols() != 3 {
        return Err(PyValueError::new_err("array must have shape (N, 3)"));
    }
    Ok(array
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect())
}
fn array<'py, T: numpy::Element + Copy>(
    py: Python<'py>,
    value: &[[T; 3]],
) -> Bound<'py, PyArray2<T>> {
    Array2::from_shape_fn((value.len(), 3), |(i, j)| value[i][j]).into_pyarray(py)
}

#[pyclass]
#[derive(Debug)]
struct FieldContext {
    residual: Option<FieldResidual>,
}

type Gradient<'py> = (
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<Complex>>,
);

#[pymethods]
impl FieldContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<Gradient<'py>> {
        let g = triples(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.len() != residual.value.len()
            || g.iter()
                .flatten()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(PyValueError::new_err(
                "cotangent must be finite and match the field shape",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let result = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            result.coefficients.into_pyarray(py),
            array(py, &result.points),
            array(py, &result.origins),
            result.ks.to_vec().into_pyarray(py),
        ))
    }
}

#[pyfunction]
fn field<'py>(
    py: Python<'py>,
    modes: Vec<(usize, i32, i32, u8)>,
    origins: Vec<[f64; 3]>,
    coefficients: PyReadonlyArray1<'py, Complex>,
    points: PyReadonlyArray2<'py, f64>,
    ks: [Complex; 2],
    helicity: bool,
    outgoing: bool,
) -> PyResult<(Bound<'py, PyArray2<Complex>>, FieldContext)> {
    let basis = make_basis(modes, origins);
    let coefficients = coefficients.to_vec()?;
    let points = triples(points)?;
    let radial = if outgoing {
        Radial::Outgoing
    } else {
        Radial::Regular
    };
    let residual = py
        .detach(move || fields::field(basis, coefficients, points, ks, helicity, radial))
        .map_err(error)?;
    Ok((
        array(py, &residual.value),
        FieldContext {
            residual: Some(residual),
        },
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FieldContext>()?;
    m.add_function(wrap_pyfunction!(field, m)?)?;
    Ok(())
}
