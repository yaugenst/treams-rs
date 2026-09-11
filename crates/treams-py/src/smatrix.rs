//! Native S-matrix blocks and owned reverse contexts.
#![allow(clippy::indexing_slicing)] // Validated four-block NumPy shapes.

use nalgebra::DMatrix;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray4, PyReadonlyArray2, PyReadonlyArray4,
    ndarray::{Array2, Array4},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    smatrix::{self, ArrayResidual, Blocks, FresnelResidual, PropagationResidual, StackResidual},
};

use crate::error;

fn from_array(value: PyReadonlyArray4<'_, Complex>) -> PyResult<Blocks> {
    let a = value.as_array();
    let s = a.shape();
    if s[0] != 2 || s[1] != 2 || s[2] == 0 || s[2] != s[3] {
        return Err(PyValueError::new_err(
            "S matrices require shape (2, 2, n, n) with n > 0",
        ));
    }
    if a.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(PyValueError::new_err("S matrices must be finite"));
    }
    Ok(std::array::from_fn(|b| {
        DMatrix::from_fn(s[2], s[3], |i, j| a[(b / 2, b % 2, i, j)])
    }))
}
fn array<'py>(py: Python<'py>, value: &Blocks) -> Bound<'py, PyArray4<Complex>> {
    let (d, c) = value[0].shape();
    Array4::from_shape_fn((2, 2, d, c), |(a, b, i, j)| value[2 * a + b][(i, j)]).into_pyarray(py)
}

#[pyclass]
#[derive(Debug)]
struct ArrayContext {
    residual: Option<ArrayResidual>,
}
type ArrayGradient<'py> = (Bound<'py, PyArray2<Complex>>, Bound<'py, PyArray4<Complex>>);
#[pymethods]
impl ArrayContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray4<'py, Complex>,
    ) -> PyResult<ArrayGradient<'py>> {
        let g = from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g[0].shape() != residual.value[0].shape() {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (response, channels) = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((crate::tmatrix::matrix(py, &response), array(py, &channels)))
    }
}
#[pyfunction]
fn smatrix_from_array<'py>(
    py: Python<'py>,
    response: PyReadonlyArray2<'py, Complex>,
    channels: PyReadonlyArray4<'py, Complex>,
) -> PyResult<(Bound<'py, PyArray4<Complex>>, ArrayContext)> {
    let response = crate::tmatrix::from_array(response)?;
    let a = channels.as_array();
    let s = a.shape();
    if s[0] != 2 || s[1] != 2 {
        return Err(PyValueError::new_err(
            "channels require shape (2, 2, multipoles, plane modes)",
        ));
    }
    let channels =
        std::array::from_fn(|b| DMatrix::from_fn(s[2], s[3], |i, j| a[(b / 2, b % 2, i, j)]));
    let residual = py
        .detach(move || smatrix::from_array(response, channels))
        .map_err(error)?;
    Ok((
        array(py, &residual.value),
        ArrayContext {
            residual: Some(residual),
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct SMatrixContext {
    residual: Option<StackResidual>,
}

type Pair<'py> = (Bound<'py, PyArray4<Complex>>, Bound<'py, PyArray4<Complex>>);

#[pymethods]
impl SMatrixContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray4<'py, Complex>,
    ) -> PyResult<Pair<'py>> {
        let g = from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g[0].shape() != residual.value[0].shape() {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (lower, upper) = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((array(py, &lower), array(py, &upper)))
    }
}

#[pyfunction]
fn smatrix_add<'py>(
    py: Python<'py>,
    lower: PyReadonlyArray4<'py, Complex>,
    upper: PyReadonlyArray4<'py, Complex>,
) -> PyResult<(Bound<'py, PyArray4<Complex>>, SMatrixContext)> {
    let lower = from_array(lower)?;
    let upper = from_array(upper)?;
    let residual = py
        .detach(move || smatrix::add(lower, upper))
        .map_err(error)?;
    Ok((
        array(py, &residual.value),
        SMatrixContext {
            residual: Some(residual),
        },
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ArrayContext>()?;
    module.add_function(wrap_pyfunction!(smatrix_from_array, module)?)?;
    module.add_class::<SMatrixContext>()?;
    module.add_function(wrap_pyfunction!(smatrix_add, module)?)?;
    module.add_class::<FresnelContext>()?;
    module.add_class::<PropagationContext>()?;
    module.add_function(wrap_pyfunction!(fresnel, module)?)?;
    module.add_function(wrap_pyfunction!(propagation, module)?)?;
    Ok(())
}

#[pyclass]
#[derive(Debug)]
struct FresnelContext {
    residual: Option<FresnelResidual>,
}

type FresnelGradient<'py> = (
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray1<Complex>>,
);

#[pymethods]
impl FresnelContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray4<'py, Complex>,
    ) -> PyResult<FresnelGradient<'py>> {
        let g = from_array(cotangent)?;
        if g[0].nrows() != 2 {
            return Err(PyValueError::new_err(
                "Fresnel cotangent requires shape (2, 2, 2, 2)",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (ks, kz, z) = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            Array2::from_shape_fn((2, 2), |(i, j)| ks[i][j]).into_pyarray(py),
            Array2::from_shape_fn((2, 2), |(i, j)| kz[i][j]).into_pyarray(py),
            z.to_vec().into_pyarray(py),
        ))
    }
}

#[pyfunction]
fn fresnel(
    py: Python<'_>,
    ks: [[Complex; 2]; 2],
    kz: [[Complex; 2]; 2],
    z: [Complex; 2],
) -> PyResult<(Bound<'_, PyArray4<Complex>>, FresnelContext)> {
    let residual = py
        .detach(move || smatrix::fresnel(ks, kz, z))
        .map_err(error)?;
    Ok((
        array(py, &residual.value),
        FresnelContext {
            residual: Some(residual),
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct PropagationContext {
    residual: Option<PropagationResidual>,
}

type PropagationGradient<'py> = (Bound<'py, PyArray2<Complex>>, Bound<'py, PyArray1<f64>>);

#[pymethods]
impl PropagationContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray4<'py, Complex>,
    ) -> PyResult<PropagationGradient<'py>> {
        let g = from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g[0].shape() != residual.value[0].shape() {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (vectors, distance) = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            Array2::from_shape_fn((vectors.len(), 3), |(i, j)| vectors[i][j]).into_pyarray(py),
            distance.to_vec().into_pyarray(py),
        ))
    }
}

#[pyfunction]
fn propagation(
    py: Python<'_>,
    vectors: Vec<[Complex; 3]>,
    distance: [f64; 3],
) -> PyResult<(Bound<'_, PyArray4<Complex>>, PropagationContext)> {
    let residual = py
        .detach(move || smatrix::propagation(vectors, distance))
        .map_err(error)?;
    Ok((
        array(py, &residual.value),
        PropagationContext {
            residual: Some(residual),
        },
    ))
}
