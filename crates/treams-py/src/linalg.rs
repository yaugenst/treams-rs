//! Native linear algebra contexts shared by scattering and band workflows.
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{Complex, linalg};

use crate::{
    error,
    tmatrix::{from_array, matrix},
};

type MatrixPair<'py> = (Bound<'py, PyArray2<Complex>>, Bound<'py, PyArray2<Complex>>);

#[pyclass]
#[derive(Debug)]
struct SolveContext {
    residual: Option<linalg::SolveResidual>,
}

#[pymethods]
impl SolveContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<MatrixPair<'py>> {
        let g = from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.shape() != residual.value.shape() {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (a, b) = py.detach(move || residual.pullback(g)).map_err(error)?;
        Ok((matrix(py, &a), matrix(py, &b)))
    }
}

#[pyfunction]
fn linear_solve<'py>(
    py: Python<'py>,
    operator: PyReadonlyArray2<'py, Complex>,
    rhs: PyReadonlyArray2<'py, Complex>,
) -> PyResult<(Bound<'py, PyArray2<Complex>>, SolveContext)> {
    let a = from_array(operator)?;
    let b = from_array(rhs)?;
    let residual = py.detach(move || linalg::solve(&a, b)).map_err(error)?;
    Ok((
        matrix(py, &residual.value),
        SolveContext {
            residual: Some(residual),
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct EigenContext {
    residual: Option<linalg::EigenResidual>,
}

#[pymethods]
impl EigenContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        eigenvalues: PyReadonlyArray1<'py, Complex>,
        eigenvectors: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<Bound<'py, PyArray2<Complex>>> {
        let values: Vec<_> = eigenvalues.as_array().iter().copied().collect();
        let vectors = from_array(eigenvectors)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if values.len() != residual.values.len()
            || vectors.shape() != residual.vectors.shape()
            || values
                .iter()
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(PyValueError::new_err(
                "cotangent shapes must match the finite eigensystem outputs",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let g = py
            .detach(move || residual.pullback(&values, vectors))
            .map_err(error)?;
        Ok(matrix(py, &g))
    }
}

type Eigensystem<'py> = (
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray2<Complex>>,
    EigenContext,
);

#[pyfunction]
fn eig<'py>(
    py: Python<'py>,
    operator: PyReadonlyArray2<'py, Complex>,
) -> PyResult<Eigensystem<'py>> {
    let a = from_array(operator)?;
    let residual = py.detach(move || linalg::eig(&a)).map_err(error)?;
    Ok((
        residual.values.clone().into_pyarray(py),
        matrix(py, &residual.vectors),
        EigenContext {
            residual: Some(residual),
        },
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SolveContext>()?;
    m.add_class::<EigenContext>()?;
    m.add_function(wrap_pyfunction!(linear_solve, m)?)?;
    m.add_function(wrap_pyfunction!(eig, m)?)?;
    Ok(())
}
