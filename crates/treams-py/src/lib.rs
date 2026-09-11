//! NumPy/PyO3 boundary for the treams Rust core.
// PyO3 extracts owned borrow guards by value; array shapes are validated before indexing.
#![allow(clippy::needless_pass_by_value, clippy::indexing_slicing)]

mod basis;
mod channels;
mod coordinates;
mod cylinder;
mod ebcm;
mod fields;
mod lattice;
mod linalg;
mod smatrix;
mod special;
mod tmatrix;
mod ufunc;

use num_complex::Complex64;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::Array2,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::coeffs::{Material, Matrix2, MieResidual, mie_forward};
use treams_core::special::{Radial, spherical};

fn error(error: treams_core::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn materials(
    epsilon: &[Complex64],
    mu: &[Complex64],
    kappa: &[Complex64],
) -> PyResult<Vec<Material>> {
    if epsilon.len() != mu.len() || epsilon.len() != kappa.len() {
        return Err(PyValueError::new_err(
            "epsilon, mu and kappa must have equal lengths",
        ));
    }
    Ok(epsilon
        .iter()
        .zip(mu)
        .zip(kappa)
        .map(|((&epsilon, &mu), &kappa)| Material { epsilon, mu, kappa })
        .collect())
}

fn matrix<'py>(py: Python<'py>, value: &Matrix2) -> Bound<'py, PyArray2<Complex64>> {
    Array2::from_shape_fn((2, 2), |(i, j)| value[(i, j)]).into_pyarray(py)
}

#[pyclass]
#[derive(Debug)]
struct MieContext {
    residual: Option<MieResidual>,
}

type Gradients<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<Complex64>>,
    Bound<'py, PyArray1<Complex64>>,
    Bound<'py, PyArray1<Complex64>>,
);

#[pymethods]
impl MieContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<Gradients<'py>> {
        let data = cotangent.as_array();
        if data.dim() != (2, 2) || data.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(PyValueError::new_err(
                "cotangent must be a finite complex array with shape (2, 2)",
            ));
        }
        let g = Matrix2::from_fn(|i, j| data[(i, j)]);
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let result = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            result.sizes.into_pyarray(py),
            result.epsilon.into_pyarray(py),
            result.mu.into_pyarray(py),
            result.kappa.into_pyarray(py),
        ))
    }
}

#[pyfunction]
fn mie<'py>(
    py: Python<'py>,
    l: u32,
    sizes: PyReadonlyArray1<'py, f64>,
    epsilon: PyReadonlyArray1<'py, Complex64>,
    mu: PyReadonlyArray1<'py, Complex64>,
    kappa: PyReadonlyArray1<'py, Complex64>,
) -> PyResult<(Bound<'py, PyArray2<Complex64>>, MieContext)> {
    let sizes = sizes.to_vec()?;
    let mat = materials(&epsilon.to_vec()?, &mu.to_vec()?, &kappa.to_vec()?)?;
    let residual = py
        .detach(move || mie_forward(l, &sizes, &mat))
        .map_err(error)?;
    Ok((
        matrix(py, &residual.value),
        MieContext {
            residual: Some(residual),
        },
    ))
}

#[pyfunction]
fn radial(l: u32, z: Complex64, outgoing: bool) -> PyResult<(Complex64, Complex64, Complex64)> {
    let kind = if outgoing {
        Radial::Outgoing
    } else {
        Radial::Regular
    };
    let result = spherical(l, z, kind).map_err(error)?;
    Ok((result.value, result.first, result.second))
}

#[pyfunction]
fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    channels::register(m)?;
    smatrix::register(m)?;
    tmatrix::register(m)?;
    cylinder::register(m)?;
    basis::register(m)?;
    fields::register(m)?;
    lattice::register(m)?;
    linalg::register(m)?;
    ebcm::register(m)?;
    special::register(m)?;
    coordinates::register(m)?;
    ufunc::register(m)?;
    m.add_class::<MieContext>()?;
    m.add_function(wrap_pyfunction!(build_profile, m)?)?;
    m.add_function(wrap_pyfunction!(mie, m)?)?;
    m.add_function(wrap_pyfunction!(radial, m)?)?;
    Ok(())
}
