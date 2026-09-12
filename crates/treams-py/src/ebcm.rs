//! Axisymmetric Q matrices and opaque surface/medium pullbacks.
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, ndarray::Array2};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{Complex, ebcm, waves::Mode};

use crate::{
    error,
    tmatrix::{from_array, owned_matrix},
};

#[pyclass]
#[derive(Debug)]
struct QContext {
    residual: Option<ebcm::QResidual>,
}

type QGradient<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray1<Complex>>,
);

#[pymethods]
impl QContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<QGradient<'py>> {
        let g = from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.shape() != residual.shape() {
            return Err(PyValueError::new_err(
                "EBCM cotangent shape must match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let gradient = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            gradient.radii.into_pyarray(py),
            gradient.slopes.into_pyarray(py),
            Array2::from_shape_fn((2, 2), |(i, j)| gradient.ks[i][j]).into_pyarray(py),
            gradient.zs.to_vec().into_pyarray(py),
        ))
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)] // Explicit numerical inputs at the Python boundary.
fn ebcm_qmat<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray2<'py, f64>,
    to: Vec<(i32, i32, u8)>,
    source: Vec<(i32, i32, u8)>,
    ks: PyReadonlyArray2<'py, Complex>,
    zs: [Complex; 2],
    singular: bool,
    legacy: bool,
) -> PyResult<(Bound<'py, PyArray2<Complex>>, QContext)> {
    let samples = samples.as_array();
    let ks = ks.as_array();
    if samples.ncols() != 4 || ks.shape() != [2, 2] {
        return Err(PyValueError::new_err(
            "EBCM requires (N, 4) surface samples and (2, 2) wavenumbers",
        ));
    }
    let surface = ebcm::Surface {
        theta: samples.column(0).to_vec(),
        weights: samples.column(1).to_vec(),
        radii: samples.column(2).to_vec(),
        slopes: samples.column(3).to_vec(),
    };
    let ks = std::array::from_fn(|i| std::array::from_fn(|j| ks[(i, j)]));
    let modes = |values: Vec<(i32, i32, u8)>| {
        values
            .into_iter()
            .map(|(l, m, pol)| Mode { l, m, pol })
            .collect()
    };
    let (value, residual) = py
        .detach(move || ebcm::qmat(modes(to), modes(source), surface, ks, zs, singular, legacy))
        .map_err(error)?;
    Ok((
        owned_matrix(py, value)?,
        QContext {
            residual: Some(residual),
        },
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<QContext>()?;
    m.add_function(wrap_pyfunction!(ebcm_qmat, m)?)?;
    Ok(())
}
