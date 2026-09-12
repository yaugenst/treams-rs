//! Shared immutable CPU factors and thin incident-field residuals.

use std::sync::Arc;

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    illumination::{Factor, Gradient, Residual},
};

use crate::{
    error,
    tmatrix::{from_array, matrix, owned_matrix},
};

#[pyclass(name = "InteractionFactor", frozen)]
#[derive(Debug)]
struct InteractionFactor {
    factor: Arc<Factor>,
    blocked: bool,
}

#[pymethods]
impl InteractionFactor {
    #[new]
    fn new(
        py: Python<'_>,
        local: PyReadonlyArray2<'_, Complex>,
        coupling: PyReadonlyArray2<'_, Complex>,
    ) -> PyResult<Self> {
        let local = from_array(local)?;
        let coupling = from_array(coupling)?;
        let factor = py
            .detach(move || Factor::new(local, coupling))
            .map_err(error)?;
        Ok(Self {
            factor: Arc::new(factor),
            blocked: false,
        })
    }

    #[staticmethod]
    fn from_blocks(
        py: Python<'_>,
        local: Vec<PyReadonlyArray2<'_, Complex>>,
        coupling: PyReadonlyArray2<'_, Complex>,
    ) -> PyResult<Self> {
        let local = local
            .into_iter()
            .map(from_array)
            .collect::<PyResult<Vec<_>>>()?;
        let coupling = from_array(coupling)?;
        let factor = py
            .detach(move || Factor::from_blocks(local, coupling))
            .map_err(error)?;
        Ok(Self {
            factor: Arc::new(factor),
            blocked: true,
        })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.factor.dimension()
    }

    fn solve<'py>(
        &self,
        py: Python<'py>,
        incident: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<Bound<'py, PyArray2<Complex>>> {
        let incident = from_array(incident)?;
        let value = py.detach(|| self.factor.solve(&incident)).map_err(error)?;
        owned_matrix(py, value)
    }

    fn record<'py>(
        &self,
        py: Python<'py>,
        incident: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<(
        Bound<'py, PyArray2<Complex>>,
        InteractionIlluminationContext,
    )> {
        let incident = from_array(incident)?;
        let residual = py.detach(|| self.factor.record(incident)).map_err(error)?;
        Ok((
            matrix(py, &residual.value),
            InteractionIlluminationContext {
                residual: Some(residual),
                blocked: self.blocked,
            },
        ))
    }
}

#[pyclass]
#[derive(Debug)]
struct InteractionIlluminationContext {
    residual: Option<Residual>,
    blocked: bool,
}

impl InteractionIlluminationContext {
    fn gradient(
        &mut self,
        py: Python<'_>,
        cotangent: PyReadonlyArray2<'_, Complex>,
    ) -> PyResult<Gradient> {
        let g = from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.shape() != residual.value.shape() {
            return Err(PyValueError::new_err(
                "cotangent shape must match scattered fields",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        py.detach(move || residual.pullback(&g)).map_err(error)
    }
}

type DenseGradient<'py> = (
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray2<Complex>>,
);
type BlockGradient<'py> = (
    Vec<Bound<'py, PyArray2<Complex>>>,
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray2<Complex>>,
);

#[pymethods]
impl InteractionIlluminationContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<DenseGradient<'py>> {
        if self.blocked {
            return Err(PyValueError::new_err(
                "use pullback_blocks for block-local factors",
            ));
        }
        let mut g = self.gradient(py, cotangent)?;
        let local = g
            .local
            .pop()
            .ok_or_else(|| PyValueError::new_err("missing local matrix gradient"))?;
        Ok((
            owned_matrix(py, local)?,
            owned_matrix(py, g.coupling)?,
            owned_matrix(py, g.incident)?,
        ))
    }

    fn pullback_blocks<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<BlockGradient<'py>> {
        if !self.blocked {
            return Err(PyValueError::new_err(
                "use pullback for dense-local factors",
            ));
        }
        let g = self.gradient(py, cotangent)?;
        Ok((
            g.local
                .into_iter()
                .map(|m| owned_matrix(py, m))
                .collect::<PyResult<Vec<_>>>()?,
            owned_matrix(py, g.coupling)?,
            owned_matrix(py, g.incident)?,
        ))
    }
}

#[pyfunction]
fn cluster_factor(
    py: Python<'_>,
    lmax: u32,
    k0: f64,
    radii: PyReadonlyArray1<'_, f64>,
    epsilon: PyReadonlyArray1<'_, Complex>,
    positions: PyReadonlyArray2<'_, f64>,
) -> PyResult<InteractionFactor> {
    let radii = radii.to_vec()?;
    let epsilon = epsilon.to_vec()?;
    let positions = positions.as_array();
    if positions.ncols() != 3 {
        return Err(PyValueError::new_err(
            "positions must have shape (particles, 3)",
        ));
    }
    let positions: Vec<_> = positions
        .rows()
        .into_iter()
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    let factor = py
        .detach(|| treams_core::tmatrix::cluster_factor(lmax, k0, &radii, &epsilon, &positions))
        .map_err(error)?;
    Ok(InteractionFactor {
        factor: Arc::new(factor),
        blocked: true,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cluster_factor, m)?)?;
    m.add_class::<InteractionFactor>()?;
    m.add_class::<InteractionIlluminationContext>()?;
    Ok(())
}
