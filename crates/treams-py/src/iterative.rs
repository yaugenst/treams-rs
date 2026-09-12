//! Native reusable matrix-free sphere operators and one-use implicit pullbacks.
use std::sync::Arc;

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::Array2,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    iterative::{Convergence, GmresOptions, IterativeResidual, SphereCluster},
};

use crate::{
    error,
    tmatrix::{from_array, matrix, owned_matrix},
};

type Reports = Vec<(usize, f64, f64)>;

fn reports(values: &[Convergence]) -> Reports {
    values
        .iter()
        .map(|report| (report.iterations, report.residual_norm, report.rhs_norm))
        .collect()
}

/// Matrix-free geometry shared across illumination calls and their residuals.
#[pyclass(name = "NativeSphereCluster")]
#[derive(Debug)]
struct Cluster {
    operator: Arc<SphereCluster>,
}

#[pymethods]
impl Cluster {
    #[new]
    fn new(
        py: Python<'_>,
        lmax: u32,
        k0: f64,
        radii: PyReadonlyArray1<'_, f64>,
        epsilon: PyReadonlyArray1<'_, Complex>,
        positions: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
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
        let operator = py
            .detach(|| SphereCluster::new(lmax, k0, &radii, &epsilon, &positions))
            .map_err(error)?;
        Ok(Self {
            operator: Arc::new(operator),
        })
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.operator.dimension()
    }

    #[pyo3(signature = (incident, *, rtol=1e-10, atol=0.0, restart=30, max_iterations=300))]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        incident: PyReadonlyArray2<'_, Complex>,
        rtol: f64,
        atol: f64,
        restart: usize,
        max_iterations: usize,
    ) -> PyResult<(Bound<'py, PyArray2<Complex>>, Reports)> {
        let incident = from_array(incident)?;
        let options = GmresOptions {
            rtol,
            atol,
            restart,
            max_iterations,
        };
        let solution = py
            .detach(|| self.operator.solve(&incident, options))
            .map_err(error)?;
        Ok((
            owned_matrix(py, solution.value)?,
            reports(&solution.reports),
        ))
    }

    #[pyo3(signature = (incident, *, rtol=1e-10, atol=0.0, restart=30, max_iterations=300))]
    fn solve_with_pullback<'py>(
        &self,
        py: Python<'py>,
        incident: PyReadonlyArray2<'_, Complex>,
        rtol: f64,
        atol: f64,
        restart: usize,
        max_iterations: usize,
    ) -> PyResult<(Bound<'py, PyArray2<Complex>>, Context, Reports)> {
        let incident = from_array(incident)?;
        let options = GmresOptions {
            rtol,
            atol,
            restart,
            max_iterations,
        };
        let residual = py
            .detach(|| self.operator.clone().record(incident, options))
            .map_err(error)?;
        let convergence = reports(&residual.solution.reports);
        Ok((
            matrix(py, &residual.solution.value),
            Context {
                residual: Some(residual),
            },
            convergence,
        ))
    }
}

/// First-order native illumination residual, consumed by one pullback.
#[pyclass(name = "IterativeContext")]
#[derive(Debug)]
struct Context {
    residual: Option<IterativeResidual>,
}

type Gradients<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<Complex>>,
    f64,
    Bound<'py, PyArray2<Complex>>,
    Reports,
);

#[pymethods]
impl Context {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'_, Complex>,
    ) -> PyResult<Gradients<'py>> {
        let g = from_array(cotangent)?;
        let shape = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?
            .solution
            .value
            .shape();
        if g.shape() != shape {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let result = py.detach(|| residual.pullback(&g)).map_err(error)?;
        let positions = Array2::from_shape_fn((result.cluster.positions.len(), 3), |(i, j)| {
            result.cluster.positions[i][j]
        });
        Ok((
            result.cluster.radii.into_pyarray(py),
            positions.into_pyarray(py),
            result.cluster.epsilon.into_pyarray(py),
            result.cluster.k0,
            owned_matrix(py, result.incident)?,
            reports(&result.reports),
        ))
    }
}

/// Register the matrix-free CPU solver independently of GPU features.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Cluster>()?;
    module.add_class::<Context>()
}
