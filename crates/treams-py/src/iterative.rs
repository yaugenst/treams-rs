//! `IterativeSphereCluster` and `IterativeContext`: the matrix-free sphere cluster,
//! solved with GMRES (`treams_core::cluster::IterativeSphereCluster`).
use std::sync::Arc;

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use treams_core::{
    Complex,
    cluster::{self as core_cluster, IterativeResidual},
    fpenv::ieee,
    linalg::{Convergence, GmresOptions},
};

use crate::{
    args::spheres,
    context::{context, detached},
    convert::{C1, C2, Cotangent, R1, R2, from_array, matrix, owned_matrix, rows_array},
};

/// GMRES convergence of each illumination as `(iterations, residual_norm, rhs_norm)`.
type ConvergenceTuples = Vec<(usize, f64, f64)>;

fn convergence_tuples(values: &[Convergence]) -> ConvergenceTuples {
    values
        .iter()
        .map(|entry| (entry.iterations, entry.residual_norm, entry.rhs_norm))
        .collect()
}

/// GMRES options of a solve.
const fn gmres(rtol: f64, atol: f64, restart: usize, max_iterations: usize) -> GmresOptions {
    GmresOptions {
        rtol,
        atol,
        restart,
        max_iterations,
    }
}

/// Matrix-free geometry shared across illumination calls and their residuals.
#[pyclass(module = "treams_rs._native", frozen)]
#[derive(Debug)]
pub(crate) struct IterativeSphereCluster {
    operator: Arc<core_cluster::IterativeSphereCluster>,
}

#[pymethods]
impl IterativeSphereCluster {
    /// The matrix-free cluster of homogeneous spheres in vacuum:
    /// `cluster::IterativeSphereCluster::new`.
    #[new]
    fn new(
        py: Python<'_>,
        lmax: u32,
        k0: f64,
        radii: PyReadonlyArray1<'_, f64>,
        epsilon: PyReadonlyArray1<'_, Complex>,
        positions: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        ieee(|| {
            let (radii, epsilon, positions) = spheres(&radii, &epsilon, positions)?;
            let operator = detached(py, || {
                core_cluster::IterativeSphereCluster::new(lmax, k0, &radii, &epsilon, &positions)
            })?;
            Ok(Self {
                operator: Arc::new(operator),
            })
        })
    }

    /// The number of rows of `I - T C`.
    #[getter]
    fn dimension(&self) -> usize {
        ieee(|| self.operator.dimension())
    }

    /// The scattered coefficients of each incident column, solved with GMRES, and the
    /// convergence of each solve.
    #[pyo3(signature = (incident, *, rtol=1e-10, atol=0.0, restart=30, max_iterations=300))]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        incident: PyReadonlyArray2<'_, Complex>,
        rtol: f64,
        atol: f64,
        restart: usize,
        max_iterations: usize,
    ) -> PyResult<(C2<'py>, ConvergenceTuples)> {
        ieee(|| {
            let incident = from_array(incident, "incident")?;
            let options = gmres(rtol, atol, restart, max_iterations);
            let solution = detached(py, || self.operator.solve(&incident, options))?;
            Ok((
                owned_matrix(py, solution.value)?,
                convergence_tuples(&solution.convergence),
            ))
        })
    }

    /// As `solve`, with the context between the coefficients and the convergence.
    #[pyo3(signature = (incident, *, rtol=1e-10, atol=0.0, restart=30, max_iterations=300))]
    fn record<'py>(
        &self,
        py: Python<'py>,
        incident: PyReadonlyArray2<'_, Complex>,
        rtol: f64,
        atol: f64,
        restart: usize,
        max_iterations: usize,
    ) -> PyResult<(C2<'py>, IterativeContext, ConvergenceTuples)> {
        ieee(|| {
            let incident = from_array(incident, "incident")?;
            let options = gmres(rtol, atol, restart, max_iterations);
            let residual = detached(py, || self.operator.clone().record(incident, options))?;
            let convergence = convergence_tuples(&residual.solution().convergence);
            // A C-ordered copy: the residual keeps the solution for the pullback.
            Ok((
                matrix(py, &residual.solution().value)?,
                IterativeContext::new(residual),
                convergence,
            ))
        })
    }
}

context!(
    /// The context of `IterativeSphereCluster.record`, whose pullback solves the
    /// adjoint systems with GMRES.
    IterativeContext(IterativeResidual)
);

#[pymethods]
impl IterativeContext {
    /// Cotangents of `(k0, radii, epsilon, positions, incident)`, in the forward
    /// argument order, then the convergence of each adjoint solve.
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'_>,
    ) -> PyResult<(f64, R1<'py>, C1<'py>, R2<'py>, C2<'py>, ConvergenceTuples)> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, IterativeResidual::shape)?;
            let result = detached(py, || residual.pullback(&g))?;
            Ok((
                result.cluster.k0,
                result.cluster.radii.into_pyarray(py),
                result.cluster.epsilon.into_pyarray(py),
                rows_array(py, result.cluster.positions)?,
                owned_matrix(py, result.incident)?,
                convergence_tuples(&result.convergence),
            ))
        })
    }
}
