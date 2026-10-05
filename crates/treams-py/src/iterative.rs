//! `IterativeSphereCluster` and `IterativeContext`: the matrix-free sphere cluster,
//! solved with GMRES (`treams_core::cluster::IterativeSphereCluster`).
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, ndarray::Ix2};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    cluster::{self as core_cluster, IterativeResidual},
    fpenv::ieee,
    linalg::{Convergence, GmresOptions},
    saved::SavedState,
};

use crate::{
    args::spheres,
    context::{context, detached, error, state_array},
    convert::{
        C1, C2, Cotangent, R1, R2, RealTangent, Tangent, finite_tangent, from_array, matrix,
        matrix_cotangent, matrix_tangent, owned_matrix, rows_array, vector_tangent,
    },
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
            let residual = detached(py, || self.operator.record(incident, options))?;
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
    #[staticmethod]
    fn _state_spec(lmax: u32, particles: usize, columns: usize) -> PyResult<usize> {
        ieee(|| IterativeResidual::state_size(lmax, particles, columns).map_err(error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(py: Python<'_>, state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| {
            let bytes = state.as_slice()?;
            detached(py, || IterativeResidual::from_state(bytes)).map(Self::new)
        })
    }

    /// Directional derivative in `(k0, radii, epsilon, positions, incident)` order,
    /// followed by the independent convergence certificate of each tangent solve.
    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        k0: f64,
        radii: RealTangent<'_>,
        epsilon: Tangent<'_>,
        positions: RealTangent<'_>,
        incident: Tangent<'_>,
    ) -> PyResult<(C2<'py>, ConvergenceTuples)> {
        ieee(|| {
            let residual = &self.residual;
            if !k0.is_finite() {
                return Err(PyValueError::new_err("k0 tangent must be finite"));
            }
            let radii = vector_tangent(&radii, residual.particles())?;
            let epsilon = vector_tangent(&epsilon, residual.particles())?;
            let positions = finite_tangent::<_, Ix2>(&positions, &[residual.particles(), 3])?
                .outer_iter()
                .map(|row| [row[0], row[1], row[2]])
                .collect::<Vec<_>>();
            let incident = matrix_tangent(&incident, residual.shape())?;
            let solution = detached(py, || {
                residual.pushforward(k0, &radii, &epsilon, &positions, &incident)
            })?;
            Ok((
                owned_matrix(py, solution.value)?,
                convergence_tuples(&solution.convergence),
            ))
        })
    }

    /// Cotangents of `(k0, radii, epsilon, positions, incident)`, in the forward
    /// argument order, then the convergence of each adjoint solve.
    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'_>,
    ) -> PyResult<(f64, R1<'py>, C1<'py>, R2<'py>, C2<'py>, ConvergenceTuples)> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
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
