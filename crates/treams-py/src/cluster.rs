//! `sphere_cluster`, `particle_cluster`, `cylindrical_particle_cluster`,
//! `interaction`, `InteractionFactor` and `sphere_cluster_factor` with their
//! contexts: cluster T-matrices, interaction solves and reusable factors
//! (`treams_core::cluster`).

use std::sync::Arc;

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    cluster::{
        self as core_cluster, IlluminateGradient, IlluminateResidual, InteractionResidual,
        SphereClusterResidual,
    },
    fpenv::ieee,
};

use crate::{
    args::spheres,
    context::{context, detached},
    convert::{C1, C2, Cotangent, R1, R2, from_array, matrix, owned_matrix, rows_array},
};

context!(SphereClusterContext(SphereClusterResidual));

#[pymethods]
impl SphereClusterContext {
    /// Cotangents of `(k0, radii, epsilon, positions)`, in the forward argument order.
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(f64, R1<'py>, C1<'py>, R2<'py>)> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, SphereClusterResidual::shape)?;
            let result = detached(py, move || residual.pullback(&g))?;
            Ok((
                result.k0,
                result.radii.into_pyarray(py),
                result.epsilon.into_pyarray(py),
                rows_array(py, result.positions)?,
            ))
        })
    }
}

/// Record the T-matrix of a cluster of homogeneous spheres: `cluster::sphere_cluster`.
#[pyfunction]
pub(crate) fn sphere_cluster<'py>(
    py: Python<'py>,
    lmax: u32,
    k0: f64,
    radii: PyReadonlyArray1<'py, f64>,
    epsilon: PyReadonlyArray1<'py, Complex>,
    positions: PyReadonlyArray2<'py, f64>,
) -> PyResult<(C2<'py>, SphereClusterContext)> {
    ieee(|| {
        let (radii, epsilon, positions) = spheres(&radii, &epsilon, positions)?;
        let residual = detached(py, move || {
            core_cluster::sphere_cluster(lmax, k0, &radii, &epsilon, &positions)
        })?;
        // A C-ordered copy: the residual keeps the interacting T-matrix.
        Ok((
            matrix(py, residual.value())?,
            SphereClusterContext::new(residual),
        ))
    })
}

context!(InteractionContext(InteractionResidual));

#[pymethods]
impl InteractionContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C2<'py>, C2<'py>)> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, InteractionResidual::shape)?;
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok((
                owned_matrix(py, gradient.local)?,
                owned_matrix(py, gradient.coupling)?,
            ))
        })
    }
}

/// Record the solution X of `(I - T C) X = T`: `cluster::interaction`.
#[pyfunction]
pub(crate) fn interaction<'py>(
    py: Python<'py>,
    local: PyReadonlyArray2<'py, Complex>,
    coupling: PyReadonlyArray2<'py, Complex>,
) -> PyResult<(C2<'py>, InteractionContext)> {
    ieee(|| {
        let local = from_array(local, "local")?;
        let coupling = from_array(coupling, "coupling")?;
        let residual = detached(py, move || core_cluster::interaction(local, coupling))?;
        // A C-ordered copy: the residual keeps the interacting T-matrix.
        Ok((
            matrix(py, residual.value())?,
            InteractionContext::new(residual),
        ))
    })
}

context!(ParticleClusterContext(
    core_cluster::ParticleClusterResidual
));

#[pymethods]
impl ParticleClusterContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(Vec<C2<'py>>, R2<'py>, C1<'py>)> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, core_cluster::ParticleClusterResidual::shape)?;
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok((
                gradient
                    .local
                    .into_iter()
                    .map(|value| owned_matrix(py, value))
                    .collect::<PyResult<_>>()?,
                rows_array(py, gradient.positions)?,
                gradient.ks.to_vec().into_pyarray(py),
            ))
        })
    }
}

/// Record a cluster T-matrix from local blocks and the core call on its basis. The
/// matrix leaves as a C-ordered copy: the residual keeps it.
fn record_particle_cluster<'py>(
    py: Python<'py>,
    local: Vec<PyReadonlyArray2<'py, Complex>>,
    cluster: impl Send
    + FnOnce(
        Vec<nalgebra::DMatrix<Complex>>,
    ) -> treams_core::Result<core_cluster::ParticleClusterResidual>,
) -> PyResult<(C2<'py>, ParticleClusterContext)> {
    let local = local
        .into_iter()
        .map(|block| from_array(block, "local"))
        .collect::<PyResult<Vec<_>>>()?;
    let residual = detached(py, move || cluster(local))?;
    Ok((
        matrix(py, residual.value())?,
        ParticleClusterContext::new(residual),
    ))
}

/// Record a particle-cluster T-matrix, spherical basis: `cluster::particle_cluster`.
#[pyfunction]
pub(crate) fn particle_cluster<'py>(
    py: Python<'py>,
    local: Vec<PyReadonlyArray2<'py, Complex>>,
    modes: Vec<(usize, i32, i32, u8)>,
    positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
) -> PyResult<(C2<'py>, ParticleClusterContext)> {
    ieee(|| {
        let basis = crate::args::make_basis(modes, positions);
        record_particle_cluster(py, local, move |local| {
            core_cluster::particle_cluster(local, basis, ks, helicity)
        })
    })
}

/// Record a particle-cluster T-matrix, cylindrical basis: `cluster::cylindrical_particle_cluster`.
#[pyfunction]
pub(crate) fn cylindrical_particle_cluster<'py>(
    py: Python<'py>,
    local: Vec<PyReadonlyArray2<'py, Complex>>,
    modes: Vec<(usize, f64, i32, u8)>,
    positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
) -> PyResult<(C2<'py>, ParticleClusterContext)> {
    ieee(|| {
        let basis = crate::args::make_cyl_basis(modes, positions);
        record_particle_cluster(py, local, move |local| {
            core_cluster::cylindrical_particle_cluster(local, basis, ks, helicity)
        })
    })
}

/// A factor of `I - T C` that solves for requested illuminations:
/// `cluster::InteractionFactor`.
#[pyclass(module = "treams_rs._native", frozen)]
#[derive(Debug)]
pub(crate) struct InteractionFactor {
    factor: Arc<core_cluster::InteractionFactor>,
    /// Whether the local T-matrix is a list of blocks, whose contexts take
    /// `pullback_blocks` instead of `pullback`.
    blocked: bool,
}

#[pymethods]
impl InteractionFactor {
    /// Factor `I - T C` of one dense local T-matrix: `cluster::InteractionFactor::new`.
    #[new]
    fn new(
        py: Python<'_>,
        local: PyReadonlyArray2<'_, Complex>,
        coupling: PyReadonlyArray2<'_, Complex>,
    ) -> PyResult<Self> {
        ieee(|| {
            let local = from_array(local, "local")?;
            let coupling = from_array(coupling, "coupling")?;
            let factor = detached(py, move || {
                core_cluster::InteractionFactor::new(local, coupling)
            })?;
            Ok(Self {
                factor: Arc::new(factor),
                blocked: false,
            })
        })
    }

    /// Factor `I - T C` of separate local blocks: `cluster::InteractionFactor::from_blocks`.
    #[staticmethod]
    fn from_blocks(
        py: Python<'_>,
        local: Vec<PyReadonlyArray2<'_, Complex>>,
        coupling: PyReadonlyArray2<'_, Complex>,
    ) -> PyResult<Self> {
        ieee(|| {
            let local = local
                .into_iter()
                .map(|block| from_array(block, "local"))
                .collect::<PyResult<Vec<_>>>()?;
            let coupling = from_array(coupling, "coupling")?;
            let factor = detached(py, move || {
                core_cluster::InteractionFactor::from_blocks(local, coupling)
            })?;
            Ok(Self {
                factor: Arc::new(factor),
                blocked: true,
            })
        })
    }

    /// The number of rows of `I - T C`.
    #[getter]
    fn dimension(&self) -> usize {
        ieee(|| self.factor.dimension())
    }

    /// The scattered coefficients of each incident column, without a context.
    fn solve<'py>(
        &self,
        py: Python<'py>,
        incident: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let incident = from_array(incident, "incident")?;
            let value = detached(py, || self.factor.solve(&incident))?;
            owned_matrix(py, value)
        })
    }

    /// The scattered coefficients of each incident column and their context.
    fn record<'py>(
        &self,
        py: Python<'py>,
        incident: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<(C2<'py>, IlluminateContext)> {
        ieee(|| {
            let incident = from_array(incident, "incident")?;
            let residual = detached(py, || self.factor.record(incident))?;
            Ok((
                matrix(py, residual.value())?,
                IlluminateContext::new(residual, self.blocked),
            ))
        })
    }
}

context!(IlluminateContext(IlluminateResidual, blocked: bool));

impl IlluminateContext {
    fn gradient(
        &mut self,
        py: Python<'_>,
        cotangent: &Cotangent<'_>,
    ) -> PyResult<IlluminateGradient> {
        let (residual, g) = self
            .residual
            .take_with_matrix(cotangent, IlluminateResidual::shape)?;
        detached(py, move || residual.pullback(&g))
    }
}

#[pymethods]
impl IlluminateContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C2<'py>, C2<'py>, C2<'py>)> {
        ieee(|| {
            if self.blocked {
                return Err(PyValueError::new_err(
                    "use pullback_blocks for block-local factors",
                ));
            }
            let mut g = self.gradient(py, &cotangent)?;
            let local = g
                .local
                .pop()
                .ok_or_else(|| PyValueError::new_err("missing local matrix gradient"))?;
            Ok((
                owned_matrix(py, local)?,
                owned_matrix(py, g.coupling)?,
                owned_matrix(py, g.incident)?,
            ))
        })
    }

    fn pullback_blocks<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(Vec<C2<'py>>, C2<'py>, C2<'py>)> {
        ieee(|| {
            if !self.blocked {
                return Err(PyValueError::new_err(
                    "use pullback for dense-local factors",
                ));
            }
            let g = self.gradient(py, &cotangent)?;
            Ok((
                g.local
                    .into_iter()
                    .map(|m| owned_matrix(py, m))
                    .collect::<PyResult<Vec<_>>>()?,
                owned_matrix(py, g.coupling)?,
                owned_matrix(py, g.incident)?,
            ))
        })
    }
}

/// Factor `I - T C` of a cluster of homogeneous spheres once: `cluster::sphere_cluster_factor`.
#[pyfunction]
pub(crate) fn sphere_cluster_factor(
    py: Python<'_>,
    lmax: u32,
    k0: f64,
    radii: PyReadonlyArray1<'_, f64>,
    epsilon: PyReadonlyArray1<'_, Complex>,
    positions: PyReadonlyArray2<'_, f64>,
) -> PyResult<InteractionFactor> {
    ieee(|| {
        let (radii, epsilon, positions) = spheres(&radii, &epsilon, positions)?;
        let factor = detached(py, || {
            core_cluster::sphere_cluster_factor(lmax, k0, &radii, &epsilon, &positions)
        })?;
        Ok(InteractionFactor {
            factor: Arc::new(factor),
            blocked: true,
        })
    })
}
