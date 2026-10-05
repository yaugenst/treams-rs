//! `sphere_cluster`, `particle_cluster`, `cylindrical_particle_cluster`,
//! `interaction`, `InteractionFactor` and `sphere_cluster_factor` with their
//! contexts: cluster T-matrices, interaction solves and reusable factors
//! (`treams_core::cluster`).

use std::sync::Arc;

use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::{Ix0, Ix1},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    cluster::{
        self as core_cluster, IlluminateGradient, IlluminateResidual, InteractionResidual,
        SphereClusterResidual,
    },
    fpenv::ieee,
    saved::SavedState,
};

use crate::{
    args::spheres,
    context::{context, detached, error, restore_state, state_array},
    convert::{
        C1, C2, Cotangent, R1, R2, RealTangent, Tangent, finite_tangent, from_array, matrix,
        matrix_cotangent, matrix_tangent, owned_matrix, rows, rows_array, vector_tangent,
    },
};

/// Convert one finite direction per recorded local block. The core owns the block
/// shapes; contexts retain no duplicate input metadata.
fn block_tangents(
    tangents: &[Tangent<'_>],
    shapes: impl ExactSizeIterator<Item = (usize, usize)>,
) -> PyResult<Vec<nalgebra::DMatrix<Complex>>> {
    if tangents.len() != shapes.len() {
        return Err(PyValueError::new_err(
            "require one tangent per recorded local block",
        ));
    }
    tangents
        .iter()
        .zip(shapes)
        .map(|(tangent, shape)| matrix_tangent(tangent, shape))
        .collect()
}

context!(SphereClusterContext(SphereClusterResidual));

#[pymethods]
impl SphereClusterContext {
    #[staticmethod]
    fn _state_spec(lmax: usize, particles: usize) -> PyResult<usize> {
        ieee(|| SphereClusterResidual::state_size(lmax, particles).map_err(error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| Ok(Self::new(restore_state(&state)?)))
    }

    /// Tangent of the interacting T-matrix for `(k0, radii, epsilon, positions)`.
    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        k0: RealTangent<'py>,
        radii: RealTangent<'py>,
        epsilon: Tangent<'py>,
        positions: RealTangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let n = residual.particles();
            let k0 = finite_tangent::<_, Ix0>(&k0, &[])?[()];
            let radii = vector_tangent(&radii, n)?;
            let epsilon = vector_tangent(&epsilon, n)?;
            let positions = rows(finite_tangent(&positions, &[n, 3])?, "position tangent")?;
            let tangent = detached(py, || {
                residual.pushforward(k0, &radii, &epsilon, &positions)
            })?;
            owned_matrix(py, tangent)
        })
    }

    /// Cotangents of `(k0, radii, epsilon, positions)`, in the forward argument order.
    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(f64, R1<'py>, C1<'py>, R2<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
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
    #[staticmethod]
    fn _state_spec(dimension: usize) -> PyResult<usize> {
        ieee(|| InteractionResidual::state_size(&[dimension]).map_err(error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| Ok(Self::new(restore_state(&state)?)))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        local: Tangent<'py>,
        coupling: Tangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let local = matrix_tangent(&local, residual.shape())?;
            let coupling = matrix_tangent(&coupling, residual.shape())?;
            let tangent = detached(py, || residual.pushforward(&local, &coupling))?;
            owned_matrix(py, tangent)
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C2<'py>, C2<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
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
    #[staticmethod]
    fn _state_spec(local_sizes: Vec<usize>, cylindrical: bool) -> PyResult<usize> {
        ieee(|| {
            if cylindrical {
                core_cluster::ParticleClusterResidual::cylindrical_state_size(&local_sizes)
            } else {
                core_cluster::ParticleClusterResidual::spherical_state_size(&local_sizes)
            }
            .map_err(error)
        })
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| Ok(Self::new(restore_state(&state)?)))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        local: Vec<Tangent<'py>>,
        positions: RealTangent<'py>,
        ks: Tangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let n = residual.local_shapes().len();
            let local = block_tangents(&local, residual.local_shapes())?;
            let positions = rows(finite_tangent(&positions, &[n, 3])?, "position tangent")?;
            let ks = finite_tangent::<_, Ix1>(&ks, &[2])?;
            let ks = [ks[0], ks[1]];
            let tangent = detached(py, || residual.pushforward(&local, &positions, ks))?;
            owned_matrix(py, tangent)
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(Vec<C2<'py>>, R2<'py>, C1<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
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

    /// Incident-field cotangent for a fixed factor, without recording any fields.
    fn pullback_incident<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let columns = cotangent.as_array().shape().get(1).copied().unwrap_or(0);
            let cotangent = matrix_cotangent(&cotangent, (self.factor.dimension(), columns))?;
            let value = detached(py, || self.factor.pullback_incident(&cotangent))?;
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
    fn gradient(&self, py: Python<'_>, cotangent: &Cotangent<'_>) -> PyResult<IlluminateGradient> {
        let residual = &self.residual;
        let g = matrix_cotangent(cotangent, residual.shape())?;
        detached(py, move || residual.pullback(&g))
    }
}

#[pymethods]
impl IlluminateContext {
    #[staticmethod]
    fn _state_spec(local_sizes: Vec<usize>, columns: usize) -> PyResult<usize> {
        ieee(|| {
            IlluminateResidual::state_size(&local_sizes, columns)
                .map_err(error)?
                .checked_add(1)
                .ok_or_else(|| PyValueError::new_err("cluster derivative state is too large"))
        })
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| {
            let mut state = detached(py, || self.residual.save_state())?;
            state.push(u8::from(self.blocked));
            Ok(state.into_pyarray(py))
        })
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| {
            let Some((&blocked, bytes)) = state.as_slice()?.split_last() else {
                return Err(PyValueError::new_err("empty cluster derivative state"));
            };
            if blocked > 1 {
                return Err(PyValueError::new_err("invalid cluster derivative state"));
            }
            let residual = IlluminateResidual::from_state(bytes).map_err(error)?;
            if blocked == 0 && residual.local_shapes().len() != 1 {
                return Err(PyValueError::new_err(
                    "dense factor requires one local matrix",
                ));
            }
            Ok(Self::new(residual, blocked == 1))
        })
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        local: Tangent<'py>,
        coupling: Tangent<'py>,
        incident: Tangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            if self.blocked {
                return Err(PyValueError::new_err(
                    "use pushforward_blocks for block-local factors",
                ));
            }
            let residual = &self.residual;
            let (n, _) = residual.shape();
            let local = matrix_tangent(&local, (n, n))?;
            let coupling = matrix_tangent(&coupling, (n, n))?;
            let incident = matrix_tangent(&incident, residual.shape())?;
            let tangent = detached(py, || {
                residual.pushforward(std::slice::from_ref(&local), &coupling, &incident)
            })?;
            owned_matrix(py, tangent)
        })
    }

    fn pushforward_blocks<'py>(
        &self,
        py: Python<'py>,
        local: Vec<Tangent<'py>>,
        coupling: Tangent<'py>,
        incident: Tangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            if !self.blocked {
                return Err(PyValueError::new_err(
                    "use pushforward for dense-local factors",
                ));
            }
            let residual = &self.residual;
            let (n, _) = residual.shape();
            let local = block_tangents(&local, residual.local_shapes())?;
            let coupling = matrix_tangent(&coupling, (n, n))?;
            let incident = matrix_tangent(&incident, residual.shape())?;
            let tangent = detached(py, || residual.pushforward(&local, &coupling, &incident))?;
            owned_matrix(py, tangent)
        })
    }

    fn pullback<'py>(
        &self,
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
        &self,
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
