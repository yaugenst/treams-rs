//! `sphere`, `cylinder` and `tmatrix_metric` with their contexts: T-matrices of
//! single spheres and cylinders, and the metrics cd, db and chi
//! (`treams_core::tmatrix`).

use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::{Ix0, Ix1},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    fpenv::ieee,
    tmatrix::{self, SphereResidual},
};

use crate::{
    args::{layer_tangents, materials},
    context::{context, detached, error, restore_state, state_array},
    convert::{
        C1, C2, Cotangent, R1, RealCotangent, RealTangent, Tangent, finite_cotangent,
        finite_tangent, from_array, matrix, matrix_cotangent, matrix_tangent, owned_matrix,
        vector_tangent,
    },
};

context!(SphereContext(SphereResidual));

#[pymethods]
impl SphereContext {
    #[staticmethod]
    fn _state_spec(lmax: usize, boundaries: usize) -> PyResult<usize> {
        ieee(|| SphereResidual::state_size(lmax, boundaries).map_err(error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        k0: RealTangent<'py>,
        radii: RealTangent<'py>,
        epsilon: Tangent<'py>,
        mu: Tangent<'py>,
        kappa: Tangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let k0 = finite_tangent::<_, Ix0>(&k0, &[])?[()];
            let (radii, materials) =
                layer_tangents(residual.boundaries(), &radii, &epsilon, &mu, &kappa)?;
            let tangent = detached(py, move || residual.pushforward(k0, &radii, &materials))?;
            owned_matrix(py, tangent)
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(f64, R1<'py>, C1<'py>, C1<'py>, C1<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
            let result = detached(py, move || residual.pullback(&g))?;
            Ok((
                result.k0,
                result.layers.radii.into_pyarray(py),
                result.layers.epsilon.into_pyarray(py),
                result.layers.mu.into_pyarray(py),
                result.layers.kappa.into_pyarray(py),
            ))
        })
    }
}

/// Record the T-matrix of a layered sphere: `tmatrix::sphere`.
#[pyfunction]
pub(crate) fn sphere<'py>(
    py: Python<'py>,
    lmax: u32,
    k0: f64,
    radii: PyReadonlyArray1<'py, f64>,
    epsilon: PyReadonlyArray1<'py, Complex>,
    mu: PyReadonlyArray1<'py, Complex>,
    kappa: PyReadonlyArray1<'py, Complex>,
) -> PyResult<(C2<'py>, SphereContext)> {
    ieee(|| {
        let radii = radii.as_array().to_vec();
        let mat = materials(&epsilon, &mu, &kappa)?;
        let (value, residual) = detached(py, move || tmatrix::sphere(lmax, k0, &radii, &mat))?;
        // A C-ordered copy of the moved T-matrix.
        Ok((matrix(py, &value)?, SphereContext::new(residual)))
    })
}

context!(CylinderContext(tmatrix::CylinderResidual));

#[pymethods]
impl CylinderContext {
    #[staticmethod]
    fn _state_spec(kz_count: usize, mmax: usize, boundaries: usize) -> PyResult<usize> {
        ieee(|| tmatrix::CylinderResidual::state_size(kz_count, mmax, boundaries).map_err(error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        kzs: RealTangent<'py>,
        k0: RealTangent<'py>,
        radii: RealTangent<'py>,
        epsilon: Tangent<'py>,
        mu: Tangent<'py>,
        kappa: Tangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let (kz_count, boundaries) = residual.input_counts();
            let kzs = vector_tangent(&kzs, kz_count)?;
            let k0 = finite_tangent::<_, Ix0>(&k0, &[])?[()];
            let (radii, materials) = layer_tangents(boundaries, &radii, &epsilon, &mu, &kappa)?;
            let tangent = detached(py, move || {
                residual.pushforward(&kzs, k0, &radii, &materials)
            })?;
            owned_matrix(py, tangent)
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R1<'py>, f64, R1<'py>, C1<'py>, C1<'py>, C1<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
            let result = detached(py, move || residual.pullback(&g))?;
            Ok((
                result.kzs.into_pyarray(py),
                result.k0,
                result.layers.radii.into_pyarray(py),
                result.layers.epsilon.into_pyarray(py),
                result.layers.mu.into_pyarray(py),
                result.layers.kappa.into_pyarray(py),
            ))
        })
    }
}

/// Record the T-matrix of a layered cylinder: `tmatrix::cylinder`.
#[pyfunction]
pub(crate) fn cylinder<'py>(
    py: Python<'py>,
    kzs: PyReadonlyArray1<'py, f64>,
    mmax: u32,
    k0: f64,
    radii: PyReadonlyArray1<'py, f64>,
    epsilon: PyReadonlyArray1<'py, Complex>,
    mu: PyReadonlyArray1<'py, Complex>,
    kappa: PyReadonlyArray1<'py, Complex>,
) -> PyResult<(C2<'py>, CylinderContext)> {
    ieee(|| {
        let kzs = kzs.as_array().to_vec();
        let radii = radii.as_array().to_vec();
        let mat = materials(&epsilon, &mu, &kappa)?;
        let (value, residual) =
            detached(py, move || tmatrix::cylinder(&kzs, mmax, k0, &radii, &mat))?;
        // A C-ordered copy of the moved T-matrix.
        Ok((matrix(py, &value)?, CylinderContext::new(residual)))
    })
}

context!(TMatrixMetricContext(tmatrix::MetricResidual));

#[pymethods]
impl TMatrixMetricContext {
    #[staticmethod]
    fn _state_spec(dimension: usize) -> PyResult<usize> {
        ieee(|| tmatrix::MetricResidual::state_size(dimension).map_err(error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| restore_state(&state).map(Self::new))
    }

    fn pushforward(
        &self,
        py: Python<'_>,
        operator: Tangent<'_>,
        ks: RealTangent<'_>,
    ) -> PyResult<f64> {
        ieee(|| {
            let matrix = matrix_tangent(&operator, self.residual.shape())?;
            let ks = finite_tangent::<_, Ix1>(&ks, &[2])?;
            let ks = [ks[0], ks[1]];
            let residual = &self.residual;
            detached(py, move || residual.pushforward(&matrix, ks))
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: RealCotangent<'py>,
    ) -> PyResult<(C2<'py>, R1<'py>)> {
        ieee(|| {
            let cotangent = finite_cotangent::<_, Ix0>(&cotangent, &[])?[()];
            let residual = &self.residual;
            let gradient = detached(py, move || residual.pullback(cotangent))?;
            Ok((
                owned_matrix(py, gradient.matrix)?,
                gradient.ks.to_vec().into_pyarray(py),
            ))
        })
    }
}

/// Record the helicity metric cd, db or chi of a T-matrix: `tmatrix::metric`.
#[pyfunction]
pub(crate) fn tmatrix_metric(
    py: Python<'_>,
    operator: PyReadonlyArray2<'_, Complex>,
    polarizations: Vec<u8>,
    ks: [f64; 2],
    kind: &str,
) -> PyResult<(f64, TMatrixMetricContext)> {
    ieee(|| {
        let kind = match kind {
            "cd" => tmatrix::Metric::CircularDichroism,
            "db" => tmatrix::Metric::DualityBreaking,
            "chi" => tmatrix::Metric::Chirality,
            _ => {
                return Err(PyValueError::new_err(
                    "T-matrix metric must be cd, db or chi",
                ));
            }
        };
        let matrix = from_array(operator, "operator")?;
        let (value, residual) = detached(py, move || {
            tmatrix::metric(&matrix, &polarizations, ks, kind)
        })?;
        Ok((value, TMatrixMetricContext::new(residual)))
    })
}
