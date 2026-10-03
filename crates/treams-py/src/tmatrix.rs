//! `sphere`, `cylinder` and `tmatrix_metric` with their contexts: T-matrices of
//! single spheres and cylinders, and the metrics cd, db and chi
//! (`treams_core::tmatrix`).

use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, ndarray::Ix0};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    fpenv::ieee,
    tmatrix::{self, SphereResidual},
};

use crate::{
    args::materials,
    context::{context, detached},
    convert::{
        C1, C2, Cotangent, R1, RealCotangent, finite_cotangent, from_array, matrix, owned_matrix,
    },
};

context!(SphereContext(SphereResidual));

#[pymethods]
impl SphereContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(f64, R1<'py>, C1<'py>, C1<'py>, C1<'py>)> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, SphereResidual::shape)?;
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
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R1<'py>, f64, R1<'py>, C1<'py>, C1<'py>, C1<'py>)> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, tmatrix::CylinderResidual::shape)?;
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
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: RealCotangent<'py>,
    ) -> PyResult<(C2<'py>, R1<'py>)> {
        ieee(|| {
            let cotangent = finite_cotangent::<_, Ix0>(&cotangent, &[])?[()];
            let residual = self.residual.take()?;
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
