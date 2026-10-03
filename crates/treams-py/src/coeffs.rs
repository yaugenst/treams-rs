//! `mie` and `mie_cyl` with their contexts: Mie coefficients of layered spheres and
//! cylinders (`treams_core::coeffs`).
use num_complex::Complex64;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use treams_core::{
    Complex,
    coeffs::{self, MieCylResidual, MieResidual},
    fpenv::ieee,
};

use crate::{
    args::materials,
    context::{context, detached},
    convert::{C1, C2, Cotangent, R1, matrix2_array, matrix2_cotangent},
};

context!(MieContext(MieResidual));

#[pymethods]
impl MieContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R1<'py>, C1<'py>, C1<'py>, C1<'py>)> {
        ieee(|| {
            let g = matrix2_cotangent(&cotangent)?;
            let residual = self.residual.take()?;
            let result = detached(py, move || residual.pullback(&g))?;
            Ok((
                result.sizes.into_pyarray(py),
                result.epsilon.into_pyarray(py),
                result.mu.into_pyarray(py),
                result.kappa.into_pyarray(py),
            ))
        })
    }
}

/// Record the Mie coefficients of a layered sphere: `coeffs::mie`.
#[pyfunction]
pub(crate) fn mie<'py>(
    py: Python<'py>,
    l: u32,
    sizes: PyReadonlyArray1<'py, f64>,
    epsilon: PyReadonlyArray1<'py, Complex64>,
    mu: PyReadonlyArray1<'py, Complex64>,
    kappa: PyReadonlyArray1<'py, Complex64>,
) -> PyResult<(C2<'py>, MieContext)> {
    ieee(|| {
        let sizes = sizes.as_array().to_vec();
        let mat = materials(&epsilon, &mu, &kappa)?;
        let residual = detached(py, move || coeffs::mie(l, &sizes, &mat))?;
        // A C-ordered copy: the residual keeps the coefficients for the pullback.
        Ok((
            matrix2_array(py, residual.value()),
            MieContext::new(residual),
        ))
    })
}

context!(MieCylContext(MieCylResidual));
#[pymethods]
impl MieCylContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(f64, f64, R1<'py>, C1<'py>, C1<'py>, C1<'py>)> {
        ieee(|| {
            let g = matrix2_cotangent(&cotangent)?;
            let residual = self.residual.take()?;
            let result = detached(py, move || residual.pullback(&g))?;
            Ok((
                result.kz,
                result.k0,
                result.layers.radii.into_pyarray(py),
                result.layers.epsilon.into_pyarray(py),
                result.layers.mu.into_pyarray(py),
                result.layers.kappa.into_pyarray(py),
            ))
        })
    }
}
/// Record the Mie coefficients of a layered cylinder: `coeffs::mie_cyl`.
#[pyfunction]
pub(crate) fn mie_cyl<'py>(
    py: Python<'py>,
    kz: f64,
    m: i32,
    k0: f64,
    radii: PyReadonlyArray1<'py, f64>,
    epsilon: PyReadonlyArray1<'py, Complex>,
    mu: PyReadonlyArray1<'py, Complex>,
    kappa: PyReadonlyArray1<'py, Complex>,
) -> PyResult<(C2<'py>, MieCylContext)> {
    ieee(|| {
        let radii = radii.as_array().to_vec();
        let mat = materials(&epsilon, &mu, &kappa)?;
        let residual = detached(py, move || coeffs::mie_cyl(kz, m, k0, &radii, &mat))?;
        // A C-ordered copy: the residual keeps the coefficients for the pullback.
        Ok((
            matrix2_array(py, residual.value()),
            MieCylContext::new(residual),
        ))
    })
}
