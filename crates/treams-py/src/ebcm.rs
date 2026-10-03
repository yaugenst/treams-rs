//! `ebcm_qmat` and `EbcmQmatContext`: the Q matrix of an axisymmetric particle by
//! the extended boundary condition method (`treams_core::ebcm::qmat`).
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{Complex, ebcm, fpenv::ieee, sw::Mode};

use crate::{
    context::{context, detached, radial},
    convert::{C1, C2, Cotangent, R1, owned_matrix, rows_array},
};

context!(EbcmQmatContext(ebcm::QmatResidual));

#[pymethods]
impl EbcmQmatContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R1<'py>, R1<'py>, C2<'py>, C1<'py>)> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, ebcm::QmatResidual::shape)?;
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok((
                gradient.radii.into_pyarray(py),
                gradient.slopes.into_pyarray(py),
                rows_array(py, Vec::from(gradient.ks))?,
                gradient.zs.to_vec().into_pyarray(py),
            ))
        })
    }
}

/// Record the EBCM Q matrix of an axisymmetric surface: `ebcm::qmat`.
#[pyfunction]
pub(crate) fn ebcm_qmat<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray2<'py, f64>,
    destination: Vec<(i32, i32, u8)>,
    source: Vec<(i32, i32, u8)>,
    ks: PyReadonlyArray2<'py, Complex>,
    zs: [Complex; 2],
    singular: bool,
    radial_area_factor: bool,
) -> PyResult<(C2<'py>, EbcmQmatContext)> {
    ieee(|| {
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
        let (value, residual) = detached(py, move || {
            ebcm::qmat(
                modes(destination),
                modes(source),
                surface,
                ks,
                zs,
                radial(singular),
                radial_area_factor,
            )
        })?;
        Ok((owned_matrix(py, value)?, EbcmQmatContext::new(residual)))
    })
}
