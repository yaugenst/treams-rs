//! `ebcm_qmat` and `EbcmQmatContext`: the Q matrix of an axisymmetric particle by
//! the extended boundary condition method (`treams_core::ebcm::qmat`).
use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::{Ix1, Ix2},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{Complex, ebcm, fpenv::ieee, sw::Mode};

use crate::{
    context::{context, detached, error, radial, restore_state, state_array},
    convert::{
        C1, C2, Cotangent, R1, RealTangent, Tangent, finite_tangent, matrix_cotangent,
        owned_matrix, rows_array, vector_tangent,
    },
};

context!(EbcmQmatContext(ebcm::QmatResidual));

#[pymethods]
impl EbcmQmatContext {
    #[staticmethod]
    fn _state_spec(samples: usize, destinations: usize, sources: usize) -> PyResult<usize> {
        ieee(|| ebcm::QmatResidual::state_size(samples, destinations, sources).map_err(error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| restore_state(&state).map(Self::new))
    }

    /// Directional derivative in `(radii, slopes, ks, zs)` input order.
    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        radii: RealTangent<'_>,
        slopes: RealTangent<'_>,
        ks: Tangent<'_>,
        zs: Tangent<'_>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let radii = vector_tangent(&radii, residual.samples())?;
            let slopes = vector_tangent(&slopes, residual.samples())?;
            let ks = finite_tangent::<_, Ix2>(&ks, &[2, 2])?;
            let ks = std::array::from_fn(|i| std::array::from_fn(|j| ks[(i, j)]));
            let zs = finite_tangent::<_, Ix1>(&zs, &[2])?;
            let zs = [zs[0], zs[1]];
            let tangent = detached(py, || residual.pushforward(&radii, &slopes, ks, zs))?;
            owned_matrix(py, tangent)
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R1<'py>, R1<'py>, C2<'py>, C1<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
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
