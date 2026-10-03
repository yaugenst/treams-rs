//! `spherical_channels` and `cylindrical_channels` with their contexts: the incidence
//! and emission channels of periodic sphere and cylinder arrays
//! (`treams_core::channels`).

use nalgebra::DMatrix;
use numpy::{
    IntoPyArray,
    ndarray::{Array4, Ix4},
};
use pyo3::prelude::*;
use treams_core::{
    Complex,
    channels::{self, ChannelGradient, CylindricalChannelsResidual, SphericalChannelsResidual},
    fpenv::ieee,
};

use crate::{
    args::{make_basis, make_cyl_basis},
    context::{OneUse, context, detached},
    convert::{C1, C4, Cotangent, R2, layout_error, merged_cotangent, rows_array},
};

context!(SphericalChannelsContext(SphericalChannelsResidual));
context!(CylindricalChannelsContext(CylindricalChannelsResidual));

/// Gradients with respect to the positions, the two wavenumbers, the transverse
/// wavevectors and the area (spheres) or the period (cylinders).
type Gradient<'py> = (R2<'py>, C1<'py>, R2<'py>, f64);

#[pymethods]
impl SphericalChannelsContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<Gradient<'py>> {
        ieee(|| {
            channel_pullback(
                py,
                &mut self.residual,
                &cotangent,
                SphericalChannelsResidual::shape,
                SphericalChannelsResidual::pullback,
            )
        })
    }
}

#[pymethods]
impl CylindricalChannelsContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<Gradient<'py>> {
        ieee(|| {
            channel_pullback(
                py,
                &mut self.residual,
                &cotangent,
                CylindricalChannelsResidual::shape,
                CylindricalChannelsResidual::pullback,
            )
        })
    }
}

/// The pullback of both channel contexts: check the cotangent against the
/// residual's `(2, 2, multipoles, channels)` shape, then consume the residual.
fn channel_pullback<'py, R: Send>(
    py: Python<'py>,
    residual: &mut OneUse<R>,
    cotangent: &Cotangent<'py>,
    shape: impl FnOnce(&R) -> (usize, usize),
    pullback: impl FnOnce(R, &DMatrix<Complex>) -> treams_core::Result<ChannelGradient> + Send,
) -> PyResult<Gradient<'py>> {
    let (residual, g) = residual.take_if(|residual| {
        // (kind, side, multipole) rows in C order are the native row index.
        let (rows, columns) = shape(residual);
        merged_cotangent::<Ix4>(cotangent, &[2, 2, rows / 4, columns], (rows, columns))
    })?;
    let result = detached(py, move || pullback(residual, &g))?;
    Ok((
        rows_array(py, result.positions)?,
        result.ks.to_vec().into_pyarray(py),
        rows_array(py, result.q)?,
        result.measure,
    ))
}

/// Record the channels of a periodic sphere array: `channels::spherical_channels`.
#[pyfunction]
pub(crate) fn spherical_channels(
    py: Python<'_>,
    modes: Vec<(usize, i32, i32, u8)>,
    positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    q: Vec<[f64; 2]>,
    polarizations: Vec<u8>,
    area: f64,
    helicity: bool,
    fixed_q: bool,
) -> PyResult<(C4<'_>, SphericalChannelsContext)> {
    ieee(|| {
        let basis = make_basis(modes, positions);
        let (value, residual) = detached(py, move || {
            channels::spherical_channels(basis, ks, q, polarizations, area, helicity, fixed_q)
        })?;
        Ok((
            channel_array(py, value)?,
            SphericalChannelsContext::new(residual),
        ))
    })
}

/// The channel matrix, moved into a `(2, 2, multipoles, channels)` array whose strides
/// follow the column-major matrix.
fn channel_array(py: Python<'_>, value: DMatrix<Complex>) -> PyResult<C4<'_>> {
    let (rows, columns) = value.shape();
    // Column-major (kind, side, multipole) rows read as (channel, kind, side, multipole).
    let value = Array4::from_shape_vec((columns, 2, 2, rows / 4), Vec::from(value.data))
        .map_err(layout_error)?
        .permuted_axes([1, 2, 3, 0]);
    Ok(value.into_pyarray(py))
}

/// Record the channels of a periodic cylinder array: `channels::cylindrical_channels`.
#[pyfunction]
pub(crate) fn cylindrical_channels(
    py: Python<'_>,
    modes: Vec<(usize, f64, i32, u8)>,
    positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    q: Vec<[f64; 2]>,
    polarizations: Vec<u8>,
    period: f64,
    helicity: bool,
    fixed_q: bool,
) -> PyResult<(C4<'_>, CylindricalChannelsContext)> {
    ieee(|| {
        let basis = make_cyl_basis(modes, positions);
        let (value, residual) = detached(py, move || {
            channels::cylindrical_channels(basis, ks, q, polarizations, period, helicity, fixed_q)
        })?;
        Ok((
            channel_array(py, value)?,
            CylindricalChannelsContext::new(residual),
        ))
    })
}
