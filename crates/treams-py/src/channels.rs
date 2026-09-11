//! Four spherical incidence/radiation channel arrays and their native residual.
#![allow(clippy::indexing_slicing)] // Validated four-dimensional NumPy shapes.

use nalgebra::DMatrix;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray4, PyReadonlyArray4,
    ndarray::{Array2, Array4},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    channels::{self, ChannelsResidual},
};

use crate::{basis::make_basis, error};

#[pyclass]
#[derive(Debug)]
struct ChannelsContext {
    residual: Option<ChannelsResidual>,
    fixed_q: bool,
}

type Gradient<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray2<f64>>,
    f64,
);
#[pymethods]
impl ChannelsContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray4<'py, Complex>,
    ) -> PyResult<Gradient<'py>> {
        let a = cotangent.as_array();
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let d = residual.value.nrows() / 4;
        let c = residual.value.ncols();
        if a.shape() != [2, 2, d, c] || a.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(PyValueError::new_err(
                "invalid channel cotangent shape or values",
            ));
        }
        let g = DMatrix::from_fn(4 * d, c, |i, j| a[(i / (2 * d), (i / d) % 2, i % d, j)]);
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let fixed_q = self.fixed_q;
        let result = py
            .detach(move || residual.pullback(&g, fixed_q))
            .map_err(error)?;
        Ok((
            Array2::from_shape_fn((result.positions.len(), 3), |(i, j)| result.positions[i][j])
                .into_pyarray(py),
            result.ks.to_vec().into_pyarray(py),
            Array2::from_shape_fn((result.q.len(), 2), |(i, j)| result.q[i][j]).into_pyarray(py),
            result.area,
        ))
    }
}

#[pyfunction]
fn spherical_channels(
    py: Python<'_>,
    modes: Vec<(usize, i32, i32, u8)>,
    positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    q: Vec<[f64; 2]>,
    polarizations: Vec<u8>,
    area: f64,
    helicity: bool,
    fixed_q: bool,
) -> PyResult<(Bound<'_, PyArray4<Complex>>, ChannelsContext)> {
    let basis = make_basis(modes, positions);
    let residual = py
        .detach(move || channels::spherical(basis, ks, q, polarizations, area, helicity))
        .map_err(error)?;
    let d = residual.value.nrows() / 4;
    let c = residual.value.ncols();
    let value = Array4::from_shape_fn((2, 2, d, c), |(kind, side, i, j)| {
        residual.value[((kind * 2 + side) * d + i, j)]
    })
    .into_pyarray(py);
    Ok((
        value,
        ChannelsContext {
            residual: Some(residual),
            fixed_q,
        },
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ChannelsContext>()?;
    module.add_function(wrap_pyfunction!(spherical_channels, module)?)?;
    Ok(())
}
