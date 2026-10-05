//! The S-matrix records `smatrix_from_array`, `smatrix_add`, `smatrix_illuminate`,
//! `smatrix_periodic`, `bands` and `smatrix_tr`, the planar records `fresnel`,
//! `interface_coefficients`, `propagation_matrix` and `layer_stack`, and the
//! chirality densities, with their contexts (`treams_core::smatrix`).
//!
//! An S-matrix is a `(2, 2, n, n)` array of four blocks `S[out][in]`, indexed by
//! direction: 0 is up, towards the positive side, and 1 is down. `S[0, 0]` is the
//! upward transmission, `S[0, 1]` the reflection of waves incident from above,
//! `S[1, 0]` the reflection of waves incident from below and `S[1, 1]` the downward
//! transmission: the order `[S00, S01, S10, S11]` of `smatrix::Blocks`.

use nalgebra::DMatrix;
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4, PyReadonlyArrayDyn,
    ndarray::{Array3, Array4, Array5, ArrayView2, ArrayView4, Axis, Ix2, Ix3, Ix4, Ix5, s},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    fpenv::ieee,
    smatrix::{self, AddResidual, Blocks, FresnelResidual, FromArrayResidual, PropagationResidual},
};

use crate::{
    context::{context, cotangent_error, detached},
    convert::{
        C1, C2, C3, C4, C5, Cotangent, Finite, LentMatrix, R1, R2, RealCotangent, RealTangent,
        Tangent, all_finite, cotangent_view, finite_cotangent, finite_tangent, from_array,
        layout_error, matrix, matrix_cotangent, matrix_from_view, matrix_tangent, owned_matrix,
        rows, rows_array, vector_cotangent, vector_tangent,
    },
};

/// The four `(n, n)` blocks of the `(2, 2, n, n)` S-matrix argument `name`, `n > 0`.
fn block_views<'a>(
    a: ArrayView4<'a, Complex>,
    name: &str,
) -> PyResult<[ArrayView2<'a, Complex>; 4]> {
    let s = a.shape();
    if s[0] != 2 || s[1] != 2 || s[2] == 0 || s[2] != s[3] {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (2, 2, n, n) with n > 0"
        )));
    }
    Ok(std::array::from_fn(|b| {
        a.slice_move(s![b / 2, b % 2, .., ..])
    }))
}
fn blocks(value: PyReadonlyArray4<'_, Complex>, name: &str) -> PyResult<Blocks> {
    Ok(block_views(value.as_array(), name)?.map(matrix_from_view))
}
/// Finite blocks of a cotangent of shape `(2, 2, rows, columns)`.
fn cotangent_blocks(
    cotangent: &Cotangent<'_>,
    (rows, columns): (usize, usize),
) -> PyResult<Blocks> {
    let expected = [2, 2, rows, columns];
    let a = cotangent_view::<_, Ix4>(cotangent, &expected)?;
    let blocks: Blocks =
        std::array::from_fn(|b| matrix_from_view(a.slice(s![b / 2, b % 2, .., ..])));
    if blocks.iter().all(|block| all_finite(block.as_slice())) {
        Ok(blocks)
    } else {
        Err(cotangent_error(&expected))
    }
}

/// One finite direction of four equally shaped scattering or channel blocks.
fn tangent_blocks(tangent: &Tangent<'_>, shape: (usize, usize)) -> PyResult<Blocks> {
    let a = finite_tangent::<_, Ix4>(tangent, &[2, 2, shape.0, shape.1])?;
    Ok(std::array::from_fn(|b| {
        matrix_from_view(a.slice(s![b / 2, b % 2, .., ..]))
    }))
}

/// A fixed-size parameter vector with the recorded input shape.
fn tangent_array<T: Finite, const N: usize>(
    tangent: &PyReadonlyArrayDyn<'_, T>,
) -> PyResult<[T; N]> {
    let a = finite_tangent::<_, numpy::ndarray::Ix1>(tangent, &[N])?;
    Ok(std::array::from_fn(|i| a[i]))
}

/// A fixed-size parameter matrix with the recorded input shape.
fn tangent_matrix<T: Finite, const M: usize, const N: usize>(
    tangent: &PyReadonlyArrayDyn<'_, T>,
) -> PyResult<[[T; N]; M]> {
    let a = finite_tangent::<_, Ix2>(tangent, &[M, N])?;
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| a[(i, j)])))
}
/// The column-major storage of four equally shaped matrices, one after another.
///
/// The first allocation grows to hold all four, so only three are copied.
/// `Vec::extend` of an owned `Vec` copies with `memcpy`, while `extend_from_slice`
/// of `Complex64` compiles to an element-by-element loop.
fn concatenated([first, rest @ ..]: [DMatrix<Complex>; 4]) -> Vec<Complex> {
    let mut data = Vec::from(first.data);
    data.reserve_exact(3 * data.len());
    for block in rest {
        data.extend(Vec::from(block.data));
    }
    data
}
/// Four equally shaped blocks as a `(2, 2, rows, columns)` array.
fn blocks_array(py: Python<'_>, value: Blocks) -> PyResult<C4<'_>> {
    let (rows, columns) = value[0].shape();
    Ok(
        Array4::from_shape_vec((2, 2, columns, rows), concatenated(value))
            .map_err(layout_error)?
            .permuted_axes([0, 1, 3, 2])
            .into_pyarray(py),
    )
}
/// Four equally shaped field matrices as a `(4, rows, columns)` array.
fn fields_array(py: Python<'_>, value: [DMatrix<Complex>; 4]) -> PyResult<C3<'_>> {
    let (rows, columns) = value[0].shape();
    Ok(
        Array3::from_shape_vec((4, columns, rows), concatenated(value))
            .map_err(layout_error)?
            .permuted_axes([0, 2, 1])
            .into_pyarray(py),
    )
}

context!(SMatrixFromArrayContext(FromArrayResidual));
#[pymethods]
impl SMatrixFromArrayContext {
    #[staticmethod]
    fn _state_spec(multipoles: usize, modes: usize) -> PyResult<usize> {
        ieee(|| FromArrayResidual::state_size(multipoles, modes).map_err(crate::context::error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        response: Tangent<'py>,
        channels: Tangent<'py>,
    ) -> PyResult<C4<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let (multipoles, modes) = residual.input_shape();
            let response = matrix_tangent(&response, (multipoles, multipoles))?;
            let channels = tangent_blocks(&channels, (multipoles, modes))?;
            blocks_array(
                py,
                detached(py, move || residual.pushforward(&response, &channels))?,
            )
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C2<'py>, C4<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = cotangent_blocks(&cotangent, residual.shape())?;
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok((
                owned_matrix(py, gradient.response)?,
                blocks_array(py, gradient.channels)?,
            ))
        })
    }
}
/// Record the S-matrix of a periodic array from its response and channels: `smatrix::from_array`.
#[pyfunction]
pub(crate) fn smatrix_from_array<'py>(
    py: Python<'py>,
    response: PyReadonlyArray2<'py, Complex>,
    channels: PyReadonlyArray4<'py, Complex>,
) -> PyResult<(C4<'py>, SMatrixFromArrayContext)> {
    ieee(|| {
        let response = from_array(response, "response")?;
        let a = channels.as_array();
        if a.shape()[..2] != [2, 2] {
            return Err(PyValueError::new_err(
                "channels require shape (2, 2, multipoles, plane modes)",
            ));
        }
        let channels = std::array::from_fn(|b| matrix_from_view(a.slice(s![b / 2, b % 2, .., ..])));
        let (value, residual) = detached(py, move || smatrix::from_array(response, channels))?;
        Ok((
            blocks_array(py, value)?,
            SMatrixFromArrayContext::new(residual),
        ))
    })
}

context!(SMatrixAddContext(AddResidual));

#[pymethods]
impl SMatrixAddContext {
    #[staticmethod]
    fn _state_spec(n: usize) -> PyResult<usize> {
        ieee(|| AddResidual::state_size(n).map_err(crate::context::error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        lower: Tangent<'py>,
        upper: Tangent<'py>,
    ) -> PyResult<C4<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let lower = tangent_blocks(&lower, residual.shape())?;
            let upper = tangent_blocks(&upper, residual.shape())?;
            blocks_array(
                py,
                detached(py, move || residual.pushforward(&lower, &upper))?,
            )
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C4<'py>, C4<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = cotangent_blocks(&cotangent, residual.shape())?;
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok((
                blocks_array(py, gradient.lower)?,
                blocks_array(py, gradient.upper)?,
            ))
        })
    }
}

/// Record the S-matrix of `upper` stacked on `lower`: `smatrix::add`.
#[pyfunction]
pub(crate) fn smatrix_add<'py>(
    py: Python<'py>,
    lower: PyReadonlyArray4<'py, Complex>,
    upper: PyReadonlyArray4<'py, Complex>,
) -> PyResult<(C4<'py>, SMatrixAddContext)> {
    ieee(|| {
        let lower = blocks(lower, "lower")?;
        let upper = blocks(upper, "upper")?;
        let (value, residual) = detached(py, move || smatrix::add(lower, upper))?;
        Ok((blocks_array(py, value)?, SMatrixAddContext::new(residual)))
    })
}

context!(ChiralityDensityContext(smatrix::ChiralityDensityResidual));

#[pymethods]
impl ChiralityDensityContext {
    #[staticmethod]
    fn _state_spec(n: usize) -> PyResult<usize> {
        ieee(|| smatrix::ChiralityDensityResidual::state_size(n).map_err(crate::context::error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        ks: Tangent<'py>,
        normal: Tangent<'py>,
        interval: RealTangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let n = residual.shape().1;
            let ks = vector_tangent(&ks, n)?;
            let normal = vector_tangent(&normal, n)?;
            let interval = tangent_array(&interval)?;
            owned_matrix(
                py,
                detached(py, move || residual.pushforward(&ks, &normal, interval))?,
            )
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C1<'py>, C1<'py>, R1<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok((
                gradient.ks.into_pyarray(py),
                gradient.normal.into_pyarray(py),
                gradient.interval.to_vec().into_pyarray(py),
            ))
        })
    }
}

/// Record the chirality-density coefficients of plane modes: `smatrix::chirality_density`.
#[pyfunction]
pub(crate) fn chirality_density<'py>(
    py: Python<'py>,
    ks: PyReadonlyArray1<'py, Complex>,
    normal: PyReadonlyArray1<'py, Complex>,
    interval: [f64; 2],
) -> PyResult<(C2<'py>, ChiralityDensityContext)> {
    ieee(|| {
        let ks = ks.as_array().to_vec();
        let normal = normal.as_array().to_vec();
        let (value, residual) =
            detached(py, move || smatrix::chirality_density(ks, normal, interval))?;
        Ok((
            owned_matrix(py, value)?,
            ChiralityDensityContext::new(residual),
        ))
    })
}

context!(OrientedChiralityContext(smatrix::OrientedChiralityResidual));

#[pymethods]
impl OrientedChiralityContext {
    #[staticmethod]
    fn _state_spec(n: usize) -> PyResult<usize> {
        ieee(|| smatrix::OrientedChiralityResidual::state_size(n).map_err(crate::context::error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        transverse: RealTangent<'py>,
        normal: Tangent<'py>,
        interval: RealTangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let n = residual.shape().1;
            let transverse = rows(
                finite_tangent::<_, Ix2>(&transverse, &[n, 2])?,
                "transverse tangent",
            )?;
            let normal = vector_tangent(&normal, n)?;
            let interval = tangent_array(&interval)?;
            owned_matrix(
                py,
                detached(py, move || {
                    residual.pushforward(&transverse, &normal, interval)
                })?,
            )
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, C1<'py>, R1<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok((
                rows_array(py, gradient.transverse)?,
                gradient.normal.into_pyarray(py),
                gradient.interval.to_vec().into_pyarray(py),
            ))
        })
    }
}

/// Record the signed helicity forms of plane modes for any normal: `smatrix::oriented_chirality`.
#[pyfunction]
pub(crate) fn oriented_chirality<'py>(
    py: Python<'py>,
    transverse: PyReadonlyArray2<'py, f64>,
    normal: PyReadonlyArray1<'py, Complex>,
    polarizations: Vec<u8>,
    axis: usize,
    interval: [f64; 2],
) -> PyResult<(C2<'py>, OrientedChiralityContext)> {
    ieee(|| {
        let transverse = rows(transverse.as_array(), "transverse")?;
        let normal = normal.as_array().to_vec();
        let (value, residual) = detached(py, move || {
            smatrix::oriented_chirality(transverse, normal, polarizations, axis, interval)
        })?;
        Ok((
            owned_matrix(py, value)?,
            OrientedChiralityContext::new(residual),
        ))
    })
}

context!(SMatrixIlluminateContext(smatrix::IlluminateResidual));

#[pymethods]
impl SMatrixIlluminateContext {
    #[staticmethod]
    fn _state_spec(n: usize, columns: usize) -> PyResult<usize> {
        ieee(|| smatrix::IlluminateResidual::state_size(n, columns).map_err(crate::context::error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        lower: Tangent<'py>,
        upper: Tangent<'py>,
        up: Tangent<'py>,
        down: Tangent<'py>,
    ) -> PyResult<C3<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let (n, columns) = residual.shape();
            let lower = tangent_blocks(&lower, (n, n))?;
            let upper = tangent_blocks(&upper, (n, n))?;
            let incoming = [
                matrix_tangent(&up, (n, columns))?,
                matrix_tangent(&down, (n, columns))?,
            ];
            fields_array(
                py,
                detached(py, move || residual.pushforward(&lower, &upper, &incoming))?,
            )
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C4<'py>, C4<'py>, C2<'py>, C2<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let (modes, columns) = residual.shape();
            let expected = [4, modes, columns];
            let a = cotangent_view::<_, Ix3>(&cotangent, &expected)?;
            let g = std::array::from_fn(|b| matrix_from_view(a.index_axis(Axis(0), b)));
            if !g
                .iter()
                .all(|field: &DMatrix<Complex>| all_finite(field.as_slice()))
            {
                return Err(cotangent_error(&expected));
            }
            let gradient = detached(py, move || residual.pullback(&g))?;
            let [up, down] = gradient.incoming;
            Ok((
                blocks_array(py, gradient.lower)?,
                blocks_array(py, gradient.upper)?,
                owned_matrix(py, up)?,
                owned_matrix(py, down)?,
            ))
        })
    }
}

/// Record the outgoing and internal fields of two stacked S-matrices: `smatrix::illuminate`.
#[pyfunction]
pub(crate) fn smatrix_illuminate<'py>(
    py: Python<'py>,
    lower: PyReadonlyArray4<'py, Complex>,
    upper: PyReadonlyArray4<'py, Complex>,
    up: PyReadonlyArray2<'py, Complex>,
    down: PyReadonlyArray2<'py, Complex>,
) -> PyResult<(C3<'py>, SMatrixIlluminateContext)> {
    ieee(|| {
        let lower = block_views(lower.as_array(), "lower")?.map(LentMatrix::new);
        let upper = block_views(upper.as_array(), "upper")?.map(LentMatrix::new);
        let lower = lower.each_ref().map(LentMatrix::faer);
        let upper = upper.each_ref().map(LentMatrix::faer);
        let incoming = [from_array(up, "up")?, from_array(down, "down")?];
        let (value, residual) = detached(py, move || smatrix::illuminate(lower, upper, incoming))?;
        Ok((
            fields_array(py, value)?,
            SMatrixIlluminateContext::new(residual),
        ))
    })
}

/// The fields of `smatrix_illuminate` without a context: `smatrix::illuminate_value`.
#[pyfunction]
pub(crate) fn smatrix_illuminate_value<'py>(
    py: Python<'py>,
    lower: PyReadonlyArray4<'py, Complex>,
    upper: PyReadonlyArray4<'py, Complex>,
    up: PyReadonlyArray2<'py, Complex>,
    down: PyReadonlyArray2<'py, Complex>,
) -> PyResult<C3<'py>> {
    ieee(|| {
        let lower = block_views(lower.as_array(), "lower")?.map(LentMatrix::new);
        let upper = block_views(upper.as_array(), "upper")?.map(LentMatrix::new);
        let incoming = [up.as_array(), down.as_array()].map(LentMatrix::new);
        let lower = lower.each_ref().map(LentMatrix::faer);
        let upper = upper.each_ref().map(LentMatrix::faer);
        let incoming = incoming.each_ref().map(LentMatrix::faer);
        let value = detached(py, || smatrix::illuminate_value(lower, upper, incoming))?;
        fields_array(py, value)
    })
}

context!(SMatrixPeriodicContext(smatrix::PeriodicResidual));

#[pymethods]
impl SMatrixPeriodicContext {
    #[staticmethod]
    fn _state_spec(n: usize) -> PyResult<usize> {
        ieee(|| smatrix::PeriodicResidual::state_size(n).map_err(crate::context::error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(&self, py: Python<'py>, smats: Tangent<'py>) -> PyResult<C2<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let n = residual.shape().0 / 2;
            let smats = tangent_blocks(&smats, (n, n))?;
            owned_matrix(py, detached(py, move || residual.pushforward(&smats))?)
        })
    }

    fn pullback<'py>(&self, py: Python<'py>, cotangent: Cotangent<'py>) -> PyResult<C4<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
            blocks_array(py, detached(py, move || residual.pullback(&g))?)
        })
    }
}

/// Record the transfer matrix of one period of a stack: `smatrix::periodic`.
#[pyfunction]
pub(crate) fn smatrix_periodic<'py>(
    py: Python<'py>,
    smats: PyReadonlyArray4<'py, Complex>,
) -> PyResult<(C2<'py>, SMatrixPeriodicContext)> {
    ieee(|| {
        let smats = blocks(smats, "smats")?;
        let (value, residual) = detached(py, move || smatrix::periodic(smats))?;
        Ok((
            owned_matrix(py, value)?,
            SMatrixPeriodicContext::new(residual),
        ))
    })
}

context!(BandsContext(smatrix::BandsResidual));

#[pymethods]
impl BandsContext {
    #[staticmethod]
    fn _state_spec(n: usize) -> PyResult<usize> {
        ieee(|| smatrix::BandsResidual::state_size(n).map_err(crate::context::error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        smats: Tangent<'py>,
        period: RealTangent<'py>,
    ) -> PyResult<(C1<'py>, C2<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let n = residual.wavenumbers().len() / 2;
            let period = finite_tangent::<_, numpy::ndarray::Ix0>(&period, &[])?[()];
            let smats = tangent_blocks(&smats, (n, n))?;
            let (values, vectors) = detached(py, move || residual.pushforward(&smats, period))?;
            Ok((values.into_pyarray(py), owned_matrix(py, vectors)?))
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        wavenumbers: Cotangent<'py>,
        eigenvectors: Cotangent<'py>,
    ) -> PyResult<(C4<'py>, f64)> {
        ieee(|| {
            let residual = &self.residual;
            let g = vector_cotangent(&wavenumbers, residual.wavenumbers().len())?;
            let vectors = matrix_cotangent(&eigenvectors, residual.vectors().shape())?;
            let gradient = detached(py, move || residual.pullback(&g, vectors))?;
            Ok((blocks_array(py, gradient.blocks)?, gradient.period))
        })
    }
}

/// Record the Bloch wavenumbers and eigenvectors of a periodic stack: `smatrix::bands`.
#[pyfunction]
pub(crate) fn bands<'py>(
    py: Python<'py>,
    smats: PyReadonlyArray4<'py, Complex>,
    period: f64,
) -> PyResult<(C1<'py>, C2<'py>, BandsContext)> {
    ieee(|| {
        let blocks = blocks(smats, "smats")?;
        let residual = detached(py, move || smatrix::bands(blocks, period))?;
        // The wavenumbers and the C-ordered eigenvectors leave as copies: the residual
        // keeps both for the pullback.
        Ok((
            residual.wavenumbers().to_vec().into_pyarray(py),
            matrix(py, residual.vectors())?,
            BandsContext::new(residual),
        ))
    })
}

context!(FresnelContext(FresnelResidual));

#[pymethods]
impl FresnelContext {
    #[staticmethod]
    fn _state_spec() -> PyResult<usize> {
        ieee(|| FresnelResidual::state_size().map_err(crate::context::error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        ks: Tangent<'py>,
        kzs: Tangent<'py>,
        zs: Tangent<'py>,
    ) -> PyResult<C4<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let ks = tangent_matrix(&ks)?;
            let kzs = tangent_matrix(&kzs)?;
            let zs = tangent_array(&zs)?;
            blocks_array(py, detached(py, move || residual.pushforward(ks, kzs, zs))?)
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C2<'py>, C2<'py>, C1<'py>)> {
        ieee(|| {
            let g = cotangent_blocks(&cotangent, (2, 2))?;
            let residual = &self.residual;
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok((
                rows_array(py, Vec::from(gradient.ks))?,
                rows_array(py, Vec::from(gradient.kz))?,
                gradient.z.to_vec().into_pyarray(py),
            ))
        })
    }
}

/// Record the Fresnel blocks of a chiral interface: `smatrix::fresnel`.
#[pyfunction]
pub(crate) fn fresnel(
    py: Python<'_>,
    ks: [[Complex; 2]; 2],
    kzs: [[Complex; 2]; 2],
    zs: [Complex; 2],
) -> PyResult<(C4<'_>, FresnelContext)> {
    ieee(|| {
        let (value, residual) = detached(py, move || smatrix::fresnel(ks, kzs, zs))?;
        Ok((blocks_array(py, value)?, FresnelContext::new(residual)))
    })
}

context!(InterfaceCoefficientsContext(smatrix::InterfaceResidual));
#[pymethods]
impl InterfaceCoefficientsContext {
    #[staticmethod]
    fn _state_spec() -> PyResult<usize> {
        ieee(|| smatrix::InterfaceResidual::state_size().map_err(crate::context::error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        ks: Tangent<'py>,
        zs: Tangent<'py>,
        q: RealTangent<'py>,
    ) -> PyResult<C4<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let ks = tangent_matrix(&ks)?;
            let zs = tangent_array(&zs)?;
            let q = tangent_array(&q)?;
            blocks_array(py, detached(py, move || residual.pushforward(ks, zs, q))?)
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C2<'py>, C1<'py>, R1<'py>)> {
        ieee(|| {
            let g = cotangent_blocks(&cotangent, (2, 2))?;
            let residual = &self.residual;
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok((
                rows_array(py, Vec::from(gradient.ks))?,
                gradient.z.to_vec().into_pyarray(py),
                gradient.q.to_vec().into_pyarray(py),
            ))
        })
    }
}
/// Record the S-matrix of a planar interface: `smatrix::interface`.
#[pyfunction]
pub(crate) fn interface_coefficients(
    py: Python<'_>,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    q: [f64; 2],
    axis: usize,
    fixed_q: bool,
) -> PyResult<(C4<'_>, InterfaceCoefficientsContext)> {
    ieee(|| {
        let (value, residual) = detached(py, move || smatrix::interface(ks, zs, q, axis, fixed_q))?;
        Ok((
            blocks_array(py, value)?,
            InterfaceCoefficientsContext::new(residual),
        ))
    })
}

context!(PropagationMatrixContext(PropagationResidual));

#[pymethods]
impl PropagationMatrixContext {
    #[staticmethod]
    fn _state_spec(n: usize) -> PyResult<usize> {
        ieee(|| PropagationResidual::state_size(n).map_err(crate::context::error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        vectors: Tangent<'py>,
        distance: RealTangent<'py>,
    ) -> PyResult<C4<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let vectors = rows(
                finite_tangent::<_, Ix2>(&vectors, &[residual.shape().0, 3])?,
                "vector tangent",
            )?;
            let distance = tangent_array(&distance)?;
            blocks_array(
                py,
                detached(py, move || residual.pushforward(&vectors, distance))?,
            )
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C2<'py>, R1<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = cotangent_blocks(&cotangent, residual.shape())?;
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok((
                rows_array(py, gradient.vectors)?,
                gradient.distance.to_vec().into_pyarray(py),
            ))
        })
    }
}

/// Record the S-matrix of propagation over a distance: `smatrix::propagation`.
#[pyfunction]
pub(crate) fn propagation_matrix(
    py: Python<'_>,
    vectors: Vec<[Complex; 3]>,
    distance: [f64; 3],
) -> PyResult<(C4<'_>, PropagationMatrixContext)> {
    ieee(|| {
        let (value, residual) = detached(py, move || smatrix::propagation(vectors, distance))?;
        Ok((
            blocks_array(py, value)?,
            PropagationMatrixContext::new(residual),
        ))
    })
}

context!(LayerStackContext(smatrix::LayerStackResidual));
#[pymethods]
impl LayerStackContext {
    #[staticmethod]
    fn _state_spec(media: usize, channels: usize) -> PyResult<usize> {
        ieee(|| {
            smatrix::LayerStackResidual::state_size(media, channels).map_err(crate::context::error)
        })
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        ks: Tangent<'py>,
        zs: Tangent<'py>,
        q: RealTangent<'py>,
        thickness: RealTangent<'py>,
    ) -> PyResult<C5<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let media = residual.medium_count();
            let ks = rows(
                finite_tangent::<_, Ix2>(&ks, &[media, 2])?,
                "wavenumber tangent",
            )?;
            let q = rows(
                finite_tangent::<_, Ix2>(&q, &[residual.channel_count(), 2])?,
                "transverse tangent",
            )?;
            let zs = vector_tangent(&zs, media)?;
            let thickness = vector_tangent(&thickness, media - 2)?;
            let values = detached(py, move || residual.pushforward(&ks, &zs, &q, &thickness))?;
            Ok(
                Array5::from_shape_fn((values.len(), 2, 2, 2, 2), |(q, a, b, i, j)| {
                    values[q][2 * a + b][(i, j)]
                })
                .into_pyarray(py),
            )
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(C2<'py>, C1<'py>, R2<'py>, R1<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let expected = [residual.channel_count(), 2, 2, 2, 2];
            let g = finite_cotangent::<_, Ix5>(&cotangent, &expected)?;
            let g = (0..g.shape()[0])
                .map(|q| {
                    std::array::from_fn(|b| {
                        DMatrix::from_fn(2, 2, |i, j| g[(q, b / 2, b % 2, i, j)])
                    })
                })
                .collect();
            let result = detached(py, move || residual.pullback(g))?;
            Ok((
                rows_array(py, result.ks)?,
                result.zs.into_pyarray(py),
                rows_array(py, result.q)?,
                result.thickness.into_pyarray(py),
            ))
        })
    }
}
/// Record the S-matrices of a layer stack, one per plane-wave channel: `smatrix::layer_stack`.
#[pyfunction]
pub(crate) fn layer_stack(
    py: Python<'_>,
    ks: Vec<[Complex; 2]>,
    zs: Vec<Complex>,
    q: Vec<[f64; 2]>,
    thickness: Vec<f64>,
    axis: usize,
    fixed_q: bool,
) -> PyResult<(C5<'_>, LayerStackContext)> {
    ieee(|| {
        let (values, residual) = detached(py, move || {
            smatrix::layer_stack(ks, &zs, q, &thickness, axis, fixed_q)
        })?;
        let array = Array5::from_shape_fn((values.len(), 2, 2, 2, 2), |(q, a, b, i, j)| {
            values[q][2 * a + b][(i, j)]
        })
        .into_pyarray(py);
        Ok((array, LayerStackContext::new(residual)))
    })
}

context!(SMatrixTrContext(smatrix::TrResidual));
#[pymethods]
impl SMatrixTrContext {
    #[staticmethod]
    fn _state_spec(n: usize, columns: usize, groups: usize) -> PyResult<usize> {
        ieee(|| smatrix::TrResidual::state_size(n, columns, groups).map_err(crate::context::error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state).map(Self::new))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        matrices: Tangent<'py>,
        incident: Tangent<'py>,
        ks: Tangent<'py>,
        zs: Tangent<'py>,
        q: RealTangent<'py>,
    ) -> PyResult<R2<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let (n, columns, groups) = residual.input_shape();
            let q = rows(
                finite_tangent::<_, Ix2>(&q, &[groups, 2])?,
                "transverse tangent",
            )?;
            let matrices = tangent_blocks(&matrices, (n, n))?;
            let incident = matrix_tangent(&incident, (n, columns))?;
            let ks = tangent_matrix(&ks)?;
            let zs = tangent_array(&zs)?;
            owned_matrix(
                py,
                detached(py, move || {
                    residual.pushforward(&matrices, &incident, ks, zs, &q)
                })?,
            )
        })
    }

    /// The powers are real, so only the real part of a complex cotangent
    /// enters the real pairing.
    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: RealCotangent<'_>,
    ) -> PyResult<(C4<'py>, C2<'py>, C2<'py>, C1<'py>, R2<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let a = finite_cotangent::<_, Ix2>(&cotangent, &<[usize; 2]>::from(residual.shape()))?;
            let g = DMatrix::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)]);
            let gradient = detached(py, move || residual.pullback(&g))?;
            Ok((
                blocks_array(py, gradient.matrices)?,
                owned_matrix(py, gradient.incident)?,
                rows_array(py, Vec::from(gradient.ks))?,
                gradient.zs.to_vec().into_pyarray(py),
                rows_array(py, gradient.q)?,
            ))
        })
    }
}

/// The blocks that light incident in `direction` meets: transmission
/// `[direction, direction]` and reflection `[1 - direction, direction]`.
fn tr_blocks(
    a: ArrayView4<'_, Complex>,
    direction: usize,
) -> PyResult<[ArrayView2<'_, Complex>; 2]> {
    let shape = a.shape();
    if shape[..2] != [2, 2] || shape[2] != shape[3] || direction > 1 {
        return Err(PyValueError::new_err(
            "require square (2,2,n,n) scattering blocks and direction 0/1",
        ));
    }
    Ok([
        a.slice_move(s![direction, direction, .., ..]),
        a.slice_move(s![1 - direction, direction, .., ..]),
    ])
}

/// Record the transmitted and reflected powers of illuminations: `smatrix::tr`.
#[pyfunction]
pub(crate) fn smatrix_tr<'py>(
    py: Python<'py>,
    matrices: PyReadonlyArray4<'_, Complex>,
    incident: PyReadonlyArray2<'_, Complex>,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    q: Vec<[f64; 2]>,
    modes: Vec<(usize, u8)>,
    axis: usize,
    helicity: bool,
    direction: usize,
    fixed_q: bool,
) -> PyResult<(R2<'py>, SMatrixTrContext)> {
    ieee(|| {
        let blocks = tr_blocks(matrices.as_array(), direction)?.map(LentMatrix::new);
        let views = blocks.each_ref().map(LentMatrix::faer);
        let incident = from_array(incident, "incident")?;
        let ports = smatrix::TrPorts {
            ks,
            zs,
            q,
            modes,
            axis,
            helicity,
            direction,
        };
        let (value, residual) = detached(py, move || {
            // The residual keeps the powers for the pullback; the output is a copy.
            let residual = smatrix::tr(views, incident, ports, fixed_q)?;
            Ok((residual.value().clone(), residual))
        })?;
        Ok((owned_matrix(py, value)?, SMatrixTrContext::new(residual)))
    })
}

/// The powers of `smatrix_tr` without a context: `smatrix::tr_value`.
#[pyfunction]
pub(crate) fn smatrix_tr_value<'py>(
    py: Python<'py>,
    matrices: PyReadonlyArray4<'_, Complex>,
    incident: PyReadonlyArray2<'_, Complex>,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    q: Vec<[f64; 2]>,
    modes: Vec<(usize, u8)>,
    axis: usize,
    helicity: bool,
    direction: usize,
) -> PyResult<R2<'py>> {
    ieee(|| {
        let blocks = tr_blocks(matrices.as_array(), direction)?.map(LentMatrix::new);
        let views = blocks.each_ref().map(LentMatrix::faer);
        let incident = from_array(incident, "incident")?;
        let ports = smatrix::TrPorts {
            ks,
            zs,
            q,
            modes,
            axis,
            helicity,
            direction,
        };
        let value = detached(py, move || smatrix::tr_value(views, &incident, &ports))?;
        owned_matrix(py, value)
    })
}
