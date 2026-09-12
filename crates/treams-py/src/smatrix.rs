//! Native S-matrix blocks and owned reverse contexts.
#![allow(clippy::indexing_slicing)] // Validated four-block NumPy shapes.

use faer::MatRef;
use nalgebra::DMatrix;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArray5, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyReadonlyArray5,
    ndarray::{Array2, Array3, Array4, Array5, ArrayView2, s},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    smatrix::{self, ArrayResidual, Blocks, FresnelResidual, PropagationResidual, StackResidual},
};

use crate::error;

fn from_array(value: PyReadonlyArray4<'_, Complex>) -> PyResult<Blocks> {
    let a = value.as_array();
    let s = a.shape();
    if s[0] != 2 || s[1] != 2 || s[2] == 0 || s[2] != s[3] {
        return Err(PyValueError::new_err(
            "S matrices require shape (2, 2, n, n) with n > 0",
        ));
    }
    Ok(std::array::from_fn(|b| {
        crate::tmatrix::matrix_from_view(a.slice(s![b / 2, b % 2, .., ..]))
    }))
}
fn cotangent_blocks(value: PyReadonlyArray4<'_, Complex>) -> PyResult<Blocks> {
    let blocks = from_array(value)?;
    if blocks
        .iter()
        .flatten()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(PyValueError::new_err("S matrices must be finite"));
    }
    Ok(blocks)
}
fn array<'py>(py: Python<'py>, value: &Blocks) -> Bound<'py, PyArray4<Complex>> {
    let (d, c) = value[0].shape();
    Array4::from_shape_fn((2, 2, c, d), |(a, b, j, i)| value[2 * a + b][(i, j)])
        .permuted_axes([0, 1, 3, 2])
        .into_pyarray(py)
}

fn array_owned(py: Python<'_>, value: Blocks) -> PyResult<Bound<'_, PyArray4<Complex>>> {
    let (rows, cols) = value[0].shape();
    let [first, second, third, fourth] = value;
    let mut data = Vec::from(first.data);
    data.reserve(3 * rows * cols);
    for block in [second, third, fourth] {
        data.extend(Vec::from(block.data));
    }
    Array4::from_shape_vec((2, 2, cols, rows), data)
        .map(|a| a.permuted_axes([0, 1, 3, 2]).into_pyarray(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[allow(clippy::expect_used)] // Caller packs every non-contiguous view.
fn borrowed_matrix<'a>(
    a: &'a ArrayView2<'_, Complex>,
    packed: Option<&'a DMatrix<Complex>>,
) -> MatRef<'a, Complex> {
    if let Some(packed) = packed {
        MatRef::from_column_major_slice(packed.as_slice(), a.nrows(), a.ncols())
    } else {
        let data = a
            .as_slice_memory_order()
            .expect("contiguous input or packed copy");
        if a.strides()[0] == 1 {
            MatRef::from_column_major_slice(data, a.nrows(), a.ncols())
        } else {
            MatRef::from_row_major_slice(data, a.nrows(), a.ncols())
        }
    }
}

#[pyclass]
#[derive(Debug)]
struct ArrayContext {
    residual: Option<ArrayResidual>,
}
type ArrayGradient<'py> = (Bound<'py, PyArray2<Complex>>, Bound<'py, PyArray4<Complex>>);
#[pymethods]
impl ArrayContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray4<'py, Complex>,
    ) -> PyResult<ArrayGradient<'py>> {
        let g = cotangent_blocks(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g[0].shape() != residual.value[0].shape() {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (response, channels) = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            crate::tmatrix::owned_matrix(py, response)?,
            array_owned(py, channels)?,
        ))
    }
}
#[pyfunction]
fn smatrix_from_array<'py>(
    py: Python<'py>,
    response: PyReadonlyArray2<'py, Complex>,
    channels: PyReadonlyArray4<'py, Complex>,
) -> PyResult<(Bound<'py, PyArray4<Complex>>, ArrayContext)> {
    let response = crate::tmatrix::from_array(response)?;
    let a = channels.as_array();
    let s = a.shape();
    if s[0] != 2 || s[1] != 2 {
        return Err(PyValueError::new_err(
            "channels require shape (2, 2, multipoles, plane modes)",
        ));
    }
    let channels =
        std::array::from_fn(|b| DMatrix::from_fn(s[2], s[3], |i, j| a[(b / 2, b % 2, i, j)]));
    let residual = py
        .detach(move || smatrix::from_array(response, channels))
        .map_err(error)?;
    Ok((
        array(py, &residual.value),
        ArrayContext {
            residual: Some(residual),
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct SMatrixContext {
    residual: Option<StackResidual>,
}

type Pair<'py> = (Bound<'py, PyArray4<Complex>>, Bound<'py, PyArray4<Complex>>);

#[pymethods]
impl SMatrixContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray4<'py, Complex>,
    ) -> PyResult<Pair<'py>> {
        let g = cotangent_blocks(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g[0].shape() != residual.value[0].shape() {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (lower, upper) = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((array_owned(py, lower)?, array_owned(py, upper)?))
    }
}

#[pyfunction]
fn smatrix_add<'py>(
    py: Python<'py>,
    lower: PyReadonlyArray4<'py, Complex>,
    upper: PyReadonlyArray4<'py, Complex>,
) -> PyResult<(Bound<'py, PyArray4<Complex>>, SMatrixContext)> {
    let lower = from_array(lower)?;
    let upper = from_array(upper)?;
    let residual = py
        .detach(move || smatrix::add(lower, upper))
        .map_err(error)?;
    Ok((
        array(py, &residual.value),
        SMatrixContext {
            residual: Some(residual),
        },
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TransmissionContext>()?;
    module.add_function(wrap_pyfunction!(smatrix_transmittance, module)?)?;
    module.add_class::<ChiralityContext>()?;
    module.add_function(wrap_pyfunction!(chirality_density, module)?)?;
    module.add_class::<OrientedChiralityContext>()?;
    module.add_function(wrap_pyfunction!(oriented_chirality, module)?)?;
    module.add_class::<IlluminationContext>()?;
    module.add_class::<SMatrixPeriodicContext>()?;
    module.add_class::<BandContext>()?;
    module.add_function(wrap_pyfunction!(smatrix_illuminate, module)?)?;
    module.add_function(wrap_pyfunction!(smatrix_illuminate_forward, module)?)?;
    module.add_function(wrap_pyfunction!(smatrix_periodic, module)?)?;
    module.add_function(wrap_pyfunction!(bands, module)?)?;
    module.add_class::<ArrayContext>()?;
    module.add_function(wrap_pyfunction!(smatrix_from_array, module)?)?;
    module.add_class::<SMatrixContext>()?;
    module.add_function(wrap_pyfunction!(smatrix_add, module)?)?;
    module.add_class::<FresnelContext>()?;
    module.add_class::<PropagationContext>()?;
    module.add_function(wrap_pyfunction!(fresnel, module)?)?;
    module.add_class::<InterfaceContext>()?;
    module.add_class::<LayersContext>()?;
    module.add_function(wrap_pyfunction!(layer_stack, module)?)?;
    module.add_function(wrap_pyfunction!(interface, module)?)?;
    module.add_function(wrap_pyfunction!(propagation, module)?)?;
    Ok(())
}

#[pyclass]
#[derive(Debug)]
struct ChiralityContext {
    residual: Option<smatrix::ChiralityResidual>,
}

type ChiralityGradient<'py> = (
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray1<f64>>,
);

#[pymethods]
impl ChiralityContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<ChiralityGradient<'py>> {
        let g = crate::tmatrix::from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.shape() != residual.shape() {
            return Err(PyValueError::new_err(
                "chirality cotangent shape must match output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let gradient = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            gradient.ks.into_pyarray(py),
            gradient.normal.into_pyarray(py),
            gradient.interval.to_vec().into_pyarray(py),
        ))
    }
}

#[pyfunction]
fn chirality_density<'py>(
    py: Python<'py>,
    ks: PyReadonlyArray1<'py, Complex>,
    normal: PyReadonlyArray1<'py, Complex>,
    interval: [f64; 2],
) -> PyResult<(Bound<'py, PyArray2<Complex>>, ChiralityContext)> {
    let ks = ks.as_array().to_vec();
    let normal = normal.as_array().to_vec();
    let (value, residual) = py
        .detach(move || smatrix::chirality_density(ks, normal, interval))
        .map_err(error)?;
    Ok((
        crate::tmatrix::owned_matrix(py, value)?,
        ChiralityContext {
            residual: Some(residual),
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct OrientedChiralityContext {
    residual: Option<smatrix::OrientedChiralityResidual>,
}

type OrientedChiralityGradient<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray1<f64>>,
);

#[pymethods]
impl OrientedChiralityContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<OrientedChiralityGradient<'py>> {
        let g = crate::tmatrix::from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.shape() != residual.shape() {
            return Err(PyValueError::new_err(
                "chirality cotangent shape must match output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let gradient = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            Array2::from_shape_fn((gradient.transverse.len(), 2), |(i, j)| {
                gradient.transverse[i][j]
            })
            .into_pyarray(py),
            gradient.normal.into_pyarray(py),
            gradient.interval.to_vec().into_pyarray(py),
        ))
    }
}

#[pyfunction]
fn oriented_chirality<'py>(
    py: Python<'py>,
    transverse: PyReadonlyArray2<'py, f64>,
    normal: PyReadonlyArray1<'py, Complex>,
    polarizations: Vec<u8>,
    axis: usize,
    interval: [f64; 2],
) -> PyResult<(Bound<'py, PyArray2<Complex>>, OrientedChiralityContext)> {
    let q = transverse.as_array();
    if q.ncols() != 2 {
        return Err(PyValueError::new_err(
            "transverse components must have shape (N, 2)",
        ));
    }
    let transverse = q.rows().into_iter().map(|row| [row[0], row[1]]).collect();
    let normal = normal.as_array().to_vec();
    let (value, residual) = py
        .detach(move || {
            smatrix::oriented_chirality(transverse, normal, polarizations, axis, interval)
        })
        .map_err(error)?;
    let output = Array2::from_shape_vec((value.ncols(), 3), Vec::from(value.data))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .reversed_axes()
        .into_pyarray(py);
    Ok((
        output,
        OrientedChiralityContext {
            residual: Some(residual),
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct IlluminationContext {
    residual: Option<smatrix::IlluminationResidual>,
}

type IlluminationGradient<'py> = (
    Bound<'py, PyArray4<Complex>>,
    Bound<'py, PyArray4<Complex>>,
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray2<Complex>>,
);

#[pymethods]
impl IlluminationContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray3<'py, Complex>,
    ) -> PyResult<IlluminationGradient<'py>> {
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (n, p) = residual.shape();
        let a = cotangent.as_array();
        if a.dim() != (4, n, p) || a.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(PyValueError::new_err(
                "cotangent must be finite and match the field coefficient shape",
            ));
        }
        let g = std::array::from_fn(|b| DMatrix::from_fn(n, p, |i, j| a[(b, i, j)]));
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (lower, upper, [up, down]) = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            array_owned(py, lower)?,
            array_owned(py, upper)?,
            crate::tmatrix::owned_matrix(py, up)?,
            crate::tmatrix::owned_matrix(py, down)?,
        ))
    }
}

#[pyfunction]
fn smatrix_illuminate<'py>(
    py: Python<'py>,
    lower: PyReadonlyArray4<'py, Complex>,
    upper: PyReadonlyArray4<'py, Complex>,
    up: PyReadonlyArray2<'py, Complex>,
    down: PyReadonlyArray2<'py, Complex>,
) -> PyResult<(Bound<'py, PyArray3<Complex>>, IlluminationContext)> {
    let arrays = [lower.as_array(), upper.as_array()];
    for a in &arrays {
        let s = a.shape();
        if s[0] != 2 || s[1] != 2 || s[2] == 0 || s[2] != s[3] {
            return Err(PyValueError::new_err(
                "S matrices require shape (2, 2, n, n) with n > 0",
            ));
        }
    }
    let blocks: [ArrayView2<'_, Complex>; 8] =
        std::array::from_fn(|i| arrays[i / 4].slice(s![i / 2 % 2, i % 2, .., ..]));
    let packed = blocks.each_ref().map(|a| {
        a.as_slice_memory_order()
            .is_none()
            .then(|| crate::tmatrix::matrix_from_view(*a))
    });
    let lower = std::array::from_fn(|i| borrowed_matrix(&blocks[i], packed[i].as_ref()));
    let upper = std::array::from_fn(|i| borrowed_matrix(&blocks[i + 4], packed[i + 4].as_ref()));
    let incoming = [
        crate::tmatrix::from_array(up)?,
        crate::tmatrix::from_array(down)?,
    ];
    let (value, residual) = py
        .detach(move || smatrix::illuminate_borrowed(lower, upper, incoming))
        .map_err(error)?;
    let (n, p) = residual.shape();
    let value = Array3::from_shape_fn((4, n, p), |(b, i, j)| value[b][(i, j)]).into_pyarray(py);
    Ok((
        value,
        IlluminationContext {
            residual: Some(residual),
        },
    ))
}

#[pyfunction]
fn smatrix_illuminate_forward<'py>(
    py: Python<'py>,
    lower: PyReadonlyArray4<'py, Complex>,
    upper: PyReadonlyArray4<'py, Complex>,
    up: PyReadonlyArray2<'py, Complex>,
    down: PyReadonlyArray2<'py, Complex>,
) -> PyResult<Bound<'py, PyArray3<Complex>>> {
    let arrays = [lower.as_array(), upper.as_array()];
    if arrays.iter().any(|a| a.shape()[0..2] != [2, 2]) {
        return Err(PyValueError::new_err(
            "S matrices require shape (2, 2, n, n)",
        ));
    }
    let blocks: [ArrayView2<'_, Complex>; 8] =
        std::array::from_fn(|i| arrays[i / 4].slice(s![i / 2 % 2, i % 2, .., ..]));
    let inputs = [up.as_array(), down.as_array()];
    let packed = blocks.each_ref().map(|a| {
        a.as_slice_memory_order()
            .is_none()
            .then(|| crate::tmatrix::matrix_from_view(*a))
    });
    let packed_inputs = inputs.each_ref().map(|a| {
        a.as_slice_memory_order()
            .is_none()
            .then(|| crate::tmatrix::matrix_from_view(*a))
    });
    let lower = std::array::from_fn(|i| borrowed_matrix(&blocks[i], packed[i].as_ref()));
    let upper = std::array::from_fn(|i| borrowed_matrix(&blocks[i + 4], packed[i + 4].as_ref()));
    let incoming = std::array::from_fn(|i| borrowed_matrix(&inputs[i], packed_inputs[i].as_ref()));
    let value = py
        .detach(|| smatrix::illuminate_forward(lower, upper, incoming))
        .map_err(error)?;
    let (n, p) = value[0].shape();
    Ok(Array3::from_shape_fn((4, n, p), |(b, i, j)| value[b][(i, j)]).into_pyarray(py))
}

#[pyclass]
#[derive(Debug)]
struct SMatrixPeriodicContext {
    residual: Option<smatrix::PeriodicResidual>,
}

#[pymethods]
impl SMatrixPeriodicContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<Bound<'py, PyArray4<Complex>>> {
        let g = crate::tmatrix::from_array(cotangent)?;
        let n = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?
            .dimension();
        if g.shape() != (2 * n, 2 * n) {
            return Err(PyValueError::new_err(
                "cotangent shape does not match transfer matrix",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let result = py.detach(move || residual.pullback(&g)).map_err(error)?;
        array_owned(py, result)
    }
}

#[pyfunction]
fn smatrix_periodic<'py>(
    py: Python<'py>,
    smats: PyReadonlyArray4<'py, Complex>,
) -> PyResult<(Bound<'py, PyArray2<Complex>>, SMatrixPeriodicContext)> {
    let smats = from_array(smats)?;
    let (value, residual) = py.detach(move || smatrix::periodic(smats)).map_err(error)?;
    Ok((
        crate::tmatrix::owned_matrix(py, value)?,
        SMatrixPeriodicContext {
            residual: Some(residual),
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct BandContext {
    residual: Option<smatrix::BandResidual>,
}

#[pymethods]
impl BandContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        wavenumbers: PyReadonlyArray1<'py, Complex>,
        eigenvectors: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<(Bound<'py, PyArray4<Complex>>, f64)> {
        let g: Vec<_> = wavenumbers.as_array().iter().copied().collect();
        let vectors = crate::tmatrix::from_array(eigenvectors)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.len() != residual.wavenumbers.len()
            || vectors.shape() != residual.vectors().shape()
            || g.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(PyValueError::new_err(
                "cotangent shapes must match finite Bloch outputs",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (smats, period) = py
            .detach(move || residual.pullback(&g, vectors))
            .map_err(error)?;
        Ok((array_owned(py, smats)?, period))
    }
}

type BandResult<'py> = (
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray2<Complex>>,
    BandContext,
);

#[pyfunction]
fn bands<'py>(
    py: Python<'py>,
    smats: PyReadonlyArray4<'py, Complex>,
    period: f64,
) -> PyResult<BandResult<'py>> {
    let blocks = from_array(smats)?;
    let residual = py
        .detach(move || smatrix::bands(blocks, period))
        .map_err(error)?;
    Ok((
        residual.wavenumbers.clone().into_pyarray(py),
        crate::tmatrix::matrix(py, residual.vectors()),
        BandContext {
            residual: Some(residual),
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct FresnelContext {
    residual: Option<FresnelResidual>,
}

type FresnelGradient<'py> = (
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray1<Complex>>,
);

#[pymethods]
impl FresnelContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray4<'py, Complex>,
    ) -> PyResult<FresnelGradient<'py>> {
        let g = cotangent_blocks(cotangent)?;
        if g[0].nrows() != 2 {
            return Err(PyValueError::new_err(
                "Fresnel cotangent requires shape (2, 2, 2, 2)",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (ks, kz, z) = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            Array2::from_shape_fn((2, 2), |(i, j)| ks[i][j]).into_pyarray(py),
            Array2::from_shape_fn((2, 2), |(i, j)| kz[i][j]).into_pyarray(py),
            z.to_vec().into_pyarray(py),
        ))
    }
}

#[pyfunction]
fn fresnel(
    py: Python<'_>,
    ks: [[Complex; 2]; 2],
    kz: [[Complex; 2]; 2],
    z: [Complex; 2],
) -> PyResult<(Bound<'_, PyArray4<Complex>>, FresnelContext)> {
    let residual = py
        .detach(move || smatrix::fresnel(ks, kz, z))
        .map_err(error)?;
    Ok((
        array(py, &residual.value),
        FresnelContext {
            residual: Some(residual),
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct InterfaceContext {
    residual: Option<smatrix::InterfaceResidual>,
    fixed_q: bool,
}
type InterfaceGradient<'py> = (
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray1<f64>>,
);
#[pymethods]
impl InterfaceContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray4<'py, Complex>,
    ) -> PyResult<InterfaceGradient<'py>> {
        let g = cotangent_blocks(cotangent)?;
        if g[0].nrows() != 2 {
            return Err(PyValueError::new_err(
                "interface cotangent requires shape (2, 2, 2, 2)",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let fixed_q = self.fixed_q;
        let (ks, z, q) = py
            .detach(move || residual.pullback(&g, fixed_q))
            .map_err(error)?;
        Ok((
            Array2::from_shape_fn((2, 2), |(i, j)| ks[i][j]).into_pyarray(py),
            z.to_vec().into_pyarray(py),
            q.to_vec().into_pyarray(py),
        ))
    }
}
#[pyfunction]
fn interface(
    py: Python<'_>,
    ks: [[Complex; 2]; 2],
    z: [Complex; 2],
    q: [f64; 2],
    axis: usize,
    fixed_q: bool,
) -> PyResult<(Bound<'_, PyArray4<Complex>>, InterfaceContext)> {
    let residual = py
        .detach(move || smatrix::interface(ks, z, q, axis))
        .map_err(error)?;
    Ok((
        array(py, &residual.value),
        InterfaceContext {
            residual: Some(residual),
            fixed_q,
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct PropagationContext {
    residual: Option<PropagationResidual>,
}

type PropagationGradient<'py> = (Bound<'py, PyArray2<Complex>>, Bound<'py, PyArray1<f64>>);

#[pymethods]
impl PropagationContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray4<'py, Complex>,
    ) -> PyResult<PropagationGradient<'py>> {
        let g = cotangent_blocks(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g[0].shape() != residual.value[0].shape() {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (vectors, distance) = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            Array2::from_shape_fn((vectors.len(), 3), |(i, j)| vectors[i][j]).into_pyarray(py),
            distance.to_vec().into_pyarray(py),
        ))
    }
}

#[pyfunction]
fn propagation(
    py: Python<'_>,
    vectors: Vec<[Complex; 3]>,
    distance: [f64; 3],
) -> PyResult<(Bound<'_, PyArray4<Complex>>, PropagationContext)> {
    let residual = py
        .detach(move || smatrix::propagation(vectors, distance))
        .map_err(error)?;
    Ok((
        array(py, &residual.value),
        PropagationContext {
            residual: Some(residual),
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct LayersContext {
    residual: Option<treams_core::layers::LayersResidual>,
    fixed_q: bool,
}
type LayersGradient<'py> = (
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
);
#[pymethods]
impl LayersContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray5<'py, Complex>,
    ) -> PyResult<LayersGradient<'py>> {
        let g = cotangent.as_array();
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.shape() != [residual.channel_count(), 2, 2, 2, 2]
            || g.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(PyValueError::new_err(
                "compact layer cotangent must be finite and match shape (channels, 2, 2, 2, 2)",
            ));
        }
        let g = (0..residual.channel_count())
            .map(|q| {
                std::array::from_fn(|b| DMatrix::from_fn(2, 2, |i, j| g[(q, b / 2, b % 2, i, j)]))
            })
            .collect();
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let fixed_q = self.fixed_q;
        let result = py
            .detach(move || residual.pullback(g, fixed_q))
            .map_err(error)?;
        Ok((
            Array2::from_shape_fn((result.ks.len(), 2), |(i, j)| result.ks[i][j]).into_pyarray(py),
            result.zs.into_pyarray(py),
            Array2::from_shape_fn((result.q.len(), 2), |(i, j)| result.q[i][j]).into_pyarray(py),
            result.thickness.into_pyarray(py),
        ))
    }
}
#[pyfunction]
fn layer_stack(
    py: Python<'_>,
    ks: Vec<[Complex; 2]>,
    zs: Vec<Complex>,
    q: Vec<[f64; 2]>,
    thickness: Vec<f64>,
    axis: usize,
    fixed_q: bool,
) -> PyResult<(Bound<'_, PyArray5<Complex>>, LayersContext)> {
    let (values, residual) = py
        .detach(move || treams_core::layers::stack(ks, &zs, q, &thickness, axis))
        .map_err(error)?;
    let array = Array5::from_shape_fn((values.len(), 2, 2, 2, 2), |(q, a, b, i, j)| {
        values[q][2 * a + b][(i, j)]
    })
    .into_pyarray(py);
    Ok((
        array,
        LayersContext {
            residual: Some(residual),
            fixed_q,
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct TransmissionContext {
    residual: Option<smatrix::TransmissionResidual>,
    fixed_q: bool,
}
type TransmissionGradient<'py> = (
    Bound<'py, PyArray4<Complex>>,
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray2<Complex>>,
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray2<f64>>,
);
#[pymethods]
impl TransmissionContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'_, Complex>,
    ) -> PyResult<TransmissionGradient<'py>> {
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let a = cotangent.as_array();
        if a.dim() != residual.value.shape()
            || a.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(PyValueError::new_err(
                "power cotangent must be finite and match output shape",
            ));
        }
        let g = DMatrix::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)].re);
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let fixed_q = self.fixed_q;
        let gradient = py
            .detach(move || residual.pullback(&g, fixed_q))
            .map_err(error)?;
        let shape = gradient.incident.shape();
        let incident =
            Array2::from_shape_vec((shape.1, shape.0), Vec::from(gradient.incident.data))
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .reversed_axes()
                .into_pyarray(py);
        Ok((
            array_owned(py, gradient.matrices)?,
            incident,
            Array2::from_shape_fn((2, 2), |(i, j)| gradient.ks[i][j]).into_pyarray(py),
            gradient.zs.to_vec().into_pyarray(py),
            Array2::from_shape_fn((gradient.q.len(), 2), |(i, j)| gradient.q[i][j])
                .into_pyarray(py),
        ))
    }
}

#[pyfunction]
fn smatrix_transmittance<'py>(
    py: Python<'py>,
    matrices: PyReadonlyArray4<'_, Complex>,
    incident: PyReadonlyArray2<'_, Complex>,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    q: Vec<[f64; 2]>,
    modes: Vec<(usize, u8)>,
    axis: usize,
    helicity: bool,
    transmission: usize,
    fixed_q: bool,
    record: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Option<TransmissionContext>)> {
    let a = matrices.as_array();
    let shape = a.shape();
    if shape[..2] != [2, 2] || shape[2] != shape[3] || transmission > 1 {
        return Err(PyValueError::new_err(
            "require square (2,2,n,n) scattering blocks and direction 0/1",
        ));
    }
    let blocks = [
        a.slice(s![transmission, transmission, .., ..]),
        a.slice(s![1 - transmission, transmission, .., ..]),
    ];
    let packed = blocks.each_ref().map(|a| {
        a.as_slice_memory_order()
            .is_none()
            .then(|| crate::tmatrix::matrix_from_view(*a))
    });
    let views = std::array::from_fn(|i| borrowed_matrix(&blocks[i], packed[i].as_ref()));
    let incident = crate::tmatrix::from_array(incident)?;
    let (value, residual) = py
        .detach(move || {
            if record {
                let residual = smatrix::transmittance(
                    views,
                    incident,
                    ks,
                    zs,
                    q,
                    modes,
                    axis,
                    helicity,
                    transmission,
                )?;
                Ok((residual.value.clone(), Some(residual)))
            } else {
                Ok((
                    smatrix::transmittance_value(
                        views,
                        &incident,
                        ks,
                        zs,
                        &q,
                        &modes,
                        axis,
                        helicity,
                        transmission,
                    )?,
                    None,
                ))
            }
        })
        .map_err(error)?;
    let shape = value.shape();
    let value = Array2::from_shape_vec((shape.1, shape.0), Vec::from(value.data))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .reversed_axes()
        .into_pyarray(py);
    Ok((
        value,
        residual.map(|residual| TransmissionContext {
            residual: Some(residual),
            fixed_q,
        }),
    ))
}
