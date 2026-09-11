//! Opaque native contexts for complete spherical and cluster operations.
#![allow(clippy::indexing_slicing)] // Validated NumPy shapes and fixed coordinates.

use nalgebra::DMatrix;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::{Array2, ArrayView2},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use treams_core::{
    Complex,
    interaction::{self, InteractionResidual},
    special::Radial,
    tmatrix::{self, ClusterResidual, SphereResidual},
    waves::{self, Mode},
};

use crate::{error, materials};

pub(crate) fn matrix<'py>(
    py: Python<'py>,
    value: &DMatrix<Complex>,
) -> Bound<'py, PyArray2<Complex>> {
    Array2::from_shape_fn(value.shape(), |(i, j)| value[(i, j)]).into_pyarray(py)
}
pub(crate) fn from_array(value: PyReadonlyArray2<'_, Complex>) -> PyResult<DMatrix<Complex>> {
    let a = value.as_array();
    if a.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(PyValueError::new_err("array must be finite"));
    }
    Ok(matrix_from_view(a))
}

pub(crate) fn matrix_from_view(a: ArrayView2<'_, Complex>) -> DMatrix<Complex> {
    // Tile the NumPy-to-column-major copy so large C-order inputs do not walk
    // one cache line per element. Both tiles fit in the CPU's L1 data cache.
    if a.strides()[0] == 1
        && let Some(data) = a.as_slice_memory_order()
    {
        return DMatrix::from_column_slice(a.nrows(), a.ncols(), data);
    }
    let mut result = DMatrix::zeros(a.nrows(), a.ncols());
    if a.is_empty() {
        return result;
    }
    let fill = |(block, columns): (usize, &mut [Complex])| {
        for row in (0..a.nrows()).step_by(32) {
            for (j, column) in columns.chunks_mut(a.nrows()).enumerate() {
                for i in row..(row + 32).min(a.nrows()) {
                    column[i] = a[(i, 32 * block + j)];
                }
            }
        }
    };
    if a.len() >= 65_536 {
        result
            .as_mut_slice()
            .par_chunks_mut(32 * a.nrows())
            .enumerate()
            .for_each(fill);
    } else {
        result
            .as_mut_slice()
            .chunks_mut(32 * a.nrows())
            .enumerate()
            .for_each(fill);
    }
    result
}

#[pyclass]
#[derive(Debug)]
struct SphereContext {
    residual: Option<SphereResidual>,
}

type SphereGradient<'py> = (
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray1<Complex>>,
);

#[pymethods]
impl SphereContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<SphereGradient<'py>> {
        let g = from_array(cotangent)?;
        let shape = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?
            .value
            .shape();
        if g.shape() != shape {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let result = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            result.k0,
            result.radii.into_pyarray(py),
            result.materials.epsilon.into_pyarray(py),
            result.materials.mu.into_pyarray(py),
            result.materials.kappa.into_pyarray(py),
        ))
    }
}

#[pyfunction]
fn sphere<'py>(
    py: Python<'py>,
    lmax: u32,
    k0: f64,
    radii: PyReadonlyArray1<'py, f64>,
    epsilon: PyReadonlyArray1<'py, Complex>,
    mu: PyReadonlyArray1<'py, Complex>,
    kappa: PyReadonlyArray1<'py, Complex>,
) -> PyResult<(Bound<'py, PyArray2<Complex>>, SphereContext)> {
    let radii = radii.to_vec()?;
    let mat = materials(&epsilon.to_vec()?, &mu.to_vec()?, &kappa.to_vec()?)?;
    let residual = py
        .detach(move || tmatrix::sphere(lmax, k0, &radii, &mat))
        .map_err(error)?;
    Ok((
        matrix(py, &residual.value),
        SphereContext {
            residual: Some(residual),
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct ClusterContext {
    residual: Option<ClusterResidual>,
}

type ClusterGradient<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<Complex>>,
    f64,
);

type MatrixPair<'py> = (Bound<'py, PyArray2<Complex>>, Bound<'py, PyArray2<Complex>>);

#[pymethods]
impl ClusterContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<ClusterGradient<'py>> {
        let g = from_array(cotangent)?;
        let shape = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?
            .value()
            .shape();
        if g.shape() != shape {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let result = py.detach(move || residual.pullback(&g)).map_err(error)?;
        let positions =
            Array2::from_shape_fn((result.positions.len(), 3), |(i, j)| result.positions[i][j]);
        Ok((
            result.radii.into_pyarray(py),
            positions.into_pyarray(py),
            result.epsilon.into_pyarray(py),
            result.k0,
        ))
    }
}

#[pyfunction]
fn cluster<'py>(
    py: Python<'py>,
    lmax: u32,
    k0: f64,
    radii: PyReadonlyArray1<'py, f64>,
    epsilon: PyReadonlyArray1<'py, Complex>,
    positions: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<Complex>>, ClusterContext)> {
    let radii = radii.to_vec()?;
    let epsilon = epsilon.to_vec()?;
    let positions = positions.as_array();
    if positions.ncols() != 3 {
        return Err(PyValueError::new_err("positions must have shape (N, 3)"));
    }
    let positions: Vec<_> = positions
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect();
    let residual = py
        .detach(move || tmatrix::cluster(lmax, k0, &radii, &epsilon, &positions))
        .map_err(error)?;
    Ok((
        matrix(py, residual.value()),
        ClusterContext {
            residual: Some(residual),
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct InteractionContext {
    residual: Option<InteractionResidual>,
}

#[pymethods]
impl InteractionContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<MatrixPair<'py>> {
        let g = from_array(cotangent)?;
        let shape = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?
            .value
            .shape();
        if g.shape() != shape {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (local, coupling) = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((matrix(py, &local), matrix(py, &coupling)))
    }
}

#[pyfunction]
fn interact<'py>(
    py: Python<'py>,
    local: PyReadonlyArray2<'py, Complex>,
    coupling: PyReadonlyArray2<'py, Complex>,
) -> PyResult<(Bound<'py, PyArray2<Complex>>, InteractionContext)> {
    let local = from_array(local)?;
    let coupling = from_array(coupling)?;
    let residual = py
        .detach(move || interaction::forward(local, coupling))
        .map_err(error)?;
    Ok((
        matrix(py, &residual.value),
        InteractionContext {
            residual: Some(residual),
        },
    ))
}

#[pyfunction]
fn translation(
    to: (i32, i32, u8),
    from: (i32, i32, u8),
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    outgoing: bool,
) -> PyResult<(Complex, [Complex; 3], Complex)> {
    let (l, m, pol) = to;
    let to = Mode { l, m, pol };
    let (l, m, pol) = from;
    let from = Mode { l, m, pol };
    let radial = if outgoing {
        Radial::Outgoing
    } else {
        Radial::Regular
    };
    let result = waves::translate(to, from, k, position, helicity, radial).map_err(error)?;
    Ok((result.value, result.position, result.k))
}

type WaveJet = ([Complex; 3], [[Complex; 3]; 3], [Complex; 3]);

#[pyfunction]
fn spherical_wave(
    mode: (i32, i32, u8),
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    outgoing: bool,
) -> PyResult<WaveJet> {
    let (l, m, pol) = mode;
    let result = treams_core::fields::spherical_wave(
        Mode { l, m, pol },
        k,
        position,
        helicity,
        if outgoing {
            Radial::Outgoing
        } else {
            Radial::Regular
        },
    )
    .map_err(error)?;
    Ok((result.value, result.position, result.k))
}

#[pyclass]
#[derive(Debug)]
struct ParticleClusterContext {
    residual: Option<tmatrix::ParticleClusterResidual>,
}

type ParticleClusterGradient<'py> = (
    Vec<Bound<'py, PyArray2<Complex>>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<Complex>>,
);

#[pymethods]
impl ParticleClusterContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<ParticleClusterGradient<'py>> {
        let g = from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.shape() != residual.value().shape() {
            return Err(PyValueError::new_err(
                "cotangent must match the cluster matrix shape",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let gradient = py.detach(move || residual.pullback(&g)).map_err(error)?;
        let local = gradient
            .local
            .into_iter()
            .map(|value| {
                Array2::from_shape_vec((value.ncols(), value.nrows()), Vec::from(value.data))
                    .map(|a| a.reversed_axes().into_pyarray(py))
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok((
            local,
            Array2::from_shape_fn((gradient.positions.len(), 3), |(i, j)| {
                gradient.positions[i][j]
            })
            .into_pyarray(py),
            gradient.ks.to_vec().into_pyarray(py),
        ))
    }
}

#[pyfunction]
fn particle_cluster<'py>(
    py: Python<'py>,
    local: Vec<PyReadonlyArray2<'py, Complex>>,
    modes: Vec<(usize, i32, i32, u8)>,
    positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
) -> PyResult<(Bound<'py, PyArray2<Complex>>, ParticleClusterContext)> {
    let local = local
        .into_iter()
        .map(from_array)
        .collect::<PyResult<Vec<_>>>()?;
    let basis = crate::basis::make_basis(modes, positions);
    let residual = py
        .detach(move || tmatrix::particle_cluster(local, basis, ks, helicity))
        .map_err(error)?;
    Ok((
        matrix(py, residual.value()),
        ParticleClusterContext {
            residual: Some(residual),
        },
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ParticleClusterContext>()?;
    m.add_function(wrap_pyfunction!(particle_cluster, m)?)?;
    m.add_class::<MetricContext>()?;
    m.add_function(wrap_pyfunction!(tmatrix_metric, m)?)?;
    m.add_class::<SphereContext>()?;
    m.add_class::<ClusterContext>()?;
    m.add_class::<InteractionContext>()?;
    m.add_function(wrap_pyfunction!(sphere, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_wave, m)?)?;
    m.add_function(wrap_pyfunction!(cluster, m)?)?;
    m.add_function(wrap_pyfunction!(interact, m)?)?;
    m.add_function(wrap_pyfunction!(translation, m)?)?;
    Ok(())
}

#[pyclass]
#[derive(Debug)]
struct MetricContext {
    residual: Option<tmatrix::MetricResidual>,
}

type MetricGradient<'py> = (Bound<'py, PyArray2<Complex>>, Bound<'py, PyArray1<f64>>);

#[pymethods]
impl MetricContext {
    fn pullback<'py>(&mut self, py: Python<'py>, cotangent: f64) -> PyResult<MetricGradient<'py>> {
        if !cotangent.is_finite() {
            return Err(PyValueError::new_err("metric cotangent must be finite"));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let (gradient, ks) = py
            .detach(move || residual.pullback(cotangent))
            .map_err(error)?;
        Ok((matrix(py, &gradient), ks.to_vec().into_pyarray(py)))
    }
}

#[pyfunction]
fn tmatrix_metric(
    py: Python<'_>,
    operator: PyReadonlyArray2<'_, Complex>,
    polarizations: Vec<u8>,
    ks: [f64; 2],
    kind: &str,
) -> PyResult<(f64, MetricContext)> {
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
    let matrix = from_array(operator)?;
    let residual = py
        .detach(move || tmatrix::metric(&matrix, &polarizations, ks, kind))
        .map_err(error)?;
    Ok((
        residual.value,
        MetricContext {
            residual: Some(residual),
        },
    ))
}
