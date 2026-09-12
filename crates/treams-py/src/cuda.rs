//! Optional persistent CUDA objects; this module is absent from CPU-only builds.
use std::sync::Arc;

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use treams_core::Complex;
use treams_cuda::gpu::{self, CudaLu, DeviceMatrix, Gpu};

use crate::tmatrix::{from_array, owned_matrix};

type MatrixPair<'py> = (Bound<'py, PyArray2<Complex>>, Bound<'py, PyArray2<Complex>>);

fn error(error: gpu::Error) -> PyErr {
    match error {
        gpu::Error::InvalidInput(message) => PyValueError::new_err(message),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// A persistent device and stream.
#[pyclass(name = "CudaDevice")]
#[derive(Debug)]
pub(crate) struct Device {
    gpu: Gpu,
}

#[pymethods]
impl Device {
    #[new]
    #[pyo3(signature = (ordinal=0))]
    fn new(py: Python<'_>, ordinal: usize) -> PyResult<Self> {
        Ok(Self {
            gpu: py.detach(|| Gpu::new(ordinal)).map_err(error)?,
        })
    }

    #[getter]
    fn name(&self) -> PyResult<String> {
        self.gpu.name().map_err(error)
    }

    fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.gpu.synchronize()).map_err(error)
    }

    fn upload(&self, py: Python<'_>, value: PyReadonlyArray2<'_, Complex>) -> PyResult<Matrix> {
        let value = from_array(value)?;
        Ok(Matrix {
            value: py.detach(|| self.gpu.upload(&value)).map_err(error)?,
        })
    }

    fn factor(&self, py: Python<'_>, operator: PyReadonlyArray2<'_, Complex>) -> PyResult<Factor> {
        let operator = from_array(operator)?;
        let factor = py
            .detach(|| self.gpu.factor_host(operator))
            .map_err(error)?;
        Ok(Factor {
            gpu: self.gpu.clone(),
            factor: Arc::new(factor),
        })
    }

    fn matmul(&self, py: Python<'_>, left: &Matrix, right: &Matrix) -> PyResult<Matrix> {
        Ok(Matrix {
            value: py
                .detach(|| self.gpu.matmul(&left.value, &right.value))
                .map_err(error)?,
        })
    }
}

/// A resident matrix that explicitly downloads to `NumPy`.
#[pyclass(name = "CudaMatrix")]
#[derive(Debug)]
pub(crate) struct Matrix {
    value: DeviceMatrix,
}

#[pymethods]
impl Matrix {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.value.shape()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.value.bytes()
    }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex>>> {
        let host = py.detach(|| self.value.download()).map_err(error)?;
        owned_matrix(py, host)
    }
}

/// Reusable factors with host and resident-RHS operations.
#[pyclass(name = "CudaFactor")]
#[derive(Debug)]
pub(crate) struct Factor {
    gpu: Gpu,
    factor: Arc<CudaLu>,
}

#[pymethods]
impl Factor {
    #[getter]
    fn dimension(&self) -> usize {
        self.factor.dimension()
    }

    #[getter]
    fn nbytes(&self) -> usize {
        self.factor.bytes()
    }

    #[pyo3(signature = (rhs, *, adjoint=false))]
    fn solve<'py>(
        &self,
        py: Python<'py>,
        rhs: PyReadonlyArray2<'_, Complex>,
        adjoint: bool,
    ) -> PyResult<Bound<'py, PyArray2<Complex>>> {
        let rhs = from_array(rhs)?;
        let value = py
            .detach(|| {
                self.factor
                    .solve(self.gpu.upload(&rhs)?, adjoint)?
                    .download()
            })
            .map_err(error)?;
        owned_matrix(py, value)
    }

    #[pyo3(signature = (rhs, *, adjoint=false))]
    fn solve_device(&self, py: Python<'_>, rhs: &Matrix, adjoint: bool) -> PyResult<Matrix> {
        Ok(Matrix {
            value: py
                .detach(|| self.factor.solve(rhs.value.try_clone()?, adjoint))
                .map_err(error)?,
        })
    }

    fn solve_with_pullback<'py>(
        &self,
        py: Python<'py>,
        rhs: PyReadonlyArray2<'_, Complex>,
    ) -> PyResult<(Bound<'py, PyArray2<Complex>>, SolveContext)> {
        let rhs = from_array(rhs)?;
        let (host, value) = py
            .detach(|| {
                let value = self.factor.solve(self.gpu.upload(&rhs)?, false)?;
                Ok::<_, gpu::Error>((value.download()?, value))
            })
            .map_err(error)?;
        Ok((
            owned_matrix(py, host)?,
            SolveContext {
                gpu: self.gpu.clone(),
                factor: self.factor.clone(),
                value: Some(value),
            },
        ))
    }
}

/// A first-order native pullback, retaining GPU solution and shared LU.
#[pyclass(name = "CudaSolveContext")]
#[derive(Debug)]
pub(crate) struct SolveContext {
    gpu: Gpu,
    factor: Arc<CudaLu>,
    value: Option<DeviceMatrix>,
}

#[pymethods]
impl SolveContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'_, Complex>,
    ) -> PyResult<MatrixPair<'py>> {
        let g = from_array(cotangent)?;
        let value = self
            .value
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.shape() != value.shape() || g.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(PyValueError::new_err("invalid CUDA solve cotangent"));
        }
        let (ga, gb) = py
            .detach(|| {
                let (ga, gb) = self.factor.pullback(value, self.gpu.upload(&g)?)?;
                Ok::<_, gpu::Error>((ga.download()?, gb.download()?))
            })
            .map_err(error)?;
        self.value = None;
        Ok((owned_matrix(py, ga)?, owned_matrix(py, gb)?))
    }
}

/// Register only when the extension was explicitly built with CUDA.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Device>()?;
    module.add_class::<Matrix>()?;
    module.add_class::<Factor>()?;
    module.add_class::<SolveContext>()?;
    #[cfg(all(feature = "cuda-tile", target_os = "linux"))]
    module.add_class::<tile::PlaneWaves>()?;
    Ok(())
}

#[cfg(all(feature = "cuda-tile", target_os = "linux"))]
mod tile {
    #![allow(clippy::indexing_slicing)] // Three columns checked before row extraction.

    use super::{Complex, PyArray2, PyReadonlyArray2, PyResult, PyRuntimeError, PyValueError};
    use numpy::{IntoPyArray, PyReadonlyArray1, ndarray::Array2};
    use pyo3::prelude::*;

    #[pyclass(name = "CudaPlaneWaves")]
    #[derive(Debug)]
    pub(super) struct PlaneWaves {
        expansion: treams_cuda_tile::PlaneWaves,
    }

    #[pymethods]
    impl PlaneWaves {
        #[new]
        fn new(
            py: Python<'_>,
            device: usize,
            vectors: PyReadonlyArray2<'_, Complex>,
            polarizations: PyReadonlyArray1<'_, u8>,
            coefficients: PyReadonlyArray1<'_, Complex>,
            helicity: bool,
        ) -> PyResult<Self> {
            let vectors = vectors.as_array();
            if vectors.ncols() != 3 {
                return Err(PyValueError::new_err(
                    "wavevectors require shape (modes, 3)",
                ));
            }
            let vectors: Vec<_> = vectors
                .rows()
                .into_iter()
                .map(|r| [r[0], r[1], r[2]])
                .collect();
            let polarizations: Vec<_> = polarizations.as_array().iter().copied().collect();
            let coefficients: Vec<_> = coefficients.as_array().iter().copied().collect();
            let expansion = py
                .detach(|| {
                    treams_cuda_tile::PlaneWaves::new(
                        device,
                        &vectors,
                        &polarizations,
                        &coefficients,
                        helicity,
                    )
                })
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            Ok(Self { expansion })
        }

        fn evaluate<'py>(
            &self,
            py: Python<'py>,
            points: PyReadonlyArray2<'_, f64>,
        ) -> PyResult<Bound<'py, PyArray2<Complex>>> {
            let points = points.as_array();
            if points.ncols() != 3 {
                return Err(PyValueError::new_err("points require shape (points, 3)"));
            }
            let points: Vec<_> = points
                .rows()
                .into_iter()
                .map(|r| [r[0], r[1], r[2]])
                .collect();
            let values = py
                .detach(|| self.expansion.evaluate(&points))
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            Ok(
                Array2::from_shape_vec((values.len(), 3), values.into_iter().flatten().collect())
                    .map_err(|error| PyValueError::new_err(error.to_string()))?
                    .into_pyarray(py),
            )
        }
    }
}
