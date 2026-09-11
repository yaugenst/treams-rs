//! Batched Ewald sums and their special functions.

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ndarray::Array2};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use treams_core::{integrals, lattice};

use crate::error;

#[derive(Debug)]
enum Periodic {
    Spherical(treams_core::basis::PeriodicResidual),
    Cylindrical(treams_core::cylwaves::PeriodicResidual),
}
impl Periodic {
    fn value(&self) -> &nalgebra::DMatrix<Complex64> {
        match self {
            Self::Spherical(r) => &r.value,
            Self::Cylindrical(r) => &r.value,
        }
    }
    fn pullback(
        self,
        g: &nalgebra::DMatrix<Complex64>,
    ) -> treams_core::Result<treams_core::basis::PeriodicGradient> {
        match self {
            Self::Spherical(r) => r.pullback(g),
            Self::Cylindrical(r) => r.pullback(g),
        }
    }
}
#[pyclass]
#[derive(Debug)]
struct PeriodicContext {
    residual: Option<Periodic>,
}
type PeriodicGradients<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<Complex64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
);
type AxialPeriodicGradients<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<Complex64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
);
impl PeriodicContext {
    fn take(
        &mut self,
        cotangent: PyReadonlyArray2<'_, Complex64>,
    ) -> PyResult<(Periodic, nalgebra::DMatrix<Complex64>)> {
        let g = crate::tmatrix::from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.shape() != residual.value().shape() {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        Ok((residual, g))
    }
}
fn periodic_gradient(
    py: Python<'_>,
    result: treams_core::basis::PeriodicGradient,
) -> PeriodicGradients<'_> {
    (
        Array2::from_shape_fn((result.expansion.destination.len(), 3), |(i, j)| {
            result.expansion.destination[i][j]
        })
        .into_pyarray(py),
        Array2::from_shape_fn((result.expansion.source.len(), 3), |(i, j)| {
            result.expansion.source[i][j]
        })
        .into_pyarray(py),
        result.expansion.ks.to_vec().into_pyarray(py),
        result.bloch.into_pyarray(py),
        Array2::from_shape_fn(result.vectors.shape(), |(i, j)| result.vectors[(i, j)])
            .into_pyarray(py),
    )
}
#[pymethods]
impl PeriodicContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<PeriodicGradients<'py>> {
        let (residual, g) = self.take(cotangent)?;
        let result = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok(periodic_gradient(py, result))
    }
    /// Also return gradients of sorted distinct shared axial wavenumbers.
    fn pullback_axial<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex64>,
    ) -> PyResult<AxialPeriodicGradients<'py>> {
        let (residual, g) = self.take(cotangent)?;
        let Periodic::Cylindrical(residual) = residual else {
            return Err(PyValueError::new_err(
                "axial periodic derivatives require two cylindrical bases",
            ));
        };
        let (result, axial) = py
            .detach(move || residual.pullback_axial(&g))
            .map_err(error)?;
        let (destination, source, ks, bloch, vectors) = periodic_gradient(py, result);
        Ok((
            destination,
            source,
            ks,
            bloch,
            vectors,
            axial.into_pyarray(py),
        ))
    }
}

#[pyfunction]
fn incgamma(n: f64, z: Complex64) -> PyResult<Complex64> {
    integrals::incgamma(n, z).map_err(error)
}

#[pyfunction]
fn intkambe(n: i32, z: Complex64, eta: Complex64) -> PyResult<Complex64> {
    integrals::intkambe(n, z, eta).map_err(error)
}

#[pyfunction]
fn lattice_sum(
    py: Python<'_>,
    spherical: bool,
    modes: Vec<(i32, i32)>,
    k: Complex64,
    bloch: Vec<f64>,
    vectors: Vec<Vec<f64>>,
    shift: [f64; 3],
    eta: Complex64,
) -> PyResult<Vec<Complex64>> {
    let lattice = lattice::Lattice::new(&vectors, &bloch).map_err(error)?;
    py.detach(move || {
        modes
            .par_iter()
            .map(|&(l, m)| {
                let wave = if spherical {
                    lattice::Wave::Spherical { l, m }
                } else {
                    lattice::Wave::Cylindrical { m }
                };
                lattice::sum(wave, k, &lattice, shift, eta)
            })
            .collect::<treams_core::Result<Vec<_>>>()
    })
    .map_err(error)
}

#[pyfunction]
fn periodic_expansion(
    py: Python<'_>,
    to: Vec<(usize, i32, i32, u8)>,
    source: Vec<(usize, i32, i32, u8)>,
    to_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    ks: [Complex64; 2],
    helicity: bool,
    bloch: Vec<f64>,
    vectors: Vec<Vec<f64>>,
    eta: Complex64,
) -> PyResult<(Bound<'_, PyArray2<Complex64>>, PeriodicContext)> {
    let to = crate::basis::make_basis(to, to_positions);
    let source = crate::basis::make_basis(source, source_positions);
    let lattice = lattice::Lattice::new(&vectors, &bloch).map_err(error)?;
    let value = py
        .detach(move || treams_core::basis::periodic(to, source, ks, helicity, lattice, eta))
        .map_err(error)?;
    Ok((
        crate::tmatrix::matrix(py, &value.value),
        PeriodicContext {
            residual: Some(Periodic::Spherical(value)),
        },
    ))
}

#[pyfunction]
fn periodic_cyl_expansion(
    py: Python<'_>,
    to: Vec<(usize, f64, i32, u8)>,
    source: Vec<(usize, f64, i32, u8)>,
    to_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    ks: [Complex64; 2],
    bloch: Vec<f64>,
    vectors: Vec<Vec<f64>>,
    eta: Complex64,
) -> PyResult<(Bound<'_, PyArray2<Complex64>>, PeriodicContext)> {
    let to = crate::basis::make_cyl_basis(to, to_positions);
    let source = crate::basis::make_cyl_basis(source, source_positions);
    let lattice = lattice::Lattice::new(&vectors, &bloch).map_err(error)?;
    let value = py
        .detach(move || treams_core::cylwaves::periodic(to, source, ks, lattice, eta))
        .map_err(error)?;
    Ok((
        crate::tmatrix::matrix(py, &value.value),
        PeriodicContext {
            residual: Some(Periodic::Cylindrical(value)),
        },
    ))
}

type LatticeDerivatives = (
    Complex64,
    Complex64,
    [Complex64; 3],
    [Complex64; 3],
    [[Complex64; 3]; 3],
);
#[pyfunction]
fn lattice_derivatives(
    py: Python<'_>,
    spherical: bool,
    mode: (i32, i32),
    k: Complex64,
    bloch: Vec<f64>,
    vectors: Vec<Vec<f64>>,
    shift: [f64; 3],
    eta: Complex64,
) -> PyResult<LatticeDerivatives> {
    let wave = if spherical {
        lattice::Wave::Spherical {
            l: mode.0,
            m: mode.1,
        }
    } else {
        lattice::Wave::Cylindrical { m: mode.1 }
    };
    let lattice = lattice::Lattice::new(&vectors, &bloch).map_err(error)?;
    let result = py
        .detach(move || lattice::derivatives(wave, k, &lattice, shift, eta))
        .map_err(error)?;
    Ok((
        result.value,
        result.k,
        result.position,
        result.bloch,
        result.vectors,
    ))
}

#[pyfunction]
fn lattice_cube(
    py: Python<'_>,
    dim: usize,
    n: i64,
    edge: bool,
) -> PyResult<Bound<'_, PyArray2<i64>>> {
    let values = py
        .detach(move || treams_core::geometry::cube(dim, n, edge))
        .map_err(error)?;
    Ok(Array2::from_shape_vec((values.len() / dim, dim), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py))
}
#[pyfunction]
fn diffraction_orders<'py>(
    py: Python<'py>,
    b: PyReadonlyArray2<'py, f64>,
    radius: f64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let b = b.as_array();
    if b.shape() != [2, 2] {
        return Err(PyValueError::new_err(
            "reciprocal lattice requires shape (2, 2)",
        ));
    }
    let b = std::array::from_fn(|i| std::array::from_fn(|j| b[(i, j)]));
    let values = py
        .detach(move || treams_core::geometry::diffraction_orders(b, radius))
        .map_err(error)?;
    Ok(Array2::from_shape_vec((values.len() / 2, 2), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py))
}
#[pyfunction]
fn first_brillouin<'py>(
    py: Python<'py>,
    k: PyReadonlyArray1<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
    dim: usize,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let k = k.as_array();
    let b = b.as_array();
    if !(2..=3).contains(&dim) || k.len() != dim || b.shape() != [dim, dim] {
        return Err(PyValueError::new_err(
            "wavevector and lattice must match dimension",
        ));
    }
    let k = std::array::from_fn(|i| if i < dim { k[i] } else { 0.0 });
    let b = std::array::from_fn(|i| {
        std::array::from_fn(|j| if i < dim && j < dim { b[(i, j)] } else { 0.0 })
    });
    let value = py
        .detach(move || treams_core::geometry::first_brillouin(k, b, dim, n))
        .map_err(error)?;
    Ok(value[..dim].to_vec().into_pyarray(py))
}
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lattice_cube, m)?)?;
    m.add_function(wrap_pyfunction!(diffraction_orders, m)?)?;
    m.add_function(wrap_pyfunction!(first_brillouin, m)?)?;
    m.add_class::<PeriodicContext>()?;
    m.add_function(wrap_pyfunction!(incgamma, m)?)?;
    m.add_function(wrap_pyfunction!(intkambe, m)?)?;
    m.add_function(wrap_pyfunction!(lattice_sum, m)?)?;
    m.add_function(wrap_pyfunction!(lattice_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_expansion, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_cyl_expansion, m)?)?;
    Ok(())
}
