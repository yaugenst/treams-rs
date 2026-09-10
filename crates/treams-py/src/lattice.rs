//! Batched Ewald sums and their special functions.

use num_complex::Complex64;
use pyo3::prelude::*;
use rayon::prelude::*;
use treams_core::{integrals, lattice};

use crate::error;

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
) -> PyResult<Bound<'_, numpy::PyArray2<Complex64>>> {
    let to = crate::basis::make_basis(to, to_positions);
    let source = crate::basis::make_basis(source, source_positions);
    let lattice = lattice::Lattice::new(&vectors, &bloch).map_err(error)?;
    let value = py
        .detach(move || treams_core::basis::periodic(&to, &source, ks, helicity, &lattice, eta))
        .map_err(error)?;
    Ok(crate::tmatrix::matrix(py, &value))
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
) -> PyResult<Bound<'_, numpy::PyArray2<Complex64>>> {
    let to = crate::basis::make_cyl_basis(to, to_positions);
    let source = crate::basis::make_cyl_basis(source, source_positions);
    let lattice = lattice::Lattice::new(&vectors, &bloch).map_err(error)?;
    let value = py
        .detach(move || treams_core::cylwaves::periodic(&to, &source, ks, &lattice, eta))
        .map_err(error)?;
    Ok(crate::tmatrix::matrix(py, &value))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(incgamma, m)?)?;
    m.add_function(wrap_pyfunction!(intkambe, m)?)?;
    m.add_function(wrap_pyfunction!(lattice_sum, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_expansion, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_cyl_expansion, m)?)?;
    Ok(())
}
