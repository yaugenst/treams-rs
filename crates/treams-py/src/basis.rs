//! Python expansion contexts.
#![allow(clippy::indexing_slicing)] // Native gradient dimensions and Cartesian triples are validated.
use crate::{
    error,
    tmatrix::{from_array, matrix},
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, ndarray::Array2};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    basis::{self, Basis, TranslationResidual},
    special::Radial,
    waves::Mode,
};

#[derive(Debug)]
enum Expansion {
    Spherical(TranslationResidual),
    Cylindrical(treams_core::cylwaves::ExpansionResidual),
    Conversion(treams_core::conversion::ConversionResidual),
}
impl Expansion {
    fn value(&self) -> &nalgebra::DMatrix<Complex> {
        match self {
            Self::Spherical(r) => &r.value,
            Self::Cylindrical(r) => &r.value,
            Self::Conversion(r) => &r.value,
        }
    }
    fn pullback(
        self,
        g: &nalgebra::DMatrix<Complex>,
    ) -> treams_core::Result<basis::TranslationGradient> {
        match self {
            Self::Spherical(r) => r.pullback(g),
            Self::Cylindrical(r) => r.pullback(g),
            Self::Conversion(r) => r.pullback(g),
        }
    }
}

#[pyclass]
#[derive(Debug)]
struct ExpansionContext {
    residual: Option<Expansion>,
}

type Gradient<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<Complex>>,
);
#[pymethods]
impl ExpansionContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<Gradient<'py>> {
        let g = from_array(cotangent)?;
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
        let result = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            Array2::from_shape_fn((result.destination.len(), 3), |(i, j)| {
                result.destination[i][j]
            })
            .into_pyarray(py),
            Array2::from_shape_fn((result.source.len(), 3), |(i, j)| result.source[i][j])
                .into_pyarray(py),
            result.ks.to_vec().into_pyarray(py),
        ))
    }
}

pub(crate) fn make_basis(modes: Vec<(usize, i32, i32, u8)>, positions: Vec<[f64; 3]>) -> Basis {
    Basis {
        modes: modes
            .into_iter()
            .map(|(p, l, m, pol)| (p, Mode { l, m, pol }))
            .collect(),
        positions,
    }
}

#[pyclass]
#[derive(Debug)]
struct RotationContext {
    residual: Option<treams_core::rotation::RotationResidual>,
}
#[pymethods]
impl RotationContext {
    fn pullback(
        &mut self,
        py: Python<'_>,
        cotangent: PyReadonlyArray2<'_, Complex>,
    ) -> PyResult<[f64; 3]> {
        let g = from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.shape() != residual.value.shape()
            || g.iter().any(|g| !g.re.is_finite() || !g.im.is_finite())
        {
            return Err(PyValueError::new_err(
                "cotangent must be finite and match the rotation shape",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        py.detach(move || residual.pullback(&g)).map_err(error)
    }
}

#[pyfunction]
fn rotation(
    py: Python<'_>,
    to: Vec<(usize, i32, i32, u8)>,
    source: Vec<(usize, i32, i32, u8)>,
    to_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    angles: [f64; 3],
) -> PyResult<(Bound<'_, PyArray2<Complex>>, RotationContext)> {
    let to = make_basis(to, to_positions);
    let source = make_basis(source, source_positions);
    let residual = py
        .detach(move || treams_core::rotation::spherical(to, source, angles))
        .map_err(error)?;
    Ok((
        matrix(py, &residual.value),
        RotationContext {
            residual: Some(residual),
        },
    ))
}

#[pyfunction]
fn cyl_rotation(
    py: Python<'_>,
    to: Vec<(usize, f64, i32, u8)>,
    source: Vec<(usize, f64, i32, u8)>,
    to_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    angles: [f64; 3],
) -> PyResult<(Bound<'_, PyArray2<Complex>>, RotationContext)> {
    let to = make_cyl_basis(to, to_positions);
    let source = make_cyl_basis(source, source_positions);
    let residual = py
        .detach(move || treams_core::rotation::cylindrical(&to, &source, angles))
        .map_err(error)?;
    Ok((
        matrix(py, &residual.value),
        RotationContext {
            residual: Some(residual),
        },
    ))
}

#[pyfunction]
fn expansion(
    py: Python<'_>,
    to: Vec<(usize, i32, i32, u8)>,
    source: Vec<(usize, i32, i32, u8)>,
    to_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
    outgoing: bool,
) -> PyResult<(Bound<'_, PyArray2<Complex>>, ExpansionContext)> {
    let to = make_basis(to, to_positions);
    let source = make_basis(source, source_positions);
    let radial = if outgoing {
        Radial::Outgoing
    } else {
        Radial::Regular
    };
    let residual = py
        .detach(move || basis::translation(to, source, ks, helicity, radial))
        .map_err(error)?;
    Ok((
        matrix(py, &residual.value),
        ExpansionContext {
            residual: Some(Expansion::Spherical(residual)),
        },
    ))
}

pub(crate) fn make_cyl_basis(
    modes: Vec<(usize, f64, i32, u8)>,
    positions: Vec<[f64; 3]>,
) -> treams_core::cylwaves::Basis {
    treams_core::cylwaves::Basis {
        modes: modes
            .into_iter()
            .map(|(p, kz, m, pol)| (p, treams_core::cylwaves::Mode { kz, m, pol }))
            .collect(),
        positions,
    }
}

#[pyfunction]
fn cyl_expansion(
    py: Python<'_>,
    to: Vec<(usize, f64, i32, u8)>,
    source: Vec<(usize, f64, i32, u8)>,
    to_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    outgoing: bool,
) -> PyResult<(Bound<'_, PyArray2<Complex>>, ExpansionContext)> {
    let to = make_cyl_basis(to, to_positions);
    let source = make_cyl_basis(source, source_positions);
    let radial = if outgoing {
        Radial::Outgoing
    } else {
        Radial::Regular
    };
    let residual = py
        .detach(move || treams_core::cylwaves::expansion(to, source, ks, radial))
        .map_err(error)?;
    Ok((
        matrix(py, &residual.value),
        ExpansionContext {
            residual: Some(Expansion::Cylindrical(residual)),
        },
    ))
}

#[pyfunction]
fn cw_to_sw(
    py: Python<'_>,
    to: Vec<(usize, i32, i32, u8)>,
    source: Vec<(usize, f64, i32, u8)>,
    to_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
) -> PyResult<(Bound<'_, PyArray2<Complex>>, ExpansionContext)> {
    let to = make_basis(to, to_positions);
    let source = make_cyl_basis(source, source_positions);
    let residual = py
        .detach(move || treams_core::conversion::cylindrical_to_spherical(to, source, ks, helicity))
        .map_err(error)?;
    Ok((
        matrix(py, &residual.value),
        ExpansionContext {
            residual: Some(Expansion::Conversion(residual)),
        },
    ))
}

type CylJet = (Complex, [Complex; 3], Complex, Complex);
#[pyfunction]
fn cyl_translation(
    to: (f64, i32, u8),
    source: (f64, i32, u8),
    k: Complex,
    position: [f64; 3],
    outgoing: bool,
) -> PyResult<CylJet> {
    let to = treams_core::cylwaves::Mode {
        kz: to.0,
        m: to.1,
        pol: to.2,
    };
    let source = treams_core::cylwaves::Mode {
        kz: source.0,
        m: source.1,
        pol: source.2,
    };
    let result = treams_core::cylwaves::translate(
        to,
        source,
        k,
        position,
        if outgoing {
            Radial::Outgoing
        } else {
            Radial::Regular
        },
    )
    .map_err(error)?;
    Ok((result.value, result.position, result.k, result.kz))
}

#[pyfunction]
fn plane_to_spherical(
    py: Python<'_>,
    modes: Vec<(usize, i32, i32, u8)>,
    positions: Vec<[f64; 3]>,
    vector: [Complex; 3],
    pol: u8,
    helicity: bool,
) -> PyResult<Bound<'_, PyArray1<Complex>>> {
    let basis = make_basis(modes, positions);
    Ok(py
        .detach(move || treams_core::plane::spherical(&basis, vector, pol, helicity))
        .map_err(error)?
        .into_pyarray(py))
}
#[pyfunction]
fn plane_polarization(vector: [Complex; 3], pol: u8, helicity: bool) -> PyResult<[Complex; 3]> {
    treams_core::plane::polarization(vector, pol, helicity).map_err(error)
}

#[pyfunction]
fn plane_to_cylindrical(
    py: Python<'_>,
    modes: Vec<(usize, f64, i32, u8)>,
    positions: Vec<[f64; 3]>,
    vector: [Complex; 3],
    pol: u8,
) -> PyResult<Bound<'_, PyArray1<Complex>>> {
    let basis = make_cyl_basis(modes, positions);
    Ok(py
        .detach(move || treams_core::plane::cylindrical(&basis, vector, pol))
        .map_err(error)?
        .into_pyarray(py))
}

#[pyclass]
#[derive(Debug)]
struct PlaneExpansionContext {
    residual: Option<treams_core::plane::ExpansionResidual>,
    fixed_vectors: bool,
}
type PlaneGradient<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<Complex>>);
#[pymethods]
impl PlaneExpansionContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<PlaneGradient<'py>> {
        let g = from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.shape() != residual.shape() {
            return Err(PyValueError::new_err(
                "cotangent shape does not match forward output",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let fixed = self.fixed_vectors;
        let gradient = py
            .detach(move || residual.pullback(&g, fixed))
            .map_err(error)?;
        Ok((
            Array2::from_shape_fn((gradient.origins.len(), 3), |(i, j)| gradient.origins[i][j])
                .into_pyarray(py),
            Array2::from_shape_fn((gradient.vectors.len(), 3), |(i, j)| gradient.vectors[i][j])
                .into_pyarray(py),
        ))
    }
}
#[pyfunction]
fn plane_expansion(
    py: Python<'_>,
    modes: Vec<(usize, i32, i32, u8)>,
    origins: Vec<[f64; 3]>,
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    helicity: bool,
    fixed_vectors: bool,
) -> PyResult<(Bound<'_, PyArray2<Complex>>, PlaneExpansionContext)> {
    let basis = make_basis(modes, origins);
    let (value, residual) = py
        .detach(move || treams_core::plane::expansion(basis, vectors, polarizations, helicity))
        .map_err(error)?;
    finish_plane_expansion(py, value, residual, fixed_vectors)
}

#[pyfunction]
fn cylindrical_plane_expansion(
    py: Python<'_>,
    modes: Vec<(usize, f64, i32, u8)>,
    origins: Vec<[f64; 3]>,
    vectors: Vec<[Complex; 3]>,
    polarizations: Vec<u8>,
    helicity: bool,
    fixed_vectors: bool,
) -> PyResult<(Bound<'_, PyArray2<Complex>>, PlaneExpansionContext)> {
    let basis = make_cyl_basis(modes, origins);
    let (value, residual) = py
        .detach(move || treams_core::plane::expansion(basis, vectors, polarizations, helicity))
        .map_err(error)?;
    finish_plane_expansion(py, value, residual, fixed_vectors)
}

fn finish_plane_expansion(
    py: Python<'_>,
    value: nalgebra::DMatrix<Complex>,
    residual: treams_core::plane::ExpansionResidual,
    fixed_vectors: bool,
) -> PyResult<(Bound<'_, PyArray2<Complex>>, PlaneExpansionContext)> {
    let array = Array2::from_shape_vec((value.ncols(), value.nrows()), Vec::from(value.data))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .reversed_axes()
        .into_pyarray(py);
    Ok((
        array,
        PlaneExpansionContext {
            residual: Some(residual),
            fixed_vectors,
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct PeriodicConversionContext {
    residual: Option<treams_core::conversion::PeriodicConversionResidual>,
}
type PeriodicConversionGradient<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray1<f64>>,
    f64,
);
#[pymethods]
impl PeriodicConversionContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<PeriodicConversionGradient<'py>> {
        let g = from_array(cotangent)?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        if g.shape() != residual.shape() || g.iter().any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(PyValueError::new_err(
                "cotangent must be finite and match the conversion shape",
            ));
        }
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let result = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            Array2::from_shape_fn((result.expansion.destination.len(), 3), |(i, j)| {
                result.expansion.destination[i][j]
            })
            .into_pyarray(py),
            Array2::from_shape_fn((result.expansion.source.len(), 3), |(i, j)| {
                result.expansion.source[i][j]
            })
            .into_pyarray(py),
            result.expansion.ks.to_vec().into_pyarray(py),
            result.kz.into_pyarray(py),
            result.period,
        ))
    }
}
#[pyfunction]
fn periodic_conversion(
    py: Python<'_>,
    to: Vec<(usize, f64, i32, u8)>,
    source: Vec<(usize, i32, i32, u8)>,
    to_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    period: f64,
    helicity: bool,
) -> PyResult<(Bound<'_, PyArray2<Complex>>, PeriodicConversionContext)> {
    let to = make_cyl_basis(to, to_positions);
    let source = make_basis(source, source_positions);
    let (value, residual) = py
        .detach(move || {
            treams_core::conversion::periodic_spherical_to_cylindrical(
                to, source, ks, period, helicity,
            )
        })
        .map_err(error)?;
    let value = Array2::from_shape_vec((value.ncols(), value.nrows()), Vec::from(value.data))
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .reversed_axes()
        .into_pyarray(py);
    Ok((
        value,
        PeriodicConversionContext {
            residual: Some(residual),
        },
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PlaneExpansionContext>()?;
    m.add_class::<PeriodicConversionContext>()?;
    m.add_function(wrap_pyfunction!(periodic_conversion, m)?)?;
    m.add_function(wrap_pyfunction!(plane_expansion, m)?)?;
    m.add_function(wrap_pyfunction!(cylindrical_plane_expansion, m)?)?;
    m.add_class::<RotationContext>()?;
    m.add_function(wrap_pyfunction!(rotation, m)?)?;
    m.add_function(wrap_pyfunction!(cyl_rotation, m)?)?;
    m.add_class::<ExpansionContext>()?;
    m.add_function(wrap_pyfunction!(plane_to_cylindrical, m)?)?;
    m.add_function(wrap_pyfunction!(plane_to_spherical, m)?)?;
    m.add_function(wrap_pyfunction!(plane_polarization, m)?)?;
    m.add_function(wrap_pyfunction!(expansion, m)?)?;
    m.add_function(wrap_pyfunction!(cyl_expansion, m)?)?;
    m.add_function(wrap_pyfunction!(cw_to_sw, m)?)?;
    m.add_function(wrap_pyfunction!(cyl_translation, m)?)?;
    Ok(())
}
