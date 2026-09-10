//! Python expansion contexts.
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
}
impl Expansion {
    fn value(&self) -> &nalgebra::DMatrix<Complex> {
        match self {
            Self::Spherical(r) => &r.value,
            Self::Cylindrical(r) => &r.value,
        }
    }
    fn pullback(
        self,
        g: &nalgebra::DMatrix<Complex>,
    ) -> treams_core::Result<basis::TranslationGradient> {
        match self {
            Self::Spherical(r) => r.pullback(g),
            Self::Cylindrical(r) => r.pullback(g),
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

fn make_cyl_basis(
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ExpansionContext>()?;
    m.add_function(wrap_pyfunction!(plane_to_cylindrical, m)?)?;
    m.add_function(wrap_pyfunction!(plane_to_spherical, m)?)?;
    m.add_function(wrap_pyfunction!(plane_polarization, m)?)?;
    m.add_function(wrap_pyfunction!(expansion, m)?)?;
    m.add_function(wrap_pyfunction!(cyl_expansion, m)?)?;
    m.add_function(wrap_pyfunction!(cyl_translation, m)?)?;
    Ok(())
}
