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

#[pyclass]
#[derive(Debug)]
struct ExpansionContext {
    residual: Option<TranslationResidual>,
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
        if g.shape() != residual.value.shape() {
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
            residual: Some(residual),
        },
    ))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ExpansionContext>()?;
    m.add_function(wrap_pyfunction!(expansion, m)?)?;
    Ok(())
}
