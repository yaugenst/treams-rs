//! `expansion`, `cylindrical_expansion`, `cw_to_sw` and `periodic_to_cw` with their
//! contexts: expansion matrices between spherical and cylindrical bases
//! (`treams_core::sw`, `treams_core::cw`).
use nalgebra::DMatrix;
use numpy::IntoPyArray;
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{Complex, basis, fpenv::ieee, sw};

use crate::{
    args::{make_basis, make_cyl_basis},
    context::{context, detached, radial},
    convert::{C1, C2, Cotangent, R1, R2, owned_matrix, rows_array},
};

#[derive(Debug)]
enum Expansion {
    Spherical(sw::ExpansionResidual),
    Cylindrical(treams_core::cw::ExpansionResidual),
    ToSw(treams_core::cw::ToSwResidual),
}
impl Expansion {
    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Spherical(r) => r.shape(),
            Self::Cylindrical(r) => r.shape(),
            Self::ToSw(r) => r.shape(),
        }
    }
    fn pullback(self, g: &DMatrix<Complex>) -> treams_core::Result<basis::ExpansionGradient> {
        match self {
            Self::Spherical(r) => r.pullback(g),
            Self::Cylindrical(r) => r.pullback(g),
            Self::ToSw(r) => r.pullback(g),
        }
    }
}

context!(ExpansionContext(Expansion));

type Gradient<'py> = (R2<'py>, R2<'py>, C1<'py>);
fn expansion_gradient(py: Python<'_>, result: basis::ExpansionGradient) -> PyResult<Gradient<'_>> {
    Ok((
        rows_array(py, result.destination)?,
        rows_array(py, result.source)?,
        result.ks.to_vec().into_pyarray(py),
    ))
}
#[pymethods]
impl ExpansionContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<Gradient<'py>> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, Expansion::shape)?;
            expansion_gradient(py, detached(py, move || residual.pullback(&g))?)
        })
    }

    /// Gradients of positions, medium ks and sorted distinct shared axial wavenumbers.
    fn pullback_axial<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, R2<'py>, C1<'py>, R1<'py>)> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, Expansion::shape)?;
            let Expansion::Cylindrical(residual) = residual else {
                return Err(PyValueError::new_err(
                    "axial expansion derivatives require two cylindrical bases",
                ));
            };
            let (result, axial) = detached(py, move || residual.pullback_axial(&g))?;
            let (destination, source, ks) = expansion_gradient(py, result)?;
            Ok((destination, source, ks, axial.into_pyarray(py)))
        })
    }
}

/// An expansion matrix, moved into an F-ordered array, and its context.
fn finish_expansion(
    py: Python<'_>,
    value: DMatrix<Complex>,
    residual: Expansion,
) -> PyResult<(C2<'_>, ExpansionContext)> {
    Ok((owned_matrix(py, value)?, ExpansionContext::new(residual)))
}

/// Record the expansion matrix between two spherical bases: `sw::expansion`.
#[pyfunction]
pub(crate) fn expansion(
    py: Python<'_>,
    destination: Vec<(usize, i32, i32, u8)>,
    source: Vec<(usize, i32, i32, u8)>,
    destination_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
    singular: bool,
) -> PyResult<(C2<'_>, ExpansionContext)> {
    ieee(|| {
        let destination = make_basis(destination, destination_positions);
        let source = make_basis(source, source_positions);
        let (value, residual) = detached(py, move || {
            sw::expansion(destination, source, ks, helicity, radial(singular))
        })?;
        finish_expansion(py, value, Expansion::Spherical(residual))
    })
}

/// Record the expansion matrix between two cylindrical bases: `cw::expansion`.
#[pyfunction]
pub(crate) fn cylindrical_expansion(
    py: Python<'_>,
    destination: Vec<(usize, f64, i32, u8)>,
    source: Vec<(usize, f64, i32, u8)>,
    destination_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    singular: bool,
) -> PyResult<(C2<'_>, ExpansionContext)> {
    ieee(|| {
        let destination = make_cyl_basis(destination, destination_positions);
        let source = make_cyl_basis(source, source_positions);
        let (value, residual) = detached(py, move || {
            treams_core::cw::expansion(destination, source, ks, radial(singular))
        })?;
        finish_expansion(py, value, Expansion::Cylindrical(residual))
    })
}

/// Record the expansion of a cylindrical basis in a spherical one: `cw::to_sw_matrix`.
#[pyfunction]
pub(crate) fn cw_to_sw(
    py: Python<'_>,
    destination: Vec<(usize, i32, i32, u8)>,
    source: Vec<(usize, f64, i32, u8)>,
    destination_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
) -> PyResult<(C2<'_>, ExpansionContext)> {
    ieee(|| {
        let destination = make_basis(destination, destination_positions);
        let source = make_cyl_basis(source, source_positions);
        let (value, residual) = detached(py, move || {
            treams_core::cw::to_sw_matrix(destination, source, ks, helicity)
        })?;
        finish_expansion(py, value, Expansion::ToSw(residual))
    })
}

context!(PeriodicToCwContext(sw::PeriodicToCwResidual));
#[pymethods]
impl PeriodicToCwContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, R2<'py>, C1<'py>, R1<'py>, f64)> {
        ieee(|| {
            let (residual, g) = self
                .residual
                .take_with_matrix(&cotangent, sw::PeriodicToCwResidual::shape)?;
            let result = detached(py, move || residual.pullback(&g))?;
            let (destination, source, ks) = expansion_gradient(py, result.expansion)?;
            Ok((
                destination,
                source,
                ks,
                result.kz.into_pyarray(py),
                result.period,
            ))
        })
    }
}
/// Record the expansion of z-periodic spherical in cylindrical waves: `sw::periodic_to_cw_matrix`.
#[pyfunction]
pub(crate) fn periodic_to_cw(
    py: Python<'_>,
    destination: Vec<(usize, f64, i32, u8)>,
    source: Vec<(usize, i32, i32, u8)>,
    destination_positions: Vec<[f64; 3]>,
    source_positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    period: f64,
    helicity: bool,
) -> PyResult<(C2<'_>, PeriodicToCwContext)> {
    ieee(|| {
        let destination = make_cyl_basis(destination, destination_positions);
        let source = make_basis(source, source_positions);
        let (value, residual) = detached(py, move || {
            sw::periodic_to_cw_matrix(destination, source, ks, period, helicity)
        })?;
        Ok((owned_matrix(py, value)?, PeriodicToCwContext::new(residual)))
    })
}
