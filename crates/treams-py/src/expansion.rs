//! `expansion`, `cylindrical_expansion`, `cw_to_sw` and `periodic_to_cw` with their
//! contexts: expansion matrices between spherical and cylindrical bases
//! (`treams_core::sw`, `treams_core::cw`).
#![allow(clippy::indexing_slicing)] // Tangent shapes are validated before fixed-axis access.
use nalgebra::DMatrix;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, ndarray::Ix2};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{Complex, basis, fpenv::ieee, saved::SavedState, sw};

use crate::{
    args::{make_basis, make_cyl_basis},
    context::{context, detached, error, radial, restore_state, state_array},
    convert::{
        C1, C2, Cotangent, R1, R2, RealTangent, Tangent, finite_tangent, matrix_cotangent,
        owned_matrix, rows, rows_array, vector_tangent,
    },
};

#[derive(Debug)]
enum Expansion {
    Spherical(sw::ExpansionResidual),
    Cylindrical(treams_core::cw::ExpansionResidual),
    ToSw(treams_core::cw::ToSwResidual),
}
impl Expansion {
    fn state_size(
        dm: usize,
        dp: usize,
        sm: usize,
        sp: usize,
        family: u8,
    ) -> treams_core::Result<usize> {
        let bytes = match family {
            0 => sw::ExpansionResidual::state_size(dm, dp, sm, sp),
            1 => treams_core::cw::ExpansionResidual::state_size(dm, dp, sm, sp),
            2 => treams_core::cw::ToSwResidual::state_size(dm, dp, sm, sp),
            _ => Err(treams_core::Error::InvalidInput(
                "invalid expansion family".into(),
            )),
        }?;
        bytes
            .checked_add(1)
            .ok_or_else(|| treams_core::Error::InvalidInput("saved expansion size overflow".into()))
    }
    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Spherical(r) => r.shape(),
            Self::Cylindrical(r) => r.shape(),
            Self::ToSw(r) => r.shape(),
        }
    }
    fn pullback(&self, g: &DMatrix<Complex>) -> treams_core::Result<basis::ExpansionGradient> {
        match self {
            Self::Spherical(r) => r.pullback(g),
            Self::Cylindrical(r) => r.pullback(g),
            Self::ToSw(r) => r.pullback(g),
        }
    }
    fn pushforward(
        &self,
        destination: &[[f64; 3]],
        source: &[[f64; 3]],
        ks: [Complex; 2],
    ) -> treams_core::Result<DMatrix<Complex>> {
        match self {
            Self::Spherical(r) => r.pushforward(destination, source, ks),
            Self::Cylindrical(r) => r.pushforward(destination, source, ks),
            Self::ToSw(r) => r.pushforward(destination, source, ks),
        }
    }
}

impl SavedState for Expansion {
    fn save_state(&self) -> treams_core::Result<Vec<u8>> {
        let (tag, data) = match self {
            Self::Spherical(r) => (0, r.save_state()?),
            Self::Cylindrical(r) => (1, r.save_state()?),
            Self::ToSw(r) => (2, r.save_state()?),
        };
        let mut writer = treams_core::saved::Writer::new(data.len() + 1);
        writer.byte(tag);
        writer.raw(&data);
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> treams_core::Result<Self> {
        let Some((&tag, data)) = bytes.split_first() else {
            return Err(treams_core::Error::InvalidInput(
                "empty expansion state".into(),
            ));
        };
        match tag {
            0 => sw::ExpansionResidual::from_state(data).map(Self::Spherical),
            1 => treams_core::cw::ExpansionResidual::from_state(data).map(Self::Cylindrical),
            2 => treams_core::cw::ToSwResidual::from_state(data).map(Self::ToSw),
            _ => Err(treams_core::Error::InvalidInput(
                "invalid expansion family".into(),
            )),
        }
    }
}

/// Preserve the two-dimensional position contract before the core checks its count.
fn position_tangent(tangent: &RealTangent<'_>) -> PyResult<Vec<[f64; 3]>> {
    let count = tangent.as_array().shape().first().copied().unwrap_or(0);
    rows(
        finite_tangent::<_, Ix2>(tangent, &[count, 3])?,
        "position tangent",
    )
}

fn wavenumber_tangent(tangent: &Tangent<'_>) -> PyResult<[Complex; 2]> {
    let values = vector_tangent(tangent, 2)?;
    Ok([values[0], values[1]])
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
    #[staticmethod]
    fn _state_spec(
        destination_modes: usize,
        destination_positions: usize,
        source_modes: usize,
        source_positions: usize,
        family: u8,
    ) -> PyResult<usize> {
        ieee(|| {
            Expansion::state_size(
                destination_modes,
                destination_positions,
                source_modes,
                source_positions,
                family,
            )
            .map_err(error)
        })
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| Ok(Self::new(restore_state(&state)?)))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        destination_positions: RealTangent<'py>,
        source_positions: RealTangent<'py>,
        ks: Tangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let destination = position_tangent(&destination_positions)?;
            let source = position_tangent(&source_positions)?;
            let ks = wavenumber_tangent(&ks)?;
            let residual = &self.residual;
            let result = detached(py, || residual.pushforward(&destination, &source, ks))?;
            owned_matrix(py, result)
        })
    }

    fn pushforward_axial<'py>(
        &self,
        py: Python<'py>,
        destination_positions: RealTangent<'py>,
        source_positions: RealTangent<'py>,
        ks: Tangent<'py>,
        kz: RealTangent<'py>,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let destination = position_tangent(&destination_positions)?;
            let source = position_tangent(&source_positions)?;
            let ks = wavenumber_tangent(&ks)?;
            let kz = vector_tangent(&kz, kz.as_array().len())?;
            let Expansion::Cylindrical(residual) = &self.residual else {
                return Err(PyValueError::new_err(
                    "axial expansion derivatives require two cylindrical bases",
                ));
            };
            let result = detached(py, || {
                residual.pushforward_axial(&destination, &source, ks, &kz)
            })?;
            owned_matrix(py, result)
        })
    }

    fn pullback<'py>(&self, py: Python<'py>, cotangent: Cotangent<'py>) -> PyResult<Gradient<'py>> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
            expansion_gradient(py, detached(py, move || residual.pullback(&g))?)
        })
    }

    /// Gradients of positions, medium ks and sorted distinct shared axial wavenumbers.
    fn pullback_axial<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, R2<'py>, C1<'py>, R1<'py>)> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
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
    #[staticmethod]
    fn _state_spec(
        destination_modes: usize,
        destination_positions: usize,
        source_modes: usize,
        source_positions: usize,
    ) -> PyResult<usize> {
        ieee(|| {
            sw::PeriodicToCwResidual::state_size(
                destination_modes,
                destination_positions,
                source_modes,
                source_positions,
            )
            .map_err(error)
        })
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u8>>> {
        ieee(|| state_array(py, &self.residual))
    }

    #[staticmethod]
    fn _from_state(state: PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| Ok(Self::new(restore_state(&state)?)))
    }

    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        destination_positions: RealTangent<'py>,
        source_positions: RealTangent<'py>,
        ks: Tangent<'py>,
        kz: RealTangent<'py>,
        period: f64,
    ) -> PyResult<C2<'py>> {
        ieee(|| {
            let destination = position_tangent(&destination_positions)?;
            let source = position_tangent(&source_positions)?;
            let ks = wavenumber_tangent(&ks)?;
            let residual = &self.residual;
            let kz = vector_tangent(&kz, residual.shape().0)?;
            let result = detached(py, || {
                residual.pushforward(&destination, &source, ks, &kz, period)
            })?;
            owned_matrix(py, result)
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(R2<'py>, R2<'py>, C1<'py>, R1<'py>, f64)> {
        ieee(|| {
            let residual = &self.residual;
            let g = matrix_cotangent(&cotangent, residual.shape())?;
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
