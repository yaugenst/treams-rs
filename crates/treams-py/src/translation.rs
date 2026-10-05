//! `spherical_translation_record` and `cylindrical_translation_record`: records of
//! the translation coefficient of one mode pair
//! (`treams_core::sw::polar_translation_array`, `cw::polar_translation_array`).
use crate::{
    broadcast::{Recorded, broadcast_context, context, record},
    context::radial,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use treams_core::{
    Complex, cw,
    fpenv::ieee,
    sw::{self, Mode},
};

broadcast_context! {
    SphericalTranslationContext(sw::PolarTranslationResidual, 3);
    CylindricalTranslationContext(cw::PolarTranslationResidual, 4);
}

/// Record translation coefficients of spherical mode pairs: `sw::polar_translation_array`.
#[pyfunction]
pub(crate) fn spherical_translation_record<'py>(
    py: Python<'py>,
    modes: Vec<[(i32, i32, u8); 2]>,
    arguments: [PyReadonlyArray1<'py, Complex>; 3],
    helicity: bool,
    singular: bool,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 3],
) -> Recorded<'py, SphericalTranslationContext> {
    ieee(|| {
        let modes = modes
            .into_iter()
            .map(|pair| pair.map(|(l, m, pol)| Mode { l, m, pol }))
            .collect();
        let arguments = arguments.map(|a| a.as_array().to_vec());
        record(py, shape, argument_shapes, move || {
            sw::polar_translation_array(modes, arguments, helicity, radial(singular))
        })
    })
}

/// Record translation coefficients of cylindrical mode pairs: `cw::polar_translation_array`.
#[pyfunction]
pub(crate) fn cylindrical_translation_record<'py>(
    py: Python<'py>,
    orders: Vec<i32>,
    arguments: [PyReadonlyArray1<'py, Complex>; 4],
    singular: bool,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 4],
) -> Recorded<'py, CylindricalTranslationContext> {
    ieee(|| {
        let arguments = arguments.map(|a| a.as_array().to_vec());
        record(py, shape, argument_shapes, move || {
            cw::polar_translation_array(orders, arguments, radial(singular))
        })
    })
}

/// Reconstruct spherical-translation inputs and static coupling plans.
#[pyfunction]
pub(crate) fn spherical_translation_context(
    modes: Vec<[(i32, i32, u8); 2]>,
    arguments: [PyReadonlyArray1<'_, Complex>; 3],
    helicity: bool,
    singular: bool,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 3],
) -> PyResult<SphericalTranslationContext> {
    ieee(|| {
        let modes = modes
            .into_iter()
            .map(|pair| pair.map(|(l, m, pol)| Mode { l, m, pol }))
            .collect();
        let arguments = arguments.map(|a| a.as_array().to_vec());
        let residual =
            sw::PolarTranslationResidual::new(modes, arguments, helicity, radial(singular))
                .map_err(crate::context::error)?;
        context(shape, argument_shapes, residual)
    })
}

/// Reconstruct cylindrical-translation inputs without evaluating coefficients.
#[pyfunction]
pub(crate) fn cylindrical_translation_context(
    orders: Vec<i32>,
    arguments: [PyReadonlyArray1<'_, Complex>; 4],
    singular: bool,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 4],
) -> PyResult<CylindricalTranslationContext> {
    ieee(|| {
        let arguments = arguments.map(|a| a.as_array().to_vec());
        let residual = cw::PolarTranslationResidual::new(orders, arguments, radial(singular))
            .map_err(crate::context::error)?;
        context(shape, argument_shapes, residual)
    })
}
