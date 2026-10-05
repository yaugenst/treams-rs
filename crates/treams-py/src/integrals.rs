//! `incgamma_record`, `intkambe_record` and their 0-d variants: records of the
//! upper incomplete gamma function and the Kambe integral (`treams_core::special`).
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use treams_core::{
    Complex,
    fpenv::ieee,
    special::{IncgammaResidual, IntkambeResidual, incgamma_array, intkambe_array},
};

use crate::{
    broadcast::{Recorded, broadcast_context, check_flat_shape, context, record, record_held},
    context::error,
};

broadcast_context! {
    IncgammaContext(IncgammaResidual, 1);
    IntkambeContext(IntkambeResidual, 2);
}

/// Record the incomplete gamma function on broadcast arguments: `special::incgamma_array`.
#[pyfunction]
pub(crate) fn incgamma_record<'py>(
    py: Python<'py>,
    degrees: PyReadonlyArray1<'py, f64>,
    arguments: PyReadonlyArray1<'py, Complex>,
    shape: Vec<usize>,
    argument_shape: Vec<usize>,
) -> Recorded<'py, IncgammaContext> {
    ieee(|| {
        let (degrees, arguments) = (degrees.as_array().to_vec(), arguments.as_array().to_vec());
        record(py, shape, [argument_shape], move || {
            incgamma_array(degrees, arguments)
        })
    })
}

/// Reconstruct the incomplete-gamma derivative context from its broadcast inputs.
#[pyfunction]
pub(crate) fn incgamma_context(
    degrees: PyReadonlyArray1<'_, f64>,
    arguments: PyReadonlyArray1<'_, Complex>,
    shape: Vec<usize>,
    argument_shape: Vec<usize>,
) -> PyResult<IncgammaContext> {
    ieee(|| {
        check_flat_shape(
            &shape,
            &[degrees.as_array().len(), arguments.as_array().len()],
        )?;
        let residual =
            IncgammaResidual::new(degrees.as_array().to_vec(), arguments.as_array().to_vec())
                .map_err(error)?;
        context(shape, [argument_shape], residual)
    })
}

/// Record the incomplete gamma function at one point: `special::incgamma_array`.
#[pyfunction]
pub(crate) fn incgamma_record_scalar(
    py: Python<'_>,
    n: f64,
    z: Complex,
) -> Recorded<'_, IncgammaContext> {
    ieee(|| record_held(py, || incgamma_array(vec![n], vec![z])))
}

/// Record the Kambe integral on broadcast arguments: `special::intkambe_array`.
#[pyfunction]
pub(crate) fn intkambe_record<'py>(
    py: Python<'py>,
    orders: PyReadonlyArray1<'py, i32>,
    z: PyReadonlyArray1<'py, Complex>,
    eta: PyReadonlyArray1<'py, Complex>,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 2],
) -> Recorded<'py, IntkambeContext> {
    ieee(|| {
        let orders = orders.as_array().to_vec();
        let arguments = [z.as_array().to_vec(), eta.as_array().to_vec()];
        record(py, shape, argument_shapes, move || {
            intkambe_array(orders, arguments)
        })
    })
}

/// Reconstruct the Kambe derivative context from its broadcast inputs.
#[pyfunction]
pub(crate) fn intkambe_context(
    orders: PyReadonlyArray1<'_, i32>,
    z: PyReadonlyArray1<'_, Complex>,
    eta: PyReadonlyArray1<'_, Complex>,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 2],
) -> PyResult<IntkambeContext> {
    ieee(|| {
        check_flat_shape(
            &shape,
            &[
                orders.as_array().len(),
                z.as_array().len(),
                eta.as_array().len(),
            ],
        )?;
        let residual = IntkambeResidual::new(
            orders.as_array().to_vec(),
            [z.as_array().to_vec(), eta.as_array().to_vec()],
        )
        .map_err(error)?;
        context(shape, argument_shapes, residual)
    })
}

/// Record the Kambe integral at one point: `special::intkambe_array`.
#[pyfunction]
pub(crate) fn intkambe_record_scalar(
    py: Python<'_>,
    n: i32,
    z: Complex,
    eta: Complex,
) -> Recorded<'_, IntkambeContext> {
    ieee(|| {
        let forward = move || intkambe_array(vec![n], [vec![z], vec![eta]]);
        // Odd orders sum incomplete-gamma series (about 30 us), even orders do not.
        if n % 2 == 0 {
            record_held(py, forward)
        } else {
            record(py, Vec::new(), Default::default(), forward)
        }
    })
}
