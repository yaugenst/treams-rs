//! `bessel_record`, `angular_record`, `wignerd_record` and their 0-d variants:
//! records of Bessel, angular and Wigner D functions (`treams_core::special`).

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use treams_core::{
    Complex,
    fpenv::ieee,
    special::{self, AngularResidual, BesselResidual, WignerDResidual},
};

use crate::{
    args::{angular_function, bessel_function},
    broadcast::{Recorded, broadcast_context, context, record, record_held},
    context::error,
};

broadcast_context! {
    BesselContext(BesselResidual, 1);
    AngularContext(AngularResidual, 1);
    WignerdContext(WignerDResidual, 3);
}

/// Record Bessel or Hankel functions or their derivatives: `special::bessel_array`.
#[pyfunction]
pub(crate) fn bessel_record<'py>(
    py: Python<'py>,
    orders: PyReadonlyArray1<'py, f64>,
    arguments: PyReadonlyArray1<'py, Complex>,
    function: &str,
    spherical: bool,
    derivative: u8,
    shape: Vec<usize>,
    argument_shape: Vec<usize>,
) -> Recorded<'py, BesselContext> {
    ieee(|| {
        let function = bessel_function(function)?;
        let (orders, arguments) = (orders.as_array().to_vec(), arguments.as_array().to_vec());
        record(py, shape, [argument_shape], move || {
            special::bessel_array(orders, arguments, function, spherical, derivative)
        })
    })
}

/// Reconstruct the Bessel derivative context from its broadcast inputs.
#[pyfunction]
pub(crate) fn bessel_context(
    orders: PyReadonlyArray1<'_, f64>,
    arguments: PyReadonlyArray1<'_, Complex>,
    function: &str,
    spherical: bool,
    derivative: u8,
    shape: Vec<usize>,
    argument_shape: Vec<usize>,
) -> PyResult<BesselContext> {
    ieee(|| {
        let residual = BesselResidual::new(
            orders.as_array().to_vec(),
            arguments.as_array().to_vec(),
            bessel_function(function)?,
            spherical,
            derivative,
        )
        .map_err(error)?;
        context(shape, [argument_shape], residual)
    })
}

/// Record a Bessel or Hankel function or its derivative at one point: `special::bessel_array`.
#[pyfunction]
pub(crate) fn bessel_record_scalar<'py>(
    py: Python<'py>,
    order: f64,
    z: Complex,
    function: &str,
    spherical: bool,
    derivative: u8,
) -> Recorded<'py, BesselContext> {
    ieee(|| {
        let function = bessel_function(function)?;
        record_held(py, || {
            special::bessel_array(vec![order], vec![z], function, spherical, derivative)
        })
    })
}

/// Record a Legendre, pi or tau function at one point: `special::angular_array`.
#[pyfunction]
pub(crate) fn angular_record_scalar<'py>(
    py: Python<'py>,
    degree: f64,
    order: f64,
    z: Complex,
    function: &str,
) -> Recorded<'py, AngularContext> {
    ieee(|| {
        let function = angular_function(function)?;
        record_held(py, || {
            special::angular_array(vec![degree], vec![order], vec![z], function)
        })
    })
}

/// Record Legendre, pi or tau functions on broadcast arguments: `special::angular_array`.
#[pyfunction]
pub(crate) fn angular_record<'py>(
    py: Python<'py>,
    degrees: PyReadonlyArray1<'py, f64>,
    orders: PyReadonlyArray1<'py, f64>,
    arguments: PyReadonlyArray1<'py, Complex>,
    function: &str,
    shape: Vec<usize>,
    argument_shape: Vec<usize>,
) -> Recorded<'py, AngularContext> {
    ieee(|| {
        let (degrees, orders, arguments) = (
            degrees.as_array().to_vec(),
            orders.as_array().to_vec(),
            arguments.as_array().to_vec(),
        );
        let function = angular_function(function)?;
        record(py, shape, [argument_shape], move || {
            special::angular_array(degrees, orders, arguments, function)
        })
    })
}

/// Reconstruct the angular derivative context from its broadcast inputs.
#[pyfunction]
pub(crate) fn angular_context(
    degrees: PyReadonlyArray1<'_, f64>,
    orders: PyReadonlyArray1<'_, f64>,
    arguments: PyReadonlyArray1<'_, Complex>,
    function: &str,
    shape: Vec<usize>,
    argument_shape: Vec<usize>,
) -> PyResult<AngularContext> {
    ieee(|| {
        let residual = AngularResidual::new(
            degrees.as_array().to_vec(),
            orders.as_array().to_vec(),
            arguments.as_array().to_vec(),
            angular_function(function)?,
        )
        .map_err(error)?;
        context(shape, [argument_shape], residual)
    })
}

/// Record Wigner D functions on broadcast angles: `special::wigner_d_array`.
#[pyfunction]
pub(crate) fn wignerd_record<'py>(
    py: Python<'py>,
    labels: Vec<[i32; 3]>,
    phi: PyReadonlyArray1<'py, Complex>,
    theta: PyReadonlyArray1<'py, Complex>,
    psi: PyReadonlyArray1<'py, Complex>,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 3],
) -> Recorded<'py, WignerdContext> {
    ieee(|| {
        let angles = [phi, theta, psi].map(|angle| angle.as_array().to_vec());
        record(py, shape, argument_shapes, move || {
            special::wigner_d_array(labels, angles)
        })
    })
}

/// Reconstruct the Wigner derivative context from its broadcast inputs.
#[pyfunction]
pub(crate) fn wignerd_context(
    labels: Vec<[i32; 3]>,
    phi: PyReadonlyArray1<'_, Complex>,
    theta: PyReadonlyArray1<'_, Complex>,
    psi: PyReadonlyArray1<'_, Complex>,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 3],
) -> PyResult<WignerdContext> {
    ieee(|| {
        let angles = [phi, theta, psi].map(|angle| angle.as_array().to_vec());
        let residual = WignerDResidual::new(labels, angles).map_err(error)?;
        context(shape, argument_shapes, residual)
    })
}

/// Record a Wigner D function at one set of angles: `special::wigner_d_array`.
#[pyfunction]
pub(crate) fn wignerd_record_scalar(
    py: Python<'_>,
    labels: [i32; 3],
    angles: [Complex; 3],
) -> Recorded<'_, WignerdContext> {
    ieee(|| {
        record_held(py, || {
            special::wigner_d_array(vec![labels], angles.map(|z| vec![z]))
        })
    })
}
