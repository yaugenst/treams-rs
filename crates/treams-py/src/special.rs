//! Broadcast mathematical functions and their native complex-argument pullbacks.

use numpy::{
    IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn,
    ndarray::{ArrayD, Axis, IxDyn},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    special::{self, Angular, AngularResidual, Bessel, BesselResidual},
};

use crate::error;

fn bessel_kind(kind: &str) -> PyResult<Bessel> {
    Ok(match kind {
        "j" => Bessel::J,
        "y" => Bessel::Y,
        "h1" => Bessel::H1,
        "h2" => Bessel::H2,
        _ => return Err(PyValueError::new_err("invalid Bessel function")),
    })
}

macro_rules! argument_context {
    ($name:ident, $residual:ty) => {
        #[pyclass]
        #[derive(Debug)]
        struct $name {
            residual: Option<$residual>,
            shape: Vec<usize>,
            argument_shape: Vec<usize>,
        }

        #[pymethods]
        impl $name {
            fn pullback<'py>(
                &mut self,
                py: Python<'py>,
                cotangent: PyReadonlyArrayDyn<'py, Complex>,
            ) -> PyResult<Bound<'py, PyArrayDyn<Complex>>> {
                let g = cotangent.as_array();
                if g.shape() != self.shape
                    || g.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
                {
                    return Err(PyValueError::new_err(
                        "Function cotangent must be finite and match output shape",
                    ));
                }
                let g: Vec<_> = g.iter().copied().collect();
                let residual = self.residual.take().ok_or_else(|| {
                    PyValueError::new_err("pullback residual has already been consumed")
                })?;
                let gradient = py.detach(move || residual.pullback(&g)).map_err(error)?;
                let shape_error =
                    |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
                let result = if gradient.len() == 1 {
                    ArrayD::from_shape_vec(IxDyn(&self.argument_shape), gradient)
                        .map_err(shape_error)?
                } else {
                    let mut result = ArrayD::from_shape_vec(IxDyn(&self.shape), gradient)
                        .map_err(shape_error)?;
                    let leading = self.shape.len() - self.argument_shape.len();
                    for axis in (0..self.shape.len()).rev() {
                        if axis < leading || self.argument_shape.get(axis - leading) == Some(&1) {
                            result = result.sum_axis(Axis(axis));
                        }
                    }
                    result
                        .into_shape_with_order(IxDyn(&self.argument_shape))
                        .map_err(shape_error)?
                };
                Ok(result.into_pyarray(py))
            }
        }
    };
}
argument_context!(BesselContext, BesselResidual);
argument_context!(AngularContext, AngularResidual);

#[pyfunction]
#[allow(clippy::too_many_arguments)] // Function selection plus broadcast metadata are static.
fn bessel<'py>(
    py: Python<'py>,
    orders: PyReadonlyArray1<'py, f64>,
    arguments: PyReadonlyArray1<'py, Complex>,
    kind: &str,
    spherical: bool,
    derivative: u8,
    shape: Vec<usize>,
    argument_shape: Vec<usize>,
) -> PyResult<(Bound<'py, PyArrayDyn<Complex>>, BesselContext)> {
    let kind = bessel_kind(kind)?;
    if argument_shape.len() > shape.len()
        || argument_shape
            .iter()
            .rev()
            .zip(shape.iter().rev())
            .any(|(&a, &b)| a != 1 && a != b)
    {
        return Err(PyValueError::new_err(
            "argument shape must broadcast to output",
        ));
    }
    let orders = orders.as_array().to_vec();
    let arguments = arguments.as_array().to_vec();
    let (value, residual) = py
        .detach(move || special::bessel_array(orders, arguments, kind, spherical, derivative))
        .map_err(error)?;
    let value = ArrayD::from_shape_vec(IxDyn(&shape), value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    Ok((
        value,
        BesselContext {
            residual: Some(residual),
            shape,
            argument_shape,
        },
    ))
}

#[pyfunction]
fn bessel_scalar<'py>(
    py: Python<'py>,
    order: f64,
    z: Complex,
    kind: &str,
    spherical: bool,
    derivative: u8,
) -> PyResult<(Bound<'py, PyArrayDyn<Complex>>, BesselContext)> {
    let (value, residual) = special::bessel_array(
        vec![order],
        vec![z],
        bessel_kind(kind)?,
        spherical,
        derivative,
    )
    .map_err(error)?;
    let value = ArrayD::from_shape_vec(IxDyn(&[]), value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    Ok((
        value,
        BesselContext {
            residual: Some(residual),
            shape: Vec::new(),
            argument_shape: Vec::new(),
        },
    ))
}

#[pyfunction]
fn hankel_scalar(order: f64, z: Complex, first: bool) -> PyResult<Complex> {
    special::bessel(
        order,
        z,
        if first { Bessel::H1 } else { Bessel::H2 },
        false,
        0,
    )
    .map_err(error)
}

fn angular_kind(kind: &str) -> PyResult<Angular> {
    Ok(match kind {
        "legendre" => Angular::Legendre,
        "pi" => Angular::Pi,
        "tau" => Angular::Tau,
        _ => return Err(PyValueError::new_err("invalid angular function")),
    })
}

#[pyfunction]
fn angular_value(degree: f64, order: f64, z: Complex, kind: &str) -> PyResult<Complex> {
    special::angular_value(degree, order, z, angular_kind(kind)?).map_err(error)
}

#[pyfunction]
fn angular_scalar<'py>(
    py: Python<'py>,
    degree: f64,
    order: f64,
    z: Complex,
    kind: &str,
) -> PyResult<(Bound<'py, PyArrayDyn<Complex>>, AngularContext)> {
    let (value, residual) =
        special::angular_array(vec![degree], vec![order], vec![z], angular_kind(kind)?)
            .map_err(error)?;
    let value = ArrayD::from_shape_vec(IxDyn(&[]), value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    Ok((
        value,
        AngularContext {
            residual: Some(residual),
            shape: Vec::new(),
            argument_shape: Vec::new(),
        },
    ))
}

#[pyfunction]
fn angular<'py>(
    py: Python<'py>,
    degrees: PyReadonlyArray1<'py, f64>,
    orders: PyReadonlyArray1<'py, f64>,
    arguments: PyReadonlyArray1<'py, Complex>,
    kind: &str,
    shape: Vec<usize>,
    argument_shape: Vec<usize>,
) -> PyResult<(Bound<'py, PyArrayDyn<Complex>>, AngularContext)> {
    if argument_shape.len() > shape.len()
        || argument_shape
            .iter()
            .rev()
            .zip(shape.iter().rev())
            .any(|(&a, &b)| a != 1 && a != b)
    {
        return Err(PyValueError::new_err(
            "argument shape must broadcast to output",
        ));
    }
    let (degrees, orders, arguments) = (
        degrees.as_array().to_vec(),
        orders.as_array().to_vec(),
        arguments.as_array().to_vec(),
    );
    let kind = angular_kind(kind)?;
    let (value, residual) = py
        .detach(move || special::angular_array(degrees, orders, arguments, kind))
        .map_err(error)?;
    let value = ArrayD::from_shape_vec(IxDyn(&shape), value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    Ok((
        value,
        AngularContext {
            residual: Some(residual),
            shape,
            argument_shape,
        },
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BesselContext>()?;
    module.add_class::<AngularContext>()?;
    module.add_function(wrap_pyfunction!(angular, module)?)?;
    module.add_function(wrap_pyfunction!(angular_value, module)?)?;
    module.add_function(wrap_pyfunction!(angular_scalar, module)?)?;
    module.add_function(wrap_pyfunction!(bessel_scalar, module)?)?;
    module.add_function(wrap_pyfunction!(hankel_scalar, module)?)?;
    module.add_function(wrap_pyfunction!(bessel, module)?)?;
    Ok(())
}
