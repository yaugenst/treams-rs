//! Broadcast mathematical functions and their native complex-argument pullbacks.

use numpy::{
    IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn,
    ndarray::{ArrayD, Axis, IxDyn},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    special::{self, Bessel, BesselResidual},
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

#[pyclass]
#[derive(Debug)]
struct BesselContext {
    residual: Option<BesselResidual>,
    shape: Vec<usize>,
    argument_shape: Vec<usize>,
}

#[pymethods]
impl BesselContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArrayDyn<'py, Complex>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex>>> {
        let g = cotangent.as_array();
        if g.shape() != self.shape || g.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(PyValueError::new_err(
                "Bessel cotangent must be finite and match output shape",
            ));
        }
        let g: Vec<_> = g.iter().copied().collect();
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let gradient = py.detach(move || residual.pullback(&g)).map_err(error)?;
        let shape_error = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
        let result = if gradient.len() == 1 {
            ArrayD::from_shape_vec(IxDyn(&self.argument_shape), gradient).map_err(shape_error)?
        } else {
            let mut result =
                ArrayD::from_shape_vec(IxDyn(&self.shape), gradient).map_err(shape_error)?;
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
fn bessel_forward<'py>(
    py: Python<'py>,
    orders: PyReadonlyArray1<'py, f64>,
    arguments: PyReadonlyArray1<'py, Complex>,
    kind: &str,
    spherical: bool,
    derivative: u8,
    shape: Vec<usize>,
) -> PyResult<Bound<'py, PyArrayDyn<Complex>>> {
    let kind = bessel_kind(kind)?;
    let orders = orders.as_slice()?;
    let arguments = arguments.as_slice()?;
    let value = py
        .detach(|| special::bessel_values(orders, arguments, kind, spherical, derivative))
        .map_err(error)?;
    Ok(ArrayD::from_shape_vec(IxDyn(&shape), value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BesselContext>()?;
    module.add_function(wrap_pyfunction!(bessel_forward, module)?)?;
    module.add_function(wrap_pyfunction!(bessel, module)?)?;
    Ok(())
}
