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

pub(crate) fn reduce_broadcast<T: numpy::ndarray::LinalgScalar>(
    gradient: Vec<T>,
    shape: &[usize],
    argument_shape: &[usize],
) -> PyResult<ArrayD<T>> {
    let shape_error = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
    if gradient.len() == argument_shape.iter().product::<usize>() {
        return ArrayD::from_shape_vec(IxDyn(argument_shape), gradient).map_err(shape_error);
    }
    let mut result = ArrayD::from_shape_vec(IxDyn(shape), gradient).map_err(shape_error)?;
    let leading = shape.len() - argument_shape.len();
    for axis in (0..shape.len()).rev() {
        if axis < leading || argument_shape.get(axis - leading) == Some(&1) {
            result = result.sum_axis(Axis(axis));
        }
    }
    result
        .into_shape_with_order(IxDyn(argument_shape))
        .map_err(shape_error)
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
                let result = reduce_broadcast(gradient, &self.shape, &self.argument_shape)?;
                Ok(result.into_pyarray(py))
            }
        }
    };
}
argument_context!(BesselContext, BesselResidual);
argument_context!(AngularContext, AngularResidual);
argument_context!(GammaContext, treams_core::integrals::GammaResidual);

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

#[pyclass]
#[derive(Debug)]
struct WignerContext {
    residual: Option<treams_core::rotation::WignerResidual>,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 3],
}

type EulerGradient<'py> = (
    Bound<'py, PyArrayDyn<Complex>>,
    Bound<'py, PyArrayDyn<Complex>>,
    Bound<'py, PyArrayDyn<Complex>>,
);

#[pymethods]
impl WignerContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArrayDyn<'py, Complex>,
    ) -> PyResult<EulerGradient<'py>> {
        let g = cotangent.as_array();
        if g.shape() != self.shape || g.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(PyValueError::new_err(
                "Wigner cotangent must be finite and match output shape",
            ));
        }
        let g: Vec<_> = g.iter().copied().collect();
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let gradients = py.detach(move || residual.pullback(&g)).map_err(error)?;
        let [a, b, c] = gradients;
        let [sa, sb, sc] = &self.argument_shapes;
        Ok((
            reduce_broadcast(a, &self.shape, sa)?.into_pyarray(py),
            reduce_broadcast(b, &self.shape, sb)?.into_pyarray(py),
            reduce_broadcast(c, &self.shape, sc)?.into_pyarray(py),
        ))
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)] // Three Euler arguments and their broadcast metadata.
fn wigner<'py>(
    py: Python<'py>,
    labels: Vec<[i32; 3]>,
    phi: PyReadonlyArray1<'py, Complex>,
    theta: PyReadonlyArray1<'py, Complex>,
    psi: PyReadonlyArray1<'py, Complex>,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 3],
) -> PyResult<(Bound<'py, PyArrayDyn<Complex>>, WignerContext)> {
    if argument_shapes.iter().any(|a| {
        a.len() > shape.len()
            || a.iter()
                .rev()
                .zip(shape.iter().rev())
                .any(|(&a, &b)| a != 1 && a != b)
    }) {
        return Err(PyValueError::new_err(
            "argument shapes must broadcast to output",
        ));
    }
    let angles = [
        phi.as_array().to_vec(),
        theta.as_array().to_vec(),
        psi.as_array().to_vec(),
    ];
    let (value, residual) = py
        .detach(move || treams_core::rotation::wigner_array(labels, angles))
        .map_err(error)?;
    let value = ArrayD::from_shape_vec(IxDyn(&shape), value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    Ok((
        value,
        WignerContext {
            residual: Some(residual),
            shape,
            argument_shapes,
        },
    ))
}

#[pyfunction]
fn wigner_scalar(
    py: Python<'_>,
    labels: [i32; 3],
    angles: [Complex; 3],
) -> PyResult<(Bound<'_, PyArrayDyn<Complex>>, WignerContext)> {
    let (value, residual) =
        treams_core::rotation::wigner_array(vec![labels], angles.map(|z| vec![z]))
            .map_err(error)?;
    let value = ArrayD::from_shape_vec(IxDyn(&[]), value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    Ok((
        value,
        WignerContext {
            residual: Some(residual),
            shape: Vec::new(),
            argument_shapes: std::array::from_fn(|_| Vec::new()),
        },
    ))
}

#[pyfunction]
fn wigner3j_scalar(j1: i32, j2: i32, j3: i32, m1: i32, m2: i32, m3: i32) -> PyResult<f64> {
    if [j1, j2, j3, m1, m2, m3]
        .iter()
        .any(|n| n.unsigned_abs() > 260)
    {
        return Err(PyValueError::new_err(
            "Wigner labels must be integers in [-260, 260]",
        ));
    }
    Ok(treams_core::angular::wigner3j(j1, j2, j3, m1, m2, m3))
}

fn gamma_result(
    py: Python<'_>,
    degrees: Vec<f64>,
    arguments: Vec<Complex>,
    shape: Vec<usize>,
    argument_shape: Vec<usize>,
) -> PyResult<(Bound<'_, PyArrayDyn<Complex>>, GammaContext)> {
    validate_argument_shapes(&shape, &[&argument_shape])?;
    let (value, residual) = py
        .detach(move || treams_core::integrals::GammaResidual::new(degrees, arguments))
        .map_err(error)?;
    let value = ArrayD::from_shape_vec(IxDyn(&shape), value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    Ok((
        value,
        GammaContext {
            residual: Some(residual),
            shape,
            argument_shape,
        },
    ))
}
#[pyfunction]
fn gamma_record<'py>(
    py: Python<'py>,
    degrees: PyReadonlyArray1<'py, f64>,
    arguments: PyReadonlyArray1<'py, Complex>,
    shape: Vec<usize>,
    argument_shape: Vec<usize>,
) -> PyResult<(Bound<'py, PyArrayDyn<Complex>>, GammaContext)> {
    gamma_result(
        py,
        degrees.as_array().to_vec(),
        arguments.as_array().to_vec(),
        shape,
        argument_shape,
    )
}
#[pyfunction]
fn gamma_record_scalar(
    py: Python<'_>,
    n: f64,
    z: Complex,
) -> PyResult<(Bound<'_, PyArrayDyn<Complex>>, GammaContext)> {
    gamma_result(py, vec![n], vec![z], Vec::new(), Vec::new())
}
fn validate_argument_shapes(shape: &[usize], arguments: &[&[usize]]) -> PyResult<()> {
    if arguments.iter().any(|a| {
        a.len() > shape.len()
            || a.iter()
                .rev()
                .zip(shape.iter().rev())
                .any(|(&a, &b)| a != 1 && a != b)
    }) {
        return Err(PyValueError::new_err(
            "argument shapes must broadcast to output",
        ));
    }
    Ok(())
}
#[pyclass]
#[derive(Debug)]
struct KambeContext {
    residual: Option<treams_core::integrals::KambeResidual>,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 2],
}
type IntegralGradient<'py> = (
    Bound<'py, PyArrayDyn<Complex>>,
    Bound<'py, PyArrayDyn<Complex>>,
);
#[pymethods]
impl KambeContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArrayDyn<'py, Complex>,
    ) -> PyResult<IntegralGradient<'py>> {
        let g = cotangent.as_array();
        if g.shape() != self.shape || g.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(PyValueError::new_err(
                "integral cotangent must be finite and match output shape",
            ));
        }
        let g = g.iter().copied().collect::<Vec<_>>();
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let [a, b] = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            reduce_broadcast(a, &self.shape, &self.argument_shapes[0])?.into_pyarray(py),
            reduce_broadcast(b, &self.shape, &self.argument_shapes[1])?.into_pyarray(py),
        ))
    }
}
fn kambe_result(
    py: Python<'_>,
    orders: Vec<i32>,
    arguments: [Vec<Complex>; 2],
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 2],
) -> PyResult<(Bound<'_, PyArrayDyn<Complex>>, KambeContext)> {
    validate_argument_shapes(&shape, &argument_shapes.each_ref().map(Vec::as_slice))?;
    let (value, residual) = py
        .detach(move || treams_core::integrals::KambeResidual::new(orders, arguments))
        .map_err(error)?;
    let value = ArrayD::from_shape_vec(IxDyn(&shape), value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    Ok((
        value,
        KambeContext {
            residual: Some(residual),
            shape,
            argument_shapes,
        },
    ))
}
#[pyfunction]
fn kambe_record<'py>(
    py: Python<'py>,
    orders: PyReadonlyArray1<'py, i32>,
    z: PyReadonlyArray1<'py, Complex>,
    eta: PyReadonlyArray1<'py, Complex>,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 2],
) -> PyResult<(Bound<'py, PyArrayDyn<Complex>>, KambeContext)> {
    kambe_result(
        py,
        orders.as_array().to_vec(),
        [z.as_array().to_vec(), eta.as_array().to_vec()],
        shape,
        argument_shapes,
    )
}
#[pyfunction]
fn kambe_record_scalar(
    py: Python<'_>,
    n: i32,
    z: Complex,
    eta: Complex,
) -> PyResult<(Bound<'_, PyArrayDyn<Complex>>, KambeContext)> {
    kambe_result(
        py,
        vec![n],
        [vec![z], vec![eta]],
        Vec::new(),
        std::array::from_fn(|_| Vec::new()),
    )
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<GammaContext>()?;
    module.add_class::<KambeContext>()?;
    module.add_function(wrap_pyfunction!(gamma_record, module)?)?;
    module.add_function(wrap_pyfunction!(gamma_record_scalar, module)?)?;
    module.add_function(wrap_pyfunction!(kambe_record, module)?)?;
    module.add_function(wrap_pyfunction!(kambe_record_scalar, module)?)?;
    module.add_function(wrap_pyfunction!(wigner3j_scalar, module)?)?;
    module.add_class::<WignerContext>()?;
    module.add_function(wrap_pyfunction!(wigner, module)?)?;
    module.add_function(wrap_pyfunction!(wigner_scalar, module)?)?;
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
