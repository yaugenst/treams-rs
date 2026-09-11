//! Recorded spherical-coordinate translation coefficients.
use crate::{error, special::reduce_broadcast};
use numpy::{
    IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn,
    ndarray::{ArrayD, IxDyn},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{Complex, polar, special::Radial, waves::Mode};

#[pyclass]
#[derive(Debug)]
struct PolarTranslationContext {
    residual: Option<polar::Residual>,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 3],
}
type Gradients<'py> = (
    Bound<'py, PyArrayDyn<Complex>>,
    Bound<'py, PyArrayDyn<Complex>>,
    Bound<'py, PyArrayDyn<Complex>>,
);
#[pymethods]
impl PolarTranslationContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArrayDyn<'py, Complex>,
    ) -> PyResult<Gradients<'py>> {
        let g = cotangent.as_array();
        if g.shape() != self.shape || g.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(PyValueError::new_err(
                "cotangent must be finite and match output shape",
            ));
        }
        let g: Vec<_> = g.iter().copied().collect();
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let [a, b, c] = py.detach(move || residual.pullback(&g)).map_err(error)?;
        let [sa, sb, sc] = &self.argument_shapes;
        Ok((
            reduce_broadcast(a, &self.shape, sa)?.into_pyarray(py),
            reduce_broadcast(b, &self.shape, sb)?.into_pyarray(py),
            reduce_broadcast(c, &self.shape, sc)?.into_pyarray(py),
        ))
    }
}
#[pyfunction]
#[allow(clippy::too_many_arguments)] // Mode pairs, three arguments and their shape metadata.
fn spherical_translation<'py>(
    py: Python<'py>,
    modes: Vec<[(i32, i32, u8); 2]>,
    arguments: [PyReadonlyArray1<'py, Complex>; 3],
    helicity: bool,
    singular: bool,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 3],
) -> PyResult<(Bound<'py, PyArrayDyn<Complex>>, PolarTranslationContext)> {
    if argument_shapes.iter().any(|s| {
        s.len() > shape.len()
            || s.iter()
                .rev()
                .zip(shape.iter().rev())
                .any(|(&a, &b)| a != 1 && a != b)
    }) {
        return Err(PyValueError::new_err(
            "argument shapes must broadcast to output",
        ));
    }
    let modes = modes
        .into_iter()
        .map(|pair| pair.map(|(l, m, pol)| Mode { l, m, pol }))
        .collect();
    let arguments = arguments.map(|a| a.as_array().to_vec());
    let radial = if singular {
        Radial::Outgoing
    } else {
        Radial::Regular
    };
    let (value, residual) = py
        .detach(move || polar::spherical(modes, arguments, helicity, radial))
        .map_err(error)?;
    let value = ArrayD::from_shape_vec(IxDyn(&shape), value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    Ok((
        value,
        PolarTranslationContext {
            residual: Some(residual),
            shape,
            argument_shapes,
        },
    ))
}
#[pyclass]
#[derive(Debug)]
struct CylindricalTranslationContext {
    residual: Option<polar::CylindricalResidual>,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 4],
}
type CylindricalGradients<'py> = (
    Bound<'py, PyArrayDyn<Complex>>,
    Bound<'py, PyArrayDyn<Complex>>,
    Bound<'py, PyArrayDyn<Complex>>,
    Bound<'py, PyArrayDyn<Complex>>,
);
#[pymethods]
impl CylindricalTranslationContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArrayDyn<'py, Complex>,
    ) -> PyResult<CylindricalGradients<'py>> {
        let g = cotangent.as_array();
        if g.shape() != self.shape || g.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(PyValueError::new_err(
                "cotangent must be finite and match output shape",
            ));
        }
        let g: Vec<_> = g.iter().copied().collect();
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let [a, b, c, d] = py.detach(move || residual.pullback(&g)).map_err(error)?;
        let [sa, sb, sc, sd] = &self.argument_shapes;
        Ok((
            reduce_broadcast(a, &self.shape, sa)?.into_pyarray(py),
            reduce_broadcast(b, &self.shape, sb)?.into_pyarray(py),
            reduce_broadcast(c, &self.shape, sc)?.into_pyarray(py),
            reduce_broadcast(d, &self.shape, sd)?.into_pyarray(py),
        ))
    }
}
#[pyfunction]
fn cylindrical_translation<'py>(
    py: Python<'py>,
    orders: Vec<i32>,
    arguments: [PyReadonlyArray1<'py, Complex>; 4],
    singular: bool,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 4],
) -> PyResult<(
    Bound<'py, PyArrayDyn<Complex>>,
    CylindricalTranslationContext,
)> {
    if argument_shapes.iter().any(|s| {
        s.len() > shape.len()
            || s.iter()
                .rev()
                .zip(shape.iter().rev())
                .any(|(&a, &b)| a != 1 && a != b)
    }) {
        return Err(PyValueError::new_err(
            "argument shapes must broadcast to output",
        ));
    }
    let arguments = arguments.map(|a| a.as_array().to_vec());
    let radial = if singular {
        Radial::Outgoing
    } else {
        Radial::Regular
    };
    let (value, residual) = py
        .detach(move || polar::cylindrical(orders, arguments, radial))
        .map_err(error)?;
    let value = ArrayD::from_shape_vec(IxDyn(&shape), value)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    Ok((
        value,
        CylindricalTranslationContext {
            residual: Some(residual),
            shape,
            argument_shapes,
        },
    ))
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CylindricalTranslationContext>()?;
    module.add_function(wrap_pyfunction!(cylindrical_translation, module)?)?;
    module.add_class::<PolarTranslationContext>()?;
    module.add_function(wrap_pyfunction!(spherical_translation, module)?)
}
