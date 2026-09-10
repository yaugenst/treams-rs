//! Cylindrical scattering bindings.
use crate::{error, materials, matrix};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    coeffs::Matrix2,
    cylinder::{self, CylinderResidual},
    special::{self, Radial},
};

#[pyclass]
#[derive(Debug)]
struct CylinderContext {
    residual: Option<CylinderResidual>,
}
type Gradient<'py> = (
    f64,
    f64,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray1<Complex>>,
    Bound<'py, PyArray1<Complex>>,
);
#[pymethods]
impl CylinderContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArray2<'py, Complex>,
    ) -> PyResult<Gradient<'py>> {
        let array = cotangent.as_array();
        if array.dim() != (2, 2) || array.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(PyValueError::new_err(
                "cotangent must be finite with shape (2, 2)",
            ));
        }
        let g = Matrix2::from_fn(|i, j| array[(i, j)]);
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let result = py.detach(move || residual.pullback(&g)).map_err(error)?;
        Ok((
            result.kz,
            result.k0,
            result.layers.sizes.into_pyarray(py),
            result.layers.epsilon.into_pyarray(py),
            result.layers.mu.into_pyarray(py),
            result.layers.kappa.into_pyarray(py),
        ))
    }
}
#[pyfunction]
fn mie_cyl<'py>(
    py: Python<'py>,
    kz: f64,
    m: i32,
    k0: f64,
    radii: PyReadonlyArray1<'py, f64>,
    epsilon: PyReadonlyArray1<'py, Complex>,
    mu: PyReadonlyArray1<'py, Complex>,
    kappa: PyReadonlyArray1<'py, Complex>,
) -> PyResult<(Bound<'py, PyArray2<Complex>>, CylinderContext)> {
    let radii = radii.to_vec()?;
    let mat = materials(&epsilon.to_vec()?, &mu.to_vec()?, &kappa.to_vec()?)?;
    let residual = py
        .detach(move || cylinder::mie_cyl(kz, m, k0, &radii, &mat))
        .map_err(error)?;
    Ok((
        matrix(py, &residual.value),
        CylinderContext {
            residual: Some(residual),
        },
    ))
}
#[pyfunction]
fn cylindrical(m: i32, z: Complex, outgoing: bool) -> PyResult<(Complex, Complex, Complex)> {
    let jet = special::cylindrical(
        m,
        z,
        if outgoing {
            Radial::Outgoing
        } else {
            Radial::Regular
        },
    )
    .map_err(error)?;
    Ok((jet.value, jet.first, jet.second))
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CylinderContext>()?;
    module.add_function(wrap_pyfunction!(mie_cyl, module)?)?;
    module.add_function(wrap_pyfunction!(cylindrical, module)?)?;
    Ok(())
}
