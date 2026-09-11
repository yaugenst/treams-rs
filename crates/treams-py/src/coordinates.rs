//! Owned coordinate pullbacks; `NumPy` shape reduction stays at the binding boundary.
use crate::{error, special::reduce_broadcast};
use numpy::{
    IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
    ndarray::{ArrayD, IxDyn},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    coordinates::{self as coord, Transform},
};

fn transform(name: &str) -> PyResult<Transform> {
    Ok(match name {
        "car2cyl" => Transform::CarToCyl,
        "car2sph" => Transform::CarToSph,
        "cyl2car" => Transform::CylToCar,
        "cyl2sph" => Transform::CylToSph,
        "sph2car" => Transform::SphToCar,
        "sph2cyl" => Transform::SphToCyl,
        "car2pol" => Transform::CarToPol,
        "pol2car" => Transform::PolToCar,
        _ => return Err(PyValueError::new_err("unknown coordinate transform")),
    })
}
fn triple<T: Copy + Default>(values: &[T]) -> [T; 3] {
    std::array::from_fn(|i| values.get(i).copied().unwrap_or_default())
}
fn array<T: numpy::Element>(values: Vec<T>, shape: &[usize]) -> PyResult<ArrayD<T>> {
    ArrayD::from_shape_vec(IxDyn(shape), values).map_err(|e| PyValueError::new_err(e.to_string()))
}
fn finite(g: &[Complex]) -> bool {
    g.iter().all(|v| v.re.is_finite() && v.im.is_finite())
}

#[pyclass]
#[derive(Debug)]
struct CoordinateContext {
    points: Option<Vec<f64>>,
    shape: Vec<usize>,
    transform: Transform,
}
#[pymethods]
impl CoordinateContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let g = cotangent.as_array();
        if g.shape() != self.shape || g.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err(
                "cotangent must be finite and match output shape",
            ));
        }
        let g: Vec<_> = g.iter().copied().collect();
        let points = self
            .points
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let transform = self.transform;
        let dim = transform.dimension();
        let result = py
            .detach(move || {
                let mut values = Vec::with_capacity(g.len());
                for (p, g) in points.chunks_exact(dim).zip(g.chunks_exact(dim)) {
                    values.extend(
                        coord::point_pullback(triple(p), transform, triple(g))?
                            .into_iter()
                            .take(dim),
                    );
                }
                Ok::<_, treams_core::Error>(values)
            })
            .map_err(error)?;
        Ok(array(result, &self.shape)?.into_pyarray(py))
    }
}
#[pyfunction]
fn coordinates<'py>(
    py: Python<'py>,
    points: PyReadonlyArrayDyn<'py, f64>,
    kind: &str,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, CoordinateContext)> {
    let transform = transform(kind)?;
    let dim = transform.dimension();
    let input = points.as_array();
    if input.shape().last() != Some(&dim) {
        return Err(PyValueError::new_err(
            "last axis must match coordinate dimension",
        ));
    }
    let shape = input.shape().to_vec();
    let points: Vec<_> = input.iter().copied().collect();
    let (values, points) = py
        .detach(move || {
            let mut values = Vec::with_capacity(points.len());
            for p in points.chunks_exact(dim) {
                values.extend(coord::point(triple(p), transform)?.into_iter().take(dim));
            }
            Ok::<_, treams_core::Error>((values, points))
        })
        .map_err(error)?;
    Ok((
        array(values, &shape)?.into_pyarray(py),
        CoordinateContext {
            points: Some(points),
            shape,
            transform,
        },
    ))
}

#[pyclass]
#[derive(Debug)]
struct VectorCoordinateContext {
    inputs: Option<(Vec<Complex>, Vec<f64>)>,
    shape: Vec<usize>,
    input_shapes: [Vec<usize>; 2],
    transform: Transform,
}
type VectorGradient<'py> = (Bound<'py, PyArrayDyn<Complex>>, Bound<'py, PyArrayDyn<f64>>);
#[pymethods]
impl VectorCoordinateContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArrayDyn<'py, Complex>,
    ) -> PyResult<VectorGradient<'py>> {
        let g = cotangent.as_array();
        if g.shape() != self.shape {
            return Err(PyValueError::new_err("cotangent must match output shape"));
        }
        let g: Vec<_> = g.iter().copied().collect();
        if !finite(&g) {
            return Err(PyValueError::new_err("cotangent must be finite"));
        }
        let (vectors, points) = self
            .inputs
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let transform = self.transform;
        let dim = transform.dimension();
        let (gv, gp) = py
            .detach(move || {
                let mut gv = vec![Complex::default(); vectors.len()];
                let mut gp = vec![0.0; points.len()];
                for (i, g) in g.chunks_exact(dim).enumerate() {
                    let vi = if vectors.len() == dim { 0 } else { i * dim };
                    let pi = if points.len() == dim { 0 } else { i * dim };
                    let (v, p) = coord::vector_pullback(
                        triple(&vectors[vi..vi + dim]),
                        triple(&points[pi..pi + dim]),
                        transform,
                        triple(g),
                    )?;
                    for (out, g) in gv[vi..vi + dim].iter_mut().zip(v) {
                        *out += g;
                    }
                    for (out, g) in gp[pi..pi + dim].iter_mut().zip(p) {
                        *out += g;
                    }
                }
                Ok::<_, treams_core::Error>((gv, gp))
            })
            .map_err(error)?;
        Ok((
            reduce_broadcast(gv, &self.shape, &self.input_shapes[0])?.into_pyarray(py),
            reduce_broadcast(gp, &self.shape, &self.input_shapes[1])?.into_pyarray(py),
        ))
    }
}
#[pyfunction]
fn vector_coordinates<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArrayDyn<'py, Complex>,
    points: PyReadonlyArrayDyn<'py, f64>,
    kind: &str,
    shape: Vec<usize>,
    input_shapes: [Vec<usize>; 2],
) -> PyResult<(Bound<'py, PyArrayDyn<Complex>>, VectorCoordinateContext)> {
    let transform = transform(kind)?;
    let dim = transform.dimension();
    let count: usize = shape.iter().product();
    if shape.last() != Some(&dim)
        || input_shapes.iter().any(|s| {
            s.last() != Some(&dim)
                || s.len() > shape.len()
                || s.iter()
                    .rev()
                    .zip(shape.iter().rev())
                    .any(|(&a, &b)| a != 1 && a != b)
        })
    {
        return Err(PyValueError::new_err(
            "input shapes must broadcast to the coordinate output",
        ));
    }
    let vectors: Vec<_> = vectors.as_array().iter().copied().collect();
    let points: Vec<_> = points.as_array().iter().copied().collect();
    if (vectors.len() != dim && vectors.len() != count)
        || (points.len() != dim && points.len() != count)
    {
        return Err(PyValueError::new_err(
            "coordinate inputs must be one vector or the broadcast output size",
        ));
    }
    let (values, vectors, points) = py
        .detach(move || {
            let mut values = Vec::with_capacity(count);
            for i in 0..count / dim {
                let vi = if vectors.len() == dim { 0 } else { i * dim };
                let pi = if points.len() == dim { 0 } else { i * dim };
                values.extend(
                    coord::vector(
                        triple(&vectors[vi..vi + dim]),
                        triple(&points[pi..pi + dim]),
                        transform,
                    )?
                    .into_iter()
                    .take(dim),
                );
            }
            Ok::<_, treams_core::Error>((values, vectors, points))
        })
        .map_err(error)?;
    Ok((
        array(values, &shape)?.into_pyarray(py),
        VectorCoordinateContext {
            inputs: Some((vectors, points)),
            shape,
            input_shapes,
            transform,
        },
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CoordinateContext>()?;
    module.add_class::<VectorCoordinateContext>()?;
    module.add_function(wrap_pyfunction!(coordinates, module)?)?;
    module.add_function(wrap_pyfunction!(vector_coordinates, module)?)?;
    Ok(())
}
