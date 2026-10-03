//! `coordinates_record` and `vector_coordinates_record` with their contexts
//! (`treams_core::special::coordinates`). The contexts keep copies of the points,
//! and the vector pullback sums gradients over broadcast axes.
use crate::{
    args::transform,
    broadcast::{BroadcastShapes, broadcast_context, check_broadcast, reduce_broadcast, shaped},
    context::{context, detached},
    convert::{CDyn, Cotangent, RDyn, RealCotangent},
};
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    fpenv::ieee,
    special::coordinates::{self as coord, Transform},
};

/// Point and vector function names of each transform, in the order of the
/// `TRANSFORM` parameter of the coordinate ufunc loops.
pub(crate) const TRANSFORMS: [(&str, &str, Transform); 8] = [
    ("car2cyl", "vcar2cyl", Transform::CarToCyl),
    ("car2sph", "vcar2sph", Transform::CarToSph),
    ("cyl2car", "vcyl2car", Transform::CylToCar),
    ("cyl2sph", "vcyl2sph", Transform::CylToSph),
    ("sph2car", "vsph2car", Transform::SphToCar),
    ("sph2cyl", "vsph2cyl", Transform::SphToCyl),
    ("car2pol", "vcar2pol", Transform::CarToPol),
    ("pol2car", "vpol2car", Transform::PolToCar),
];

/// The first three values, padded with zeros: a point or vector of two or three
/// components as the core's three.
fn triple<T: Copy + Default>(values: &[T]) -> [T; 3] {
    std::array::from_fn(|i| values.get(i).copied().unwrap_or_default())
}

context!(
    /// The context of `coordinates_record`: a copy of the points, whose shape the
    /// values and the gradients share.
    CoordinatesContext(Vec<f64>, shape: Vec<usize>, transform: Transform)
);
#[pymethods]
impl CoordinatesContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: RealCotangent<'py>,
    ) -> PyResult<RDyn<'py>> {
        ieee(|| {
            let g = crate::broadcast::cotangent(&cotangent, &self.shape)?;
            let points = self.residual.take()?;
            let transform = self.transform;
            let dim = transform.dimension();
            let result = detached(py, move || {
                let mut values = Vec::with_capacity(g.len());
                for (p, g) in points.chunks_exact(dim).zip(g.chunks_exact(dim)) {
                    values.extend(
                        coord::point_pullback(triple(p), transform, triple(g))?
                            .into_iter()
                            .take(dim),
                    );
                }
                Ok(values)
            })?;
            shaped(py, result, &self.shape)
        })
    }
}
/// Record points in another coordinate system: `special::coordinates::point`.
#[pyfunction]
pub(crate) fn coordinates_record<'py>(
    py: Python<'py>,
    points: PyReadonlyArrayDyn<'py, f64>,
    kind: &str,
) -> PyResult<(RDyn<'py>, CoordinatesContext)> {
    ieee(|| {
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
        let (values, points) = detached(py, move || {
            let mut values = Vec::with_capacity(points.len());
            for p in points.chunks_exact(dim) {
                values.extend(coord::point(triple(p), transform)?.into_iter().take(dim));
            }
            Ok((values, points))
        })?;
        Ok((
            shaped(py, values, &shape)?,
            CoordinatesContext::new(points, shape, transform),
        ))
    })
}

broadcast_context!(
    /// The context of `vector_coordinates_record`: copies of the vectors and the points.
    VectorCoordinatesContext((Vec<Complex>, Vec<f64>), [Vec<usize>; 2], transform: Transform)
);
#[pymethods]
impl VectorCoordinatesContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(CDyn<'py>, RDyn<'py>)> {
        ieee(|| {
            let g = crate::broadcast::cotangent(&cotangent, &self.shapes.shape)?;
            let (vectors, points) = self.residual.take()?;
            let transform = self.transform;
            let dim = transform.dimension();
            let (gv, gp) = detached(py, move || {
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
                Ok((gv, gp))
            })?;
            let BroadcastShapes {
                shape,
                argument_shapes: [vector_shape, point_shape],
            } = &self.shapes;
            Ok((
                reduce_broadcast(gv, shape, vector_shape)?.into_pyarray(py),
                reduce_broadcast(gp, shape, point_shape)?.into_pyarray(py),
            ))
        })
    }
}
/// Record vector components in another coordinate system: `special::coordinates::vector`.
#[pyfunction]
pub(crate) fn vector_coordinates_record<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArrayDyn<'py, Complex>,
    points: PyReadonlyArrayDyn<'py, f64>,
    kind: &str,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 2],
) -> PyResult<(CDyn<'py>, VectorCoordinatesContext)> {
    ieee(|| {
        let transform = transform(kind)?;
        let dim = transform.dimension();
        let count: usize = shape.iter().product();
        if shape.last() != Some(&dim) || argument_shapes.iter().any(|s| s.last() != Some(&dim)) {
            return Err(PyValueError::new_err(
                "last axis must match coordinate dimension",
            ));
        }
        check_broadcast(&shape, &argument_shapes)?;
        let vectors: Vec<_> = vectors.as_array().iter().copied().collect();
        let points: Vec<_> = points.as_array().iter().copied().collect();
        if (vectors.len() != dim && vectors.len() != count)
            || (points.len() != dim && points.len() != count)
        {
            return Err(PyValueError::new_err(
                "coordinate inputs must be one vector or the broadcast output size",
            ));
        }
        let (values, vectors, points) = detached(py, move || {
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
            Ok((values, vectors, points))
        })?;
        Ok((
            shaped(py, values, &shape)?,
            VectorCoordinatesContext::new(
                (vectors, points),
                BroadcastShapes {
                    shape,
                    argument_shapes,
                },
                transform,
            ),
        ))
    })
}
