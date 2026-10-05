//! `coordinates_record` and `vector_coordinates_record` with their contexts
//! (`treams_core::special::coordinates`). The contexts keep copies of the points,
//! and the vector pullback sums gradients over broadcast axes.
use crate::{
    args::transform,
    broadcast::{BroadcastShapes, broadcast_context, check_broadcast, reduce_broadcast, shaped},
    context::{context, detached},
    convert::{CDyn, Cotangent, RDyn, RealCotangent, RealTangent, Tangent},
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
    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        points_tangent: RealTangent<'py>,
    ) -> PyResult<RDyn<'py>> {
        ieee(|| {
            let tangent = crate::broadcast::tangent(&points_tangent, &self.shape, &self.shape)?;
            let points = &self.residual;
            let transform = self.transform;
            let dim = transform.dimension();
            let result = detached(py, move || {
                let mut values = Vec::with_capacity(points.len());
                for (point, tangent) in points.chunks_exact(dim).zip(tangent.chunks_exact(dim)) {
                    values.extend(
                        coord::point_pushforward(triple(point), transform, triple(tangent))?
                            .into_iter()
                            .take(dim),
                    );
                }
                Ok(values)
            })?;
            shaped(py, result, &self.shape)
        })
    }

    fn pullback<'py>(&self, py: Python<'py>, cotangent: RealCotangent<'py>) -> PyResult<RDyn<'py>> {
        ieee(|| {
            let g = crate::broadcast::cotangent(&cotangent, &self.shape)?;
            let points = &self.residual;
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
        let context = coordinates_context(py, points, kind)?;
        let dim = context.transform.dimension();
        let values = detached(py, || {
            let mut values = Vec::with_capacity(context.residual.len());
            for p in context.residual.chunks_exact(dim) {
                values.extend(
                    coord::point(triple(p), context.transform)?
                        .into_iter()
                        .take(dim),
                );
            }
            Ok(values)
        })?;
        Ok((shaped(py, values, &context.shape)?, context))
    })
}

/// Restore an input-only point context without evaluating converted coordinates.
#[pyfunction]
pub(crate) fn coordinates_context(
    _py: Python<'_>,
    points: PyReadonlyArrayDyn<'_, f64>,
    kind: &str,
) -> PyResult<CoordinatesContext> {
    ieee(|| {
        let transform = transform(kind)?;
        let input = points.as_array();
        if input.shape().last() != Some(&transform.dimension()) {
            return Err(PyValueError::new_err(
                "last axis must match coordinate dimension",
            ));
        }
        Ok(CoordinatesContext::new(
            input.iter().copied().collect(),
            input.shape().to_vec(),
            transform,
        ))
    })
}

broadcast_context!(
    /// The context of `vector_coordinates_record`: copies of the vectors and the points.
    VectorCoordinatesContext((Vec<Complex>, Vec<f64>), [Vec<usize>; 2], transform: Transform)
);
#[pymethods]
impl VectorCoordinatesContext {
    fn pushforward<'py>(
        &self,
        py: Python<'py>,
        vectors_tangent: Tangent<'py>,
        points_tangent: RealTangent<'py>,
    ) -> PyResult<CDyn<'py>> {
        ieee(|| {
            let vector_tangent = crate::broadcast::tangent(
                &vectors_tangent,
                &self.shapes.argument_shapes[0],
                &self.shapes.shape,
            )?;
            let point_tangent = crate::broadcast::tangent(
                &points_tangent,
                &self.shapes.argument_shapes[1],
                &self.shapes.shape,
            )?;
            let (vectors, points) = &self.residual;
            let transform = self.transform;
            let dim = transform.dimension();
            let result = detached(py, move || {
                let mut values = Vec::with_capacity(vector_tangent.len());
                for (i, (dv, dp)) in vector_tangent
                    .chunks_exact(dim)
                    .zip(point_tangent.chunks_exact(dim))
                    .enumerate()
                {
                    let vi = if vectors.len() == dim { 0 } else { i * dim };
                    let pi = if points.len() == dim { 0 } else { i * dim };
                    values.extend(
                        coord::vector_pushforward(
                            triple(&vectors[vi..vi + dim]),
                            triple(&points[pi..pi + dim]),
                            transform,
                            triple(dv),
                            triple(dp),
                        )?
                        .into_iter()
                        .take(dim),
                    );
                }
                Ok(values)
            })?;
            shaped(py, result, &self.shapes.shape)
        })
    }

    fn pullback<'py>(
        &self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<(CDyn<'py>, RDyn<'py>)> {
        ieee(|| {
            let g = crate::broadcast::cotangent(&cotangent, &self.shapes.shape)?;
            let (vectors, points) = &self.residual;
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
        let context =
            vector_coordinates_context(py, vectors, points, kind, shape, argument_shapes)?;
        let dim = context.transform.dimension();
        let count =
            crate::broadcast::shape_size(&context.shapes.shape).map_err(crate::context::error)?;
        let (vectors, points) = &context.residual;
        let values = detached(py, || {
            let mut values = Vec::with_capacity(count);
            for i in 0..count / dim {
                let vi = if vectors.len() == dim { 0 } else { i * dim };
                let pi = if points.len() == dim { 0 } else { i * dim };
                values.extend(
                    coord::vector(
                        triple(&vectors[vi..vi + dim]),
                        triple(&points[pi..pi + dim]),
                        context.transform,
                    )?
                    .into_iter()
                    .take(dim),
                );
            }
            Ok(values)
        })?;
        Ok((shaped(py, values, &context.shapes.shape)?, context))
    })
}

/// Restore an input-only vector context without evaluating converted vectors.
#[pyfunction]
pub(crate) fn vector_coordinates_context(
    _py: Python<'_>,
    vectors: PyReadonlyArrayDyn<'_, Complex>,
    points: PyReadonlyArrayDyn<'_, f64>,
    kind: &str,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; 2],
) -> PyResult<VectorCoordinatesContext> {
    ieee(|| {
        let transform = transform(kind)?;
        let dim = transform.dimension();
        let count = crate::broadcast::shape_size(&shape).map_err(crate::context::error)?;
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
        Ok(VectorCoordinatesContext::new(
            (vectors, points),
            BroadcastShapes {
                shape,
                argument_shapes,
            },
            transform,
        ))
    })
}
