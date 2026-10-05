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
    saved::{Reader, SavedState, Writer},
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

fn invalid_state() -> treams_core::Error {
    treams_core::Error::InvalidInput("invalid native coordinate state".into())
}

fn transform_tag(transform: Transform) -> u8 {
    match transform {
        Transform::CarToCyl => 0,
        Transform::CarToSph => 1,
        Transform::CylToCar => 2,
        Transform::CylToSph => 3,
        Transform::SphToCar => 4,
        Transform::SphToCyl => 5,
        Transform::CarToPol => 6,
        Transform::PolToCar => 7,
    }
}

fn read_transform(reader: &mut Reader<'_>) -> treams_core::Result<Transform> {
    TRANSFORMS
        .get(usize::from(reader.byte()?))
        .map(|entry| entry.2)
        .ok_or_else(invalid_state)
}

fn coordinate_state_size(shape: &[usize]) -> treams_core::Result<usize> {
    if !matches!(shape.last(), Some(2 | 3)) {
        return Err(invalid_state());
    }
    let shapes = BroadcastShapes {
        shape: shape.to_vec(),
        argument_shapes: [],
    };
    crate::broadcast::shape_size(shape)?
        .checked_mul(8)
        .and_then(|size| size.checked_add(1))
        .and_then(|size| {
            shapes
                .state_size()
                .ok()
                .and_then(|metadata| size.checked_add(metadata))
        })
        .ok_or_else(invalid_state)
}

impl SavedState for CoordinatesContext {
    fn save_state(&self) -> treams_core::Result<Vec<u8>> {
        let mut writer = Writer::new(coordinate_state_size(&self.shape)?);
        BroadcastShapes {
            shape: self.shape.clone(),
            argument_shapes: [],
        }
        .write_state(&mut writer)?;
        writer.byte(transform_tag(self.transform));
        for &point in &self.residual {
            writer.f64(point);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> treams_core::Result<Self> {
        let mut reader = Reader::new(bytes);
        let shapes = BroadcastShapes::<[Vec<usize>; 0]>::read_state(&mut reader)?;
        let transform = read_transform(&mut reader)?;
        if shapes.shape.last() != Some(&transform.dimension())
            || bytes.len() != coordinate_state_size(&shapes.shape)?
        {
            return Err(invalid_state());
        }
        let points = (0..crate::broadcast::shape_size(&shapes.shape)?)
            .map(|_| reader.f64())
            .collect::<treams_core::Result<Vec<_>>>()?;
        reader.finish()?;
        Ok(Self::new(points, shapes.shape, transform))
    }
}

fn coordinate_lengths(
    shapes: &BroadcastShapes<[Vec<usize>; 2]>,
) -> treams_core::Result<[usize; 2]> {
    let dim = *shapes.shape.last().ok_or_else(invalid_state)?;
    if !matches!(dim, 2 | 3)
        || shapes
            .argument_shapes
            .iter()
            .any(|shape| shape.last() != Some(&dim))
    {
        return Err(invalid_state());
    }
    let size = crate::broadcast::shape_size(&shapes.shape)?;
    let length = |shape: &[usize]| -> treams_core::Result<usize> {
        Ok(if crate::broadcast::shape_size(shape)? == dim {
            dim
        } else {
            size
        })
    };
    let [vectors, points] = &shapes.argument_shapes;
    Ok([length(vectors)?, length(points)?])
}

fn vector_coordinate_state_size(
    shapes: &BroadcastShapes<[Vec<usize>; 2]>,
) -> treams_core::Result<usize> {
    let [vectors, points] = coordinate_lengths(shapes)?;
    let metadata = shapes.state_size()?;
    vectors
        .checked_mul(16)
        .and_then(|size| {
            points
                .checked_mul(8)
                .and_then(|points| size.checked_add(points))
        })
        .and_then(|size| size.checked_add(metadata))
        .and_then(|size| size.checked_add(1))
        .ok_or_else(invalid_state)
}

impl SavedState for VectorCoordinatesContext {
    fn save_state(&self) -> treams_core::Result<Vec<u8>> {
        let mut writer = Writer::new(vector_coordinate_state_size(&self.shapes)?);
        self.shapes.write_state(&mut writer)?;
        writer.byte(transform_tag(self.transform));
        let (vectors, points) = &self.residual;
        for &value in vectors {
            writer.complex(value);
        }
        for &value in points {
            writer.f64(value);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> treams_core::Result<Self> {
        let mut reader = Reader::new(bytes);
        let shapes = BroadcastShapes::read_state(&mut reader)?;
        shapes.validate_broadcast()?;
        let transform = read_transform(&mut reader)?;
        if shapes.shape.last() != Some(&transform.dimension())
            || bytes.len() != vector_coordinate_state_size(&shapes)?
        {
            return Err(invalid_state());
        }
        let [vectors, points] = coordinate_lengths(&shapes)?;
        let vectors = (0..vectors)
            .map(|_| reader.complex())
            .collect::<treams_core::Result<Vec<_>>>()?;
        let points = (0..points)
            .map(|_| reader.f64())
            .collect::<treams_core::Result<Vec<_>>>()?;
        reader.finish()?;
        Ok(Self::new((vectors, points), shapes, transform))
    }
}

context!(
    /// The context of `coordinates_record`: a copy of the points, whose shape the
    /// values and the gradients share.
    CoordinatesContext(Vec<f64>, shape: Vec<usize>, transform: Transform)
);
#[pymethods]
impl CoordinatesContext {
    #[staticmethod]
    fn _state_spec(shape: Vec<usize>) -> PyResult<usize> {
        ieee(|| coordinate_state_size(&shape).map_err(crate::context::error))
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, self))
    }

    #[staticmethod]
    fn _from_state(state: numpy::PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state))
    }

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
    #[staticmethod]
    fn _state_spec(shape: Vec<usize>, argument_shapes: [Vec<usize>; 2]) -> PyResult<usize> {
        ieee(|| {
            check_broadcast(&shape, &argument_shapes)?;
            vector_coordinate_state_size(&BroadcastShapes {
                shape,
                argument_shapes,
            })
            .map_err(crate::context::error)
        })
    }

    fn _state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<u8>>> {
        ieee(|| crate::context::state_array(py, self))
    }

    #[staticmethod]
    fn _from_state(state: numpy::PyReadonlyArray1<'_, u8>) -> PyResult<Self> {
        ieee(|| crate::context::restore_state(&state))
    }

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
