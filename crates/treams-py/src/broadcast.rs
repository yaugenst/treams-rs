//! Broadcast records: output shapes, finite cotangents and gradients summed back to
//! the argument shapes. The `_record` functions of the special functions, integrals,
//! coordinates, translations, vector waves and lattice sums share them.
use numpy::{
    Element, IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn,
    ndarray::{ArrayD, Axis, IxDyn},
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyTuple};
use treams_core::Complex;

use crate::{
    context::{detached, error},
    convert::{CDyn, Finite, Tangent, finite_cotangent, finite_tangent, layout_error},
};

/// Require every argument shape to broadcast to the output `shape`.
pub(crate) fn check_broadcast<S: AsRef<[usize]>>(shape: &[usize], arguments: &[S]) -> PyResult<()> {
    shape_size(shape).map_err(error)?;
    for argument in arguments {
        shape_size(argument.as_ref()).map_err(error)?;
    }
    let broadcasts = |argument: &[usize]| {
        argument.len() <= shape.len()
            && (argument.iter().rev())
                .zip(shape.iter().rev())
                .all(|(&a, &b)| a == 1 || a == b)
    };
    if arguments
        .iter()
        .all(|argument| broadcasts(argument.as_ref()))
    {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "argument shapes must broadcast to output",
        ))
    }
}

/// Check the declared output shape against the retained scalar-or-array inputs.
pub(crate) fn check_flat_shape(shape: &[usize], lengths: &[usize]) -> PyResult<()> {
    let actual = if lengths.contains(&0) {
        0
    } else {
        lengths.iter().copied().max().unwrap_or_default()
    };
    if shape_size(shape).map_err(error)? != actual {
        return Err(PyValueError::new_err(
            "output shape must match broadcast inputs",
        ));
    }
    Ok(())
}

/// Row-major values as an array of `shape`.
pub(crate) fn shaped<'py, T: Element>(
    py: Python<'py>,
    values: Vec<T>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    Ok(ArrayD::from_shape_vec(IxDyn(shape), values)
        .map_err(layout_error)?
        .into_pyarray(py))
}

/// Row-major values of a finite cotangent with the recorded output `shape`.
pub(crate) fn cotangent<T: Finite>(
    cotangent: &PyReadonlyArrayDyn<'_, T>,
    shape: &[usize],
) -> PyResult<Vec<T>> {
    Ok(finite_cotangent::<_, IxDyn>(cotangent, shape)?
        .iter()
        .copied()
        .collect())
}

/// A finite argument tangent, checked against the saved shape and broadcast in output
/// order. Tangents have the original input shape, even when a record stores
/// already-broadcast inputs.
pub(crate) fn tangent<T: Finite>(
    tangent: &PyReadonlyArrayDyn<'_, T>,
    argument_shape: &[usize],
    shape: &[usize],
) -> PyResult<Vec<T>> {
    let tangent = finite_tangent::<_, IxDyn>(tangent, argument_shape)?;
    Ok(tangent
        .broadcast(IxDyn(shape))
        .ok_or_else(|| PyValueError::new_err("tangent shape must broadcast to output"))?
        .iter()
        .copied()
        .collect())
}

/// Parse a variable-arity tuple of complex argument tangents against one record.
pub(crate) fn tangents(
    tangents: &Bound<'_, PyTuple>,
    argument_shapes: &[Vec<usize>],
    shape: &[usize],
) -> PyResult<Vec<Vec<Complex>>> {
    if tangents.len() != argument_shapes.len() {
        return Err(PyValueError::new_err(format!(
            "expected {} argument tangents",
            argument_shapes.len()
        )));
    }
    tangents
        .iter()
        .zip(argument_shapes)
        .map(|(value, argument_shape)| {
            let value = value.extract::<Tangent<'_>>()?;
            // Native elementwise kernels accept one scalar for all outputs. Keep
            // that representation for shared parameters of large output arrays.
            let output_shape = if argument_shape.iter().product::<usize>() == 1 {
                argument_shape.as_slice()
            } else {
                shape
            };
            tangent(&value, argument_shape, output_shape)
        })
        .collect()
}

/// A gradient of the broadcast output `shape`, summed over the axes that the
/// argument broadcasts along, as an array of `argument_shape`.
pub(crate) fn reduce_broadcast<T: numpy::ndarray::LinalgScalar>(
    gradient: Vec<T>,
    shape: &[usize],
    argument_shape: &[usize],
) -> PyResult<ArrayD<T>> {
    if gradient.len() == argument_shape.iter().product::<usize>() {
        return ArrayD::from_shape_vec(IxDyn(argument_shape), gradient).map_err(layout_error);
    }
    let mut result = ArrayD::from_shape_vec(IxDyn(shape), gradient).map_err(layout_error)?;
    let leading = shape.len() - argument_shape.len();
    for axis in (0..shape.len()).rev() {
        if axis < leading || argument_shape.get(axis - leading) == Some(&1) {
            result = result.sum_axis(Axis(axis));
        }
    }
    result
        .into_shape_with_order(IxDyn(argument_shape))
        .map_err(layout_error)
}

/// Gradients of `N` recorded arguments, each reduced to its argument shape.
pub(crate) trait Gradients<const N: usize> {
    /// One array for a single argument, otherwise a tuple in argument order.
    fn reduce<'py>(
        self,
        py: Python<'py>,
        shapes: &BroadcastShapes<[Vec<usize>; N]>,
    ) -> PyResult<Bound<'py, PyAny>>;
}
impl Gradients<1> for Vec<Complex> {
    fn reduce<'py>(
        self,
        py: Python<'py>,
        shapes: &BroadcastShapes<[Vec<usize>; 1]>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let [argument_shape] = &shapes.argument_shapes;
        Ok(reduce_broadcast(self, &shapes.shape, argument_shape)?
            .into_pyarray(py)
            .into_any())
    }
}
impl<const N: usize> Gradients<N> for [Vec<Complex>; N] {
    fn reduce<'py>(
        self,
        py: Python<'py>,
        shapes: &BroadcastShapes<[Vec<usize>; N]>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let gradients = self
            .into_iter()
            .zip(&shapes.argument_shapes)
            .map(|(gradient, argument_shape)| {
                Ok(reduce_broadcast(gradient, &shapes.shape, argument_shape)?.into_pyarray(py))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, gradients)?.into_any())
    }
}

/// Recorded values and their pullback context.
pub(crate) type Recorded<'py, C> = PyResult<(CDyn<'py>, C)>;

/// The broadcast shape of a record's values and the shapes of its arguments.
#[derive(Debug)]
pub(crate) struct BroadcastShapes<A> {
    /// The shape that every argument broadcasts to.
    pub(crate) shape: Vec<usize>,
    /// The shape of each argument, in argument order.
    pub(crate) argument_shapes: A,
}

/// Number of elements, checked before using shape metadata to allocate storage.
pub(crate) fn shape_size(shape: &[usize]) -> treams_core::Result<usize> {
    shape.iter().try_fold(1_usize, |size, &dim| {
        size.checked_mul(dim)
            .ok_or_else(|| treams_core::Error::InvalidInput("broadcast shape is too large".into()))
    })
}

/// A pullback context of `N` broadcast arguments.
pub(crate) trait Record<const N: usize> {
    type Residual: Send;
    /// The context of `residual`, recorded with `shapes`.
    fn recorded(residual: Self::Residual, shapes: BroadcastShapes<[Vec<usize>; N]>) -> Self;
}

/// Retain a reconstructed residual with the caller's original broadcast shapes.
pub(crate) fn context<const N: usize, C: Record<N>>(
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; N],
    residual: C::Residual,
) -> PyResult<C> {
    check_broadcast(&shape, &argument_shapes)?;
    Ok(C::recorded(
        residual,
        BroadcastShapes {
            shape,
            argument_shapes,
        },
    ))
}

/// Run a recorded forward pass without the GIL and keep its residual.
///
/// The flat values take the broadcast output `shape`; each argument shape
/// must broadcast to it, and its gradient is reduced back in the pullback.
pub(crate) fn record<const N: usize, C: Record<N>>(
    py: Python<'_>,
    shape: Vec<usize>,
    argument_shapes: [Vec<usize>; N],
    forward: impl FnOnce() -> treams_core::Result<(Vec<Complex>, C::Residual)> + Send,
) -> Recorded<'_, C> {
    check_broadcast(&shape, &argument_shapes)?;
    let (value, residual) = detached(py, forward)?;
    Ok((
        shaped(py, value, &shape)?,
        C::recorded(
            residual,
            BroadcastShapes {
                shape,
                argument_shapes,
            },
        ),
    ))
}

/// Record one 0-d evaluation of a microsecond-scale kernel with the GIL held.
///
/// Calibrated: releasing and reacquiring the GIL costs about 0.09 us, 5-9% of these
/// calls.
pub(crate) fn record_held<const N: usize, C: Record<N>>(
    py: Python<'_>,
    forward: impl FnOnce() -> treams_core::Result<(Vec<Complex>, C::Residual)>,
) -> Recorded<'_, C> {
    let (value, residual) = forward().map_err(error)?;
    Ok((
        shaped(py, value, &[])?,
        C::recorded(
            residual,
            BroadcastShapes {
                shape: Vec::new(),
                argument_shapes: std::array::from_fn(|_| Vec::new()),
            },
        ),
    ))
}

/// Define pullback contexts of broadcast records: [`context!`](crate::context::context)
/// structs whose `shapes` field holds the [`BroadcastShapes`].
///
/// `Name(Residual, N);` also defines `pullback` for `N` arguments whose residual
/// `pullback(&[Complex])` returns their gradients, summed back to the argument
/// shapes; the calling module imports `pyo3::prelude::*` and `fpenv::ieee`.
/// `Name(Residual, Shapes, field: Type, ...)` defines the struct only, with
/// `Shapes` as the type of the argument shapes, for contexts with their own pullback.
macro_rules! broadcast_context {
    ($($(#[$doc:meta])* $name:ident($residual:ty, $count:literal);)+) => {$(
        $crate::context::context!(
            $(#[$doc])*
            $name($residual, shapes: $crate::broadcast::BroadcastShapes<[Vec<usize>; $count]>)
        );

        #[pymethods]
        impl $name {
            #[pyo3(signature = (*tangents))]
            fn pushforward<'py>(
                &self,
                py: Python<'py>,
                tangents: &Bound<'py, pyo3::types::PyTuple>,
            ) -> PyResult<$crate::convert::CDyn<'py>> {
                ieee(|| {
                    let tangents = $crate::broadcast::tangents(
                        tangents, &self.shapes.argument_shapes, &self.shapes.shape,
                    )?;
                    let residual = &self.residual;
                    let tangent = $crate::context::detached(py, move || {
                        residual.pushforward(std::array::from_fn(|i| tangents[i].as_slice()))
                    })?;
                    $crate::broadcast::shaped(py, tangent, &self.shapes.shape)
                })
            }

            fn pullback<'py>(
                &self,
                py: Python<'py>,
                cotangent: $crate::convert::Cotangent<'py>,
            ) -> PyResult<Bound<'py, PyAny>> {
                ieee(|| {
                    let g = $crate::broadcast::cotangent(&cotangent, &self.shapes.shape)?;
                    let residual = &self.residual;
                    let gradients = $crate::context::detached(py, move || residual.pullback(&g))?;
                    $crate::broadcast::Gradients::<$count>::reduce(gradients, py, &self.shapes)
                })
            }
        }


        impl $crate::broadcast::Record<$count> for $name {
            type Residual = $residual;
            fn recorded(
                residual: $residual,
                shapes: $crate::broadcast::BroadcastShapes<[Vec<usize>; $count]>,
            ) -> Self {
                Self::new(residual, shapes)
            }
        }
    )+};
    ($(#[$doc:meta])* $name:ident($residual:ty, $shapes:ty $(, $(#[$meta:meta])* $field:ident: $type:ty)* $(,)?)) => {
        $crate::context::context!(
            $(#[$doc])*
            $name(
                $residual,
                shapes: $crate::broadcast::BroadcastShapes<$shapes>
                $(, $(#[$meta])* $field: $type)*
            )
        );
    };
}
pub(crate) use broadcast_context;
