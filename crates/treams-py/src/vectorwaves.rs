//! `vector_wave_record` and `VectorWaveContext`: records of vector spherical
//! harmonics and of spherical, cylindrical and plane vector waves
//! (`treams_core::vectorwaves`).
use crate::{
    args::family,
    broadcast::{BroadcastShapes, broadcast_context, check_broadcast, reduce_broadcast, shaped},
    context::detached,
    convert::{CDyn, Cotangent, finite_cotangent},
};
use numpy::{
    IntoPyArray, PyReadonlyArray1,
    ndarray::{Axis, IxDyn},
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyTuple};
use treams_core::{
    Complex,
    fpenv::ieee,
    vectorwaves::{self, VectorWaveResidual, WaveLabel},
};

broadcast_context!(VectorWaveContext(
    VectorWaveResidual,
    Vec<Vec<usize>>,
    /// Whether the values are scalar (`sph_harm`) rather than Cartesian vectors.
    scalar: bool,
));
#[pymethods]
impl VectorWaveContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: Cotangent<'py>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        ieee(|| {
            let BroadcastShapes {
                shape,
                argument_shapes,
            } = &self.shapes;
            // Vector waves append the Cartesian axis to the broadcast shape.
            let mut expected = shape.clone();
            if !self.scalar {
                expected.push(3);
            }
            let g = finite_cotangent::<_, IxDyn>(&cotangent, &expected)?;
            let g: Vec<_> = if self.scalar {
                g.iter()
                    .map(|&v| vectorwaves::sph_harm_from_vsh_z_pullback(v))
                    .collect()
            } else {
                g.lanes(Axis(g.ndim() - 1))
                    .into_iter()
                    .map(|v| [v[0], v[1], v[2]])
                    .collect()
            };
            let residual = self.residual.take()?;
            let gradients = detached(py, move || residual.pullback(&g))?;
            // One tuple item per argument: two to six, depending on the function.
            let gradients = gradients
                .into_iter()
                .zip(argument_shapes)
                .map(|(g, s)| Ok(reduce_broadcast(g, shape, s)?.into_pyarray(py)))
                .collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(py, gradients)
        })
    }
}

/// Record vector spherical harmonics or vector waves: `vectorwaves::vector_wave_array`.
#[pyfunction]
pub(crate) fn vector_wave_record<'py>(
    py: Python<'py>,
    function: &str,
    labels: Vec<(i32, i32, u8)>,
    arguments: Vec<PyReadonlyArray1<'py, Complex>>,
    shape: Vec<usize>,
    argument_shapes: Vec<Vec<usize>>,
) -> PyResult<(CDyn<'py>, VectorWaveContext)> {
    ieee(|| {
        let (family, count, pol) = family(function)?;
        if arguments.len() != count || argument_shapes.len() != count {
            return Err(PyValueError::new_err(
                "wave arguments must match the function",
            ));
        }
        check_broadcast(&shape, &argument_shapes)?;
        let labels = labels
            .into_iter()
            .map(|(l, m, p)| WaveLabel {
                l,
                m,
                pol: pol.unwrap_or(p),
            })
            .collect();
        let args = std::array::from_fn(|i| {
            arguments
                .get(i)
                .map_or_else(|| vec![Complex::default()], |a| a.as_array().to_vec())
        });
        let (values, residual) = detached(py, move || {
            vectorwaves::vector_wave_array(family, labels, args, pol.is_none())
        })?;
        let mut output_shape = shape.clone();
        let scalar = function == "sph_harm";
        let values = if scalar {
            values
                .into_iter()
                .map(vectorwaves::sph_harm_from_vsh_z)
                .collect()
        } else {
            output_shape.push(3);
            values.into_flattened()
        };
        Ok((
            shaped(py, values, &output_shape)?,
            VectorWaveContext::new(
                residual,
                BroadcastShapes {
                    shape,
                    argument_shapes,
                },
                scalar,
            ),
        ))
    })
}
