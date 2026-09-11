//! Owned low-level vector-wave pullbacks and `NumPy` broadcast reduction.
use crate::{error, special::reduce_broadcast};
use numpy::{
    IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn,
    ndarray::{ArrayD, Axis, IxDyn},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use treams_core::{
    Complex,
    special::Radial,
    vectorwaves::{self, Family, Residual},
    waves::Mode,
};

fn family(kind: &str) -> PyResult<(Family, usize, Option<u8>)> {
    let radial = if kind.contains("_r") {
        Radial::Regular
    } else {
        Radial::Outgoing
    };
    let (family, count) = match kind {
        "vsh_X" => (Family::HarmonicX, 2),
        "vsh_Y" => (Family::HarmonicY, 2),
        "vsh_Z" | "sph_harm" => (Family::HarmonicZ, 2),
        "vsw_M" | "vsw_N" | "vsw_A" | "vsw_rM" | "vsw_rN" | "vsw_rA" => {
            (Family::Spherical(radial), 3)
        }
        "vcw_M" | "vcw_rM" => (Family::Cylindrical(radial), 4),
        "vcw_N" | "vcw_A" | "vcw_rN" | "vcw_rA" => (Family::Cylindrical(radial), 5),
        "vpw_M" | "vpw_N" | "vpw_A" => (Family::Plane, 6),
        _ => return Err(PyValueError::new_err("unknown vector-wave function")),
    };
    let pol = if kind.ends_with('A') {
        None
    } else {
        Some(u8::from(kind.ends_with('N')))
    };
    Ok((family, count, pol))
}

#[pyclass]
#[derive(Debug)]
struct WaveContext {
    residual: Option<Residual>,
    shape: Vec<usize>,
    scalar: bool,
    argument_shapes: Vec<Vec<usize>>,
}
#[pymethods]
impl WaveContext {
    fn pullback<'py>(
        &mut self,
        py: Python<'py>,
        cotangent: PyReadonlyArrayDyn<'py, Complex>,
    ) -> PyResult<Vec<Bound<'py, PyArrayDyn<Complex>>>> {
        let g = cotangent.as_array();
        if g.ndim() != self.shape.len() + usize::from(!self.scalar)
            || (!self.scalar && g.shape().last() != Some(&3))
            || g.shape()[..self.shape.len()] != self.shape
            || g.iter().any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(PyValueError::new_err(
                "wave cotangent must be finite and match output shape",
            ));
        }
        let g: Vec<_> = if self.scalar {
            g.iter()
                .map(|&v| [Complex::i() * v, Complex::default(), Complex::default()])
                .collect()
        } else {
            g.lanes(Axis(g.ndim() - 1))
                .into_iter()
                .map(|v| [v[0], v[1], v[2]])
                .collect()
        };
        let residual = self
            .residual
            .take()
            .ok_or_else(|| PyValueError::new_err("pullback residual has already been consumed"))?;
        let gradients = py.detach(move || residual.pullback(&g)).map_err(error)?;
        gradients
            .into_iter()
            .zip(&self.argument_shapes)
            .map(|(g, s)| Ok(reduce_broadcast(g, &self.shape, s)?.into_pyarray(py)))
            .collect()
    }
}

#[pyfunction]
fn vector_wave<'py>(
    py: Python<'py>,
    kind: &str,
    labels: Vec<(i32, i32, u8)>,
    arguments: Vec<PyReadonlyArray1<'py, Complex>>,
    shape: Vec<usize>,
    argument_shapes: Vec<Vec<usize>>,
) -> PyResult<(Bound<'py, PyArrayDyn<Complex>>, WaveContext)> {
    let (family, count, pol) = family(kind)?;
    if arguments.len() != count
        || argument_shapes.len() != count
        || argument_shapes.iter().any(|s| {
            s.len() > shape.len()
                || s.iter()
                    .rev()
                    .zip(shape.iter().rev())
                    .any(|(&a, &b)| a != 1 && a != b)
        })
    {
        return Err(PyValueError::new_err(
            "wave arguments must match the function and broadcast to output",
        ));
    }
    let modes = labels
        .into_iter()
        .map(|(l, m, p)| Mode {
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
    let (values, residual) = py
        .detach(move || vectorwaves::array(family, modes, args, pol.is_none()))
        .map_err(error)?;
    let mut output_shape = shape.clone();
    let scalar = kind == "sph_harm";
    let values = if scalar {
        values.into_iter().map(|v| -Complex::i() * v[0]).collect()
    } else {
        output_shape.push(3);
        values.into_flattened()
    };
    let value = ArrayD::from_shape_vec(IxDyn(&output_shape), values)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    Ok((
        value,
        WaveContext {
            residual: Some(residual),
            shape,
            scalar,
            argument_shapes,
        },
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<WaveContext>()?;
    module.add_function(wrap_pyfunction!(vector_wave, module)?)
}
