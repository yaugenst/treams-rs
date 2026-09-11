//! `NumPy` owns broadcasting, masking, overlap buffering and output allocation.
//! The strided C loop calls the same Rust kernels as the recorded boundary.

#![allow(unsafe_code)] // NumPy requires a raw C callback; safety obligations are confined here.
#![allow(clippy::cast_ptr_alignment)] // Slice alignment is checked; all other access is unaligned.

use std::{
    ffi::{c_char, c_long, c_void},
    sync::OnceLock,
};

use numpy::npyffi::{NPY_TYPES, PY_UFUNC_API, PyUFuncGenericFunction, npy_intp};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods, ndarray::Array2,
};
use pyo3::{
    prelude::*,
    types::{PyCapsule, PyComplex, PyDict, PyFloat, PyInt, PyList, PyTuple},
};
use rayon::prelude::*;

static PW_TRANSLATE: OnceLock<Py<PyAny>> = OnceLock::new();
static PLANE_WAVES: OnceLock<Vec<Py<PyAny>>> = OnceLock::new();
static CELLS: OnceLock<Vec<Py<PyAny>>> = OnceLock::new();
static COORDINATES: OnceLock<Vec<Py<PyAny>>> = OnceLock::new();
static FP_CLEAR: OnceLock<(Py<PyCapsule>, unsafe extern "C" fn())> = OnceLock::new();
use treams_core::{
    Complex, Error,
    special::{self, Angular, Bessel},
};

// A single coordinate needs no generalized-ufunc shape resolution. Preserve the
// original loop for broadcasting, output options and ndarray subclass dispatch.
fn coordinate_input<T>(value: &Bound<'_, PyAny>, dim: usize) -> Option<[T; 3]>
where
    T: numpy::Element + Copy + Default + for<'a, 'py> FromPyObject<'a, 'py>,
{
    if let Ok(array) = value.cast_exact::<PyArray1<T>>() {
        let array = array.readonly();
        let values = array.as_slice().ok()?;
        if values.len() == dim {
            let mut result = [T::default(); 3];
            result.get_mut(..dim)?.copy_from_slice(values);
            return Some(result);
        }
    } else if value.is_exact_instance_of::<PyList>() || value.is_exact_instance_of::<PyTuple>() {
        return if dim == 2 {
            value
                .extract::<[T; 2]>()
                .ok()
                .map(|[a, b]| [a, b, T::default()])
        } else {
            value.extract::<[T; 3]>().ok()
        };
    }
    None
}

#[allow(clippy::indexing_slicing)] // The transform fixes the component dimension at two or three.
fn coordinate_call<'py, const KIND: u8>(
    py: Python<'py>,
    points: &Bound<'py, PyAny>,
    vector: Option<&Bound<'py, PyAny>>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let transform = coordinate_transform(KIND);
    let dim = transform.dimension();
    if args.is_empty()
        && kwargs.is_none_or(PyDictMethods::is_empty)
        && let Some(position) = coordinate_input::<f64>(points, dim)
    {
        if let Some(vector) = vector {
            if let Some(value) = coordinate_input::<f64>(vector, dim) {
                let result =
                    treams_core::coordinates::vector(value.map(Complex::from), position, transform)
                        .map_err(crate::error)?
                        .map(|x| x.re);
                return Ok(PyArray1::from_slice(py, &result[..dim]).into_any());
            }
            if let Some(value) = coordinate_input::<Complex>(vector, dim) {
                let result = treams_core::coordinates::vector(value, position, transform)
                    .map_err(crate::error)?;
                return Ok(PyArray1::from_slice(py, &result[..dim]).into_any());
            }
        } else {
            let result =
                treams_core::coordinates::point(position, transform).map_err(crate::error)?;
            return Ok(PyArray1::from_slice(py, &result[..dim]).into_any());
        }
    }
    let arguments = PyTuple::new(
        py,
        vector
            .into_iter()
            .chain(std::iter::once(points))
            .cloned()
            .chain(args.iter())
            .collect::<Vec<_>>(),
    )?;
    COORDINATES
        .get()
        .and_then(|functions| functions.get(2 * usize::from(KIND) + usize::from(vector.is_some())))
        .ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("coordinate ufunc is not initialized")
        })?
        .bind(py)
        .call(arguments, kwargs)
}

fn plane_wave_call<'py, const POL: u8>(
    py: Python<'py>,
    values: [&Bound<'py, PyAny>; 6],
    label: Option<&Bound<'py, PyAny>>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    if args.is_empty()
        && kwargs.is_none_or(PyDictMethods::is_empty)
        && label.is_none_or(PyAnyMethods::is_instance_of::<PyInt>)
        && let [Some(kx), Some(ky), Some(kz), Some(x), Some(y), Some(z)] = values.map(scalar_number)
        && x.im == 0.0
        && y.im == 0.0
        && z.im == 0.0
    {
        let pol = label.map_or(Ok(POL), |p| {
            polarization_label(p.extract::<c_long>()?).map_err(crate::error)
        })?;
        let polarization =
            treams_core::plane::polarization([kx, ky, kz], pol, POL == 2).map_err(crate::error)?;
        let result = treams_core::plane::field_value(polarization, [kx, ky, kz], [x, y, z])
            .map_err(crate::error)?;
        return Ok(PyArray1::from_slice(py, &result).into_any());
    }
    let arguments = PyTuple::new(
        py,
        values
            .into_iter()
            .chain(label)
            .cloned()
            .chain(args.iter())
            .collect::<Vec<_>>(),
    )?;
    PLANE_WAVES
        .get()
        .and_then(|functions| functions.get(usize::from(POL)))
        .ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("plane wave ufunc is not initialized")
        })?
        .bind(py)
        .call(arguments, kwargs)
}

#[allow(clippy::indexing_slicing)] // Shape is validated as a square matrix of dimension 1..=3.
fn cell_call<'py, const RECIPROCAL: bool>(
    py: Python<'py>,
    cell: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    if args.is_empty()
        && kwargs.is_none_or(PyDictMethods::is_empty)
        && let Ok(array) = cell.cast_exact::<PyArray2<f64>>()
        && array.is_c_contiguous()
        && let [dim, columns] = *array.shape()
        && dim == columns
        && (1..=3).contains(&dim)
    {
        let array = array.readonly();
        if let Ok(values) = array.as_slice() {
            let matrix = std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    if i < dim && j < dim {
                        values[i * dim + j]
                    } else {
                        0.0
                    }
                })
            });
            if RECIPROCAL {
                let result =
                    treams_core::geometry::reciprocal(matrix, dim).map_err(crate::error)?;
                return Ok(Array2::from_shape_fn((dim, dim), |(i, j)| result[i][j])
                    .into_pyarray(py)
                    .into_any());
            }
            return Ok(treams_core::geometry::volume(matrix, dim)
                .map_err(crate::error)?
                .into_pyobject(py)?
                .into_any());
        }
    }
    let arguments = PyTuple::new(
        py,
        std::iter::once(cell.clone())
            .chain(args.iter())
            .collect::<Vec<_>>(),
    )?;
    CELLS
        .get()
        .and_then(|functions| functions.get(usize::from(RECIPROCAL)))
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("cell ufunc is not initialized"))?
        .bind(py)
        .call(arguments, kwargs)
}

#[pyfunction]
fn cylindrical_rotation_scalar(
    kz: f64,
    mu: c_long,
    p: c_long,
    qz: f64,
    m: c_long,
    q: c_long,
    phi: f64,
) -> PyResult<Complex> {
    cylinder_rotation(kz, mu, p, qz, m, q, phi).map_err(crate::error)
}

#[pyfunction]
fn cylindrical_translation_scalar(
    kz: f64,
    mu: c_long,
    qz: f64,
    m: c_long,
    kr: Complex,
    phi: f64,
    z: f64,
    singular: bool,
) -> PyResult<Complex> {
    cylindrical_translation(
        kz,
        mu,
        qz,
        m,
        [kr, phi.into(), z.into()],
        if singular {
            special::Radial::Outgoing
        } else {
            special::Radial::Regular
        },
    )
    .map_err(crate::error)
}
#[pyfunction]
fn plane_permutation_scalar(
    kx: Complex,
    ky: Complex,
    kz: Complex,
    p: c_long,
    q: c_long,
    helicity: bool,
    inverse: bool,
) -> PyResult<Complex> {
    treams_core::plane::permutation_coefficient(
        [kx, ky, kz],
        polarization_label(p).map_err(crate::error)?,
        polarization_label(q).map_err(crate::error)?,
        if inverse { 2 } else { 1 },
        helicity,
    )
    .map_err(crate::error)
}

// Scalar Python numbers need no NumPy shape/type dispatch. Arrays and ufunc
// options still use the registered loop, including masks and overlapping output.
fn scalar_number(value: &Bound<'_, PyAny>) -> Option<Complex> {
    if let Ok(value) = value.cast::<PyFloat>() {
        Some(value.value().into())
    } else if let Ok(value) = value.cast::<PyComplex>() {
        Some(Complex::new(value.real(), value.imag()))
    } else if value.is_instance_of::<PyInt>() {
        value.extract::<f64>().ok().map(Into::into)
    } else {
        None
    }
}
#[pyfunction]
#[pyo3(signature=(kx, ky, kz, x, y, z, *args, **kwargs))]
fn pw_translate<'py>(
    py: Python<'py>,
    kx: &Bound<'py, PyAny>,
    ky: &Bound<'py, PyAny>,
    kz: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    z: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    if args.is_empty()
        && kwargs.is_none_or(PyDictMethods::is_empty)
        && let [Some(kx), Some(ky), Some(kz), Some(x), Some(y), Some(z)] =
            [kx, ky, kz, x, y, z].map(scalar_number)
        && x.im == 0.0
        && y.im == 0.0
        && z.im == 0.0
    {
        return Ok(
            treams_core::plane::translation([kx, ky, kz], [x.re, y.re, z.re])
                .map_err(crate::error)?
                .into_pyobject(py)?
                .into_any(),
        );
    }
    let operands = PyTuple::new(
        py,
        [kx, ky, kz, x, y, z]
            .into_iter()
            .cloned()
            .chain(args.iter())
            .collect::<Vec<_>>(),
    )?;
    PW_TRANSLATE
        .get()
        .ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("plane translation is not registered")
        })?
        .bind(py)
        .call(operands, kwargs)
}

unsafe extern "C" fn bessel_loop<const KIND: u8, const SPHERICAL: bool, const DERIVATIVE: u8>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    let kind = match KIND {
        0 => Bessel::J,
        1 => Bessel::Y,
        2 => Bessel::H1,
        _ => Bessel::H2,
    };
    // SAFETY: NumPy supplies three valid operands with the registered dD->D
    // layout, one nonnegative loop count and three byte strides. Loads/stores
    // allow unaligned arrays. NumPy handles cross-element input/output overlap;
    // each scalar iteration reads both operands before writing the result.
    // The parallel path finishes all reads into an owned buffer before writing.
    // No Python API or borrowed Python object enters the Rayon workers.
    let result = std::panic::catch_unwind(|| unsafe {
        let n = usize::try_from(*dimensions).unwrap_or_default();
        let order = *args;
        let argument = *args.add(1);
        let output = *args.add(2);
        let order_step = *steps;
        let argument_step = *steps.add(1);
        let output_step = *steps.add(2);
        if n >= 64
            && (order_step == 0 || order_step == 8)
            && (argument_step == 0 || argument_step == 16)
            && order.cast::<f64>().is_aligned()
            && argument.cast::<Complex>().is_aligned()
        {
            let orders = std::slice::from_raw_parts(
                order.cast::<f64>(),
                if order_step == 0 { 1 } else { n },
            );
            let arguments = std::slice::from_raw_parts(
                argument.cast::<Complex>(),
                if argument_step == 0 { 1 } else { n },
            );
            let values = special::bessel_values(orders, arguments, kind, SPHERICAL, DERIVATIVE)?;
            let mut out = output;
            for value in values {
                out.cast::<Complex>().write_unaligned(value);
                out = out.wrapping_offset(output_step);
            }
        } else {
            let (mut order, mut argument, mut output) = (order, argument, output);
            for _ in 0..n {
                let value = special::bessel(
                    order.cast::<f64>().read_unaligned(),
                    argument.cast::<Complex>().read_unaligned(),
                    kind,
                    SPHERICAL,
                    DERIVATIVE,
                )?;
                output.cast::<Complex>().write_unaligned(value);
                order = order.wrapping_offset(order_step);
                argument = argument.wrapping_offset(argument_step);
                output = output.wrapping_offset(output_step);
            }
        }
        Ok::<_, Error>(())
    });
    finish_loop(result);
}

#[inline]
fn finish_loop(result: std::thread::Result<Result<(), Error>>) {
    // The Bessel algorithm sets incidental flags for finite real-axis results.
    // Result and finite-value checks are authoritative. Clearing through the
    // cached C function avoids attaching Python on successful numerical calls.
    // SAFETY: register initializes FP_CLEAR before exposing any callback and
    // retains the capsule. Slot 27 clears only this thread's floating-point flags.
    unsafe {
        (FP_CLEAR.get().unwrap_unchecked().1)();
    }
    report_loop(result);
}

#[inline]
fn report_loop(result: std::thread::Result<Result<(), Error>>) {
    // A raw C callback must not unwind across the interpreter. As in PyO3's
    // function wrappers, report failures with a Python exception on this thread.
    let result = result.unwrap_or_else(|_| {
        Err(Error::SpecialFunction(
            "native special-function loop panicked".into(),
        ))
    });
    if let Err(error) = result {
        Python::attach(|py| crate::error(error).restore(py));
    }
}

unsafe extern "C" fn angular_loop<const KIND: u8>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    let kind = match KIND {
        0 => Angular::Legendre,
        1 => Angular::Pi,
        _ => Angular::Tau,
    };
    // SAFETY: NumPy provides four operands with the registered ddD->D layout.
    // The strides/count are valid; unaligned scalar reads precede writes. The
    // borrowed parallel path completes its reads before any output is written.
    let result = std::panic::catch_unwind(|| unsafe {
        let n = usize::try_from(*dimensions).unwrap_or_default();
        let (a, b) = if KIND == 0 { (1, 0) } else { (0, 1) };
        let (mut degree, mut order, mut argument, mut output) =
            (*args.add(a), *args.add(b), *args.add(2), *args.add(3));
        let (ds, ms, zs, os) = (*steps.add(a), *steps.add(b), *steps.add(2), *steps.add(3));
        if n >= 1024
            && (ds == 0 || ds == 8)
            && (ms == 0 || ms == 8)
            && (zs == 0 || zs == 16)
            && degree.cast::<f64>().is_aligned()
            && order.cast::<f64>().is_aligned()
            && argument.cast::<Complex>().is_aligned()
        {
            let degrees =
                std::slice::from_raw_parts(degree.cast::<f64>(), if ds == 0 { 1 } else { n });
            let orders =
                std::slice::from_raw_parts(order.cast::<f64>(), if ms == 0 { 1 } else { n });
            let arguments =
                std::slice::from_raw_parts(argument.cast::<Complex>(), if zs == 0 { 1 } else { n });
            for value in special::angular_values(degrees, orders, arguments, kind)? {
                output.cast::<Complex>().write_unaligned(value);
                output = output.wrapping_offset(os);
            }
        } else {
            for _ in 0..n {
                let value = special::angular_value(
                    degree.cast::<f64>().read_unaligned(),
                    order.cast::<f64>().read_unaligned(),
                    argument.cast::<Complex>().read_unaligned(),
                    kind,
                )?;
                output.cast::<Complex>().write_unaligned(value);
                degree = degree.wrapping_offset(ds);
                order = order.wrapping_offset(ms);
                argument = argument.wrapping_offset(zs);
                output = output.wrapping_offset(os);
            }
        }
        Ok::<_, Error>(())
    });
    finish_loop(result);
}

// Read-only views exist only while a NumPy callback is active. Parallel kernels
// collect owned output before any store, so even aliased inputs remain unchanged.
struct Input {
    pointer: *const c_char,
    stride: npy_intp,
}
// SAFETY: NumPy holds the input buffers for the callback's lifetime; workers only
// read them and the blocking Rayon collection finishes before output is written.
unsafe impl Sync for Input {}
impl Input {
    unsafe fn read<T: Copy>(&self, i: usize) -> T {
        // SAFETY: The registration fixes T to the operand dtype; NumPy provides
        // valid strides and i is less than its nonnegative npy_intp loop count.
        unsafe {
            self.pointer
                .wrapping_offset(
                    self.stride
                        .wrapping_mul(isize::try_from(i).unwrap_or_default()),
                )
                .cast::<T>()
                .read_unaligned()
        }
    }
}

// Scalar and three-component kernels share the outer NumPy iteration contract.
// The output type determines whether NumPy supplies a component stride.
trait LoopOutput: Copy + Send {
    const VECTOR: bool;
    unsafe fn store(self, pointer: *mut c_char, component_stride: npy_intp);
}
impl LoopOutput for f64 {
    const VECTOR: bool = false;
    #[inline]
    unsafe fn store(self, pointer: *mut c_char, _: npy_intp) {
        // SAFETY: The callback registration fixes a writable double output.
        unsafe {
            pointer.cast::<Self>().write_unaligned(self);
        }
    }
}
impl LoopOutput for Complex {
    const VECTOR: bool = false;
    #[inline]
    unsafe fn store(self, pointer: *mut c_char, _: npy_intp) {
        // SAFETY: The callback registration fixes a writable complex output.
        unsafe {
            pointer.cast::<Self>().write_unaligned(self);
        }
    }
}
impl<const N: usize> LoopOutput for [Complex; N] {
    const VECTOR: bool = true;
    #[inline]
    unsafe fn store(self, pointer: *mut c_char, component_stride: npy_intp) {
        // SAFETY: The gufunc signature guarantees N complex components and
        // the supplied byte stride. Values were copied before any output store.
        unsafe {
            for (i, value) in self.into_iter().enumerate() {
                pointer
                    .wrapping_offset(
                        component_stride.wrapping_mul(isize::try_from(i).unwrap_or_default()),
                    )
                    .cast::<Complex>()
                    .write_unaligned(value);
            }
        }
    }
}

impl LoopOutput for [f64; 2] {
    const VECTOR: bool = true;
    unsafe fn store(self, pointer: *mut c_char, component_stride: npy_intp) {
        // SAFETY: The registered output has two double components with the supplied stride.
        unsafe {
            for (i, value) in self.into_iter().enumerate() {
                pointer
                    .wrapping_offset(
                        component_stride.wrapping_mul(isize::try_from(i).unwrap_or_default()),
                    )
                    .cast::<f64>()
                    .write_unaligned(value);
            }
        }
    }
}

// Fixed arity scalar kernels share NumPy's masking, buffering and strided loop
// contract. Every input is copied before writing, including in-place operations.
macro_rules! scalar_loop {
    ($name:ident $(<$t:ident>)?, $output:ty, $count:literal, $( $index:literal => $argument:ident : $ty:ty ),+ => $body:expr) => {
        scalar_loop!(@1024, $name $(<$t>)?, $output, $count, $($index => $argument : $ty),+ => $body);
    };
    (@$parallel:expr, $name:ident $(<$t:ident>)?, $output:ty, $count:literal, $( $index:literal => $argument:ident : $ty:ty ),+ => $body:expr) => {
        unsafe extern "C" fn $name $(<$t: Into<Complex> + Copy + Send + Sync>)? (args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, _data: *mut c_void) {
            // SAFETY: register supplies this exact operand signature, and NumPy
            // provides valid byte strides/counts and buffers overlapping arrays.
            // Scalar copies permit unaligned storage and all reads precede writes.
            let result = std::panic::catch_unwind(|| unsafe {
                let n = usize::try_from(*dimensions).unwrap_or_default();
                let component_stride=if <$output as LoopOutput>::VECTOR {*steps.add($count+1)}else{0};
                let mut pointers: [*mut c_char; $count + 1] = std::array::from_fn(|i| *args.add(i));
                if n >= $parallel {
                    let inputs: [Input; $count] = std::array::from_fn(|i|Input{pointer:*args.add(i),stride:*steps.add(i)});
                    let evaluate=|i| {
                        $(let $argument=inputs.get_unchecked($index).read::<$ty>(i);)+
                        $body
                    };
                    let values: Vec<$output>=(0..n).into_par_iter().map(evaluate).collect::<treams_core::Result<_>>()?;
                    let mut output=*args.add($count);
                    for value in values {
                        value.store(output,component_stride);
                        output=output.wrapping_offset(*steps.add($count));
                    }
                    return Ok(());
                }
                for _ in 0..n {
                    $(let $argument = pointers.get_unchecked($index).cast::<$ty>().read_unaligned();)+
                    let value: $output = ($body)?;
                    value.store(*pointers.get_unchecked($count),component_stride);
                    for (i,pointer) in pointers.iter_mut().enumerate() {
                        *pointer = pointer.wrapping_offset(*steps.add(i));
                    }
                }
                Ok::<_, Error>(())
            });
            finish_loop(result);
        }
    };
}

#[allow(clippy::cast_possible_truncation)] // Checked integral labels within the supported domain.
fn label(value: f64) -> treams_core::Result<i32> {
    if value.fract() != 0.0 || !(-260.0..=260.0).contains(&value) {
        return Err(Error::InvalidInput(
            "special-function labels must be integers in [-260, 260]".into(),
        ));
    }
    Ok(value as i32)
}

scalar_loop!(legendre_real_loop, f64, 3, 0=>m:f64, 1=>l:f64, 2=>x:f64 => {
    special::angular_value(l, m, x.into(), Angular::Legendre).map(|v| v.re)
});
scalar_loop!(wigner_small_loop, Complex, 4, 0=>l:f64, 1=>m:f64, 2=>k:f64, 3=>theta:Complex => {
    treams_core::rotation::wigner_small(label(l)?,label(m)?,label(k)?,theta)
});
scalar_loop!(wigner_loop, Complex, 6, 0=>l:f64, 1=>m:f64, 2=>k:f64, 3=>phi:Complex, 4=>theta:Complex, 5=>psi:Complex => {
    treams_core::rotation::wigner(label(l)?,label(m)?,label(k)?,[phi,theta,psi])
});
scalar_loop!(wigner3j_loop, f64, 6, 0=>l1:f64, 1=>l2:f64, 2=>l3:f64, 3=>m1:f64, 4=>m2:f64, 5=>m3:f64 => {
    Ok(treams_core::angular::wigner3j(label(l1)?,label(l2)?,label(l3)?,label(m1)?,label(m2)?,label(m3)?))
});
scalar_loop!(gamma_loop, Complex, 2, 0=>n:f64, 1=>z:Complex => treams_core::integrals::incgamma(n,z));
scalar_loop!(kambe_loop, Complex, 3, 0=>n:f64, 1=>z:Complex, 2=>eta:Complex => treams_core::integrals::intkambe(label(n)?,z,eta));

scalar_loop!(@usize::MAX, refractive_indices_loop, [Complex;2], 3,0=>epsilon:Complex,1=>mu:Complex,2=>kappa:Complex=>Ok(treams_core::coeffs::Material{epsilon,mu,kappa}.indices()));
scalar_loop!(@usize::MAX, refractive_indices_real_loop, [f64;2], 3,0=>epsilon:f64,1=>mu:f64,2=>kappa:f64=>{let n=(epsilon*mu).sqrt();Ok([n-kappa,n+kappa])});
scalar_loop!(@usize::MAX, wave_vector_z_loop<T>, Complex, 3,0=>kx:T,1=>ky:T,2=>k:T=>Ok(treams_core::plane::wave_vector_z(kx.into(),ky.into(),k.into())));
scalar_loop!(first_brillouin_1d_loop, f64, 2,0=>k:f64,1=>b:f64=>treams_core::geometry::first_brillouin_1d(k,b));

#[inline]
unsafe fn cell_input<T: Copy + Default>(
    pointer: *mut c_char,
    steps: *mut npy_intp,
    dim: usize,
) -> [[T; 3]; 3] {
    // SAFETY: The caller validated dimension <=3 and the registered (i,i) input.
    // Core strides support transposes, reversed and unaligned inputs.
    unsafe {
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i < dim && j < dim {
                    pointer
                        .wrapping_offset(
                            (*steps.add(2)).wrapping_mul(isize::try_from(i).unwrap_or_default())
                                + (*steps.add(3))
                                    .wrapping_mul(isize::try_from(j).unwrap_or_default()),
                        )
                        .cast::<T>()
                        .read_unaligned()
                } else {
                    T::default()
                }
            })
        })
    }
}
unsafe extern "C" fn volume_loop<
    T: Copy
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>,
>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    // SAFETY: Registered double or transparent Wrapping<C-long> buffers use
    // (i,i)->(). NumPy owns broadcasting and overlap buffering; each matrix is
    // fully copied before any output write. Panics never cross the C callback.
    let result = std::panic::catch_unwind(|| unsafe {
        let n = usize::try_from(*dimensions).unwrap_or_default();
        let dim = usize::try_from(*dimensions.add(1)).unwrap_or_default();
        if !(1..=3).contains(&dim) {
            return Err(Error::InvalidInput(
                "cell dimension must be 1, 2 or 3".into(),
            ));
        }
        let mut input = *args;
        let mut output = *args.add(1);
        for _ in 0..n {
            let value = match dim {
                1 => treams_core::geometry::volume(cell_input::<T>(input, steps, 1), 1)?,
                2 => treams_core::geometry::volume(cell_input::<T>(input, steps, 2), 2)?,
                _ => treams_core::geometry::volume(cell_input::<T>(input, steps, 3), 3)?,
            };
            output.cast::<T>().write_unaligned(value);
            input = input.wrapping_offset(*steps);
            output = output.wrapping_offset(*steps.add(1));
        }
        Ok(())
    });
    report_loop(result);
}
unsafe extern "C" fn reciprocal_loop(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    // SAFETY: Registered (i,i)->(i,i) double arrays provide two core strides per
    // operand. All input values are copied before output, including in-place use.
    let result = std::panic::catch_unwind(|| unsafe {
        let n = usize::try_from(*dimensions).unwrap_or_default();
        let dim = usize::try_from(*dimensions.add(1)).unwrap_or_default();
        if !(1..=3).contains(&dim) {
            return Err(Error::InvalidInput(
                "cell dimension must be 1, 2 or 3".into(),
            ));
        }
        let mut input = *args;
        let mut output = *args.add(1);
        for _ in 0..n {
            let value =
                treams_core::geometry::reciprocal(cell_input::<f64>(input, steps, dim), dim)?;
            for (i, row) in value.iter().enumerate().take(dim) {
                for (j, &v) in row.iter().enumerate().take(dim) {
                    output
                        .wrapping_offset(
                            (*steps.add(4)).wrapping_mul(isize::try_from(i).unwrap_or_default())
                                + (*steps.add(5))
                                    .wrapping_mul(isize::try_from(j).unwrap_or_default()),
                        )
                        .cast::<f64>()
                        .write_unaligned(v);
                }
            }
            input = input.wrapping_offset(*steps);
            output = output.wrapping_offset(*steps.add(1));
        }
        Ok(())
    });
    report_loop(result);
}

unsafe extern "C" fn lattice_loop<
    const SPHERICAL: bool,
    const DIM: usize,
    const SHIFTED: bool,
    const PART: u8,
    const INTEGER: bool,
>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    // SAFETY: Registration fixes all scalar dtypes and vector/matrix dimensions.
    // NumPy supplies valid byte strides and buffers overlapping operands. Each
    // evaluation copies inputs; parallel output is collected before any stores.
    let result = std::panic::catch_unwind(|| unsafe {
        use treams_core::lattice::{self, Lattice, SumPart, Wave};
        let has_order = SPHERICAL && (DIM != 1 || SHIFTED);
        let k_index = 1 + usize::from(has_order);
        let count = k_index + 5;
        let n = usize::try_from(*dimensions).unwrap_or_default();
        // At most seven inputs; a six-input loop has an unused output
        // descriptor in the last slot. No heap allocation on scalar calls.
        let inputs: [Input; 7] = std::array::from_fn(|j| Input {
            pointer: *args.add(j),
            stride: *steps.add(j),
        });
        let strides: [npy_intp; 4] = std::array::from_fn(|j| {
            if j < (if DIM == 1 { usize::from(SHIFTED) } else { 4 }) {
                *steps.add(count + 1 + j)
            } else {
                0
            }
        });
        let component = |operand: usize, index: usize, offset: npy_intp| {
            let input = &inputs[operand];
            input
                .pointer
                .wrapping_offset(
                    input
                        .stride
                        .wrapping_mul(isize::try_from(index).unwrap_or_default())
                        + offset,
                )
                .cast::<f64>()
                .read_unaligned()
        };
        let geometry = |i| {
            let q = std::array::from_fn(|j| {
                if j < DIM {
                    component(
                        k_index + 1,
                        i,
                        isize::try_from(j).unwrap_or_default() * strides[0],
                    )
                } else {
                    0.0
                }
            });
            let a = std::array::from_fn(|j| {
                std::array::from_fn(|h| {
                    if j < DIM && h < DIM {
                        component(
                            k_index + 2,
                            i,
                            if DIM == 1 {
                                0
                            } else {
                                isize::try_from(j).unwrap_or_default() * strides[1]
                                    + isize::try_from(h).unwrap_or_default() * strides[2]
                            },
                        )
                    } else {
                        0.0
                    }
                })
            });
            Lattice::from_array(a, q, DIM)
        };
        let fixed = if n > 0 && inputs[k_index + 1].stride == 0 && inputs[k_index + 2].stride == 0 {
            Some(geometry(0)?)
        } else {
            None
        };
        let mode_label = |operand: usize, i| {
            if INTEGER {
                i32::try_from(inputs[operand].read::<c_long>(i))
                    .map_err(|_| Error::InvalidInput("lattice mode exceeds i32 range".into()))
            } else {
                label(inputs[operand].read::<f64>(i))
            }
        };
        let evaluate = |i| {
            let first = mode_label(0, i)?;
            let wave = if SPHERICAL {
                Wave::Spherical {
                    l: first,
                    m: if has_order { mode_label(1, i)? } else { 0 },
                }
            } else {
                Wave::Cylindrical { m: first }
            };
            let varying;
            let lattice = if let Some(ref lattice) = fixed {
                lattice
            } else {
                varying = geometry(i)?;
                &varying
            };
            let mut r = [0.0; 3];
            if DIM == 1 && !SHIFTED {
                r[if SPHERICAL { 2 } else { 0 }] = component(k_index + 3, i, 0);
            } else {
                let stride = strides[if DIM == 1 { 0 } else { 3 }];
                for (j, r) in r
                    .iter_mut()
                    .enumerate()
                    .take(if SPHERICAL && (DIM == 3 || SHIFTED) {
                        3
                    } else {
                        2
                    })
                {
                    *r = component(
                        k_index + 3,
                        i,
                        isize::try_from(j).unwrap_or_default() * stride,
                    );
                }
            }
            let (eta, part) = match PART {
                3 => (
                    Complex::default(),
                    SumPart::Direct(inputs[k_index + 4].read::<c_long>(i)),
                ),
                _ => (
                    inputs[k_index + 4].read::<Complex>(i),
                    match PART {
                        0 => SumPart::Full,
                        1 => SumPart::Real,
                        _ => SumPart::Reciprocal,
                    },
                ),
            };
            lattice::sum_part(
                wave,
                inputs[k_index].read::<Complex>(i),
                lattice,
                r,
                eta,
                part,
            )
        };
        let mut output = *args.add(count);
        if n >= 8 {
            let values: Vec<Complex> = (0..n)
                .into_par_iter()
                .map(evaluate)
                .collect::<treams_core::Result<_>>()?;
            for value in values {
                output.cast::<Complex>().write_unaligned(value);
                output = output.wrapping_offset(*steps.add(count));
            }
        } else {
            for i in 0..n {
                let value = evaluate(i)?;
                output.cast::<Complex>().write_unaligned(value);
                output = output.wrapping_offset(*steps.add(count));
            }
        }
        Ok(())
    });
    finish_loop(result);
}

fn wave_mode(l: f64, m: f64, pol: f64) -> treams_core::Result<treams_core::waves::Mode> {
    let pol = match label(pol)? {
        0 => 0,
        1 => 1,
        _ => return Err(Error::InvalidInput("polarization must be 0 or 1".into())),
    };
    Ok(treams_core::waves::Mode {
        l: label(l)?,
        m: label(m)?,
        pol,
    })
}
scalar_loop!(sph_harm_loop<T>, Complex, 4, 0=>m:f64, 1=>l:f64, 2=>phi:f64, 3=>theta:T => {
    treams_core::vectorwaves::value(treams_core::vectorwaves::Family::HarmonicZ,wave_mode(l,m,0.0)?,[theta.into(),phi.into(),Complex::default(),Complex::default(),Complex::default(),Complex::default()],false).map(|v|-Complex::i()*v[0])
});
fn integer_wave_mode(
    l: c_long,
    m: c_long,
    pol: c_long,
) -> treams_core::Result<treams_core::waves::Mode> {
    let invalid = || Error::InvalidInput("invalid integer wave labels".into());
    Ok(treams_core::waves::Mode {
        l: i32::try_from(l).map_err(|_| invalid())?,
        m: i32::try_from(m).map_err(|_| invalid())?,
        pol: u8::try_from(pol).map_err(|_| invalid())?,
    })
}
scalar_loop!(vsh_x_loop<T>, [Complex;3], 4, 0=>l:c_long, 1=>m:c_long, 2=>theta:T, 3=>phi:f64 => {
    use treams_core::vectorwaves::{self,Family};
    vectorwaves::value(Family::HarmonicX,integer_wave_mode(l,m,0)?,[theta.into(),phi.into(),Complex::default(),Complex::default(),Complex::default(),Complex::default()],false)
});
scalar_loop!(vsh_y_loop<T>, [Complex;3], 4, 0=>l:c_long, 1=>m:c_long, 2=>theta:T, 3=>phi:f64 => {
    use treams_core::vectorwaves::{self,Family};
    vectorwaves::value(Family::HarmonicY,integer_wave_mode(l,m,0)?,[theta.into(),phi.into(),Complex::default(),Complex::default(),Complex::default(),Complex::default()],false)
});
scalar_loop!(vsh_z_loop<T>, [Complex;3], 4, 0=>l:c_long, 1=>m:c_long, 2=>theta:T, 3=>phi:f64 => {
    use treams_core::vectorwaves::{self,Family};
    vectorwaves::value(Family::HarmonicZ,integer_wave_mode(l,m,0)?,[theta.into(),phi.into(),Complex::default(),Complex::default(),Complex::default(),Complex::default()],false)
});
scalar_loop!(vsw_m_loop<T>, [Complex;3], 5, 0=>l:c_long, 1=>m:c_long, 2=>kr:Complex, 3=>theta:T, 4=>phi:f64 => {
    use treams_core::{vectorwaves::{self,Family},special::Radial};
    vectorwaves::value(Family::Spherical(Radial::Outgoing),integer_wave_mode(l,m,0)?,[kr,theta.into(),phi.into(),Complex::default(),Complex::default(),Complex::default()],false)
});
scalar_loop!(vsw_n_loop<T>, [Complex;3], 5, 0=>l:c_long, 1=>m:c_long, 2=>kr:Complex, 3=>theta:T, 4=>phi:f64 => {
    use treams_core::{vectorwaves::{self,Family},special::Radial};
    vectorwaves::value(Family::Spherical(Radial::Outgoing),integer_wave_mode(l,m,1)?,[kr,theta.into(),phi.into(),Complex::default(),Complex::default(),Complex::default()],false)
});
scalar_loop!(vsw_a_loop<T>, [Complex;3], 6, 0=>l:c_long, 1=>m:c_long, 2=>kr:Complex, 3=>theta:T, 4=>phi:f64, 5=>pol:c_long => {
    use treams_core::{vectorwaves::{self,Family},special::Radial};
    vectorwaves::value(Family::Spherical(Radial::Outgoing),integer_wave_mode(l,m,pol)?,[kr,theta.into(),phi.into(),Complex::default(),Complex::default(),Complex::default()],true)
});
scalar_loop!(vsw_rm_loop<T>, [Complex;3], 5, 0=>l:c_long, 1=>m:c_long, 2=>kr:Complex, 3=>theta:T, 4=>phi:f64 => {
    use treams_core::{vectorwaves::{self,Family},special::Radial};
    vectorwaves::value(Family::Spherical(Radial::Regular),integer_wave_mode(l,m,0)?,[kr,theta.into(),phi.into(),Complex::default(),Complex::default(),Complex::default()],false)
});
scalar_loop!(vsw_rn_loop<T>, [Complex;3], 5, 0=>l:c_long, 1=>m:c_long, 2=>kr:Complex, 3=>theta:T, 4=>phi:f64 => {
    use treams_core::{vectorwaves::{self,Family},special::Radial};
    vectorwaves::value(Family::Spherical(Radial::Regular),integer_wave_mode(l,m,1)?,[kr,theta.into(),phi.into(),Complex::default(),Complex::default(),Complex::default()],false)
});
scalar_loop!(vsw_ra_loop<T>, [Complex;3], 6, 0=>l:c_long, 1=>m:c_long, 2=>kr:Complex, 3=>theta:T, 4=>phi:f64, 5=>pol:c_long => {
    use treams_core::{vectorwaves::{self,Family},special::Radial};
    vectorwaves::value(Family::Spherical(Radial::Regular),integer_wave_mode(l,m,pol)?,[kr,theta.into(),phi.into(),Complex::default(),Complex::default(),Complex::default()],true)
});
scalar_loop!(vcw_m_loop, [Complex;3], 5, 0=>kz:f64, 1=>m:c_long, 2=>kr:Complex, 3=>phi:f64, 4=>z:f64 => {
    use treams_core::{vectorwaves::{self,Family},special::Radial};
    vectorwaves::value(Family::Cylindrical(Radial::Outgoing),integer_wave_mode(0,m,0)?,[kz.into(),kr,phi.into(),z.into(),Complex::default(),Complex::default()],false)
});
scalar_loop!(vcw_n_loop, [Complex;3], 6, 0=>kz:f64, 1=>m:c_long, 2=>kr:Complex, 3=>phi:f64, 4=>z:f64, 5=>k:Complex => {
    use treams_core::{vectorwaves::{self,Family},special::Radial};
    vectorwaves::value(Family::Cylindrical(Radial::Outgoing),integer_wave_mode(0,m,1)?,[kz.into(),kr,phi.into(),z.into(),k,Complex::default()],false)
});
scalar_loop!(vcw_a_loop, [Complex;3], 7, 0=>kz:f64, 1=>m:c_long, 2=>kr:Complex, 3=>phi:f64, 4=>z:f64, 5=>k:Complex, 6=>pol:c_long => {
    use treams_core::{vectorwaves::{self,Family},special::Radial};
    vectorwaves::value(Family::Cylindrical(Radial::Outgoing),integer_wave_mode(0,m,pol)?,[kz.into(),kr,phi.into(),z.into(),k,Complex::default()],true)
});
scalar_loop!(vcw_rm_loop, [Complex;3], 5, 0=>kz:f64, 1=>m:c_long, 2=>kr:Complex, 3=>phi:f64, 4=>z:f64 => {
    use treams_core::{vectorwaves::{self,Family},special::Radial};
    vectorwaves::value(Family::Cylindrical(Radial::Regular),integer_wave_mode(0,m,0)?,[kz.into(),kr,phi.into(),z.into(),Complex::default(),Complex::default()],false)
});
scalar_loop!(vcw_rn_loop, [Complex;3], 6, 0=>kz:f64, 1=>m:c_long, 2=>kr:Complex, 3=>phi:f64, 4=>z:f64, 5=>k:Complex => {
    use treams_core::{vectorwaves::{self,Family},special::Radial};
    vectorwaves::value(Family::Cylindrical(Radial::Regular),integer_wave_mode(0,m,1)?,[kz.into(),kr,phi.into(),z.into(),k,Complex::default()],false)
});
scalar_loop!(vcw_ra_loop, [Complex;3], 7, 0=>kz:f64, 1=>m:c_long, 2=>kr:Complex, 3=>phi:f64, 4=>z:f64, 5=>k:Complex, 6=>pol:c_long => {
    use treams_core::{vectorwaves::{self,Family},special::Radial};
    vectorwaves::value(Family::Cylindrical(Radial::Regular),integer_wave_mode(0,m,pol)?,[kz.into(),kr,phi.into(),z.into(),k,Complex::default()],true)
});

unsafe extern "C" fn plane_wave_loop<T: Into<Complex> + Copy + Send + Sync, const POL: u8>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    // SAFETY: The real/complex registrations have six scalar operands plus an
    // optional long polarization, and a three-complex-component output. NumPy
    // supplies the complete outer/core strides and buffers overlapping operands.
    let result = std::panic::catch_unwind(|| unsafe {
        let n = usize::try_from(*dimensions).unwrap_or_default();
        if n == 0 {
            return Ok(());
        }
        let count = if POL == 2 { 7 } else { 6 };
        let inputs: [Input; 6] = std::array::from_fn(|i| Input {
            pointer: *args.add(i),
            stride: *steps.add(i),
        });
        let polarization = |i| -> treams_core::Result<u8> {
            if POL != 2 {
                return Ok(POL);
            }
            let p = Input {
                pointer: *args.add(6),
                stride: *steps.add(6),
            }
            .read::<c_long>(i);
            u8::try_from(p).map_err(|_| Error::InvalidInput("invalid polarization".into()))
        };
        let fixed =
            (n > 1 && (0..3).all(|i| *steps.add(i) == 0) && (POL != 2 || *steps.add(6) == 0))
                .then(|| {
                    treams_core::plane::polarization(
                        std::array::from_fn(|i| inputs[i].read::<T>(0).into()),
                        polarization(0)?,
                        POL == 2,
                    )
                })
                .transpose()?;
        // Copy the optional label input before entering workers; raw pointers
        // are accessed only through Input's documented disjoint-read contract.
        let label = if POL == 2 {
            Some(Input {
                pointer: *args.add(6),
                stride: *steps.add(6),
            })
        } else {
            None
        };
        let evaluate = |i| {
            let k = std::array::from_fn(|a| inputs[a].read::<T>(i).into());
            let position = std::array::from_fn(|a| Complex::from(inputs[a + 3].read::<f64>(i)));
            let p = if let Some(p) = fixed {
                p
            } else {
                let pol = match &label {
                    Some(label) => u8::try_from(label.read::<c_long>(i))
                        .map_err(|_| Error::InvalidInput("invalid polarization".into()))?,
                    None => POL,
                };
                treams_core::plane::polarization(k, pol, POL == 2)?
            };
            treams_core::plane::field_value(p, k, position)
        };
        let mut output = *args.add(count);
        let stride = *steps.add(count + 1);
        if n >= 1024 {
            let values: Vec<_> = (0..n)
                .into_par_iter()
                .map(evaluate)
                .collect::<treams_core::Result<_>>()?;
            for v in values {
                v.store(output, stride);
                output = output.wrapping_offset(*steps.add(count));
            }
        } else {
            for i in 0..n {
                evaluate(i)?.store(output, stride);
                output = output.wrapping_offset(*steps.add(count));
            }
        }
        Ok(())
    });
    report_loop(result);
}

#[allow(clippy::float_cmp)] // Axial wave labels use exact equality, as in basis expansions.
fn cylindrical_translation(
    kz: f64,
    mu: c_long,
    qz: f64,
    m: c_long,
    args: [Complex; 3],
    radial: special::Radial,
) -> treams_core::Result<Complex> {
    if !kz.is_finite() || !qz.is_finite() || mu.unsigned_abs() > 128 || m.unsigned_abs() > 128 {
        return Err(Error::InvalidInput(
            "finite axial labels and |orders| <= 128 required".into(),
        ));
    }
    if kz != qz {
        return Ok(Complex::default());
    }
    let order = i32::try_from(m - mu)
        .map_err(|_| Error::InvalidInput("invalid cylindrical order".into()))?;
    treams_core::polar::cylindrical_value(order, [args[0], args[1], args[2], kz.into()], radial)
}
scalar_loop!(@64, tl_vcw_loop, Complex, 7, 0=>kz:f64, 1=>mu:c_long, 2=>qz:f64, 3=>m:c_long, 4=>kr:Complex, 5=>phi:f64, 6=>z:f64 => {
    cylindrical_translation(kz,mu,qz,m,[kr,phi.into(),z.into()],special::Radial::Outgoing)
});
scalar_loop!(@64, tl_vcw_r_loop<T>, Complex, 7, 0=>kz:f64, 1=>mu:c_long, 2=>qz:f64, 3=>m:c_long, 4=>kr:T, 5=>phi:f64, 6=>z:f64 => {
    cylindrical_translation(kz,mu,qz,m,[kr.into(),phi.into(),z.into()],special::Radial::Regular)
});

unsafe extern "C" fn polar_translation_loop<
    T: Into<Complex> + Copy + Send + Sync,
    const KIND: u8,
    const REGULAR: bool,
    const ARGS: usize,
>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    // SAFETY: Seven operands for special coefficients, nine for polarized
    // coefficients. Registrations use C-long labels and the documented real/
    // complex argument positions, feeding one complex scalar output. NumPy supplies the matching strides and
    // handles overlaps. Every row is read before writing; workers finish before
    // buffered output is stored, preserving noncontiguous and unaligned arrays.
    let result = std::panic::catch_unwind(|| unsafe {
        let n = usize::try_from(*dimensions).unwrap_or_default();
        if n == 0 {
            return Ok(());
        }
        let inputs: [Input; ARGS] = std::array::from_fn(|i| Input {
            pointer: *args.add(i),
            stride: *steps.add(i),
        });
        let labels = ARGS - 3;
        let prepare = |i| {
            let (to, from) = if ARGS == 9 {
                (
                    integer_wave_mode(
                        inputs[0].read::<c_long>(i),
                        inputs[1].read::<c_long>(i),
                        inputs[2].read::<c_long>(i),
                    )?,
                    integer_wave_mode(
                        inputs[3].read::<c_long>(i),
                        inputs[4].read::<c_long>(i),
                        inputs[5].read::<c_long>(i),
                    )?,
                )
            } else {
                (
                    integer_wave_mode(inputs[0].read::<c_long>(i), inputs[1].read::<c_long>(i), 0)?,
                    integer_wave_mode(
                        inputs[2].read::<c_long>(i),
                        inputs[3].read::<c_long>(i),
                        c_long::from(KIND),
                    )?,
                )
            };
            treams_core::polar::SphericalTranslation::new(
                to,
                from,
                KIND == 3,
                if REGULAR {
                    special::Radial::Regular
                } else {
                    special::Radial::Outgoing
                },
            )
        };
        let plan = if n > 1 && (0..labels).all(|i| *steps.add(i) == 0) {
            Some(prepare(0)?)
        } else {
            None
        };
        let evaluate = |i| {
            let arguments = if ARGS == 9 {
                [
                    inputs[6].read::<T>(i).into(),
                    inputs[7].read::<f64>(i).into(),
                    inputs[8].read::<f64>(i).into(),
                ]
            } else {
                [
                    inputs[4].read::<Complex>(i),
                    inputs[5].read::<T>(i).into(),
                    inputs[6].read::<f64>(i).into(),
                ]
            };
            if ARGS == 9 && !REGULAR && arguments[0].norm() < 1e-16 {
                return Ok(Complex::default());
            }
            match &plan {
                Some(p) => p.value(arguments),
                None => prepare(i)?.value(arguments),
            }
        };
        let mut output = *args.add(ARGS);
        if n >= 1024 {
            let values: Vec<_> = (0..n)
                .into_par_iter()
                .map(evaluate)
                .collect::<treams_core::Result<_>>()?;
            for v in values {
                output.cast::<Complex>().write_unaligned(v);
                output = output.wrapping_offset(*steps.add(ARGS));
            }
        } else {
            for i in 0..n {
                output.cast::<Complex>().write_unaligned(evaluate(i)?);
                output = output.wrapping_offset(*steps.add(ARGS));
            }
        }
        Ok(())
    });
    finish_loop(result);
}

fn polarization_label(p: c_long) -> treams_core::Result<u8> {
    match p {
        0 => Ok(0),
        1 => Ok(1),
        _ => Err(Error::InvalidInput("polarization must be 0 or 1".into())),
    }
}
scalar_loop!(sw_rotate_loop, Complex, 9, 0=>lambda:c_long, 1=>mu:c_long, 2=>p:c_long, 3=>l:c_long, 4=>m:c_long, 5=>q:c_long, 6=>phi:f64, 7=>theta:f64, 8=>psi:f64 => {
    let to=integer_wave_mode(lambda,mu,p)?;let from=integer_wave_mode(l,m,q)?;
    to.validate()?;from.validate()?;
    if to.l!=from.l || to.pol!=from.pol {Ok(Complex::default())}else{treams_core::rotation::wigner(to.l,to.m,from.m,[phi.into(),theta.into(),psi.into()])}
});
#[allow(clippy::float_cmp)] // Exact axial labels select cylindrical coefficients.
fn cylinder_rotation(
    kz: f64,
    mu: c_long,
    p: c_long,
    qz: f64,
    m: c_long,
    q: c_long,
    phi: f64,
) -> treams_core::Result<Complex> {
    let p = polarization_label(p)?;
    let q = polarization_label(q)?;
    if !kz.is_finite()
        || !qz.is_finite()
        || !phi.is_finite()
        || mu.unsigned_abs() > 128
        || m.unsigned_abs() > 128
    {
        return Err(Error::InvalidInput(
            "finite axial/angle arguments and |orders| <= 128 required".into(),
        ));
    }
    if kz != qz || mu != m || p != q {
        Ok(Complex::default())
    } else {
        let order =
            f64::from(i32::try_from(m).map_err(|_| Error::InvalidInput("invalid order".into()))?);
        let (sin, cos) = (-order * phi).sin_cos();
        Ok(Complex::new(cos, sin))
    }
}
scalar_loop!(cw_rotate_loop, Complex, 7, 0=>kz:f64, 1=>mu:c_long, 2=>p:c_long, 3=>qz:f64, 4=>m:c_long, 5=>q:c_long, 6=>phi:f64 => cylinder_rotation(kz,mu,p,qz,m,q,phi));
macro_rules! cylindrical_translate_loop {
    ($name:ident,$regular:expr)=> {
        scalar_loop!(@64, $name<T>, Complex, 9, 0=>kz:f64, 1=>mu:c_long, 2=>p:c_long, 3=>qz:f64, 4=>m:c_long, 5=>q:c_long, 6=>kr:T, 7=>phi:f64, 8=>z:f64 => {
            let p=polarization_label(p)?;let q=polarization_label(q)?;let kr:Complex=kr.into();
            if p!=q || (!$regular && kr.norm()<1e-16 && z.abs()<1e-16) {Ok(Complex::default())}else{cylindrical_translation(kz,mu,qz,m,[kr,phi.into(),z.into()],if $regular {special::Radial::Regular}else{special::Radial::Outgoing})}
        });
    };
}
cylindrical_translate_loop!(cw_translate_s_loop, false);
cylindrical_translate_loop!(cw_translate_r_loop, true);
scalar_loop!(@512, pw_translate_loop<T>, Complex, 6, 0=>kx:T, 1=>ky:T, 2=>kz:T, 3=>x:f64, 4=>y:f64, 5=>z:f64 => treams_core::plane::translation([kx.into(),ky.into(),kz.into()],[x,y,z]));
macro_rules! plane_to_spherical_loop {
    ($name:ident,$helicity:expr)=> {
        scalar_loop!($name<T>, Complex, 7, 0=>l:c_long, 1=>m:c_long, 2=>p:c_long, 3=>kx:T, 4=>ky:T, 5=>kz:T, 6=>q:c_long => treams_core::plane::to_spherical(integer_wave_mode(l,m,p)?,[kx.into(),ky.into(),kz.into()],polarization_label(q)?,$helicity));
    };
}
plane_to_spherical_loop!(pw_to_sw_h_loop, true);
plane_to_spherical_loop!(pw_to_sw_p_loop, false);
scalar_loop!(pw_to_cw_loop<T>, Complex, 7, 0=>kz:f64, 1=>m:c_long, 2=>p:c_long, 3=>kx:f64, 4=>ky:T, 5=>qz:f64, 6=>q:c_long => treams_core::plane::to_cylindrical(treams_core::cylwaves::Mode{kz,m:i32::try_from(m).map_err(|_|Error::InvalidInput("invalid order".into()))?,pol:polarization_label(p)?},[kx.into(),ky.into(),qz.into()],polarization_label(q)?));
macro_rules! cylindrical_to_spherical_loop {
    ($name:ident,$helicity:expr)=> {
        scalar_loop!($name, Complex, 7, 0=>l:c_long, 1=>m:c_long, 2=>p:c_long, 3=>kz:f64, 4=>mu:c_long, 5=>q:c_long, 6=>k:Complex => treams_core::conversion::to_spherical(integer_wave_mode(l,m,p)?,treams_core::cylwaves::Mode{kz,m:i32::try_from(mu).map_err(|_|Error::InvalidInput("invalid order".into()))?,pol:polarization_label(q)?},k,$helicity));
    };
}
cylindrical_to_spherical_loop!(cw_to_sw_h_loop, true);
cylindrical_to_spherical_loop!(cw_to_sw_p_loop, false);
macro_rules! plane_permutation_loop {
    ($name:ident,$helicity:expr,$turns:expr)=> {
        scalar_loop!($name<T>, Complex, 5, 0=>kx:T, 1=>ky:T, 2=>kz:T, 3=>p:c_long, 4=>q:c_long => treams_core::plane::permutation_coefficient([kx.into(),ky.into(),kz.into()],polarization_label(p)?,polarization_label(q)?,$turns,$helicity));
    };
}
plane_permutation_loop!(pw_permute_h_loop, true, 1);
plane_permutation_loop!(pw_permute_p_loop, false, 1);
plane_permutation_loop!(pw_inverse_h_loop, true, 2);
plane_permutation_loop!(pw_inverse_p_loop, false, 2);

macro_rules! spherical_radiation_loop {
    ($name:ident,$helicity:expr)=> {
        scalar_loop!($name<T>, Complex, 8, 0=>kx:f64, 1=>ky:f64, 2=>kz:T, 3=>p:c_long, 4=>l:c_long, 5=>m:c_long, 6=>q:c_long, 7=>area:f64 => treams_core::channels::spherical_to_plane(integer_wave_mode(l,m,q)?,[kx.into(),ky.into(),kz.into()],polarization_label(p)?,area,$helicity));
    };
}
spherical_radiation_loop!(sw_to_pw_h_loop, true);
spherical_radiation_loop!(sw_to_pw_p_loop, false);
scalar_loop!(cw_to_pw_loop<T>, Complex, 8, 0=>kx:f64, 1=>ky:T, 2=>kz:f64, 3=>p:c_long, 4=>qz:f64, 5=>m:c_long, 6=>q:c_long, 7=>period:f64 => treams_core::channels::cylindrical_to_plane(treams_core::cylwaves::Mode{kz:qz,m:i32::try_from(m).map_err(|_|Error::InvalidInput("invalid order".into()))?,pol:polarization_label(q)?},[kx.into(),ky.into(),kz.into()],polarization_label(p)?,period));
macro_rules! spherical_to_cylindrical_loop {
    ($name:ident,$helicity:expr)=> {
        scalar_loop!($name, Complex, 8, 0=>kz:f64, 1=>mu:c_long, 2=>p:c_long, 3=>l:c_long, 4=>m:c_long, 5=>q:c_long, 6=>k:Complex, 7=>period:f64 => treams_core::conversion::periodic_to_cylindrical(treams_core::cylwaves::Mode{kz,m:i32::try_from(mu).map_err(|_|Error::InvalidInput("invalid order".into()))?,pol:polarization_label(p)?},integer_wave_mode(l,m,q)?,k,period,$helicity));
    };
}
spherical_to_cylindrical_loop!(sw_to_cw_h_loop, true);
spherical_to_cylindrical_loop!(sw_to_cw_p_loop, false);

fn coordinate_transform(index: u8) -> treams_core::coordinates::Transform {
    use treams_core::coordinates::Transform;
    match index {
        0 => Transform::CarToCyl,
        1 => Transform::CarToSph,
        2 => Transform::CylToCar,
        3 => Transform::CylToSph,
        4 => Transform::SphToCar,
        5 => Transform::SphToCyl,
        6 => Transform::CarToPol,
        _ => Transform::PolToCar,
    }
}
#[allow(clippy::cast_possible_wrap)] // Component indices are bounded by three.
unsafe extern "C" fn coordinate_loop<
    const TRANSFORM: u8,
    const VECTOR: bool,
    const COMPLEX: bool,
>(
    args: *mut *mut c_char,
    dimensions: *mut npy_intp,
    steps: *mut npy_intp,
    _data: *mut c_void,
) {
    // SAFETY: The generalized-ufunc registration fixes the core dimension (2/3)
    // and scalar dtypes. NumPy supplies outer strides followed by one component
    // stride per operand. Copy every input component before writing, permitting
    // in-place vectors, arbitrary component axes and unaligned arrays.
    let result = std::panic::catch_unwind(|| unsafe {
        let n = usize::try_from(*dimensions).unwrap_or_default();
        let transform = coordinate_transform(TRANSFORM);
        let dim = transform.dimension();
        let operands = if VECTOR { 3 } else { 2 };
        let output_index = operands - 1;
        let position_index = usize::from(VECTOR);
        let mut input = *args;
        let mut position = *args.add(position_index);
        let mut output = *args.add(output_index);
        let input_step = *steps.add(operands);
        let position_step = *steps.add(operands + position_index);
        let output_step = *steps.add(operands + output_index);
        for _ in 0..n {
            let p = std::array::from_fn(|axis| {
                if axis < dim {
                    position
                        .wrapping_offset(position_step * axis as isize)
                        .cast::<f64>()
                        .read_unaligned()
                } else {
                    0.0
                }
            });
            let value = if VECTOR {
                let v = std::array::from_fn(|axis| {
                    if axis >= dim {
                        Complex::default()
                    } else {
                        let ptr = input.wrapping_offset(input_step * axis as isize);
                        if COMPLEX {
                            ptr.cast::<Complex>().read_unaligned()
                        } else {
                            Complex::new(ptr.cast::<f64>().read_unaligned(), 0.0)
                        }
                    }
                });
                treams_core::coordinates::vector(v, p, transform)?
            } else {
                treams_core::coordinates::point(p, transform)?.map(Complex::from)
            };
            for (axis, value) in value.into_iter().take(dim).enumerate() {
                let ptr = output.wrapping_offset(output_step * axis as isize);
                if COMPLEX {
                    ptr.cast::<Complex>().write_unaligned(value);
                } else {
                    ptr.cast::<f64>().write_unaligned(value.re);
                }
            }
            input = input.wrapping_offset(*steps);
            position = position.wrapping_offset(*steps.add(position_index));
            output = output.wrapping_offset(*steps.add(output_index));
        }
        Ok::<_, Error>(())
    });
    report_loop(result);
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let capsule = module
        .py()
        .import("numpy._core.umath")?
        .getattr("_UFUNC_API")?
        .cast_into::<PyCapsule>()?;
    let table = capsule
        .pointer_checked(None)?
        .as_ptr()
        .cast::<*const c_void>();
    // SAFETY: NumPy >=2 is required; UFunc C API slot 27 is the void(void)
    // PyUFunc_clearfperr function. Its capsule stays alive with the cached pointer.
    let clear =
        unsafe { std::mem::transmute::<*const c_void, unsafe extern "C" fn()>(*table.add(27)) };
    let _ = FP_CLEAR.set((capsule.unbind(), clear));
    macro_rules! add {
        ($name:literal, $function:expr, $types:expr, $inputs:literal) => {
            add!($name, $function, $types, $inputs, std::ptr::null());
        };
        ($name:literal, $function:expr, $types:expr, $inputs:literal, $signature:expr) => {
            add!(@loops $name, [Some($function)], [$types], $inputs, 1, $signature);
        };
        (@loops $name:literal, $functions:expr, $types:expr, $inputs:literal, $count:literal, $signature:expr) => {{
            // NumPy retains these tables for the ufunc lifetime. Mutable static
            // storage follows its C API and permits its loop-replacement API;
            // Rust never creates references to or accesses the tables afterward.
            static mut LOOPS: [PyUFuncGenericFunction; $count] = $functions;
            static mut TYPES: [[c_char; $inputs + 1]; $count] = $types;
            // SAFETY: The tables have static storage, the matching loop takes
            // the declared input count and one output; both strings are NUL terminated.
            // The returned new reference is either owned by Bound or a Python error.
            let function = unsafe {
                let pointer = PY_UFUNC_API.PyUFunc_FromFuncAndDataAndSignature(
                    module.py(),
                    std::ptr::addr_of_mut!(LOOPS).cast(),
                    std::ptr::null_mut(),
                    std::ptr::addr_of_mut!(TYPES).cast(),
                    $count,
                    $inputs,
                    1,
                    -1,
                    concat!($name, "\0").as_ptr().cast(),
                    c"Rust special function with NumPy broadcasting and output arrays.".as_ptr(),
                    0,
                    $signature,
                );
                Bound::from_owned_ptr_or_err(module.py(), pointer)?
            };
            module.add($name, function)?;
        }};
    }
    const BESSEL_TYPES: [c_char; 3] = [
        NPY_TYPES::NPY_DOUBLE as c_char,
        NPY_TYPES::NPY_CDOUBLE as c_char,
        NPY_TYPES::NPY_CDOUBLE as c_char,
    ];
    const ANGULAR_TYPES: [c_char; 4] = [
        NPY_TYPES::NPY_DOUBLE as c_char,
        NPY_TYPES::NPY_DOUBLE as c_char,
        NPY_TYPES::NPY_CDOUBLE as c_char,
        NPY_TYPES::NPY_CDOUBLE as c_char,
    ];
    add!("jv", bessel_loop::<0, false, 0>, BESSEL_TYPES, 2);
    add!("yv", bessel_loop::<1, false, 0>, BESSEL_TYPES, 2);
    add!("hankel1", bessel_loop::<2, false, 0>, BESSEL_TYPES, 2);
    add!("hankel2", bessel_loop::<3, false, 0>, BESSEL_TYPES, 2);
    add!("jv_d", bessel_loop::<0, false, 1>, BESSEL_TYPES, 2);
    add!("yv_d", bessel_loop::<1, false, 1>, BESSEL_TYPES, 2);
    add!("hankel1_d", bessel_loop::<2, false, 1>, BESSEL_TYPES, 2);
    add!("hankel2_d", bessel_loop::<3, false, 1>, BESSEL_TYPES, 2);
    add!("spherical_jn", bessel_loop::<0, true, 0>, BESSEL_TYPES, 2);
    add!("spherical_yn", bessel_loop::<1, true, 0>, BESSEL_TYPES, 2);
    add!(
        "spherical_hankel1",
        bessel_loop::<2, true, 0>,
        BESSEL_TYPES,
        2
    );
    add!(
        "spherical_hankel2",
        bessel_loop::<3, true, 0>,
        BESSEL_TYPES,
        2
    );
    add!("spherical_jn_d", bessel_loop::<0, true, 1>, BESSEL_TYPES, 2);
    add!("spherical_yn_d", bessel_loop::<1, true, 1>, BESSEL_TYPES, 2);
    add!(
        "spherical_hankel1_d",
        bessel_loop::<2, true, 1>,
        BESSEL_TYPES,
        2
    );
    add!(
        "spherical_hankel2_d",
        bessel_loop::<3, true, 1>,
        BESSEL_TYPES,
        2
    );
    add!(@loops "lpmv", [Some(legendre_real_loop), Some(angular_loop::<0>)], [[NPY_TYPES::NPY_DOUBLE as c_char; 4], ANGULAR_TYPES], 3, 2, std::ptr::null());
    add!("pi_fun", angular_loop::<1>, ANGULAR_TYPES, 3);
    add!("tau_fun", angular_loop::<2>, ANGULAR_TYPES, 3);
    const D: c_char = NPY_TYPES::NPY_DOUBLE as c_char;
    const Z: c_char = NPY_TYPES::NPY_CDOUBLE as c_char;
    add!("wignersmalld", wigner_small_loop, [D, D, D, Z, Z], 4);
    add!("wignerd", wigner_loop, [D, D, D, Z, Z, Z, Z], 6);
    add!("wigner3j", wigner3j_loop, [D, D, D, D, D, D, D], 6);
    add!("incgamma_ufunc", gamma_loop, [D, Z, Z], 2);
    add!("intkambe_ufunc", kambe_loop, [D, Z, Z, Z], 3);
    const I: c_char = NPY_TYPES::NPY_LONG as c_char;
    add!(@loops "refractive_indices",[Some(refractive_indices_real_loop),Some(refractive_indices_loop)],[[D,D,D,D],[Z,Z,Z,Z]],3,2,c"(),(),()->(2)".as_ptr());
    add!(@loops "wave_vector_z",[Some(wave_vector_z_loop::<f64>),Some(wave_vector_z_loop::<Complex>)],[[D,D,D,Z],[Z,Z,Z,Z]],3,2,std::ptr::null());
    add!(
        "lsumsw1d",
        lattice_loop::<true, 1, false, 0, false>,
        [D, Z, D, D, D, Z, Z],
        6,
        std::ptr::null()
    );
    add!(
        "lsumsw1d_shift",
        lattice_loop::<true, 1, true, 0, false>,
        [D, D, Z, D, D, D, Z, Z],
        7,
        c"(),(),(),(),(),(3),()->()".as_ptr()
    );
    add!(
        "lsumsw2d",
        lattice_loop::<true, 2, false, 0, false>,
        [D, D, Z, D, D, D, Z, Z],
        7,
        c"(),(),(),(2),(2,2),(2),()->()".as_ptr()
    );
    add!(
        "lsumsw2d_shift",
        lattice_loop::<true, 2, true, 0, false>,
        [D, D, Z, D, D, D, Z, Z],
        7,
        c"(),(),(),(2),(2,2),(3),()->()".as_ptr()
    );
    add!(
        "lsumsw3d",
        lattice_loop::<true, 3, false, 0, false>,
        [D, D, Z, D, D, D, Z, Z],
        7,
        c"(),(),(),(3),(3,3),(3),()->()".as_ptr()
    );
    add!(
        "lsumcw1d",
        lattice_loop::<false, 1, false, 0, false>,
        [D, Z, D, D, D, Z, Z],
        6,
        std::ptr::null()
    );
    add!(
        "lsumcw1d_shift",
        lattice_loop::<false, 1, true, 0, false>,
        [D, Z, D, D, D, Z, Z],
        6,
        c"(),(),(),(),(2),()->()".as_ptr()
    );
    add!(
        "lsumcw2d",
        lattice_loop::<false, 2, false, 0, false>,
        [D, Z, D, D, D, Z, Z],
        6,
        c"(),(),(2),(2,2),(2),()->()".as_ptr()
    );
    add!(
        "realsumsw1d",
        lattice_loop::<true, 1, false, 1, false>,
        [D, Z, D, D, D, Z, Z],
        6,
        std::ptr::null()
    );
    add!(
        "realsumsw1d_shift",
        lattice_loop::<true, 1, true, 1, false>,
        [D, D, Z, D, D, D, Z, Z],
        7,
        c"(),(),(),(),(),(3),()->()".as_ptr()
    );
    add!(
        "realsumsw2d",
        lattice_loop::<true, 2, false, 1, false>,
        [D, D, Z, D, D, D, Z, Z],
        7,
        c"(),(),(),(2),(2,2),(2),()->()".as_ptr()
    );
    add!(
        "realsumsw2d_shift",
        lattice_loop::<true, 2, true, 1, false>,
        [D, D, Z, D, D, D, Z, Z],
        7,
        c"(),(),(),(2),(2,2),(3),()->()".as_ptr()
    );
    add!(
        "realsumsw3d",
        lattice_loop::<true, 3, false, 1, false>,
        [D, D, Z, D, D, D, Z, Z],
        7,
        c"(),(),(),(3),(3,3),(3),()->()".as_ptr()
    );
    add!(
        "realsumcw1d",
        lattice_loop::<false, 1, false, 1, false>,
        [D, Z, D, D, D, Z, Z],
        6,
        std::ptr::null()
    );
    add!(
        "realsumcw1d_shift",
        lattice_loop::<false, 1, true, 1, false>,
        [D, Z, D, D, D, Z, Z],
        6,
        c"(),(),(),(),(2),()->()".as_ptr()
    );
    add!(
        "realsumcw2d",
        lattice_loop::<false, 2, false, 1, false>,
        [D, Z, D, D, D, Z, Z],
        6,
        c"(),(),(2),(2,2),(2),()->()".as_ptr()
    );
    add!(
        "recsumsw1d",
        lattice_loop::<true, 1, false, 2, false>,
        [D, Z, D, D, D, Z, Z],
        6,
        std::ptr::null()
    );
    add!(
        "recsumsw1d_shift",
        lattice_loop::<true, 1, true, 2, false>,
        [D, D, Z, D, D, D, Z, Z],
        7,
        c"(),(),(),(),(),(3),()->()".as_ptr()
    );
    add!(
        "recsumsw2d",
        lattice_loop::<true, 2, false, 2, false>,
        [D, D, Z, D, D, D, Z, Z],
        7,
        c"(),(),(),(2),(2,2),(2),()->()".as_ptr()
    );
    add!(
        "recsumsw2d_shift",
        lattice_loop::<true, 2, true, 2, false>,
        [D, D, Z, D, D, D, Z, Z],
        7,
        c"(),(),(),(2),(2,2),(3),()->()".as_ptr()
    );
    add!(
        "recsumsw3d",
        lattice_loop::<true, 3, false, 2, false>,
        [D, D, Z, D, D, D, Z, Z],
        7,
        c"(),(),(),(3),(3,3),(3),()->()".as_ptr()
    );
    add!(
        "recsumcw1d",
        lattice_loop::<false, 1, false, 2, false>,
        [D, Z, D, D, D, Z, Z],
        6,
        std::ptr::null()
    );
    add!(
        "recsumcw1d_shift",
        lattice_loop::<false, 1, true, 2, false>,
        [D, Z, D, D, D, Z, Z],
        6,
        c"(),(),(),(),(2),()->()".as_ptr()
    );
    add!(
        "recsumcw2d",
        lattice_loop::<false, 2, false, 2, false>,
        [D, Z, D, D, D, Z, Z],
        6,
        c"(),(),(2),(2,2),(2),()->()".as_ptr()
    );
    add!(@loops "dsumsw1d", [Some(lattice_loop::<true, 1, false, 3, true>), Some(lattice_loop::<true, 1, false, 3, false>)], [[I, Z, D, D, D, I, Z], [D, Z, D, D, D, I, Z]], 6, 2, std::ptr::null());
    add!(@loops "dsumsw1d_shift", [Some(lattice_loop::<true, 1, true, 3, true>), Some(lattice_loop::<true, 1, true, 3, false>)], [[I, I, Z, D, D, D, I, Z], [D, D, Z, D, D, D, I, Z]], 7, 2, c"(),(),(),(),(),(3),()->()".as_ptr());
    add!(@loops "dsumsw2d", [Some(lattice_loop::<true, 2, false, 3, true>), Some(lattice_loop::<true, 2, false, 3, false>)], [[I, I, Z, D, D, D, I, Z], [D, D, Z, D, D, D, I, Z]], 7, 2, c"(),(),(),(2),(2,2),(2),()->()".as_ptr());
    add!(@loops "dsumsw2d_shift", [Some(lattice_loop::<true, 2, true, 3, true>), Some(lattice_loop::<true, 2, true, 3, false>)], [[I, I, Z, D, D, D, I, Z], [D, D, Z, D, D, D, I, Z]], 7, 2, c"(),(),(),(2),(2,2),(3),()->()".as_ptr());
    add!(@loops "dsumsw3d", [Some(lattice_loop::<true, 3, false, 3, true>), Some(lattice_loop::<true, 3, false, 3, false>)], [[I, I, Z, D, D, D, I, Z], [D, D, Z, D, D, D, I, Z]], 7, 2, c"(),(),(),(3),(3,3),(3),()->()".as_ptr());
    add!(@loops "dsumcw1d", [Some(lattice_loop::<false, 1, false, 3, true>), Some(lattice_loop::<false, 1, false, 3, false>)], [[I, Z, D, D, D, I, Z], [D, Z, D, D, D, I, Z]], 6, 2, std::ptr::null());
    add!(@loops "dsumcw1d_shift", [Some(lattice_loop::<false, 1, true, 3, true>), Some(lattice_loop::<false, 1, true, 3, false>)], [[I, Z, D, D, D, I, Z], [D, Z, D, D, D, I, Z]], 6, 2, c"(),(),(),(),(2),()->()".as_ptr());
    add!(@loops "dsumcw2d", [Some(lattice_loop::<false, 2, false, 3, true>), Some(lattice_loop::<false, 2, false, 3, false>)], [[I, Z, D, D, D, I, Z], [D, Z, D, D, D, I, Z]], 6, 2, c"(),(),(2),(2,2),(2),()->()".as_ptr());
    add!("first_brillouin_1d", first_brillouin_1d_loop, [D, D, D], 2);
    add!(@loops "cell_volume",[Some(volume_loop::<std::num::Wrapping<c_long>>),Some(volume_loop::<f64>)],[[I,I],[D,D]],1,2,c"(i,i)->()".as_ptr());
    add!(
        "cell_reciprocal",
        reciprocal_loop,
        [D, D],
        1,
        c"(i,i)->(i,i)".as_ptr()
    );

    add!(@loops "sph_harm",[Some(sph_harm_loop::<f64>),Some(sph_harm_loop::<Complex>)],[[D,D,D,D,Z],[D,D,D,Z,Z]],4,2,std::ptr::null());
    add!(@loops "vsh_X",[Some(vsh_x_loop::<f64>),Some(vsh_x_loop::<Complex>)],[[I,I,D,D,Z],[I,I,Z,D,Z]],4,2,c"(),(),(),()->(3)".as_ptr());
    add!(@loops "vsh_Y",[Some(vsh_y_loop::<f64>),Some(vsh_y_loop::<Complex>)],[[I,I,D,D,Z],[I,I,Z,D,Z]],4,2,c"(),(),(),()->(3)".as_ptr());
    add!(@loops "vsh_Z",[Some(vsh_z_loop::<f64>),Some(vsh_z_loop::<Complex>)],[[I,I,D,D,Z],[I,I,Z,D,Z]],4,2,c"(),(),(),()->(3)".as_ptr());
    add!(@loops "vsw_M",[Some(vsw_m_loop::<f64>),Some(vsw_m_loop::<Complex>)],[[I,I,Z,D,D,Z],[I,I,Z,Z,D,Z]],5,2,c"(),(),(),(),()->(3)".as_ptr());
    add!(@loops "vsw_N",[Some(vsw_n_loop::<f64>),Some(vsw_n_loop::<Complex>)],[[I,I,Z,D,D,Z],[I,I,Z,Z,D,Z]],5,2,c"(),(),(),(),()->(3)".as_ptr());
    add!(@loops "vsw_A",[Some(vsw_a_loop::<f64>),Some(vsw_a_loop::<Complex>)],[[I,I,Z,D,D,I,Z],[I,I,Z,Z,D,I,Z]],6,2,c"(),(),(),(),(),()->(3)".as_ptr());
    add!(@loops "vsw_rM",[Some(vsw_rm_loop::<f64>),Some(vsw_rm_loop::<Complex>)],[[I,I,Z,D,D,Z],[I,I,Z,Z,D,Z]],5,2,c"(),(),(),(),()->(3)".as_ptr());
    add!(@loops "vsw_rN",[Some(vsw_rn_loop::<f64>),Some(vsw_rn_loop::<Complex>)],[[I,I,Z,D,D,Z],[I,I,Z,Z,D,Z]],5,2,c"(),(),(),(),()->(3)".as_ptr());
    add!(@loops "vsw_rA",[Some(vsw_ra_loop::<f64>),Some(vsw_ra_loop::<Complex>)],[[I,I,Z,D,D,I,Z],[I,I,Z,Z,D,I,Z]],6,2,c"(),(),(),(),(),()->(3)".as_ptr());
    add!(
        "vcw_M",
        vcw_m_loop,
        [D, I, Z, D, D, Z],
        5,
        c"(),(),(),(),()->(3)".as_ptr()
    );
    add!(
        "vcw_N",
        vcw_n_loop,
        [D, I, Z, D, D, Z, Z],
        6,
        c"(),(),(),(),(),()->(3)".as_ptr()
    );
    add!(
        "vcw_A",
        vcw_a_loop,
        [D, I, Z, D, D, Z, I, Z],
        7,
        c"(),(),(),(),(),(),()->(3)".as_ptr()
    );
    add!(
        "vcw_rM",
        vcw_rm_loop,
        [D, I, Z, D, D, Z],
        5,
        c"(),(),(),(),()->(3)".as_ptr()
    );
    add!(
        "vcw_rN",
        vcw_rn_loop,
        [D, I, Z, D, D, Z, Z],
        6,
        c"(),(),(),(),(),()->(3)".as_ptr()
    );
    add!(
        "vcw_rA",
        vcw_ra_loop,
        [D, I, Z, D, D, Z, I, Z],
        7,
        c"(),(),(),(),(),(),()->(3)".as_ptr()
    );
    add!(@loops "vpw_M",[Some(plane_wave_loop::<f64,0>),Some(plane_wave_loop::<Complex,0>)],[[D,D,D,D,D,D,Z],[Z,Z,Z,D,D,D,Z]],6,2,c"(),(),(),(),(),()->(3)".as_ptr());
    add!(@loops "vpw_N",[Some(plane_wave_loop::<f64,1>),Some(plane_wave_loop::<Complex,1>)],[[D,D,D,D,D,D,Z],[Z,Z,Z,D,D,D,Z]],6,2,c"(),(),(),(),(),()->(3)".as_ptr());
    add!(@loops "vpw_A",[Some(plane_wave_loop::<f64,2>),Some(plane_wave_loop::<Complex,2>)],[[D,D,D,D,D,D,I,Z],[Z,Z,Z,D,D,D,I,Z]],7,2,c"(),(),(),(),(),(),()->(3)".as_ptr());
    add!("tl_vcw", tl_vcw_loop, [D, I, D, I, Z, D, D, Z], 7);
    add!(@loops "tl_vcw_r",[Some(tl_vcw_r_loop::<f64>),Some(tl_vcw_r_loop::<Complex>)],[[D,I,D,I,D,D,D,Z],[D,I,D,I,Z,D,D,Z]],7,2,std::ptr::null());
    add!(
        "sw_rotate",
        sw_rotate_loop,
        [I, I, I, I, I, I, D, D, D, Z],
        9
    );
    add!("cw_rotate", cw_rotate_loop, [D, I, I, D, I, I, D, Z], 7);
    add!(@loops "sw_to_pw_h",[Some(sw_to_pw_h_loop::<f64>),Some(sw_to_pw_h_loop::<Complex>)],[[D,D,D,I,I,I,I,D,Z],[D,D,Z,I,I,I,I,D,Z]],8,2,std::ptr::null());
    add!(@loops "sw_to_pw_p",[Some(sw_to_pw_p_loop::<f64>),Some(sw_to_pw_p_loop::<Complex>)],[[D,D,D,I,I,I,I,D,Z],[D,D,Z,I,I,I,I,D,Z]],8,2,std::ptr::null());
    add!(@loops "cw_to_pw",[Some(cw_to_pw_loop::<f64>),Some(cw_to_pw_loop::<Complex>)],[[D,D,D,I,D,I,I,D,Z],[D,Z,D,I,D,I,I,D,Z]],8,2,std::ptr::null());
    add!(
        "sw_to_cw_h",
        sw_to_cw_h_loop,
        [D, I, I, I, I, I, Z, D, Z],
        8
    );
    add!(
        "sw_to_cw_p",
        sw_to_cw_p_loop,
        [D, I, I, I, I, I, Z, D, Z],
        8
    );
    add!(@loops "sw_translate_sh",[Some(polar_translation_loop::<f64,3,false,9>),Some(polar_translation_loop::<Complex,3,false,9>)],[[I,I,I,I,I,I,D,D,D,Z],[I,I,I,I,I,I,Z,D,D,Z]],9,2,std::ptr::null());
    add!(@loops "sw_translate_rh",[Some(polar_translation_loop::<f64,3,true,9>),Some(polar_translation_loop::<Complex,3,true,9>)],[[I,I,I,I,I,I,D,D,D,Z],[I,I,I,I,I,I,Z,D,D,Z]],9,2,std::ptr::null());
    add!(@loops "sw_translate_sp",[Some(polar_translation_loop::<f64,2,false,9>),Some(polar_translation_loop::<Complex,2,false,9>)],[[I,I,I,I,I,I,D,D,D,Z],[I,I,I,I,I,I,Z,D,D,Z]],9,2,std::ptr::null());
    add!(@loops "sw_translate_rp",[Some(polar_translation_loop::<f64,2,true,9>),Some(polar_translation_loop::<Complex,2,true,9>)],[[I,I,I,I,I,I,D,D,D,Z],[I,I,I,I,I,I,Z,D,D,Z]],9,2,std::ptr::null());
    add!(@loops "cw_translate_s",[Some(cw_translate_s_loop::<f64>),Some(cw_translate_s_loop::<Complex>)],[[D,I,I,D,I,I,D,D,D,Z],[D,I,I,D,I,I,Z,D,D,Z]],9,2,std::ptr::null());
    add!(@loops "cw_translate_r",[Some(cw_translate_r_loop::<f64>),Some(cw_translate_r_loop::<Complex>)],[[D,I,I,D,I,I,D,D,D,Z],[D,I,I,D,I,I,Z,D,D,Z]],9,2,std::ptr::null());
    add!(@loops "pw_translate_ufunc",[Some(pw_translate_loop::<f64>),Some(pw_translate_loop::<Complex>)],[[D,D,D,D,D,D,Z],[Z,Z,Z,D,D,D,Z]],6,2,std::ptr::null());
    let translation = module.getattr("pw_translate_ufunc")?.unbind();
    let _ = PW_TRANSLATE.set(translation);
    module.add_function(wrap_pyfunction!(pw_translate, module)?)?;
    module.add_function(wrap_pyfunction!(cylindrical_rotation_scalar, module)?)?;
    module.add_function(wrap_pyfunction!(cylindrical_translation_scalar, module)?)?;
    module.add_function(wrap_pyfunction!(plane_permutation_scalar, module)?)?;
    add!(@loops "pw_to_sw_h",[Some(pw_to_sw_h_loop::<f64>),Some(pw_to_sw_h_loop::<Complex>)],[[I,I,I,D,D,D,I,Z],[I,I,I,Z,Z,Z,I,Z]],7,2,std::ptr::null());
    add!(@loops "pw_to_sw_p",[Some(pw_to_sw_p_loop::<f64>),Some(pw_to_sw_p_loop::<Complex>)],[[I,I,I,D,D,D,I,Z],[I,I,I,Z,Z,Z,I,Z]],7,2,std::ptr::null());
    add!(@loops "pw_to_cw",[Some(pw_to_cw_loop::<f64>),Some(pw_to_cw_loop::<Complex>)],[[D,I,I,D,D,D,I,Z],[D,I,I,D,Z,D,I,Z]],7,2,std::ptr::null());
    add!("cw_to_sw_h", cw_to_sw_h_loop, [I, I, I, D, I, I, Z, Z], 7);
    add!("cw_to_sw_p", cw_to_sw_p_loop, [I, I, I, D, I, I, Z, Z], 7);
    add!(@loops "pw_permute_h",[Some(pw_permute_h_loop::<f64>),Some(pw_permute_h_loop::<Complex>)],[[D,D,D,I,I,Z],[Z,Z,Z,I,I,Z]],5,2,std::ptr::null());
    add!(@loops "pw_permute_p",[Some(pw_permute_p_loop::<f64>),Some(pw_permute_p_loop::<Complex>)],[[D,D,D,I,I,Z],[Z,Z,Z,I,I,Z]],5,2,std::ptr::null());
    add!(@loops "pw_inverse_h",[Some(pw_inverse_h_loop::<f64>),Some(pw_inverse_h_loop::<Complex>)],[[D,D,D,I,I,Z],[Z,Z,Z,I,I,Z]],5,2,std::ptr::null());
    add!(@loops "pw_inverse_p",[Some(pw_inverse_p_loop::<f64>),Some(pw_inverse_p_loop::<Complex>)],[[D,D,D,I,I,Z],[Z,Z,Z,I,I,Z]],5,2,std::ptr::null());
    add!(@loops "tl_vsw_A",[Some(polar_translation_loop::<f64,0,false,7>),Some(polar_translation_loop::<Complex,0,false,7>)],[[I,I,I,I,Z,D,D,Z],[I,I,I,I,Z,Z,D,Z]],7,2,std::ptr::null());
    add!(@loops "tl_vsw_B",[Some(polar_translation_loop::<f64,1,false,7>),Some(polar_translation_loop::<Complex,1,false,7>)],[[I,I,I,I,Z,D,D,Z],[I,I,I,I,Z,Z,D,Z]],7,2,std::ptr::null());
    add!(@loops "tl_vsw_rA",[Some(polar_translation_loop::<f64,0,true,7>),Some(polar_translation_loop::<Complex,0,true,7>)],[[I,I,I,I,Z,D,D,Z],[I,I,I,I,Z,Z,D,Z]],7,2,std::ptr::null());
    add!(@loops "tl_vsw_rB",[Some(polar_translation_loop::<f64,1,true,7>),Some(polar_translation_loop::<Complex,1,true,7>)],[[I,I,I,I,Z,D,D,Z],[I,I,I,I,Z,Z,D,Z]],7,2,std::ptr::null());
    let mut plane_functions = Vec::with_capacity(3);
    macro_rules! plane_function {
        ($name:ident, $public:literal, $pol:literal $(, $label:ident)?) => {{
            #[pyfunction(name = $public)]
            #[pyo3(signature=(kx,ky,kz,x,y,z $(,$label)?, *args, **kwargs))]
            fn $name<'py>(py: Python<'py>, kx: &Bound<'py, PyAny>, ky: &Bound<'py, PyAny>, kz: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, z: &Bound<'py, PyAny> $(,$label: &Bound<'py, PyAny>)?, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Bound<'py, PyAny>> {
                let label = None;
                $(let label = label.or(Some($label));)?
                plane_wave_call::<$pol>(py, [kx,ky,kz,x,y,z], label, args, kwargs)
            }
            plane_functions.push(module.getattr($public)?.unbind());
            module.add_function(wrap_pyfunction!($name, module)?)?;
        }};
    }
    plane_function!(vpw_m, "vpw_M", 0);
    plane_function!(vpw_n, "vpw_N", 1);
    plane_function!(vpw_a, "vpw_A", 2, pol);
    let _ = PLANE_WAVES.set(plane_functions);
    let mut cell_functions = Vec::with_capacity(2);
    macro_rules! cell_function {
        ($name:ident, $reciprocal:literal) => {{
            #[pyfunction]
            #[pyo3(signature=(cell, *args, **kwargs))]
            fn $name<'py>(
                py: Python<'py>,
                cell: &Bound<'py, PyAny>,
                args: &Bound<'py, PyTuple>,
                kwargs: Option<&Bound<'py, PyDict>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                cell_call::<$reciprocal>(py, cell, args, kwargs)
            }
            cell_functions.push(module.getattr(stringify!($name))?.unbind());
            module.add_function(wrap_pyfunction!($name, module)?)?;
        }};
    }
    cell_function!(cell_volume, false);
    cell_function!(cell_reciprocal, true);
    let _ = CELLS.set(cell_functions);
    let mut coordinate_functions = Vec::with_capacity(16);
    macro_rules! coordinates {
        ($point:ident,$vector:ident,$kind:literal,$point_signature:literal,$vector_signature:literal)=>{{
            static mut POINT_LOOP:[PyUFuncGenericFunction;1]=[Some(coordinate_loop::<$kind,false,false>)];
            static mut VECTOR_LOOPS:[PyUFuncGenericFunction;2]=[Some(coordinate_loop::<$kind,true,false>),Some(coordinate_loop::<$kind,true,true>)];
            static mut POINT_TYPES:[c_char;2]=[D,D];
            static mut VECTOR_TYPES:[c_char;6]=[D,D,D,Z,D,Z];
            // SAFETY: Static tables and signatures match the callback's input
            // count, fixed component dimension and dtype. NumPy owns the new ref.
            unsafe {
                for (name,signature,loops,types,count,inputs) in [
                    (concat!(stringify!($point),"\0"),concat!($point_signature,"\0"),std::ptr::addr_of_mut!(POINT_LOOP).cast(),std::ptr::addr_of_mut!(POINT_TYPES).cast(),1,1),
                    (concat!(stringify!($vector),"\0"),concat!($vector_signature,"\0"),std::ptr::addr_of_mut!(VECTOR_LOOPS).cast(),std::ptr::addr_of_mut!(VECTOR_TYPES).cast(),2,2),
                ] {
                    let ptr=PY_UFUNC_API.PyUFunc_FromFuncAndDataAndSignature(module.py(),loops,std::ptr::null_mut(),types,count,inputs,1,-1,name.as_ptr().cast(),c"Rust coordinate transform; vector positions use the input coordinate system.".as_ptr(),0,signature.as_ptr().cast());
                    let function = Bound::from_owned_ptr_or_err(module.py(),ptr)?;
                    module.add(name.trim_end_matches('\0'), &function)?;
                    coordinate_functions.push(function.unbind());
                }
            }
            #[pyfunction]
            #[pyo3(signature=(points, *args, **kwargs))]
            fn $point<'py>(py: Python<'py>, points: &Bound<'py, PyAny>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Bound<'py, PyAny>> {
                coordinate_call::<$kind>(py, points, None, args, kwargs)
            }
            #[pyfunction]
            #[pyo3(signature=(vector, points, *args, **kwargs))]
            fn $vector<'py>(py: Python<'py>, vector: &Bound<'py, PyAny>, points: &Bound<'py, PyAny>, args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<Bound<'py, PyAny>> {
                coordinate_call::<$kind>(py, points, Some(vector), args, kwargs)
            }
            module.add_function(wrap_pyfunction!($point, module)?)?;
            module.add_function(wrap_pyfunction!($vector, module)?)?;
        }};
    }
    coordinates!(car2cyl, vcar2cyl, 0, "(3)->(3)", "(3),(3)->(3)");
    coordinates!(car2sph, vcar2sph, 1, "(3)->(3)", "(3),(3)->(3)");
    coordinates!(cyl2car, vcyl2car, 2, "(3)->(3)", "(3),(3)->(3)");
    coordinates!(cyl2sph, vcyl2sph, 3, "(3)->(3)", "(3),(3)->(3)");
    coordinates!(sph2car, vsph2car, 4, "(3)->(3)", "(3),(3)->(3)");
    coordinates!(sph2cyl, vsph2cyl, 5, "(3)->(3)", "(3),(3)->(3)");
    coordinates!(car2pol, vcar2pol, 6, "(2)->(2)", "(2),(2)->(2)");
    coordinates!(pol2car, vpol2car, 7, "(2)->(2)", "(2),(2)->(2)");
    let _ = COORDINATES.set(coordinate_functions);
    Ok(())
}
