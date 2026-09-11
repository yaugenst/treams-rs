//! `NumPy` owns broadcasting, masking, overlap buffering and output allocation.
//! The strided C loop calls the same Rust kernels as the recorded boundary.

#![allow(unsafe_code)] // NumPy requires a raw C callback; safety obligations are confined here.
#![allow(clippy::cast_ptr_alignment)] // Slice alignment is checked; all other access is unaligned.

use std::{
    ffi::{c_char, c_void},
    sync::OnceLock,
};

use numpy::npyffi::{NPY_TYPES, PY_UFUNC_API, PyUFuncGenericFunction, npy_intp};
use pyo3::{prelude::*, types::PyCapsule};

static FP_CLEAR: OnceLock<(Py<PyCapsule>, unsafe extern "C" fn())> = OnceLock::new();
use treams_core::{
    Complex, Error,
    special::{self, Angular, Bessel},
};

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
    // A raw C callback must not unwind across the interpreter. As in PyO3's
    // function wrappers, report failures with a Python exception on this thread.
    let result = result.unwrap_or_else(|_| {
        Err(Error::SpecialFunction(
            "native special-function loop panicked".into(),
        ))
    });
    // The Bessel algorithm sets incidental flags for finite real-axis results.
    // Result and finite-value checks are authoritative. Clearing through the
    // cached C function avoids attaching Python on successful numerical calls.
    // SAFETY: register initializes FP_CLEAR before exposing any callback and
    // retains the capsule. Slot 27 clears only this thread's floating-point flags.
    unsafe {
        (FP_CLEAR.get().unwrap_unchecked().1)();
    }
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
        ($name:literal, $function:expr, $types:expr, $inputs:literal) => {{
            // NumPy retains these tables for the ufunc lifetime. Mutable static
            // storage follows its C API and permits its loop-replacement API;
            // Rust never creates references to or accesses the tables afterward.
            static mut LOOPS: [PyUFuncGenericFunction; 1] = [Some($function)];
            static mut TYPES: [c_char; $inputs + 1] = $types;
            // SAFETY: The tables have static storage, the matching loop takes
            // the declared input count and one output; both strings are NUL terminated.
            // The returned new reference is either owned by Bound or a Python error.
            let function = unsafe {
                let pointer = PY_UFUNC_API.PyUFunc_FromFuncAndData(
                    module.py(),
                    std::ptr::addr_of_mut!(LOOPS).cast(),
                    std::ptr::null_mut(),
                    std::ptr::addr_of_mut!(TYPES).cast(),
                    1,
                    $inputs,
                    1,
                    -1,
                    concat!($name, "\0").as_ptr().cast(),
                    c"Rust special function with NumPy broadcasting, out and where.".as_ptr(),
                    0,
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
    add!("lpmv", angular_loop::<0>, ANGULAR_TYPES, 3);
    add!("pi_fun", angular_loop::<1>, ANGULAR_TYPES, 3);
    add!("tau_fun", angular_loop::<2>, ANGULAR_TYPES, 3);
    Ok(())
}
