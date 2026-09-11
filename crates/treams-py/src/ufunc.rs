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
    special::{self, Bessel},
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
    // A raw C callback must not unwind across the interpreter. As in PyO3's
    // function wrappers, report failures with a Python exception on this thread.
    let result = result
        .unwrap_or_else(|_| Err(Error::SpecialFunction("native Bessel loop panicked".into())));
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
        ($name:literal, $kind:literal, $spherical:literal, $derivative:literal) => {{
            // NumPy retains these tables for the ufunc lifetime. Mutable static
            // storage follows its C API and permits its loop-replacement API;
            // Rust never creates references to or accesses the tables afterward.
            static mut LOOPS: [PyUFuncGenericFunction; 1] =
                [Some(bessel_loop::<$kind, $spherical, $derivative>)];
            static mut TYPES: [c_char; 3] = [
                NPY_TYPES::NPY_DOUBLE as c_char,
                NPY_TYPES::NPY_CDOUBLE as c_char,
                NPY_TYPES::NPY_CDOUBLE as c_char,
            ];
            // SAFETY: The tables have static storage, the matching loop takes
            // two inputs and one output, and both C strings are NUL terminated.
            // The returned new reference is either owned by Bound or a Python error.
            let function = unsafe {
                let pointer = PY_UFUNC_API.PyUFunc_FromFuncAndData(
                    module.py(),
                    std::ptr::addr_of_mut!(LOOPS).cast(),
                    std::ptr::null_mut(),
                    std::ptr::addr_of_mut!(TYPES).cast(),
                    1,
                    2,
                    1,
                    -1,
                    concat!($name, "\0").as_ptr().cast(),
                    c"Rust Bessel evaluation with NumPy broadcasting, out and where.".as_ptr(),
                    0,
                );
                Bound::from_owned_ptr_or_err(module.py(), pointer)?
            };
            module.add($name, function)?;
        }};
    }
    add!("jv", 0, false, 0);
    add!("yv", 1, false, 0);
    add!("hankel1", 2, false, 0);
    add!("hankel2", 3, false, 0);
    add!("jv_d", 0, false, 1);
    add!("yv_d", 1, false, 1);
    add!("hankel1_d", 2, false, 1);
    add!("hankel2_d", 3, false, 1);
    add!("spherical_jn", 0, true, 0);
    add!("spherical_yn", 1, true, 0);
    add!("spherical_hankel1", 2, true, 0);
    add!("spherical_hankel2", 3, true, 0);
    add!("spherical_jn_d", 0, true, 1);
    add!("spherical_yn_d", 1, true, 1);
    add!("spherical_hankel1_d", 2, true, 1);
    add!("spherical_hankel2_d", 3, true, 1);
    Ok(())
}
