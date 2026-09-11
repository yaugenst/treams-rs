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
use rayon::prelude::*;

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

// Fixed arity scalar kernels share NumPy's masking, buffering and strided loop
// contract. Every input is copied before writing, including in-place operations.
macro_rules! scalar_loop {
    ($name:ident, $output:ty, $count:literal, $( $index:literal => $argument:ident : $ty:ty ),+ => $body:expr) => {
        unsafe extern "C" fn $name(args: *mut *mut c_char, dimensions: *mut npy_intp, steps: *mut npy_intp, _data: *mut c_void) {
            // SAFETY: register supplies this exact operand signature, and NumPy
            // provides valid byte strides/counts and buffers overlapping arrays.
            // Scalar copies permit unaligned storage and all reads precede writes.
            let result = std::panic::catch_unwind(|| unsafe {
                let n = usize::try_from(*dimensions).unwrap_or_default();
                let mut pointers: [*mut c_char; $count + 1] = std::array::from_fn(|i| *args.add(i));
                if n >= 1024 {
                    let inputs: [Input; $count] = std::array::from_fn(|i|Input{pointer:*args.add(i),stride:*steps.add(i)});
                    let evaluate=|i| {
                        $(let $argument=inputs.get_unchecked($index).read::<$ty>(i);)+
                        $body
                    };
                    let values: Vec<$output>=(0..n).into_par_iter().map(evaluate).collect::<treams_core::Result<_>>()?;
                    let mut output=*args.add($count);
                    for value in values {
                        output.cast::<$output>().write_unaligned(value);
                        output=output.wrapping_offset(*steps.add($count));
                    }
                    return Ok(());
                }
                for _ in 0..n {
                    $(let $argument = pointers.get_unchecked($index).cast::<$ty>().read_unaligned();)+
                    let value: $output = ($body)?;
                    pointers.get_unchecked($count).cast::<$output>().write_unaligned(value);
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
    const D: c_char = NPY_TYPES::NPY_DOUBLE as c_char;
    const Z: c_char = NPY_TYPES::NPY_CDOUBLE as c_char;
    add!("wignersmalld", wigner_small_loop, [D, D, D, Z, Z], 4);
    add!("wignerd", wigner_loop, [D, D, D, Z, Z, Z, Z], 6);
    add!("wigner3j", wigner3j_loop, [D, D, D, D, D, D, D], 6);
    add!("incgamma_ufunc", gamma_loop, [D, Z, Z], 2);
    add!("intkambe_ufunc", kambe_loop, [D, Z, Z, Z], 3);
    macro_rules! coordinates {
        ($point:literal,$vector:literal,$kind:literal,$point_signature:literal,$vector_signature:literal)=>{{
            static mut POINT_LOOP:[PyUFuncGenericFunction;1]=[Some(coordinate_loop::<$kind,false,false>)];
            static mut VECTOR_LOOPS:[PyUFuncGenericFunction;2]=[Some(coordinate_loop::<$kind,true,false>),Some(coordinate_loop::<$kind,true,true>)];
            static mut POINT_TYPES:[c_char;2]=[D,D];
            static mut VECTOR_TYPES:[c_char;6]=[D,D,D,Z,D,Z];
            // SAFETY: Static tables and signatures match the callback's input
            // count, fixed component dimension and dtype. NumPy owns the new ref.
            unsafe {
                for (name,signature,loops,types,count,inputs) in [
                    (concat!($point,"\0"),concat!($point_signature,"\0"),std::ptr::addr_of_mut!(POINT_LOOP).cast(),std::ptr::addr_of_mut!(POINT_TYPES).cast(),1,1),
                    (concat!($vector,"\0"),concat!($vector_signature,"\0"),std::ptr::addr_of_mut!(VECTOR_LOOPS).cast(),std::ptr::addr_of_mut!(VECTOR_TYPES).cast(),2,2),
                ] {
                    let ptr=PY_UFUNC_API.PyUFunc_FromFuncAndDataAndSignature(module.py(),loops,std::ptr::null_mut(),types,count,inputs,1,-1,name.as_ptr().cast(),c"Rust coordinate transform; vector positions use the input coordinate system.".as_ptr(),0,signature.as_ptr().cast());
                    module.add(name.trim_end_matches('\0'),Bound::from_owned_ptr_or_err(module.py(),ptr)?)?;
                }
            }
        }};
    }
    coordinates!("car2cyl", "vcar2cyl", 0, "(3)->(3)", "(3),(3)->(3)");
    coordinates!("car2sph", "vcar2sph", 1, "(3)->(3)", "(3),(3)->(3)");
    coordinates!("cyl2car", "vcyl2car", 2, "(3)->(3)", "(3),(3)->(3)");
    coordinates!("cyl2sph", "vcyl2sph", 3, "(3)->(3)", "(3),(3)->(3)");
    coordinates!("sph2car", "vsph2car", 4, "(3)->(3)", "(3),(3)->(3)");
    coordinates!("sph2cyl", "vsph2cyl", 5, "(3)->(3)", "(3),(3)->(3)");
    coordinates!("car2pol", "vcar2pol", 6, "(2)->(2)", "(2),(2)->(2)");
    coordinates!("pol2car", "vpol2car", 7, "(2)->(2)", "(2),(2)->(2)");
    Ok(())
}
