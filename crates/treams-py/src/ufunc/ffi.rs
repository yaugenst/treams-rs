//! The `NumPy` C interface of the loops: typed operands, the inner-loop driver,
//! the guard that keeps panics and floating-point flags inside a loop, and ufunc
//! creation.

// NumPy passes loop operands as raw pointers and strides, and its C API takes raw tables.
#![allow(unsafe_code)]

use std::{
    ffi::{CStr, CString, c_char, c_int, c_long, c_void},
    marker::PhantomData,
    num::Wrapping,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::OnceLock,
};

use numpy::npyffi::{NPY_TYPES, PY_UFUNC_API, npy_intp};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyCapsule};
use rayon::prelude::*;

use treams_core::{Complex, Error, Result, fpenv::ieee};

/// Byte offset of element `index` along `stride`.
pub(super) fn offset(stride: npy_intp, index: usize) -> isize {
    stride.wrapping_mul(isize::try_from(index).unwrap_or_default())
}

/// A `NumPy` dtype (its type number), named by the Rust type loops read or write.
pub(super) trait Dtype: Copy + Send {
    const TYPE: c_char;
}
macro_rules! dtypes {
    ($($ty:ty: $number:ident),+) => {
        $(impl Dtype for $ty {
            const TYPE: c_char = NPY_TYPES::$number as c_char;
        })+
    };
}
// `Wrapping` is transparent; integer cells wrap on overflow like `NumPy` ones.
dtypes!(f64: NPY_DOUBLE, Complex: NPY_CDOUBLE, c_long: NPY_LONG, Wrapping<c_long>: NPY_LONG);

/// One operand of dtype `T` in the current inner-loop call: the address of its
/// first element and the byte stride between elements.
#[derive(Clone, Copy)]
pub(super) struct Operand<T> {
    pointer: *mut c_char,
    stride: npy_intp,
    dtype: PhantomData<T>,
}
// SAFETY: Operands exist only during a NumPy inner-loop call, which keeps every
// buffer alive until it returns. Rayon workers only read inputs; outputs are
// written on the calling thread after the blocking parallel collection.
unsafe impl<T: Dtype> Send for Operand<T> {}
// SAFETY: See `Send`; shared access never writes.
unsafe impl<T: Dtype> Sync for Operand<T> {}

impl<T: Dtype> Operand<T> {
    /// Whether every element is the first one (a zero stride).
    pub(super) fn broadcast(self) -> bool {
        self.stride == 0
    }

    /// Address of element `index`, moved by `extra` bytes along core dimensions.
    fn at(self, index: usize, extra: isize) -> *mut c_char {
        self.pointer
            .wrapping_offset(offset(self.stride, index).wrapping_add(extra))
    }

    /// Read element `index`, offset by `extra` core bytes, without alignment.
    ///
    /// # Safety
    /// `index` is below the loop count and `extra` within the core.
    pub(super) unsafe fn read_at(self, index: usize, extra: isize) -> T {
        // SAFETY: Guaranteed by the caller; unaligned reads permit any layout.
        unsafe { self.at(index, extra).cast::<T>().read_unaligned() }
    }

    /// Read scalar element `index`.
    ///
    /// # Safety
    /// As for [`Operand::read_at`].
    pub(super) unsafe fn read(self, index: usize) -> T {
        // SAFETY: Guaranteed by the caller.
        unsafe { self.read_at(index, 0) }
    }

    /// Write element `index` of the output, offset by `extra` core bytes.
    ///
    /// # Safety
    /// As for [`Operand::read_at`].
    pub(super) unsafe fn write_at(self, index: usize, extra: isize, value: T) {
        // SAFETY: Guaranteed by the caller.
        unsafe { self.at(index, extra).cast::<T>().write_unaligned(value) }
    }
}

/// Operands of one dtype: `T`, or `[T; K]` for `K` consecutive inputs or the
/// `K` components of a vector output along its core dimension.
pub(super) trait Group: Copy + Send {
    type Dtype: Dtype;
    type Operands: Copy;
    /// The number of operands, or of components of vector outputs.
    const COUNT: usize;
    /// Arrays are vectors as outputs.
    const VECTOR: bool;
    /// Take the operands from the pointers and outer strides `next` yields.
    fn take(next: &mut impl FnMut() -> (*mut c_char, npy_intp)) -> Self::Operands;
    /// Write element `index`, with components `component` bytes apart.
    ///
    /// # Safety
    /// As for [`Operand::write_at`]; vectors own the output core dimension.
    unsafe fn store(self, output: Operand<Self::Dtype>, index: usize, component: npy_intp);
}
impl<T: Dtype> Group for T {
    type Dtype = T;
    type Operands = Operand<T>;
    const COUNT: usize = 1;
    const VECTOR: bool = false;
    fn take(next: &mut impl FnMut() -> (*mut c_char, npy_intp)) -> Operand<T> {
        let (pointer, stride) = next();
        Operand {
            pointer,
            stride,
            dtype: PhantomData,
        }
    }
    unsafe fn store(self, output: Operand<T>, index: usize, _: npy_intp) {
        // SAFETY: Guaranteed by the caller.
        unsafe { output.write_at(index, 0, self) }
    }
}
impl<T: Dtype, const K: usize> Group for [T; K] {
    type Dtype = T;
    type Operands = [Operand<T>; K];
    const COUNT: usize = K;
    const VECTOR: bool = true;
    fn take(next: &mut impl FnMut() -> (*mut c_char, npy_intp)) -> Self::Operands {
        std::array::from_fn(|_| T::take(next))
    }
    unsafe fn store(self, output: Operand<T>, index: usize, component: npy_intp) {
        for (k, value) in self.into_iter().enumerate() {
            // SAFETY: Guaranteed by the caller for each of the K components.
            unsafe { output.write_at(index, offset(component, k), value) }
        }
    }
}

/// The inputs of a loop: a tuple of groups in operand order.
pub(super) trait Inputs {
    type Operands: Copy;
    /// The dtype and the number of operands of each group.
    const GROUPS: &'static [(c_char, usize)];
    fn take(next: &mut impl FnMut() -> (*mut c_char, npy_intp)) -> Self::Operands;
}
macro_rules! inputs {
    () => {};
    ($first:ident $($group:ident)*) => {
        impl<$first: Group $(, $group: Group)*> Inputs for ($first, $($group,)*) {
            type Operands = ($first::Operands, $($group::Operands,)*);
            const GROUPS: &'static [(c_char, usize)] =
                &[($first::Dtype::TYPE, $first::COUNT) $(, ($group::Dtype::TYPE, $group::COUNT))*];
            fn take(next: &mut impl FnMut() -> (*mut c_char, npy_intp)) -> Self::Operands {
                ($first::take(next), $($group::take(next),)*)
            }
        }
        inputs!($($group)*);
    };
}
inputs!(G0 G1 G2 G3 G4 G5 G6 G7 G8);

/// The operand pointers of a loop with input groups `I` and output values `O`.
///
/// Loops take them first, and [`Loop::new`] derives their dtype row from these types.
#[repr(transparent)]
pub(super) struct Args<I, O>(*mut *mut c_char, PhantomData<fn() -> (I, O)>);

/// The operands of one inner-loop call.
pub(super) struct Call<I: Inputs, O: Group> {
    pub(super) len: usize,
    pub(super) inputs: I::Operands,
    pub(super) output: Operand<O::Dtype>,
    nin: usize,
    args: *const *mut c_char,
    dimensions: *const npy_intp,
    steps: *const npy_intp,
}

impl<I: Inputs, O: Group> Args<I, O> {
    /// Collect the operands of an inner-loop call.
    ///
    /// # Safety
    /// The pointers are the arguments of a `NumPy` call of the loop.
    pub(super) unsafe fn call(
        self,
        dimensions: *const npy_intp,
        steps: *const npy_intp,
    ) -> Call<I, O> {
        let mut operands = 0;
        let mut next = || {
            // SAFETY: NumPy passes a pointer and an outer stride per operand
            // of the dtype row: the inputs of I, then the output.
            let operand = unsafe { (*self.0.add(operands), *steps.add(operands)) };
            operands += 1;
            operand
        };
        let (inputs, output) = (I::take(&mut next), O::Dtype::take(&mut next));
        // SAFETY: NumPy passes the outer loop count first.
        let len = usize::try_from(unsafe { *dimensions }).unwrap_or_default();
        let nin = operands - 1;
        Call {
            len,
            inputs,
            output,
            nin,
            args: self.0,
            dimensions,
            steps,
        }
    }
}

impl<I: Inputs, O: Group> Call<I, O> {
    /// Size of core dimension `index`; core sizes follow the loop count.
    ///
    /// # Safety
    /// The signature has more than `index` core dimensions.
    pub(super) unsafe fn core_len(&self, index: usize) -> usize {
        // SAFETY: Guaranteed by the caller.
        usize::try_from(unsafe { *self.dimensions.add(1 + index) }).unwrap_or_default()
    }

    /// Byte stride of core dimension `index` of all operands; core strides
    /// follow the outer strides of the inputs and the output.
    ///
    /// # Safety
    /// The operands have more than `index` core dimensions in total.
    pub(super) unsafe fn core_stride(&self, index: usize) -> npy_intp {
        // SAFETY: Guaranteed by the caller.
        unsafe { *self.steps.add(self.nin + 1 + index) }
    }

    /// Whether an input overlaps the output other than element for element.
    ///
    /// `NumPy`'s `reduce`, `accumulate` and `reduceat` pass the accumulator as
    /// both an input and the output and need element-by-element evaluation, which
    /// the parallel path does not provide: it evaluates every element before the
    /// first store. Other overlap of outer ranges is serialized conservatively; an
    /// in-place call (identical pointer and nonzero stride) stays parallel.
    fn accumulates(&self) -> bool {
        let last = i128::try_from(self.len).unwrap_or(i128::MAX) - 1;
        // SAFETY: NumPy passes a pointer and an outer stride per operand, the
        // inputs and then the output.
        let operand = |j: usize| unsafe { (*self.args.add(j), *self.steps.add(j)) };
        let span = |(pointer, step): (*mut c_char, npy_intp)| {
            let start = i128::try_from(pointer.addr()).unwrap_or_default();
            let end = start + i128::try_from(step).unwrap_or_default() * last;
            (start.min(end), start.max(end))
        };
        let output = operand(self.nin);
        let (low, high) = span(output);
        self.len > 1
            && (0..self.nin).map(operand).any(|input| {
                let (start, end) = span(input);
                start <= high && low <= end && !(input == output && input.1 != 0)
            })
    }

    /// Evaluate every element and store it in the output.
    ///
    /// # Safety
    /// A vector `O` has the signature's only core dimension.
    pub(super) unsafe fn store(
        &self,
        parallel: usize,
        kernel: impl Fn(usize) -> Result<O> + Sync,
    ) -> Result<()> {
        let component = if O::VECTOR {
            // SAFETY: Guaranteed by the caller.
            unsafe { self.core_stride(0) }
        } else {
            0
        };
        let output = self.output;
        // SAFETY: Indices stay below the loop count.
        let store = |index, value: O| unsafe { value.store(output, index, component) };
        // SAFETY: NumPy passes an outer stride per input.
        if self.len > 1 && (0..self.nin).all(|j| unsafe { *self.steps.add(j) } == 0) {
            // Every element is the same deterministic function of one input row.
            let value = kernel(0)?;
            (0..self.len).for_each(|index| store(index, value));
            return Ok(());
        }
        let parallel = if self.accumulates() {
            usize::MAX
        } else {
            parallel
        };
        drive(self.len, parallel, kernel, store)
    }
}

type LoopFn = unsafe extern "C" fn(*mut *mut c_char, *mut npy_intp, *mut npy_intp, *mut c_void);
/// A `NumPy` loop whose first parameter names its operand types.
type TypedLoop<I, O> = unsafe extern "C" fn(Args<I, O>, *mut npy_intp, *mut npy_intp, *mut c_void);

/// A loop with the dtype row (inputs, then the output) of its operand types.
#[derive(Clone, Copy)]
pub(super) struct Loop {
    function: LoopFn,
    /// The row; [`Loop::new`] panics beyond nine inputs, at compile time in
    /// constants such as [`UFUNCS`](super::registry::UFUNCS).
    types: [c_char; 10],
    nin: usize,
    /// Components of a vector output.
    pub(super) components: Option<usize>,
}

impl Loop {
    pub(super) const fn new<I: Inputs, O: Group>(function: TypedLoop<I, O>) -> Self {
        let (mut types, mut nin, mut group) = ([0; 10], 0, 0);
        while group < I::GROUPS.len() {
            let (dtype, count) = I::GROUPS[group];
            let end = nin + count;
            while nin < end {
                types[nin] = dtype;
                nin += 1;
            }
            group += 1;
        }
        types[nin] = O::Dtype::TYPE;
        Self {
            // SAFETY: `Args` is a transparent wrapper of the operand pointer
            // array, so both function pointer types have the same ABI.
            function: unsafe { std::mem::transmute::<TypedLoop<I, O>, LoopFn>(function) },
            types,
            nin,
            components: if O::VECTOR { Some(O::COUNT) } else { None },
        }
    }

    /// The dtype row.
    pub(super) const fn types(&self) -> &[c_char] {
        self.types.split_at(self.nin + 1).0
    }
}

/// Evaluate `kernel` for every index, then store the values in index order.
///
/// Parallel evaluation collects every value before the first store, and serial
/// evaluation reads element `i` before writing it, so outputs may alias inputs.
pub(super) fn drive<O: Send>(
    len: usize,
    parallel: usize,
    kernel: impl Fn(usize) -> Result<O> + Sync,
    mut store: impl FnMut(usize, O),
) -> Result<()> {
    if len >= parallel && treams_core::threads::current_num_threads() > 1 {
        let values = treams_core::threads::install(|| {
            let chunk = (len / rayon::current_num_threads().saturating_mul(4)).max(1);
            (0..len)
                .into_par_iter()
                .with_min_len(chunk)
                .map(&kernel)
                .collect::<Result<Vec<_>>>()
                .map_err(|error| {
                    // Report the first failing element, as the serial loop does.
                    match (0..len)
                        .into_par_iter()
                        .map(&kernel)
                        .find_first(Result::is_err)
                    {
                        Some(Err(first)) => first,
                        _ => error,
                    }
                })
        })?;
        for (index, value) in values.into_iter().enumerate() {
            store(index, value);
        }
    } else {
        for index in 0..len {
            store(index, kernel(index)?);
        }
    }
    Ok(())
}

/// Run one loop body without unwinding into `NumPy`, then report its error.
///
/// Special functions set incidental floating-point flags for finite results;
/// with `clear_fp`, these are cleared so `NumPy` does not warn. Results and
/// finite-value checks are authoritative.
pub(super) fn guard(clear_fp: bool, body: impl FnOnce() -> Result<()>) {
    ieee(|| {
        let result = catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|_| {
            Err(Error::SpecialFunction(
                "native special-function loop panicked".into(),
            ))
        });
        // A failed loop discards its output, so flags it set must not turn its
        // exception into NumPy's floating-point warning.
        if (clear_fp || result.is_err())
            && let Some((_, clear)) = FP_CLEAR.get()
        {
            // SAFETY: UFunc C API slot 27 is PyUFunc_clearfperr, a void(void)
            // function that only clears this thread's flags; the capsule is retained.
            unsafe { clear() };
        }
        if let Err(error) = result {
            // A raw C callback reports failures as the Python exception of its
            // thread. NumPy may run the loop with the GIL released while the
            // calling PyO3 wrapper still counts the thread as attached, so
            // `Python::attach` alone would not take the GIL back.
            // SAFETY: NumPy only calls loops while the interpreter runs.
            // PyGILState_Ensure is re-entrant: it reacquires a GIL that NumPy
            // released on this thread and only counts one this thread holds.
            let state = unsafe { pyo3::ffi::PyGILState_Ensure() };
            Python::attach(|py| crate::context::error(error).restore(py));
            // SAFETY: Restores the state saved by the matching call above,
            // on the same thread; no Python token outlives it.
            unsafe { pyo3::ffi::PyGILState_Release(state) };
        }
    });
}

static FP_CLEAR: OnceLock<(Py<PyCapsule>, unsafe extern "C" fn())> = OnceLock::new();

/// Cache `NumPy`'s `PyUFunc_clearfperr`, which [`guard`] calls after a loop.
pub(super) fn load_fp_clear(py: Python<'_>) -> PyResult<()> {
    let capsule = py
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
    Ok(())
}

/// Create a one-output ufunc of loops of one arity, with the core signature
/// `core` or, for vector outputs, the derived `(),...->(K)`.
///
/// `NumPy` keeps the loop and type tables, name and doc for the lifetime of
/// the ufunc and copies only the signature, so the tables are leaked once per
/// module initialization (`PyO3` forbids initializing the module twice).
pub(super) fn create<'py>(
    py: Python<'py>,
    name: &str,
    loops: &[Loop],
    core: Option<&str>,
    doc: &'static CStr,
) -> PyResult<Bound<'py, PyAny>> {
    let invalid = || PyValueError::new_err(format!("invalid loops of ufunc {name}"));
    let &[first, ..] = loops else {
        return Err(invalid());
    };
    let (nin, components) = (first.nin, first.components);
    // One arity for all loops; vector outputs take only their derived core.
    let shape = (nin, components);
    if loops.iter().any(|l| (l.nin, l.components) != shape) || core.and(components).is_some() {
        return Err(invalid());
    }
    let vector = components.map(|k| format!("{}->({k})", vec!["()"; nin].join(",")));
    let signature = core.map(str::to_owned).or(vector).map(CString::new);
    let signature = signature.transpose()?;
    let count =
        |n: usize| c_int::try_from(n).map_err(|_| PyValueError::new_err("ufunc is too large"));
    let (ntypes, nin) = (count(loops.len())?, count(nin)?);
    let name: &CStr = Box::leak(CString::new(name)?.into_boxed_c_str());
    let functions: &mut [_] = Box::leak(loops.iter().map(|l| Some(l.function)).collect());
    let types: &mut [c_char] = Box::leak(loops.iter().flat_map(Loop::types).copied().collect());
    // SAFETY: The leaked tables live for the process and hold one row of nin
    // inputs and one output per loop, derived from the operand types that loop
    // reads and writes; vector outputs have only their derived core dimension.
    // Name and doc are NUL-terminated strings that live for the process. The
    // result is a new reference or NULL with a Python error set.
    unsafe {
        let pointer = PY_UFUNC_API.PyUFunc_FromFuncAndDataAndSignature(
            py,
            functions.as_mut_ptr(),
            std::ptr::null_mut(),
            types.as_mut_ptr(),
            ntypes,
            nin,
            1,
            -1,
            name.as_ptr(),
            doc.as_ptr(),
            0,
            signature.as_deref().map_or(std::ptr::null(), CStr::as_ptr),
        );
        Bound::from_owned_ptr_or_err(py, pointer)
    }
}
