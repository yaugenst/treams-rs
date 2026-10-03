//! `NumPy` ufuncs that run the Rust kernels, and their Python-scalar fast paths.
//!
//! `NumPy` owns broadcasting, masking, overlap buffering and output allocation.
//! The strided C loop calls the same Rust kernels as the record functions.
//!
//! - `ffi`: typed operands, the inner-loop driver, the guard that keeps panics
//!   and floating-point flags inside a loop, and ufunc creation. All raw-pointer
//!   access to operands happens here.
//! - `kinds`: the const-generic kind parameters of the loops and the inner-loop
//!   sizes from which loops run in parallel.
//! - `loops`: the inner loops, one per kernel family.
//! - `registry`: names, loops, core signatures and docs of the ufuncs.
//! - `fast_paths`: entry points that skip `NumPy` dispatch for single elements.
//!   Wrappers such as `car2sph` take a ufunc's `*args, **kwargs` and pass any
//!   other call to a hidden ufunc; typed functions such as `hankel_scalar`
//!   evaluate one element, and the Python namespaces call them for scalar
//!   arguments.

pub(crate) mod fast_paths;
mod ffi;
mod kinds;
mod loops;
mod registry;

use pyo3::prelude::*;

use fast_paths::{FALLBACKS, Fallbacks};
use ffi::{create, load_fp_clear};
use kinds::ewald::{FULL, REAL, RECIPROCAL};
use loops::Eta;
use registry::{HIDDEN, UFUNCS, Ufunc, coordinate_ufuncs, lattice_family};

/// Add every ufunc and the renamed wrappers to `module`, and keep the hidden
/// ufuncs of the fast paths.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    load_fp_clear(py)?;

    for ufunc in UFUNCS {
        let function = create(py, ufunc.name, ufunc.loops, ufunc.core, ufunc.doc)?;
        module.add(ufunc.attribute, function)?;
    }
    lattice_family::<Eta<FULL>>(module, "lsum", "The Ewald sum")?;
    lattice_family::<Eta<REAL>>(module, "realsum", "The real-space part of the Ewald sum")?;
    lattice_family::<Eta<RECIPROCAL>>(
        module,
        "recsum",
        "The reciprocal-space part, with the self correction, of the Ewald sum",
    )?;
    lattice_family::<i64>(module, "dsum", "Shell ``i`` of the direct sum")?;

    // Wrappers that a namespace exposes directly carry the public name under a
    // unique attribute. pickle finds a function through `__module__` and
    // `__name__`, so `__module__` names that namespace.
    for (attribute, namespace, function) in [
        (
            "pw_translate",
            "treams_rs.pw",
            wrap_pyfunction!(fast_paths::pw_translate, module)?,
        ),
        (
            "cell_volume",
            "treams_rs.lattice",
            wrap_pyfunction!(fast_paths::cell_volume, module)?,
        ),
        (
            "cell_reciprocal",
            "treams_rs.lattice",
            wrap_pyfunction!(fast_paths::cell_reciprocal, module)?,
        ),
    ] {
        function.setattr("__module__", namespace)?;
        module.add(attribute, function)?;
    }

    let hidden = |ufunc: Ufunc| {
        Ok::<_, PyErr>(create(py, ufunc.name, ufunc.loops, ufunc.core, ufunc.doc)?.unbind())
    };
    let [pw_translate, vpw_m, vpw_n, vpw_a, volume, reciprocal] = HIDDEN.map(hidden);
    let _ = FALLBACKS.set(Fallbacks {
        pw_translate: pw_translate?,
        plane: [vpw_m?, vpw_n?, vpw_a?],
        cells: [volume?, reciprocal?],
        coordinates: [
            coordinate_ufuncs::<0>(py)?,
            coordinate_ufuncs::<1>(py)?,
            coordinate_ufuncs::<2>(py)?,
            coordinate_ufuncs::<3>(py)?,
            coordinate_ufuncs::<4>(py)?,
            coordinate_ufuncs::<5>(py)?,
            coordinate_ufuncs::<6>(py)?,
            coordinate_ufuncs::<7>(py)?,
        ],
    });
    Ok(())
}
