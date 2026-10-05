//! Python bindings of treams-core: the extension module `treams_rs._native`.
//!
//! The bindings turn `NumPy` arrays and Python values into core inputs, run the
//! core with the GIL released, and return `NumPy` arrays and pullback contexts.
//! Every formula and every analytic derivative lives in treams-core; this crate
//! converts arrays, checks shapes and sums gradients over broadcast axes. Users
//! never import `_native`: the package `treams_rs` wraps it, and the stub
//! `python/treams_rs/_native.pyi` declares every name it exports.
//!
//! # Glossary
//!
//! A record is a function that returns a value and a reusable context. The context stores what is needed to compute derivatives later: `context.pullback(g)` takes the gradient `g` of a real-valued loss with respect to the value and returns the gradients with respect to the inputs, one for each differentiable input, in the order of the arguments. Gradients follow the convention dL = Re Σ conj(g)·dx.
//!
//! In the Rust core, a function returns `(value, XResidual)`, and `XResidual::pullback(&self, cotangent)` returns the input gradients as `XGradient`.
//!
//! - *Core residual*: the `XResidual` that the core function `X` returns beside its
//!   value. It holds what derivatives need; pushforwards and pullbacks borrow it.
//! - *Binding context*: a Python class, `<DiffName>Context`, that owns one core
//!   residual, plus the shapes or flags its derivatives need. Derivative methods
//!   validate tangents or cotangents and borrow the residual, allowing repeated
//!   calls in either direction. Rejected inputs leave the context unchanged.
//! - *Record*: a native function or method that returns `(value, context)`.
//! - *Pullback*: `context.pullback(cotangent)`, which returns the gradients with
//!   respect to the differentiable inputs.
//! - *Cotangent*: the gradient `g` of the loss with respect to the value. It has the
//!   shape of the value.
//! - *Real pairing*: the convention dL = Re Σ conj(g)·dx. A real input gets a real
//!   gradient, and the pullback of a real output reads only the real part of `g`
//!   (`convert::RealCotangent`).
//! - `<DiffName>Context`, `_record` and `_scalar`: the names of contexts, broadcast
//!   records and Python-scalar fast paths; see the naming rules below.
//!
//! # Conventions
//!
//! - **Floating-point mode.** The whole body of every `#[pyfunction]` and
//!   `#[pymethods]` fn is one `treams_core::fpenv::ieee(|| ...)` call. A caller may
//!   flush subnormals to zero, as XLA does inside `jax.pure_callback`; `ieee` keeps
//!   them for the native work, argument checks included, and then restores the
//!   caller's mode. `tests/bindings/test_float_environment.py` scans these sources
//!   and fails on any other body, and `scripts/float_environment.py` compares every
//!   scalar binding and ufunc loop on a flushing thread with an ordinary one. The
//!   pool of `treams_core::threads` starts lazily, in the first parallel region;
//!   its workers keep subnormals whatever the mode of the thread that starts them.
//! - **The GIL.** Native work runs with the GIL released, through
//!   `context::detached`, which also raises core errors as `context::error` maps
//!   them. Calls of about a microsecond keep the GIL, because releasing and taking
//!   it back would cost a noticeable share of them: the 0-d records `*_record_scalar`
//!   (`broadcast::record_held`), the Python-scalar fast paths of
//!   `ufunc/fast_paths.rs`, `plane_polarization`, the `dimension` getters and
//!   `build_profile`. Odd orders of `intkambe_record_scalar` sum incomplete-gamma
//!   series and release it.
//! - **Unsafe code.** The workspace denies `unsafe_code`. In this crate only
//!   `ufunc/ffi.rs` (the `NumPy` C interface and all raw-pointer access to
//!   operands) and `ufunc/loops.rs` (the inner loops that `NumPy` calls) allow it,
//!   with a `SAFETY` comment on every block. `threads.rs` allows it for one item,
//!   the unmangled `treams_rs_num_threads` that threadpoolctl looks up. In
//!   treams-core only `fpenv` does.
//! - **The stub and its tests.** `_native.pyi` declares every export with its
//!   parameters. `tests/bindings/test_native_contexts.py` compares the stub with
//!   the module and runs one `CASES` entry per context method: deterministic
//!   records, reusable contexts that own their inputs, and every memory layout.
//!   `tests/bindings/test_ufunc_contract.py` requires the stub to declare exactly
//!   the module, keeps the dtype rows and core signature of every ufunc in
//!   `REGISTRY_ROWS`, and runs one case per ufunc. `scripts/float_environment.py`
//!   calls every scalar binding that the stub declares.
//! - **Errors.** A core `Error` raises with its message (`context::error`):
//!   `OutOfMemory` as `MemoryError`, every other variant as `ValueError`. Every
//!   input check of this crate raises `ValueError`, and a C-order copy that the
//!   system refuses (`convert::matrix`) raises `MemoryError`. A cotangent of the
//!   wrong shape or with a non-finite entry raises "cotangent must be finite with
//!   shape (2, 2, 3, 3)"; an input error names its argument, as in "points must
//!   have shape (N, 3)".
//! - **Memory order.** Outputs may be F-ordered or carry the strides of
//!   column-major storage. A matrix that the residual does not keep moves into an
//!   array without a copy (`convert::owned_matrix`); a matrix that the residual
//!   keeps leaves as a C-ordered copy (`convert::matrix`). Python code must not
//!   assume C order.
//! - **Validation.** Python validates for messages and treams compatibility; Rust
//!   validates for safety. The Python layer checks label bounds, argument types
//!   and the conventions treams users know, and raises messages in their terms.
//!   The bindings check every shape and length they index with, and the core
//!   checks its own domain, so a direct `_native` call with a wrong shape raises
//!   `ValueError` instead of reading out of bounds.
//! - **Gradient order.** A pullback returns one gradient per differentiable
//!   argument of the `treams_rs.diff` function that calls the record, in the order
//!   of that function's arguments: `diff.sphere_cluster(lmax, k0, radii, epsilon,
//!   positions)` gives `(k0, radii, epsilon, positions)`. A basis argument gives the
//!   gradient of its positions. One gradient comes alone, several as a tuple.
//!   `PeriodicToCwContext` puts the gradient of the destination kz after ks, and
//!   the methods below append their extra items at the end.
//! - **Rust practice.** Fix a conversion defect here, once, and test the supported
//!   strided inputs. A numerical change belongs in treams-core and needs a physical,
//!   algebraic or adjoint check there (proptest over valid bounded domains, with
//!   preserved regressions), and a run of the matching Python workflow against a
//!   rebuilt extension.
//!
//! # Pullback methods
//!
//! - `InteractionFactor.record` returns an `IlluminateContext`, whose pullback
//!   method the factor fixes at construction. A factor of one dense local matrix,
//!   `InteractionFactor(local, coupling)`, takes `pullback`, which returns
//!   `(local, coupling, incident)`. A factor of separate local blocks,
//!   `InteractionFactor.from_blocks` or `sphere_cluster_factor`, takes
//!   `pullback_blocks`, which returns the list of block gradients first. The other
//!   method raises `ValueError`.
//! - `IterativeContext.pullback` returns `(k0, radii, epsilon, positions, incident,
//!   convergence)`: the gradients in the order of `IterativeSphereCluster(lmax, k0,
//!   radii, epsilon, positions).record(incident)`, then the GMRES convergence of
//!   each adjoint solve as `(iterations, residual_norm, rhs_norm)`, which is no
//!   gradient. The forward reports its convergence the same way: `solve` returns
//!   `(value, convergence)` and `record` returns `(value, context, convergence)`.
//! - `pullback_axial` of `ExpansionContext`, `LatticeExpansionContext`,
//!   `FieldContext` and `FieldOperatorContext` serves cylindrical bases only. It
//!   returns the `pullback` tuple and then the gradients of the axial wavenumbers
//!   kz: one per sorted distinct kz for expansions, one per mode for fields.
//! - `EigContext.pullback(eigenvalues, eigenvectors)` and
//!   `BandsContext.pullback(wavenumbers, eigenvectors)` take one cotangent per
//!   output.
//!
//! # Naming rules
//!
//! - A native function that serves a `treams_rs.diff` function has its name. The
//!   cylindrical twin that a diff function calls for cylindrical bases starts with
//!   `cylindrical_` (`cylindrical_expansion`).
//! - Broadcast records end in `_record` (`bessel_record`), and their 0-d variants
//!   in `_record_scalar` (`bessel_record_scalar`).
//! - A context is `<DiffName>Context`, the diff name in `CamelCase` with `SMatrix`
//!   and `TMatrix` spelled so: `diff.smatrix_tr` returns an `SMatrixTrContext`. The
//!   spherical and cylindrical twins share one context.
//! - Solver objects record with a method named `record`, whose context is that of
//!   the matching diff function (`InteractionFactor.record` returns an
//!   `IlluminateContext`) or a short `<Class>Context` (`IterativeContext`). Their
//!   method `solve` computes the value without a context. A function that returns
//!   the value of a record without a context ends in `_value`
//!   (`smatrix_illuminate_value`, `smatrix_tr_value`).
//! - Ufuncs keep the upstream names. A variant behind a Python function is
//!   `<ns>_<upstream function>_<variant>`: `sw_translate_sh` serves `sw.translate`
//!   for singular waves in the helicity basis. A ufunc that a namespace exposes as
//!   itself reports the public name as `__name__`, while its `_native` attribute
//!   stays unique: the attribute `cw_periodic_to_pw` holds the ufunc
//!   `periodic_to_pw`. The wrappers `pw_translate`, `cell_volume` and
//!   `cell_reciprocal` report `translate`, `volume` and `reciprocal` the same way.
//! - `<ufunc or family>_scalar` is a Python-scalar fast path that records nothing:
//!   `cw_rotate_scalar` replaces a one-element call of `cw_rotate`. `hankel_scalar`
//!   (`hankel1`, `hankel2`) and `angular_scalar` (`lpmv`, `pi_fun`, `tau_fun`)
//!   serve a family.
//! - Test hooks end in `_jet` and live in `testing.rs`.
//! - Parameters use the names of the diff functions: `singular`, `destination` and
//!   `source`, `destination_positions` and `source_positions`, `positions`,
//!   `argument_shapes`, `kpar`, `a`, `r`, `eta`, `kzs`, `zs`, `direction` and
//!   `function`.
//!
//! # Adding a binding
//!
//! 1. Write the core function in treams-core, with its tests. A record returns
//!    `(value, XResidual)`.
//! 2. Add the binding to the file of its core module: a `#[pyfunction]` with a
//!    one-line doc that names the core function, whose whole body is one
//!    `ieee(|| ...)` call and whose core call runs in `detached`. Define its context
//!    with `context!` (`broadcast_context!` for broadcast records) and the
//!    `pullback` in a `#[pymethods]` block. A ufunc instead takes a loop in
//!    `ufunc/loops.rs` and a row with its doc literal in `ufunc/registry.rs`.
//! 3. Export it in the `#[pymodule_export] use` line of its file below.
//! 4. Declare it in `python/treams_rs/_native.pyi`.
//! 5. Add its tests: a `CASES` entry per context method in
//!    `tests/bindings/test_native_contexts.py`, or for a ufunc a `REGISTRY_ROWS` row
//!    and a `CASES` entry in `tests/bindings/test_ufunc_contract.py`.
//! 6. Call it from `treams_rs.diff` or a namespace, and regenerate the reference
//!    and `llms.txt` with `just docs`.
//!
//! Before a commit, run `just rust-fmt-check rust-lint rust-test` and the Python
//! tests of the changed functions.
// PyO3 extracts owned borrow guards by value; array shapes are validated before indexing.
#![allow(clippy::needless_pass_by_value, clippy::indexing_slicing)]

// Shared infrastructure: pullback contexts, broadcasting, array conversion and
// argument parsing. The domain files follow by layer of treams-core.
mod args;
mod broadcast;
mod context;
mod convert;

// L0: numerical support.
mod linalg;
mod threads;

// L1: special functions (treams.special).
mod coordinates;
mod integrals;
mod special;

// L2: lattice sums and lattice geometry (treams.lattice).
mod lattice;

// L3: wave families (expansions, translations, rotations, channels, fields).
mod channels;
mod expansion;
mod fields;
mod plane;
mod rotation;
mod translation;
mod vectorwaves;

// L4: scattering (Mie coefficients, T-matrices, EBCM, clusters).
mod cluster;
mod coeffs;
mod ebcm;
mod iterative;
mod tmatrix;

// L5: planar and periodic S-matrices (treams SMatrices).
mod smatrix;

// Hooks for tests and scripts.
mod testing;

// NumPy ufuncs and their Python-scalar fast paths.
mod ufunc;

use pyo3::prelude::*;

// Free-threaded CPython keeps the GIL enabled until the module is audited for it.
#[pymodule(gil_used = true)]
mod _native {
    use pyo3::prelude::*;

    // One `use` per binding file, in the order of the `mod` declarations.
    #[pymodule_export]
    use super::linalg::{
        EigContext, EigvalsContext, SolveContext, SvdvalsContext, eig, eigvals, solve, svdvals,
    };
    #[pymodule_export]
    use super::threads::{after_fork, set_num_threads, thread_info};

    #[pymodule_export]
    use super::coordinates::{
        CoordinatesContext, VectorCoordinatesContext, coordinates_record, vector_coordinates_record,
    };
    #[pymodule_export]
    use super::integrals::{
        IncgammaContext, IntkambeContext, incgamma_context, incgamma_record,
        incgamma_record_scalar, intkambe_context, intkambe_record, intkambe_record_scalar,
    };
    #[pymodule_export]
    use super::special::{
        AngularContext, BesselContext, WignerdContext, angular_context, angular_record,
        angular_record_scalar, bessel_context, bessel_record, bessel_record_scalar,
        wignerd_context, wignerd_record, wignerd_record_scalar,
    };

    #[pymodule_export]
    use super::lattice::{
        LatticeExpansionContext, LatticeExpansionFromTableContext, LatticeSumContext,
        cylindrical_lattice_expansion, diffraction_orders, first_brillouin, lattice_cube,
        lattice_expansion, lattice_expansion_from_table, lattice_sum_record,
    };

    #[pymodule_export]
    use super::channels::{
        CylindricalChannelsContext, SphericalChannelsContext, cylindrical_channels,
        spherical_channels,
    };
    #[pymodule_export]
    use super::expansion::{
        ExpansionContext, PeriodicToCwContext, cw_to_sw, cylindrical_expansion, expansion,
        periodic_to_cw,
    };
    #[pymodule_export]
    use super::fields::{
        FieldContext, FieldOperatorContext, cylindrical_field, cylindrical_field_operator, field,
        field_operator,
    };
    #[pymodule_export]
    use super::plane::{
        PlaneExpansionContext, PlaneFieldContext, PlanePermutationContext, PlanePhasesContext,
        cylindrical_plane_expansion, plane_expansion, plane_field, plane_permutation, plane_phases,
        plane_polarization,
    };
    #[pymodule_export]
    use super::rotation::{RotationContext, cylindrical_rotation, rotation};
    #[pymodule_export]
    use super::translation::{
        CylindricalTranslationContext, SphericalTranslationContext,
        cylindrical_translation_context, cylindrical_translation_record,
        spherical_translation_context, spherical_translation_record,
    };
    #[pymodule_export]
    use super::vectorwaves::{VectorWaveContext, vector_wave_context, vector_wave_record};

    #[pymodule_export]
    use super::cluster::{
        IlluminateContext, InteractionContext, InteractionFactor, ParticleClusterContext,
        SphereClusterContext, cylindrical_particle_cluster, interaction, particle_cluster,
        sphere_cluster, sphere_cluster_factor,
    };
    #[pymodule_export]
    use super::coeffs::{MieContext, MieCylContext, mie, mie_cyl};
    #[pymodule_export]
    use super::ebcm::{EbcmQmatContext, ebcm_qmat};
    #[pymodule_export]
    use super::iterative::{IterativeContext, IterativeSphereCluster};
    #[pymodule_export]
    use super::tmatrix::{
        CylinderContext, SphereContext, TMatrixMetricContext, cylinder, sphere, tmatrix_metric,
    };

    #[pymodule_export]
    use super::smatrix::{
        BandsContext, ChiralityDensityContext, FresnelContext, InterfaceCoefficientsContext,
        LayerStackContext, OrientedChiralityContext, PropagationMatrixContext, SMatrixAddContext,
        SMatrixFromArrayContext, SMatrixIlluminateContext, SMatrixPeriodicContext,
        SMatrixTrContext, bands, chirality_density, fresnel, interface_coefficients, layer_stack,
        oriented_chirality, propagation_matrix, smatrix_add, smatrix_from_array,
        smatrix_illuminate, smatrix_illuminate_value, smatrix_periodic, smatrix_tr,
        smatrix_tr_value,
    };

    #[pymodule_export]
    use super::testing::{
        build_profile, cartesian_translation_jet, cylindrical_cartesian_translation_jet,
        cylindrical_radial_jet, rayon_global_pool_unused, run_flushing_for_tests,
        spherical_wave_jet,
    };

    #[pymodule_export]
    use super::ufunc::fast_paths::{
        angular_scalar, car2cyl, car2pol, car2sph, cw_rotate_scalar, cw_translate_scalar, cyl2car,
        cyl2sph, dsumcw1d_scalar, dsumcw1d_shift_scalar, dsumcw2d_scalar, hankel_scalar,
        incgamma_scalar, intkambe_scalar, lpmv_real_scalar, pol2car, pw_permute_xyz_scalar,
        sph2car, sph2cyl, tl_vcw_scalar, vcar2cyl, vcar2pol, vcar2sph, vcyl2car, vcyl2sph,
        vpol2car, vpw_a, vpw_m, vpw_n, vsph2car, vsph2cyl, wigner3j_scalar,
    };

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        super::ufunc::register(module)
    }
}
