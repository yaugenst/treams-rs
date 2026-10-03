---
description: Which Rust module, binding file, Python module and test directory own each part of treams-rs, and the rules that hold across them.
---

# Source ownership

treams-rs has three layers. The Rust crate `treams-core` computes every value
and every derivative. The bindings crate `treams-py` builds the extension
module `treams_rs._native`: it converts NumPy arrays and holds the data that a
gradient needs. The Python package `treams_rs` gives the results their physical
meaning: bases, media, polarization conventions and named results. The
[design pages](../design/index.md) explain why.

## Ownership table

Each row names the files that change together. Paths are relative to
`crates/treams-core/src/`, `crates/treams-py/src/`, `python/treams_rs/` and the
repository root. Run the Python checks as `uv run --no-sync pytest ...` after
`just build-ext`.

| Rust module (`treams-core`) | Binding file (`treams-py`) | Python module | Tests | Focused check |
|---|---|---|---|---|
| `fpenv`, `numerics` | `lib.rs`, `context.rs`, `convert.rs`, `broadcast.rs`, `args.rs`, `ufunc/` | `_native.pyi` | `tests/bindings/` | `pytest tests/bindings`; `cargo test -p treams-core -- fpenv numerics` |
| `linalg` | `linalg.rs` | `diff.py` (`solve`, `eig`, `svdvals`) | `tests/linalg/`, `properties/linalg.rs` | `pytest tests/linalg`; `cargo test -p treams-core -- linalg` |
| `special` | `special.rs`, `integrals.rs`, `coordinates.rs`, `ufunc/registry.rs` | `special.py` | `tests/special/`, `properties/special.rs` | `pytest tests/special`; `cargo test -p treams-core -- special` |
| `lattice` | `lattice.rs` | `lattice.py`, `_lattice.py`, `_periodic.py`, `misc.py` | `tests/lattice/`, `properties/lattice/` | `pytest tests/lattice`; `cargo test -p treams-core -- lattice` |
| `sw`, `cw`, `basis`, `rotation`, `vectorwaves`, `fields` | `translation.rs`, `expansion.rs`, `rotation.rs`, `vectorwaves.rs`, `fields.rs` | `sw.py`, `cw.py`, `_bases.py`, `_modes.py`, `_waves.py`, `_fields.py` | `tests/waves/`, `properties/waves.rs` | `pytest tests/waves`; `cargo test -p treams-core -- sw:: cw:: vectorwaves fields properties::waves` |
| `pw`, `channels` | `plane.rs`, `channels.rs` | `pw.py`, `_bases.py` (`PlaneWaveBasis`, `PlaneWavePorts`), `_waves.py` (plane waves) | `tests/plane/`, `properties/plane.rs` | `pytest tests/plane`; `cargo test -p treams-core -- channels properties::plane` |
| `coeffs`, `tmatrix`, `ebcm` | `coeffs.rs`, `tmatrix.rs`, `ebcm.rs` | `coeffs.py`, `ebcm.py`, `_material.py`, `_tmatrix.py` | `tests/tmatrix/`, `properties/tmatrix.rs` | `pytest tests/tmatrix`; `cargo test -p treams-core -- coeffs ebcm properties::tmatrix` |
| `cluster` | `cluster.rs`, `iterative.rs` | `_cluster.py`, `iterative.py` | `tests/tmatrix/`, `properties/tmatrix.rs` | `pytest tests/tmatrix/test_illumination.py tests/tmatrix/test_iterative.py tests/tmatrix/test_particle_cluster.py`; `cargo test -p treams-core -- cluster:: properties::tmatrix` |
| `smatrix` | `smatrix.rs` | `_smatrix.py`, `coeffs.py` (`fresnel`) | `tests/smatrix/`, `properties/smatrix.rs` | `pytest tests/smatrix`; `cargo test -p treams-core -- smatrix` |
| - | - | `_array.py`, `_operators.py`, `_operator_objects.py`, `operators.py`, `_polarization.py`, `_results.py`, `_validation.py` | `tests/api/`, `tests/waves/test_polarization.py` | `pytest tests/api/test_operators.py tests/api/test_matrix_methods.py tests/api/test_physics_api.py` |
| - | every record and context | `diff.py`, `_records.py`, `testing.py` | `tests/autodiff/test_testing.py`, `tests/autodiff/test_adjoint_identities.py`, the `gradients` tests of every domain | `pytest -m gradients` |
| - | - | `advect.py`, `jax.py`, `torch.py`, `_framework.py`, `_framework_backend.py`, `_framework_waves.py`, `_framework_tmatrix.py`, `_framework_smatrix.py` | `tests/autodiff/` | `pytest tests/autodiff` |
| - | - | `__init__.py`, `__main__.py`, `_catalog.py`, `_upstream.py` | `tests/api/test_support_catalog.py`, `tests/api/test_namespaces.py`, `tests/api/test_docs.py` | `pytest tests/api/test_support_catalog.py tests/api/test_namespaces.py`; `just docs-check` |
| - | - | `io.py` | `tests/api/test_io.py` | `pytest tests/api/test_io.py` |
| `test_support` | `testing.rs` (`*_jet` hooks) | - | `tests/_support.py`, `tests/_scripts.py`, `tests/test_suite_rules.py` | `pytest tests/test_suite_rules.py` |

The [crate docs](https://yaugenst.github.io/treams-rs/rust/treams_core/#module-map)
and the [design crosswalk](../design/index.md#crosswalk) pair each Rust module
with its `treams_rs` and treams namespaces. [Testing](testing.md) describes the
test directories.

After a change to Rust or to the bindings, run `just build-ext`, then the
module's Rust tests and its Python test directory. The Python tests check what
the Rust tests cannot: broadcasting, memory layouts, argument checks and
complete workflows through the public API.

## Rules

These rules hold everywhere:

- **Floating-point guard.** The whole body of every `#[pyfunction]` and
  `#[pymethods]` function is one `treams_core::fpenv::ieee(|| ...)` call. It
  keeps subnormal numbers when the caller flushes them to zero, as JAX does.
  `tests/bindings/test_float_environment.py` checks every body
  ([floating-point environment](../design/floating-point.md)).
- **One thread pool.** Every Rayon parallel iterator, `rayon::join` and faer
  call with parallelism runs inside `treams_core::threads::install` (or its
  `join`, `dense` and `product` helpers), on a pool that treams-rs owns, never
  on Rayon's global pool. The pool starts lazily, is rebuilt in a forked child,
  and its workers keep subnormals.
  `tests/bindings/test_thread_pool.py` rejects other parallel code in `crates/`,
  and `tests/conftest.py` fails the session if any test started Rayon's global
  pool ([parallelism](../design/parallelism.md)).
- **Unsafe code.** The workspace denies `unsafe_code`. Only
  `treams_core::fpenv` (the floating-point control register) and the NumPy
  ufunc code of `treams-py` (`ufunc/ffi.rs` and `ufunc/loops.rs`) allow it,
  with a `SAFETY` comment on every block. `treams-py`'s `threads.rs` allows it
  for one item, the unmangled `treams_rs_num_threads` that threadpoolctl looks
  up.
- **Analytic gradients.** Every derivative is computed analytically in Rust.
  Finite differences appear only in tests and in `treams_rs.testing`.
- **The stub.** `python/treams_rs/_native.pyi` declares every name of
  `treams_rs._native` with its parameters.
  `tests/bindings/test_native_contexts.py` and
  `tests/bindings/test_ufunc_contract.py` compare the stub with the module.
- **One API catalog.** `treams_rs.support_catalog()` reads the public modules
  and their docstrings. The reference pages, the generated tables and
  `llms.txt` come from it through `just docs`; never edit them by hand, and
  keep no second list of capabilities.
- **Recorded evidence.** Nothing under `benchmarks/results/` changes
  ([benchmarks](benchmarks.md)).
- **Renames.** Every public rename gets a bullet under "Unreleased" in
  `CHANGELOG.md`. A public name that differs from its treams counterpart also
  gets an entry in `python/treams_rs/_upstream.py`, which feeds the
  [name map](../coming-from-treams/names.md) and the `AttributeError` for the
  treams name.

## Adding a binding

The crate docs of `treams-py`, in
[`crates/treams-py/src/lib.rs`](../../crates/treams-py/src/lib.rs), hold the
checklist under "Adding a binding". In short: the core function and its tests
come first; the binding goes into the file of its core module, with its
context and its `pullback`; then the export line, the stub, a `CASES` entry in
`tests/bindings/test_native_contexts.py` (or a `REGISTRY_ROWS` row for a
ufunc), the Python caller and `just docs`.

## Adding a physics feature

A feature travels through every layer. For a new function `X` that computes a
value (a forward):

1. **Rust.** Write `X` in the module of its layer in `treams-core`. If it
   supports gradients, it returns `(value, XResidual)`, where the residual
   holds what the gradient needs. When the pullback reads the value, it
   returns only `XResidual` with a `value()` accessor, as `coeffs::mie` does
   (see 'Names' in the crate docs). Its pullback,
   `XResidual::pullback(self, cotangent)`, turns the cotangent (the gradient of
   a real loss with respect to the value) into the input gradients `XGradient`
   ([glossary](../reference/glossary.md#records-and-gradients)). Test the
   implementation in the module's inline `tests` module and the physics in
   `properties/<domain>.rs`.
2. **Binding.** Follow the checklist above.
3. **Python.** Add the record, a function that returns the value and a context
   for its gradients, to `diff.py`: `diff.x(...) -> (value, XContext)`. Then
   add the upstream-style function of its namespace or the method of the
   physics object. Check label bounds and argument types in Python, with
   messages in treams terms.
4. **Framework adapters.** If a physics object exposes the feature, route it
   through `_framework_*.py`, so that Advect, JAX and PyTorch differentiate it.
5. **Tests.** Add tests to `tests/<domain>/`: a comparison with treams 0.4.5 or
   another reference, a physical identity, and `check_pullback` for the
   gradients ([adding a test](testing.md#adding-a-test)).
6. **Docs.** Write the docstring with units, shapes and conventions, add a
   runnable example to the matching guide page, update
   [capabilities](../validation/capabilities.md), run `just docs` and add a
   `CHANGELOG.md` bullet.

## Naming rules

The [glossary](../reference/glossary.md) gives one name per concept across Rust,
the bindings, Python and treams. The crate docs state the rules for each
layer: "Names" in the
[`treams-core` docs](https://yaugenst.github.io/treams-rs/rust/treams_core/#names)
and "Naming rules" in
[`crates/treams-py/src/lib.rs`](../../crates/treams-py/src/lib.rs). The rules
that cross layers:

- **Residuals and contexts.** The Rust forward `X` returns `XResidual`, and its
  pullback returns `XGradient`. The native context of the record `diff.x` is
  `<X>Context`, the diff name in CamelCase: `diff.smatrix_tr` returns an
  `SMatrixTrContext`.
- **Value-only twins.** A function that returns the value of a record without a
  context ends in `_value`: `smatrix::tr_value` in Rust,
  `smatrix_tr_value` in the bindings.
- **treams counterparts.** The doc of a Rust item that mirrors treams ends with
  ``Upstream: `treams.sw.translate`.``, followed by `Differences: ...` where it
  deviates. A Python docstring says ```Mirrors ``treams.sw.translate``.``` and
  names its differences. A module without a counterpart says
  "treams-rs extension".
- **Labels.** The degree is `l` and the order is `m`; radial kinds are
  "regular" and "singular"; two-basis functions take `destination` and
  `source`; expansion centres are `positions`.
