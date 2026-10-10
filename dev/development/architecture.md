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
| - | - | `_dispatch.py`, `_promotion.py`, `_autodiff_functions.py`, `advect.py`, `jax.py`, `torch.py`, `autograd.py`, `_framework*.py` | `tests/autodiff/` | `pytest tests/autodiff` |
| - | - | `__init__.py`, `__main__.py`, `_catalog.py`, `_upstream.py` | `tests/api/test_support_catalog.py`, `tests/api/test_namespaces.py`, `tests/api/test_docs.py` | `pytest tests/api/test_support_catalog.py tests/api/test_namespaces.py`; `just docs-check` |
| - | - | `io.py` | `tests/api/test_io.py` | `pytest tests/api/test_io.py` |
| `test_support` | `testing.rs` (`*_jet` hooks) | - | `tests/_support.py`, `tests/_scripts.py`, `tests/test_suite_rules.py` | `pytest tests/test_suite_rules.py` |

The [crate docs](../rust/treams_core/#module-map)
and the [design module table](../design/index.md#crosswalk) pair each Rust module
with its `treams_rs` and treams namespaces. [Testing](testing.md) describes the
test directories.

After a change to Rust or to the bindings, run `just build-ext`, then the
module's Rust tests and its Python test directory. The Python tests check what
the Rust tests cannot: broadcasting, memory layouts, argument checks and
complete workflows through the public API.

## Rules

Use these rules for new and changed code. They describe the intended contracts;
the checks and open gaps below distinguish what is enforced today. The core's
[conventions](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/crates/treams-core/src/lib.rs) define names, numerical
conventions and derivative state in more detail.

- **Keep formulas in the core.** Rust owns numerical values, analytic
  pushforwards and pullbacks. Bindings and frameworks adapt ownership, arrays
  and execution. Finite differences belong only in tests and `treams_rs.testing`.
- **Follow the dependency layers.** Production code uses its own or lower
  layers in the core's module table; tests may cross layers. Keep one numerical
  core crate until measured build or distribution needs justify a split.
- **Establish invariants at boundaries.** Validate public inputs and restored
  state before indexing, allocating or computing. Prefer constructors and
  private fields that preserve validated domains over repeated checks or
  combinations of policy booleans. Bindings still validate their array shapes
  and Python still provides messages in treams terms; those are distinct duties.
- **Own only useful derivative state.** A residual keeps what later derivative
  calls need, with private invariant-bearing fields. Both `pushforward(&self, …)`
  and `pullback(&self, …)` borrow it for repeated calls at fixed primal inputs.
  Returned gradient records may be plain data. Consume an owned argument when
  its caller no longer needs it, as `linalg::solve_owned` does, and reuse its
  storage where possible; do not clone merely to fit an interface.
- **Keep saved state coherent.** Shape-derived sizes, writing and restoration
  must agree. Restore validates lengths, tags, domains and allocation sizes;
  round trips preserve derivatives and deferred errors, including zero
  directions. These bytes are private invocation state, not a persistent file
  format. A mechanical refactor preserves the layout; changing a payload or
  padding requires an explicit compatibility decision and matching `state_spec`.
- **Share concrete work.** Use the smallest helper or closure that removes
  actual duplicated assembly or numerical work. Avoid a generic residual
  hierarchy, speculative extension points or a second framework implementation
  of a core derivative.
- **Branch on error identity.** Use a typed reason when code branches on an
  error or encodes it; display wording must not decide behavior. Numerical
  fallback catches only the failures it can repair and propagates resource
  and programming errors. Document which failures leave partial output.
- **Specify parallel arithmetic.** Route parallel work through the owned
  `threads` pool. Define reduction chunks and accumulation order independently
  of the worker count. Wrap Python entry-point bodies in `fpenv::ieee`, and
  preserve the caller's floating-point mode, including failure and single-thread
  paths. Benchmark before changing thresholds or regrouping sums.
- **Make unsafe obligations local.** Keep unsafe code in the existing allowed
  modules, with minimal blocks and `SAFETY` comments. Express operand lifetimes,
  read/write ownership, aliasing and dtype layout in types where possible.
  Contain panics before they can cross a C ABI boundary; do not disguise them
  as numerical failures.
- **State numerical and resource limits.** Reserve advertised large outputs
  and workspaces fallibly with the existing `numerics` helpers. Define finite
  output and accuracy contracts, check both value and derivative paths, and
  return an appropriate error when a result cannot meet them. This is not a
  promise that every internal allocation is fallible.
- **Require evidence.** Start a numerical repair with a reproducible failing
  case and an independent reference or physical identity. Check adjoint
  pairings separately from finite differences, cover both sides of thresholds,
  and record measured reasons for tuning constants. Never loosen a tolerance
  to hide a failure. Measure proposed caches, allocation tradeoffs and numerical
  regrouping before adopting them; preserve archived measurements under the
  [evidence policy](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/benchmarks/README.md#privacy-and-provenance).

### One source and its checks

Each formula, layout, capability and shared limit has one owner. Generate other
representations from it, or check their agreement when generation would obscure
the code. Do not add a second hand-maintained registry to make a check pass.

| Contract | Existing example or check | Open work |
| --- | --- | --- |
| Core layers and saved state | [`SavedState`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/crates/treams-core/src/saved.rs), [`linalg` codecs](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/crates/treams-core/src/linalg/saved.rs), [`test_native_contexts.py`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/tests/bindings/test_native_contexts.py) | [#30](https://github.com/yaugenst/treams-rs/issues/30): codec composition, fallible writer and Rust-only layer/codec checks; [#27](https://github.com/yaugenst/treams-rs/issues/27): deferred metric errors |
| Validated domains and shared label bounds | [`BlochLattice`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/crates/treams-core/src/lattice/cell.rs) validates construction | [#32](https://github.com/yaugenst/treams-rs/issues/32): basis/residual invariants and Python/Rust bound agreement |
| Error identity, finite outputs and allocation | [`Error`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/crates/treams-core/src/lib.rs), [`numerics::memory`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/crates/treams-core/src/numerics/memory.rs) | [#33](https://github.com/yaugenst/treams-rs/issues/33): typed reasons and deterministic errors |
| Owned pool, IEEE mode and reduction order | [`test_thread_pool.py`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/tests/bindings/test_thread_pool.py), [`test_float_environment.py`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/tests/bindings/test_float_environment.py), [`parallel`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/crates/treams-core/src/numerics/parallel.rs) | [#11](https://github.com/yaugenst/treams-rs/issues/11), [#35](https://github.com/yaugenst/treams-rs/issues/35): one-thread, failure, foreign-pool and extraction paths; [#34](https://github.com/yaugenst/treams-rs/issues/34): ufunc ownership and reduction aliasing |
| Native names, contexts and ufunc signatures | `_native.pyi` checked against the module by [`test_native_contexts.py`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/tests/bindings/test_native_contexts.py) and [`test_ufunc_contract.py`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/tests/bindings/test_ufunc_contract.py) | Extend these checks with each binding |
| Public API and page inventory | `support_catalog()` reads modules/docstrings; `mkdocs.yml` owns page order. `just docs` generates reference pages, tables and `llms.txt`; `just docs-check` checks them | Never edit generated output by hand |
| Shared metadata and reference evidence | [`generate_references.py`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/scripts/generate_references.py), [`test_support`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/crates/treams-core/src/test_support.rs) | [#40](https://github.com/yaugenst/treams-rs/issues/40): coverage, determinism/default metadata agreement and fixture provenance enforcement |

The ordinary Python API selects an optional adapter in `_dispatch.py` before
array coercion. `_promotion.py` converts constant physics objects,
`_autodiff_functions.py` preserves NumPy callable contracts, and `_framework*.py`
objects compose records; adapters supply the framework bridge. Keep this map
current as part of [#9](https://github.com/yaugenst/treams-rs/issues/9), which owns
package restructuring and import checks.

Every public rename gets a bullet under "Unreleased" in `CHANGELOG.md`. A name
that differs from treams also gets an entry in `_upstream.py`, which owns the
[name map](../coming-from-treams/names.md) and the treams-name `AttributeError`.

## Adding a binding

The crate docs of `treams-py`, in
[`crates/treams-py/src/lib.rs`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/crates/treams-py/src/lib.rs), hold the
checklist under "Adding a binding". Start with the core function and its tests.
Add the binding, context, `pushforward` and `pullback` to the file for that core module.
Then add the export, stub, Python caller and a `CASES` entry in
`tests/bindings/test_native_contexts.py` (or a `REGISTRY_ROWS` row for a ufunc).
Finish with `just docs`.

## Adding a physics feature

For a new function `X` that computes a value:

1. **Rust.** Write `X` in the relevant `treams-core` module. If it
   supports gradients, it returns `(value, XResidual)`, where the residual
   holds what both derivative directions need. When the pullback reads the value, it
   returns only `XResidual` with a `value()` accessor, as `coeffs::mie` does
   (see 'Names' in the crate docs). Its pullback,
   `XResidual::pullback(&self, cotangent)`, turns the cotangent (the gradient of
   a real loss with respect to the value) into the input gradients `XGradient`
   ([glossary](../reference/glossary.md#records-and-gradients)). Add the
   pushforward with the matching input tangents as well. Test the
   implementation in the module's inline `tests` module and the physics in
   `properties/<domain>.rs`.
2. **Binding.** Follow the checklist above.
3. **Python.** Add the record, a function that returns the value and a context
   for its gradients, to `diff.py`: `diff.x(...) -> (value, XContext)`. Then
   add the upstream-style function of its namespace or the method of the
   physics object. Check label bounds and argument types in Python, with
   messages in treams terms.
4. **Framework adapters.** If a physics object exposes the feature, route it
   through `_framework_*.py`, so that Advect, JAX, PyTorch and HIPS Autograd differentiate it.
5. **Tests.** Add tests to `tests/<domain>/`: a comparison with the pinned
   treams version or another reference, a physical identity, and `check_pushforward`/`check_pullback`
   for the derivatives ([adding a test](testing.md#adding-a-test)).
6. **Docs.** Write the docstring with units, shapes and conventions, add a
   runnable example to the matching guide page, update
   [capabilities](../validation/capabilities.md), run `just docs` and add a
   `CHANGELOG.md` bullet.

## Naming rules

The [glossary](../reference/glossary.md) gives one name per concept across Rust,
the bindings, Python and treams. The crate docs state the rules for each
layer: "Names" in the
[`treams-core` docs](../rust/treams_core/#names)
and "Naming rules" in
[`crates/treams-py/src/lib.rs`](https://github.com/yaugenst/treams-rs/blob/3a54b233f9f9c4d32b3e96d7578a7730507be3b2/crates/treams-py/src/lib.rs). The rules
that cross layers:

- **Residuals and contexts.** The Rust function `X` returns `XResidual`, and its
  pullback returns `XGradient`. The native context of the record `diff.x` is
  `<X>Context`, the diff name in CamelCase: `diff.smatrix_tr` returns an
  `SMatrixTrContext`.
- **Value-only functions.** A function that returns the value of a record
  without a context ends in `_value`: `smatrix::tr_value` in Rust,
  `smatrix_tr_value` in the bindings.
- **treams counterparts.** The doc of a Rust item that mirrors treams ends with
  ``Upstream: `treams.sw.translate`.``, followed by `Differences: ...` where it
  deviates. A Python docstring says ```Mirrors ``treams.sw.translate``.``` and
  names its differences. A module without a counterpart says
  "treams-rs extension".
- **Labels.** The degree is `l` and the order is `m`; radial kinds are
  "regular" and "singular"; two-basis functions take `destination` and
  `source`; expansion centres are `positions`.
