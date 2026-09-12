# Python and adapters

Read [development](../docs/development.md) for setup and test routing and
[architecture](../docs/architecture.md) for the native boundary.

- `treams_rs` owns physics metadata, array semantics and framework adapters.
  Numerical execution and analytic derivatives belong in `crates/treams-core`.
- Keep `_native.pyi` synchronized with PyO3 signatures and context results.
  Public functions need useful signatures and docstrings, including units,
  shapes, conventions and pullback ordering where applicable.
- Preserve ordinary NumPy arrays, broadcasting and supported strides. Validate
  at the owning boundary; use the shared binding conversion rather than adding
  a Python copy workaround for a native conversion defect.
- Optional Advect/JAX/PyTorch, HDF5 and CUDA imports must remain optional.
  Reuse `_adapters.py` for framework composition; retain the documented CPU,
  first-order and one-use native context contracts.
- Update capability metadata at its source and regenerate documentation for API
  changes. Do not maintain a second support list in prose or generated files.
- Run the matching Python suite and `just py-lint py-format-check py-types`.
  Rebuild the extension first if Rust or bindings changed. Use Hypothesis for
  meaningful boundary invariants and the public derivative helpers for custom
  recording functions; see [testing](../docs/testing.md).
