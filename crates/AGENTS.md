# Rust and bindings

Read [development](../docs/development.md) for check commands and
[architecture](../docs/architecture.md) before changing numerical ownership.

- `treams-core` owns shared mathematics, complex128 conventions and analytic
  pullbacks. Reuse its operators across Python and optional GPU paths.
- `treams-py` owns array conversion, GIL release and context transfer. Update
  `python/treams_rs/_native.pyi` with exposed signatures. Fix shared conversion
  defects at that boundary and test the supported strided Python inputs.
- `treams-cuda` and `treams-cuda-tile` own device execution, buffers and transfers.
  Keep both optional and absent from default CPU builds. Share mathematics
  with the core; do not silently reduce precision or fall back to CPU execution.
- Numerical changes need a native physical/algebraic or adjoint check. Use
  proptest over valid bounded domains with shrinking, and preserve regressions.
  Rebuild bindings and exercise the corresponding Python workflow too.
- Run `just rust-fmt-check rust-lint rust-test`. Follow the development guide's
  GPU lane when those targets are affected; a successful host compilation
  alone does not prove GPU execution.
