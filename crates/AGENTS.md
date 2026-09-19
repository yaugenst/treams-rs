# Rust and bindings

Read [development](../docs/development.md) for check commands and
[architecture](../docs/architecture.md) before changing numerical ownership.

- `treams-core` owns shared mathematics, complex128 conventions and analytic
  pullbacks. Keep Python numerical execution in this crate.
- `treams-py` owns array conversion, GIL release and context transfer. Update
  `python/treams_rs/_native.pyi` with exposed signatures. Fix shared conversion
  defects at that boundary and test the supported strided Python inputs.
- Numerical changes need a native physical/algebraic or adjoint check. Use
  proptest over valid bounded domains with shrinking, and preserve regressions.
  Rebuild bindings and exercise the corresponding Python workflow too.
- Run `just rust-fmt-check rust-lint rust-test` and the corresponding Python tests.
