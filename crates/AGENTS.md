# Rust and bindings

Read [source ownership](../docs/development/architecture.md) and the crate docs in `treams-core/src/lib.rs` and `treams-py/src/lib.rs`.

- The whole body of every `#[pyfunction]` and `#[pymethods]` fn is one `fpenv::ieee(|| ...)` call; the treams-rs pool (`treams_core::threads`) starts lazily inside it, and every parallel region runs on it.
- `unsafe` appears only in `treams_core::fpenv` and `treams-py`'s `ufunc/ffi.rs` and `ufunc/loops.rs`, with a `SAFETY` comment on every block; `treams-py`'s `threads.rs` allows `unsafe_code` for the unmangled `treams_rs_num_threads` alone.
- `python/treams_rs/_native.pyi` declares every export; follow "Adding a binding" in `treams-py/src/lib.rs`.
- Implementation tests go inline, physics and gradient properties in `src/properties/<domain>.rs` ([testing](../docs/development/testing.md#rust)).
