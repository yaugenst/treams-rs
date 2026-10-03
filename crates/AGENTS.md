# Rust and bindings

Read [source ownership](../docs/development/architecture.md) and the crate docs in `treams-core/src/lib.rs` and `treams-py/src/lib.rs`.

- The whole body of every `#[pyfunction]` and `#[pymethods]` function is one `fpenv::ieee(|| ...)` call. The treams-rs thread pool (`treams_core::threads`) starts on first use inside that call; every parallel region runs on this pool.
- `unsafe` appears only in `treams_core::fpenv` and `treams-py`'s `ufunc/ffi.rs` and `ufunc/loops.rs`, with a `SAFETY` comment on every block; `treams-py`'s `threads.rs` allows `unsafe_code` for the unmangled `treams_rs_num_threads` alone.
- `python/treams_rs/_native.pyi` declares every export; follow "Adding a binding" in `treams-py/src/lib.rs`.
- Implementation tests go inline, physics and gradient properties in `src/properties/<domain>.rs` ([testing](../docs/development/testing.md#rust)).
