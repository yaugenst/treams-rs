# NumPy array functions

This module exposes core kernels as NumPy universal functions (ufuncs).
NumPy handles broadcasting, `where` masks, overlapping inputs and outputs,
and output allocation. The Rust loops read each input and call the same
kernels as the gradient-recording functions.

| File | Job |
|---|---|
| [`mod.rs`](mod.rs) | Register the functions in `treams_rs._native`. |
| [`registry.rs`](registry.rs) | Define names, accepted data types, array dimensions and docstrings. |
| [`loops.rs`](loops.rs) | Connect each kernel family to NumPy's loop interface. |
| [`ffi.rs`](ffi.rs) | Read and write array elements, create ufuncs and keep Rust panics inside the C interface. |
| [`kinds.rs`](kinds.rs) | Name kernel variants and the array sizes that justify parallel execution. |
| [`fast_paths.rs`](fast_paths.rs) | Call kernels directly for scalar arguments; send array options through NumPy. |

Raw-pointer access stays in `ffi.rs`; `loops.rs` calls it under NumPy's array
contracts. Each call preserves subnormal floating-point values. Parallel
loops collect results before writing them on the calling thread.

[`test_ufunc_contract.py`](../../../../tests/bindings/test_ufunc_contract.py)
checks names, signatures, data types and array behavior. The remaining
[binding tests](../../../../tests/bindings/) check floating-point modes and
thread use. See the [binding guide](../../README.md) when adding a function.
