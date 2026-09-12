//! Optional CUDA linear algebra using the same complex128 matrices as the CPU core.
//!
//! With default features this crate contains no CUDA dependency or runtime code.
//! Enable `cuda` explicitly and use `gpu` to retain operators, LU factors,
//! right-hand sides and pullbacks on one CUDA stream.

#[cfg(feature = "cuda")]
pub mod gpu;
