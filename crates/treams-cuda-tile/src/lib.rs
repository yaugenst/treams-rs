//! Optional pure-Rust CUDA Tile kernels. CPU and WASM builds leave `cuda-tile` off.

#[cfg(all(feature = "cuda-tile", target_os = "linux"))]
mod plane;

#[cfg(all(feature = "cuda-tile", target_os = "linux"))]
pub use plane::PlaneWaves;
