# Rust CUDA Tile plane fields

`treams-cuda-tile` evaluates a weighted plane-wave expansion directly at requested
points, in complex double precision:

\[
\mathbf E(\mathbf r_i)=\sum_m c_m\mathbf e_m
\exp(i\mathbf k_m\cdot\mathbf r_i).
\]

The kernel accepts complex wavevectors and real Cartesian points, including
evanescent and lossy waves. The CPU core computes the polarization vectors, so
helicity/parity labels and normalization have one implementation. CUDA Tile
evaluates the complex exponential and accumulates the three field components.
Each tile owns its output points. Modes remain on the device between evaluations.
There is no point-by-mode field matrix. Explicit device arrays occupy
`128 * modes + 96 * points` bytes; CUDA context, JIT, and allocator overhead are
additional. This is a forward field path; it does not implement a GPU pullback.

The crate has **no default features or runtime dependencies**. CPU and WASM builds leave
`cuda-tile` disabled. Enabling it brings in the exact `cutile = 0.3.1` release and
the shared CPU core on Linux. An all-features build on macOS or WASM still
excludes these dependencies and this Linux-only API. CUDA Tile is an optional
backend, not a CPU fallback.

## Measured result

[Recorded qualification](../crates/treams-cuda-tile/qualification-rtx4080.json)
on an RTX 4080 SUPER and Ryzen 9950X, with 16 physical CPU cores:

| Modes | Points | CPU | GPU, including transfers | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 16,384 | 0.983 ms | 0.698 ms | 1.41× |
| 256 | 131,072 | 28.761 ms | 14.014 ms | 2.05× |
| 1,024 | 262,144 | 225.370 ms | 85.162 ms | 2.65× |

The largest case evaluates 268 million point-mode pairs with 24.1 MiB of explicit
device arrays. Its full three-component sampling operator would occupy 12 GiB;
the existing CPU weighted-field API also avoids that operator. Absolute field
errors against the CPU core were at most `3.5e-16` in these cases. First-call
times were 0.45–0.53 seconds and are not included in the warm speedup. The GPU
benefit is measured against the parallel Rust CPU implementation in the same
precision, not against a full-matrix Python implementation. Ordinary background
host services remained running; no competing GPU compute process was observed.

## Build and qualify

Use Linux, stable Rust 1.94+, an NVIDIA GPU with compute capability at least 8.0,
and CUDA Toolkit 13.3. The build requires the CUDA and cuRAND headers and
libclang for binding generation. Kernel JIT requires `tileiras`; the current
cuTile compiler requires version 13.2 or newer. Set `CUDA_TOOLKIT_PATH` to the
toolkit root and put its `bin` and `lib` directories on `PATH` and
`LD_LIBRARY_PATH`. No system installation is required.

```sh
cargo test -p treams-cuda-tile --release --features cuda-tile -- --ignored
RAYON_NUM_THREADS=16 cargo run -p treams-cuda-tile --release \
  --features cuda-tile --example qualify_plane -- 64 16384
```

The benchmark compares against the existing parallel CPU weighted-field path.
It reports first-call time separately from seven warm evaluations. Warm times
include point uploads, output downloads, and finite-result validation; uploaded
mode data are retained. Tests compare complex fields against the CPU core across
tile boundaries and check linearity and point permutation for both polarization
conventions. A 32-case proptest independently checks the complex plane-wave
translation identity, including attenuation, with shrinking and replay.

## Choice of NVIDIA Rust track

The requested [NVIDIA CUDA Rust article](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/)
recommends starting with Tile and using SIMT where explicit thread and memory
control is needed. This regular field computation fits Tile and does not need
cuda-oxide's pinned nightly compiler. Numerical work remains `f64`; the tensor
core half-precision throughput quoted in cuTile's examples does not describe
this electromagnetic workload.

Primary references: [cuTile Rust 0.3.1 source](https://github.com/NVlabs/cutile-rs/tree/2eed75e8f552be31216ddf2019288f03dfec4939),
[CUDA Tile 13.3 operations and precision](https://docs.nvidia.com/cuda/tile-ir/13.3/sections/operations.html),
and [CUDA 13.3 redistributions](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.0.json).
