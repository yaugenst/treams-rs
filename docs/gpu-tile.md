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
The three complex components accumulate separately, using explicit nearest-even
f64 fused multiply-add with gradual underflow. Exactly real wavevectors skip the
attenuation exponential; a nonzero imaginary part, however small, uses the full
complex-wave path. Trigonometric functions retain their full f64 implementation.
There is no point-by-mode field matrix. Explicit device arrays occupy
`128 * modes + 96 * points` bytes; CUDA context, JIT, and allocator overhead are
additional. This is a forward field path; it does not implement a GPU pullback.

The crate has **no default features or runtime dependencies**. CPU and WASM builds leave
`cuda-tile` disabled. Enabling it brings in the exact `cutile = 0.3.1` release and
the shared CPU core on Linux. An all-features build on macOS or WASM still
excludes these dependencies and this Linux-only API. CUDA Tile is an optional
backend, not a CPU fallback.

## Measured result

[Field qualification](../benchmarks/gpu-fields-qualification.json) uses an RTX
4080 SUPER and Ryzen 9950X. Both CPU and GPU retain coefficient-weighted
polarizations; the CPU also receives the exact real-wave shortcut. The stronger
CPU comparison was selected from 16 and 32 workers on all 16 physical cores.
The original CPU API is timed separately in the raw results.

| Modes | Points | Wavevectors | Prepared CPU | GPU, including transfers | Speedup |
| ---: | ---: | --- | ---: | ---: | ---: |
| 64 | 16,384 | Complex | 0.946 ms | 0.620 ms | 1.53× |
| 256 | 131,072 | Complex | 23.128 ms | 11.320 ms | 2.04× |
| 1,024 | 262,144 | Complex | 182.434 ms | 66.630 ms | 2.74× |
| 1,024 | 262,144 | Real | 137.604 ms | 49.731 ms | 2.77× |

The largest case evaluates 268 million point-mode pairs with 24.1 MiB of explicit
device arrays. Its full three-component sampling operator would occupy 12 GiB;
the existing CPU weighted-field API also avoids that operator. Absolute field
errors against the CPU core were at most `3.4e-16` in these cases. First-call
times were 0.54–0.61 seconds and are not included in the warm speedup. The larger
kernel takes roughly 0.1 seconds longer to compile initially. Repeated complex
field calls take 12–22% less time than the original GPU implementation across
these three shapes. Ordinary background host services remained running; no
competing GPU compute process was observed.

The [initial qualification](../crates/treams-cuda-tile/qualification-rtx4080.json)
is retained for provenance. The field report includes a prepared CPU baseline and a controlled kernel probe. For the largest case, the original kernel took 78.1 ms;
separate component accumulation took 65.8 ms, and explicit f64 FMA reduced that
to 57.9 ms before transfers and host processing. Tiling the mode reduction was
slower than separate component accumulation and was not adopted. Retaining
points alone could not explain a dramatic gain: point upload and field download
accounted for about 4 ms in that decomposition.

## Build and qualify

Use Linux, stable Rust 1.94+, an NVIDIA GPU with compute capability at least 8.0,
and CUDA Toolkit 13.3. The build requires the CUDA and cuRAND headers and
libclang for binding generation. Kernel JIT requires `tileiras`; the pinned
cuTile compiler requires version 13.2 or newer. Set `CUDA_TOOLKIT_PATH` to the
toolkit root and put its `bin` and `lib` directories on `PATH` and
`LD_LIBRARY_PATH`. No system installation is required.

```sh
cargo test -p treams-cuda-tile --release --features cuda-tile -- --ignored
RAYON_NUM_THREADS=16 cargo run -p treams-cuda-tile --release \
  --features cuda-tile --example qualify_plane -- 64 16384
```

The benchmark compares against both the existing parallel CPU weighted-field
path and a prepared CPU loop with the same reusable data as the GPU.
It reports first-call time separately from seven warm evaluations. Warm times
include point uploads, output downloads, and finite-result validation; uploaded
mode data are retained. Tests compare complex fields against the CPU core across
tile boundaries and check linearity and point permutation for both polarization
conventions. A 32-case proptest independently checks the complex plane-wave
translation identity, including attenuation, with shrinking and replay.
Additional checks exercise 129-mode boundaries, cancellation between separated
opposite expansions, and attenuation of `1e-12` over a path of length `1e12`.

For many coefficient updates at fixed points, [cached sampling](gpu-sampling.md)
provides another option with explicit memory and setup costs. [The opportunity
assessment](gpu-opportunities.md) separates further GPU work from algorithmic
improvements that would also benefit the CPU.

## Choice of NVIDIA Rust track

The [NVIDIA CUDA Rust article](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/)
recommends starting with Tile and using SIMT where explicit thread and memory
control is needed. This regular field computation fits Tile and does not need
cuda-oxide's pinned nightly compiler. Numerical work remains `f64`; the tensor
core half-precision throughput quoted in cuTile's examples does not describe
this electromagnetic workload.

Primary references: [cuTile Rust 0.3.1 source](https://github.com/NVlabs/cutile-rs/tree/2eed75e8f552be31216ddf2019288f03dfec4939),
[CUDA Tile 13.3 operations and precision](https://docs.nvidia.com/cuda/tile-ir/13.3/sections/operations.html),
and [CUDA 13.3 redistributions](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.0.json).
