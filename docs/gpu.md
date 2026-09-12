# Optional CUDA execution

CUDA is explicit and compiled out of default builds. The CPU core and WASM
crate depend on neither CUDA crate. `cuda` enables complex128 cuBLAS/cuSOLVER
operations; `cuda-tile` also enables the pure Rust field kernel described in
[GPU field qualification](gpu-tile.md). There is no reduced-precision mode or
automatic CPU/GPU fallback.

NVIDIA's [CUDA Rust announcement](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/)
describes Tile and SIMT programming models. The cuTile Rust kernel/JIT path
evaluates fused plane-wave superposition without a large intermediate operator.
Dense LU and matrix multiplication use NVIDIA's existing numerical libraries
through Rust bindings. Reimplementing those libraries in a new kernel would add
work without a demonstrated benefit. The alternative cuda-oxide SIMT track
requires a pinned nightly compiler and is not a dependency of this project.

The mathematical core is shared: plane-wave polarization and normalization,
complex128 matrix conventions, multipole operator assembly, and LU equilibration
are CPU-core code. GPU execution owns its stream, buffers, factors, and implicit
linear-solve pullback. The pairing remains
`dL = Re(sum(conj(g) * dX))`.

## Build and use

On Linux with an NVIDIA driver and CUDA 13 libraries:

```sh
uv run --no-sync maturin develop --release --features cuda
# Also compile the Rust field kernel; needs CUDA 13.3 headers/tileiras and libclang:
uv run --no-sync maturin develop --release --features cuda-tile
```

Dense execution requires `libcuda.so.1`, `libcublas.so.13`, and
`libcusolver.so.12` on the loader path. It does not need nvcc. cuTile additionally
needs CUDA headers during compilation and tileiras at runtime. CPU-only builds
need none of these. Even all-features builds compile on macOS without CUDA;
constructing a CUDA device there returns an explicit unsupported-host error.

```python
import numpy as np
from treams_rs import cuda

device = cuda.Device(0)
factor = device.factor(operator)  # host complex matrix; equilibrate, upload, LU once
solution = factor.solve(rhs)  # one or several requested RHS; returns NumPy
adjoint = factor.solve(cotangent, adjoint=True)

# Keep intermediate matrices on the device:
rhs_device = device.upload(rhs)
solution_device = factor.solve_device(rhs_device)
product_device = device.matmul(device.upload(operator), solution_device)
np.testing.assert_allclose(product_device.numpy(), rhs)

# A one-use first-order pullback shares the resident LU:
solution, context = factor.solve_with_pullback(rhs)
operator_bar, rhs_bar = context.pullback(cotangent)
```

`solve_device` copies its RHS on the device so the supplied matrix remains usable.
`numpy()` is an explicit download and synchronization. Factor and matrix `nbytes`
report owned resident arrays, excluding the allocator and temporary library
workspace. A dense operator gradient still costs O(n²) storage; use a requested
RHS solve and contract its adjoint directly when a full operator gradient is not
needed. The fused cuTile field kernel supports forward evaluation only. A fixed
sampling matrix supports a coefficient pullback through a resident Hermitian
product, as shown in [repeated sampling](gpu-sampling.md).

```python
expansion = cuda.PlaneWaves(wavevectors, polarizations, coefficients)
fields = expansion.evaluate(points)  # complex128, shape (number_of_points, 3)
```

Complex wavevectors include evanescent and lossy waves. The existing helicity and
parity conventions are selected by `poltype`. Only requested points are evaluated;
storage scales with points plus modes. The first call includes JIT initialization,
which must be accounted for separately in interactive applications.

## Qualification and performance boundaries

The RTX 4080 SUPER qualification uses the same complex128 precision as the CPU.
It checks pivoted solves, products, adjoint solves, implicit pullbacks, shared
equilibration, singular inputs and device ownership. The optional Python tests
also cover sphere-cluster scattering and the pure Rust field path.

```sh
cargo test -p treams-cuda --features cuda --release -- --ignored
cargo test -p treams-cuda-tile --features cuda-tile --release -- --ignored
TREAMS_TEST_CUDA=1 TREAMS_TEST_CUDA_TILE=1 uv run --no-sync pytest tests/test_cuda.py
cargo run -p treams-cuda --features cuda --release --example benchmark -- 4096 64 5
```

The field kernel has measured **1.53–2.77× end-to-end speedups** over a prepared
Rust CPU implementation, including point upload and field download. Initial JIT
cost is reported separately. See the [field proof](../benchmarks/gpu-fields-qualification.json).
For repeated coefficient updates, [cached physical sampling operators](gpu-sampling.md)
exercise the GPU's memory bandwidth and support the coefficient pullback with
`device.matmul(operator, cotangent, adjoint_left=True)`. Geometry and wavevectors
are fixed in that use case.

Dense operations have important crossovers on this consumer GPU. Small solves,
wide products and operations requiring a host round trip can be faster on the
CPU. Retained factors and matrices avoid repeated setup and transfers, but a fair
comparison must also let the CPU reuse its factors. Dense benchmark reports
therefore separate host-to-host work, resident work, and initialization, and sweep
CPU thread counts rather than choosing an unnecessarily slow CPU baseline.

The [dense proof and 25 raw runs](../benchmarks/gpu-qualification.json) sweep
1, 2, 4, 8 and 16 CPU threads, taking the best CPU median for each operation.
The same runs compare five repetitions per workload:

| Workload | Best CPU | GPU | CPU/GPU |
|---|---:|---:|---:|
| LU + 64 RHS, 4096 channels, including transfers | 442.47 ms | 389.24 ms | 1.14× |
| LU + 64 RHS, 1024 channels, including transfers | 12.00 ms | 17.87 ms | 0.67× |
| Reused LU, 64 RHS, 4096 channels, resident | 25.07 ms | 30.41 ms | 0.82× |
| Matrix action, 64 RHS, 4096 channels, resident | 20.75 ms | 10.29 ms | 2.02× |
| Matrix action, 512 RHS, 2048 channels, resident | 17.54 ms | 24.70 ms | 0.71× |

These dense workloads use synthetic well-conditioned complex matrices; the
separate sphere-cluster tests provide physical qualification. No comparison
credits the GPU with LU reuse while making the CPU refactor. The proof records
both the timing executable and the final validated extension/source hashes;
the final numerical tests include the subsequently corrected negative-stride
Python boundary and explicit sequential faer configuration.

The most promising GPU paths are large requested-point field evaluation,
repeated resident matrix actions, and many illuminations sharing an operator.
An 8-byte real or 16-byte complex representation is preserved throughout.
This GPU's 16 GiB device memory is a limit, not a reason to move every problem
to CUDA; CPU matrix-free execution remains useful for problems larger than VRAM.
