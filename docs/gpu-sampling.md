# Repeated sampling and coefficient pullbacks on CUDA

At fixed wavevectors and observation points, changing an illumination only
changes its coefficients: `E = F c`. Keeping the complex128 sampling matrix `F`
on the GPU gives a larger acceleration than repeatedly evaluating every phase.
This is useful for beam shaping or repeated illuminations when the matrix fits
in memory and enough applications amortize its construction and upload.

Qualification on an RTX 4080 SUPER and Ryzen 9950X measured the following warm
applications. Every GPU round trip includes coefficient upload,
output allocation, synchronization, and downloading all three field components.
CPU timings include output allocation. Both processors can reuse the same `F`.

| Points × modes | Cached operator | Best CPU alternative | Rust GPU round trip | Speedup | Python GPU round trip |
| --- | --- | --- | --- | --- | --- |
| 16,384 × 512 | 384 MiB | 5.046 ms | 0.709 ms | 7.12× | 0.785 ms |
| 32,768 × 1,024 | 1.5 GiB | 19.483 ms | 2.516 ms | 7.74× | 2.600 ms |

The CPU comparison includes native weighted fields, a prepared weighted sum with
cached polarizations, cached faer products in both layouts, and independent
SciPy/OpenBLAS products in both layouts. Thread counts 1, 4, 8, 16, and 32 were
tested; the prepared weighted sum with 32 workers was fastest in both cases.
Thus the headline comparison also gives the CPU the option to recompute phases
and avoid storing `F`. The Python calls are 6.43× and 7.49× faster than that same
Rust CPU baseline. These are median timings of seven changing coefficient
vectors on an otherwise idle machine, not universal speedup guarantees.

The points form an irregular three-dimensional cloud. Wavevectors are real,
nonaxial, unit length, and span the upper hemisphere, with alternating helicity.
`F` comes from the existing native plane-field implementation. The largest
forward discrepancy was below 1.2e-16 in the Rust sampling grid. No FP32, tensor
core approximation, or relaxed-accuracy Fourier transform is involved.

## Setup and memory matter

Python operator construction took 73 ms and 279 ms. Device initialization plus
operator upload took 490 ms and 1,143 ms, and the first applications took another
24 ms. Using those observed costs and the warm medians, the Python path needs
approximately **138 or 86 forward applications** to recover the cold setup cost
against the prepared CPU sum. These estimates assume steady timings and charge
the GPU for constructing `F`; they are not measured optimization-loop totals.
The Rust API has lower observed setup costs, with estimates of 79 and 39
applications. A warm context or an already required sampling matrix changes
that tradeoff. The reports retain each setup component rather than hiding it in
a resident-only speedup.

The operator consumes 16 × 3 × points × modes bytes. Forward evaluation also
needs its coefficients and output, while the adjoint needs its output cotangent
and coefficient gradient. Library workspaces and allocator state are additional.
The benchmark's process RSS includes two host layouts and transfer staging; it
is not a measurement of production peak VRAM. For 262,144 points and 1,024 modes,
`F` alone would take 12 GiB. The [fused field path](gpu-tile.md), which avoids this
matrix, remains useful for larger maps and occasional evaluations. There is no
automatic caching policy or claim that avoiding a matrix is unique to GPUs.

## Reuse the same matrix for coefficient gradients

For fixed `F`, a field cotangent `g_E` pulls back to `Fᴴ g_E`. The optional
Hermitian product uses the original device allocation; a second transposed
operator is neither built nor uploaded.

```python
from treams_rs import cuda, diff

device = cuda.Device()
operator, _ = diff.plane_field(None, points, vectors, polarizations, poltype="helicity")
resident = device.upload(operator.reshape(-1, len(vectors)))
field = device.matmul(resident, device.upload(coefficients[:, None])).numpy()
field = field.reshape(-1, 3)

coefficient_cotangent = (
    device.matmul(
        resident,
        device.upload(field_cotangent.reshape(-1, 1)),
        adjoint_left=True,
    )
    .numpy()
    .ravel()
)
```

The Rust round-trip adjoints took 0.707 ms and 2.640 ms, including uploading the
complete field cotangent and downloading the coefficient gradient. The fastest
cached faer adjoints across the tested layouts and thread counts took 6.712 ms
and 28.554 ms: 9.49× and 10.81× slower. **That adjoint comparison is against
cached CPU products only; a prepared fused CPU adjoint was not measured.**
Actual Python round trips took 0.783 ms and 2.776 ms. GPU-resident adjoint times,
which exclude transfers, are recorded separately in the raw results.

This is a coefficient-only pullback for a fixed operator. It does not supply
wavevector or geometry derivatives, a cuTile field recording API, or automatic
GPU execution through the JAX, PyTorch, or Advect adapters. Both left and right
Hermitian flags are available for general rectangular device matrix products.

## Reproduce and qualify

The [machine-readable qualification](../benchmarks/gpu-fields-qualification.json)
links the full thread grid, independent BLAS check, Python API timings, numerical
errors, source hashes, and binary hashes. Earlier exploratory Cartesian-grid
measurements are retained as pilots and excluded from the headline comparison.

```bash
# Source your CUDA 13.3 environment first.
uv run --no-sync maturin develop --release --features cuda-tile
cargo build --release -p treams-cuda --features cuda --example benchmark_sampling
for threads in 1 4 8 16 32; do
    RAYON_NUM_THREADS=$threads taskset -c "0-$((threads - 1))" \
        target/release/examples/benchmark_sampling 16384 512
done
RAYON_NUM_THREADS=32 taskset -c 0-31 uv run --no-sync python \
    scripts/benchmark_sampling_blas.py 16384 512 --cuda
```

Use 32,768 points and 1,024 modes for the larger case. CPU affinity is specific
to the recorded host's topology. Focused hardware tests cover rectangular
Hermitian products, complex inner-product pairing, physical field coefficient
pullbacks, strided Python inputs, and Hypothesis-generated shapes. The cuTile
tests also cover real and lossy waves, cancellation, tile boundaries, and a tiny
nonzero imaginary wavevector over a long propagation distance.
