# Requested illuminations and larger clusters

This page retains the September measurements. For the subsequent CPU changes
against treams-rs 0.1.0, including multiple simultaneous illuminations and a
strongly coupled case, see [CPU speedups](cpu-speedups.md).

These results compare three ways to solve a sphere cluster for a few incident
fields: the full interacting T-matrix ("Full T"), a dense LU solve for the
requested columns only ("Selected LU") and a matrix-free GMRES solve. The grid
has nine cases with 46 isolated worker measurements, recorded on 2026-09-12 on
two hosts:

| Host | CPU | Numerical workers |
| --- | --- | --- |
| Linux | AMD Ryzen 9 9950X | 4, pinned to CPUs 8–11 |
| Mac | Apple M3 (8 cores, 24 GiB RAM) | 4 |

Both hosts use the same Python sources, benchmark script, complex128 precision
and GMRES tolerance `1e-10`.

## Results

For 512 weakly scattering spheres, `lmax=1`, one illumination and 3,072 channels:

| Host | Method | Fresh solve (s) | Reuse (ms) | Physical pullback (s) | Forward peak RSS (MiB) | Adjoint peak RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|
| [Linux](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-linux-n512-l1-p1.json) | Full T | 0.940 | 2.684 | 1.885 | 640.4 | 1367.7 |
| [Linux](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-linux-n512-l1-p1.json) | Selected LU | 0.381 | 4.317 | 0.764 | 351.6 | 861.6 |
| [Linux](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-linux-n512-l1-p1.json) | Matrix-free | 0.544 | 541.090 | 0.711 | 44.9 | 45.6 |
| [Mac](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-mac-n512-l1-p1.json) | Full T | 2.902 | 2.495 | 6.342 | 690.7 | 1461.0 |
| [Mac](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-mac-n512-l1-p1.json) | Selected LU | 0.959 | 6.571 | 0.741 | 367.1 | 919.5 |
| [Mac](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-mac-n512-l1-p1.json) | Matrix-free | 0.794 | 854.252 | 1.070 | 41.0 | 42.1 |

On Linux, the selected LU solve is **2.47×** faster than the full T-matrix for
a fresh solve. The matrix-free physical pullback, the reverse pass that gives
gradients with respect to radii, positions, wavenumber and incident
amplitudes, uses **30× less peak memory** than the full-T path and 18.9× less
than the selected-LU path. On the Mac, the full-T reduction is 34.7×. These are
peaks for the whole process, including the Python runtime and matrix storage.

For 1,024 spheres and 6,144 channels, forward and adjoint GMRES both converge
in six iterations:

| Host | Fresh solve (s) | Physical pullback (s) | Adjoint peak RSS (MiB) | Radius-direction derivative relative error |
|---|---:|---:|---:|---:|
| [Linux](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-linux-n1024-l1-p1.json) | 2.192 | 2.826 | 47.9 | 2.19e-08 |
| [Mac](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-mac-n1024-l1-p1.json) | 3.377 | 4.411 | 44.9 | 2.19e-08 |

Dimension 6,144 exceeds the dense limit of 3,072, so this case has no dense
reference and no same-size dense speedup. True residuals, physical gradient
invariants and an independent finite difference check it instead.

Matrix-free is not the fastest choice for every workload. For 128 spheres with
moderate scattering and four illuminations on Linux, it takes 0.208 s against
0.0183 s for a fresh selected LU. With a stored full T-matrix or LU, further
responses take milliseconds, while matrix-free recomputes the translations.
Choose the method by memory budget, number of illuminations and convergence.

The rest of the grid, including the slower matrix-free cases:

- 128 spheres, dipole order, one weak-scattering illumination:
  [Linux](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-linux-n128-l1-p1.json),
  [Mac](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-mac-n128-l1-p1.json).
- 128 spheres, dipole order, four moderate-scattering illuminations:
  [Linux](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-linux-n128-l1-p4-moderate.json),
  [Mac](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-mac-n128-l1-p4-moderate.json).
- 64 spheres, `lmax=2`, two moderate-scattering illuminations:
  [Mac](https://github.com/yaugenst/treams-rs/blob/de582b9ab44368cb363a2e3fa85f1fdcccd5242e/benchmarks/results/illumination-mac-n64-l2-p2-moderate.json).

Across the cases with a reference, physical gradients of the different
treams-rs methods differ by at most `6.58e-11` relative, and outputs differ from
treams by at most `1.99e-15`. Radius-direction finite differences agree within
`3.02e-8` relative in all nine cases. These checks cover the nine stated
configurations, not arbitrary high-order or resonant clusters.

## Method

Each method runs its forward and adjoint measurements in fresh processes with
limited BLAS and Rayon threads: one warmup and three timed samples. Each result
records SHA-256 hashes of the native library, Python sources and benchmark
script. Setup (geometry, assembly, factorization), a complete fresh solve, a
solve that reuses geometry or factors, and the complete physical pullback are
timed separately. Building the plane-wave coefficients is timed on its own and
left out of the solver comparisons. Peak RSS includes imports, inputs, warmup
and the recorded run; the finite-difference check runs after memory is measured.

All methods use the same geometry, multipole cutoff and requested plane-wave
columns. The full-cluster API of treams-rs gives the full-T reference, and small
cases are also checked against treams 0.4.5. Matrix-free forward and adjoint
solves recompute their true residuals and enforce their tolerances. Physical
cotangents (gradients with respect to the outputs) are compared across methods,
checked against translation, scaling and illumination-amplitude invariants,
and, in the largest matrix-free case, against a radius-direction finite
difference. A multipole cutoff of one is a dipole model, not a converged
multipole expansion.

## Reproduce

```sh
just build-ext-release
uv run --no-sync python scripts/benchmark_illumination.py \
  --particles 512 --lmax 1 --columns 1 --threads 4 \
  --output benchmarks/results/local/illumination-n512-l1-p1.json
uv run --no-sync python scripts/benchmark_illumination.py \
  --particles 1024 --lmax 1 --columns 1 --threads 4 \
  --output benchmarks/results/local/illumination-n1024-l1-p1.json
```

`benchmarks/results/local/` is ignored by git; the recorded results above stay
unchanged.
