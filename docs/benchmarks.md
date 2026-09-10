# Finite-sphere cluster benchmarks

Measured on [redacted-host], AMD Ryzen 9 9950X (16 physical cores), Linux x86-64,
Python 3.13.1, treams 0.4.5. Rust uses the optimized build, faer and Rayon.
These are measured local results for the implemented sphere-cluster path,
not a performance claim about all of treams.

| Spheres | lmax | Matrix dimension | Threads | treams ms | Rust ms | Speedup | treams / Rust peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 3 | 240 | 1 | 88.23 | 3.01 | 29.3× | 71.8 / 44.2 |
| 16 | 3 | 480 | 1 | 389.64 | 13.91 | 28.0× | 90.3 / 57.2 |
| 8 | 5 | 560 | 1 | 686.49 | 22.88 | 30.0× | 98.4 / 65.1 |
| 32 | 3 | 960 | 1 | 1677.50 | 88.36 | 19.0× | 154.9 / 113.2 |
| 8 | 3 | 240 | 4 | 90.63 | 1.96 | 46.2× | 73.6 / 42.7 |
| 16 | 3 | 480 | 4 | 391.17 | 9.35 | 41.9× | 91.6 / 57.8 |
| 8 | 5 | 560 | 4 | 691.96 | 12.69 | 54.5× | 100.2 / 64.5 |
| 32 | 3 | 960 | 4 | 1636.14 | 43.41 | 37.7× | 156.4 / 113.6 |

Raw samples and environment details are in `benchmarks/results/release-*.json`.
Each row uses seven samples after a warmup. A separate correctness process first
compares the entire complex T-matrix with treams at `rtol=2e-9, atol=1e-12`.
Each backend then runs in its own process. BLAS and Rayon get equal thread budgets.
The benchmark refuses a Rust debug build and records the native binary hash.

Both timings include particle coefficient construction, translations and the full
dense interacting T-matrix. Rust also retains its pullback residual. They exclude
imports. Peak process RSS includes imports, allocator retention and benchmark
bookkeeping; import baselines are recorded separately. It is not a pure measure
of the numerical kernel's working memory. Tests used chains of spheres with radii
0.15–0.25, spacing 0.8, relative permittivity 4+0.1i and vacuum wavenumber 1.3.
No CPU affinity, frequency or thermal isolation was imposed.

The major improvements were reusing Wigner couplings across pairs, computing each
radial/angular term once per displacement, replacing generic LU/matmul with faer,
storing local particle matrices as blocks, and using disjoint Rayon workers to
assemble coupling columns. Dense interacting output and LU still cost O(D²)
space; factorization remains O(D³). No claim is made here about GPU performance,
large periodic systems, cylinders, gradient runtime, or arbitrary user geometries.

Reproduce after `uv sync --locked --group dev` and `just build-ext-release`:

```sh
uv run --no-sync python scripts/benchmark_cluster.py --particles 16 --lmax 3 --threads 4
```

## Batched spherical fields

The same harness accepts `--workload field --samples 2048`. It compares weighted
outgoing electric fields, including a retained native pullback context, against
`treams.efield(...) @ coefficients`. Basis setup and input generation are outside
the timed region for both implementations; upstream's field operator allocation
is part of its evaluation. Correctness is checked before isolated timing.

For four origins, order 3 (120 modes), 2,048 samples and four threads, the measured
median was 40.82 ms in Rust versus 822.27 ms in treams (20.14x). Peak process RSS
was 40.98 MiB versus 96.94 MiB; post-import baselines were 40.22 and 64.00 MiB.
Raw samples and binary identity: `benchmarks/results/fields-n4-l3-p2048-t4.json`.
The native path contracts amplitudes as it evaluates each sample, so it does not
allocate upstream's sample-by-component-by-mode matrix. Reverse mode recomputes
local wave derivatives and reduces cotangents with Rayon; no dense Jacobian is
retained. This is one qualified field workload, not a claim about all field sizes.
