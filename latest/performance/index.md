# Performance

In the recorded Linux broad grid of 527 cases, treams-rs is faster than treams
0.4.5 in every case, with a median speedup of 5.12× and a median of 0.66 times
the peak memory.
The [evidence provenance](evidence.md#evidence-provenance) table gives the
date, source commit, host and raw files of each set of measurements.

## Headline results

Speedup is treams time divided by treams-rs time. The RSS ratio is the
treams-rs peak resident memory divided by the treams peak, for the whole
process. Medians are unweighted over distinct cases.

| Measurement | Cases | Median speedup | Median RSS ratio | Exceptions |
| --- | ---: | ---: | ---: | --- |
| Linux broad grid | 527 | 5.12× | 0.658 | 2 cases use more memory |
| Linux size scaling | 107 | 26.68× | 0.657 | 4 cases use more memory |
| Linux thread scaling | 30 | 18.71× | 0.474 | 1 case uses more memory |
| macOS dispatch grid | 30 | 3.15× (minimum 1.30×) | 0.664 | None |
| Requested illumination, selected dense LU (Linux) | 19 | 33.84× | 0.550 | None |
| Requested illumination, matrix-free (Linux) | 19 | 12.70× | 0.372 | 1 slower case: 0.68× at 32 columns |

The Linux rows come from the
[Linux core performance record](https://github.com/yaugenst/treams-rs/blob/5947c78e5b88a364952fb554548e53b40321abe9/benchmarks/linux-core-performance.json),
summarized in [Linux core validation](https://github.com/yaugenst/treams-rs/blob/5947c78e5b88a364952fb554548e53b40321abe9/benchmarks/linux-core-qualification.md).
That record also lists 33 edge cases (median 1.77×): mostly internal
illuminations at the matrix sizes and column counts where the solve switches
method, plus scalar `tl_vcw_r` calls; three of them are slower than treams.
The macOS row comes from the [macOS dispatch record](https://github.com/yaugenst/treams-rs/blob/5947c78e5b88a364952fb554548e53b40321abe9/benchmarks/mac-qualification.json).

The cases that use more memory keep data for computing gradients later. This
is called a recorded call; its pullback turns the gradient of a result into
gradients of the inputs. treams computes only the forward result.

For large sphere clusters, solving only for the requested incident fields can
save time and memory ([large problems](large-problems.md)):

- 512 spheres, one illumination: a dense solve for the requested column takes
  0.381 s against 0.940 s for the full interacting T-matrix (2.47× faster).
- The same cluster: the matrix-free gradient peaks at 45.6 MiB against
  1,367.7 MiB for the full T-matrix (30× less).
- 1,024 spheres: the matrix-free solve takes 2.19 s and its gradient 2.83 s,
  within 47.9 MiB.

### Selected operations from the reference grid

The [reference grid](https://github.com/yaugenst/treams-rs/blob/5947c78e5b88a364952fb554548e53b40321abe9/benchmarks/complete-qualification.json) has 527
cases on Linux. Its smallest speedup is 1.02×, and its largest RSS ratio among
the cases that check memory is 0.912.

| Operation | Input size | Speedup | treams-rs / treams peak RSS |
| --- | ---: | ---: | ---: |
| Cartesian to spherical coordinates | 1 | 1.83× | 0.65 |
| Cartesian to spherical vector components | 1 | 1.58× | 0.65 |
| Plane-wave M field | 1 | 3.84× | 0.66 |
| Cylindrical rotation | 128 | 1.13× | 0.65 |
| Two-dimensional cell volume | 1 | 2.56× | 0.65 |
| Two-dimensional reciprocal cell | 1 | 1.98× | 0.65 |
| EBCM, degree 3, 96 quadrature nodes | 30 modes | 83.89× | 0.53 |
| EBCM, degree 4, 96 quadrature nodes | 48 modes | 127.85× | 0.53 |

The EBCM rows evaluate the treams surface integral, which omits a radial area
factor, so both packages compute the same quantity
([differences from treams](../coming-from-treams/differences.md)).

## Methodology

- **Agreement first.** Every case compares the treams-rs result with treams
  before timing it, at a relative tolerance of 2e-9 and an absolute tolerance
  of 1e-12.
- **Isolated processes.** Each package runs in its own process, so importing
  one does not add to the memory of the other.
- **Matched threads.** Both packages get the same BLAS and Rayon thread limit:
  four, unless the case measures thread scaling.
- **Small calls.** When both medians are below 1 ms, the two packages also run
  in one process in alternating order: 14 paired samples, each a batch of at
  least 20 ms. The speedup is the median of the paired ratios.
- **Peak memory.** Peak RSS is the highest resident memory observed for the
  process, including imports, inputs and memory retained for reuse.
- **Pass criteria.** A case passes when treams-rs is at least as fast as treams
  (speedup ≥ 1) and uses no more peak memory (RSS ratio ≤ 1).
- **Two memory exceptions.** Recorded internal illumination at 1,024 channels
  keeps owned copies of its inputs for the pullback (the reverse pass that
  computes gradients). In the Linux broad grid, with one and eight incident
  columns, it runs 1.51× and 1.47× faster than treams but peaks at 1.44 times
  its memory; that run lists both as cases that use more memory. The
  [reference replay grid](https://github.com/yaugenst/treams-rs/blob/5947c78e5b88a364952fb554548e53b40321abe9/benchmarks/complete-qualification.json)
  exempts the same two cases from the memory check. There they run 1.40× and
  1.28× faster at 1.56 and 1.75 times the memory, and the same calls without
  recording run 1.83× and 1.71× faster at 0.91 times the memory. They still
  must be at least as fast as treams.
- **Reverse passes.** treams has no reverse pass, so reverse timings stand on
  their own and are never compared with treams.
- **Every result counts.** Slower results stay in the records; no case is
  discarded or rerun to get a better number.

## Caveats

- Results can differ with input size, host and numerical conditioning.
- Each result belongs to the platform and source revision recorded with it.
  Later revisions were not rerun through every measurement.
- Speed and memory checks are separate from the accuracy checks on the
  [validation](../validation/index.md) pages.

## Reproduce

Run the benchmarks on an otherwise idle host:

```sh
just bench
```

`just bench` builds the optimized extension, then reruns every case of the
reference grid with its recorded arguments, thread count and pass criteria. It
continues after a failing case and exits with an error if any case failed.
Select a group of workloads with `just bench-performance`, `bench-geometry`,
`bench-lattice`, `bench-api`, `bench-power` or `bench-all`. Results go to
`benchmarks/results/local/`, which git ignores; the files under
`benchmarks/results/` stay as recorded. The reference run pinned both packages
to CPUs 8–11 of a Ryzen 9 9950X.

## More detail

- [Large problems](large-problems.md): requested illuminations for clusters of
  up to 1,024 spheres on Linux and macOS.
- [Platform comparison](platform-comparison.md): the 527-case grid and 209
  more cases on an Apple M3 and a Ryzen 9 9950X, with every outcome.
- [Evidence](evidence.md): the provenance table and archived measurements of
  individual kernels.
- [Benchmark files](https://github.com/yaugenst/treams-rs/blob/5947c78e5b88a364952fb554548e53b40321abe9/benchmarks/README.md): the layout of `benchmarks/`.
