# Linux core validation and performance

These measurements compare treams-rs with treams 0.4.5 on one Linux host: an
AMD Ryzen 9 9950X (16 physical cores) with Python 3.13.1. They cover all 1,300
cases of [`correction-plan.json`](correction-plan.json): 615 correctness,
gradient and boundary cases, then 685 performance cases. Boundary cases are
mostly internal illuminations at the matrix sizes and column counts where the
solve switches method, plus scalar `tl_vcw_r` calls.

| Record | Date | Source commit | Cases | Outcome |
| --- | --- | --- | ---: | --- |
| [Correctness](linux-core-qualification.json) | 2026-09-19 | `2843a70` | 615 | All pass |
| [Performance](linux-core-performance.json) | 2026-09-19 to 2026-09-20 | `2470fbe` | 685 (769 with the retained gradient and boundary cases) | All measured; exceptions below |

Both runs used the same numerical source, Python source and optimized extension
(native SHA-256 `8c51f988…`). The records hold build identities, manifest and
audit checksums, exclusions and every measured exception. Every audit reports
zero integrity errors and zero failed numerical checks. No case timed out, hit
the memory limit or failed.

## Performance and memory

Speedup is upstream time divided by treams-rs time. The RSS ratio is the
treams-rs peak resident memory divided by the upstream peak, both for the whole
process, including imports and memory the allocator keeps. The table gives
unweighted medians over distinct cases, not the speedup of an application.

| Group | Cases | Median speedup | Median RSS ratio | Slower / higher-RSS cases |
| --- | ---: | ---: | ---: | ---: |
| Broad grid | 527 | 5.12× | 0.658 | 0 / 2 |
| Size scaling | 107 | 26.68× | 0.657 | 0 / 4 |
| Thread scaling | 30 | 18.71× | 0.474 | 0 / 1 |
| Boundary cases | 33 | 1.77× | 0.823 | 3 / 10 |

All 664 broad, size and thread comparisons are faster than treams. Seven of them
use more memory:

- the two large recorded internal-illumination cases;
- the 64-particle spherical and cylindrical public cluster workflows;
- the 128-layer, 64-channel slab;
- the two-particle, order-24 rotation;
- the 16-thread slab.

A recorded call keeps the data its pullback needs; the pullback is the reverse
pass that turns the gradient of a result into gradients of the inputs. Upstream
computes only the forward result, so recorded calls can cost more. Three
boundary cases are slower for this reason: recorded internal illumination at
order 255 with one or 16 columns (0.849× and 0.841×) and the scalar recorded
`tl_vcw_r` call (0.428×). Ten larger recorded internal contexts use more memory.
The performance record lists every case ID and ratio.

## Gradients

All 51 gradient cases keep their numerical checks and arrays. Against the 14
exact dense linear adjoints that upstream can compute, treams-rs is faster in
every case and uses less memory (median 5.07×). Finite differences are a
different algorithm; their much larger ratios appear separately in the record.

## Requested illumination

The 21 illumination cases compare three ways to solve a cluster for a few
incident fields: the full interacting T-matrix, a dense solve for the requested
columns only, and a matrix-free iterative solve.

- Full and selected dense solves are faster than upstream in all 19 comparisons
  that include setup, with lower forward peak memory.
- Matrix-free solves use less memory but are slower than upstream's selected LU
  at 32 columns (0.681×).
- The 2,048- and 4,096-particle cases exceed the size limit of the dense
  reference, so they carry no upstream speed or memory comparison.

Reusing a factorization and running the reverse pass are measured separately; a
faster first solve does not imply faster reuse. Sixteen threads slow the small
cluster case down relative to one thread.

The dense 256-particle forward and reverse pass peaks at 7,316.1 MiB, against
8,219.6 MiB for the earlier build kept in the evidence archive (11.0% less). The
128-particle case peaks at 1,880.7 MiB against 2,107.8 MiB (10.8% less).
Forward-only peaks are essentially equal. The two builds ran at different
times, so this is not a controlled same-run comparison.

## Evidence archives

The archives hold the nine manifests, raw JSON, gradient arrays, CSV tables and
a generated HTML report with plots and a PDF. Extract them from the repository
root:

```sh
tar -xzf benchmarks/results/linux-core-performance-20260919.tar.gz
cat benchmarks/results/linux-qualification-report-20260920.tar.gz.part-00 \
  benchmarks/results/linux-qualification-report-20260920.tar.gz.part-01 | tar -xz
tar -xzf benchmarks/results/linux-core-qualification-20260919.tar.gz
```

Open `benchmarks/linux-qualification-report/index.html` for the report. The
performance record lists archive checksums and file counts; every archived file
matches its source byte for byte. All 2,660 local links of the report resolve,
and its 40 tests pass.

The correctness archive also keeps a first accuracy run that does not count: the
installed extension changed partway through it. The accepted accuracy run used
a copied package that matches the binary of the gradient and boundary checks.
The 18 reference inputs beyond the float64 range and the five raw upstream
cluster disagreements stay listed in the record; the independent cluster
references and the full thermal reproduction pass.

## Repeat the audits

Install the measured package and numerical source, extract the archives, then
run:

```sh
uv run --script scripts/audit_qualification.py "$PWD" linux \
  accuracy-references accuracy-physics accuracy-upstream --cohort corrected \
  --results-root benchmarks/results/linux-core-correctness-frozen-20260919
uv run --script scripts/audit_qualification.py "$PWD" linux \
  gradient correction-boundary --cohort corrected \
  --results-root benchmarks/results/linux-core-correctness-20260919
uv run --script scripts/audit_qualification.py "$PWD" linux \
  broad scaling thread-scaling illumination --cohort corrected \
  --results-root benchmarks/results/linux-core-performance-20260919
```

The audits check source and build identity, case coverage, tolerances, pass
and fail fields, timing consistency and the integrity of archived arrays. They
do not rerun the solvers.

The runner kept the declared repeats and timeouts per case and stopped any
process group above 16 GiB. Ordinary cases ran on CPUs 8–11; thread scaling
used physical CPUs 0–15. The sample spread covers one run on one day. Larger
cases always ran upstream first, so the data cannot show drift that depends on
the order.

## Scope

These are finite measurements on one host. They do not cover macOS, Python
3.12, other hosts, browsers or GPUs, and they do not certify arbitrary
parameter ranges, multipole convergence or a speed or memory advantage for
every input. The slower and higher-memory cases above stay in the record.
