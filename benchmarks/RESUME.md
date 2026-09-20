# Linux core qualification — complete

The September 19–20 campaign is complete on this Ryzen 9 9950X Linux machine
with Python 3.13.1. All 1,300 cases in `correction-plan.json` have accepted evidence:
615 correctness/gradient/boundary cases, then the remaining 685 performance cases.
The performance run used commit `2470fbe` and the same numerical source, Python
source and frozen optimized extension as the accepted correctness run.
No numerical implementation or tolerance changes were needed.

The [correctness record](linux-core-qualification.json) and
[performance record](linux-core-performance.json) contain build identities,
manifest/audit checksums, exclusions and measured exceptions. All accepted audits
have zero integrity errors and zero unresolved native numerical gates. The
performance run had no timeout, resource-limit stop or failed case.
Luna executed fixed commands only; the parent selected scope and reviewed results.

## Performance and memory

Speedup is upstream time divided by native time; RSS ratio is native divided by
upstream full-process peak RSS. These are unweighted medians across distinct cases,
not an aggregate application speedup. Imports and allocator retention count in RSS.

| Group | Cases | Median speedup | Median RSS ratio | Slower / higher-RSS cases |
| --- | ---: | ---: | ---: | ---: |
| Broad grid | 527 | 5.12× | 0.658 | 0 / 2 |
| Size scaling | 107 | 26.68× | 0.657 | 0 / 4 |
| Thread scaling | 30 | 18.71× | 0.474 | 0 / 1 |
| Retained boundary cases | 33 | 1.77× | see record | 3 / 10 |

All 664 broad/scaling/thread comparisons are faster. Their seven higher-RSS cases
are the two large recorded internal-illumination contexts, the 64-particle
spherical/cylindrical public cluster workflows, the 128-layer/64-channel slab,
the two-particle order-24 rotation and the 16-thread slab. Every case ID and ratio
is retained in the performance record.

The boundary slowdowns are recorded internal illumination at order 255 with
one or 16 columns (0.849× and 0.841×), and the scalar recorded `tl_vcw_r` call
(0.428×). Native calls retain derivative context while upstream computes the
forward result. Ten larger recorded internal contexts also use more memory.
These measurements remain unchanged; universal runtime/RSS targets are not met.

All 51 gradient cases retain their original numerical checks and arrays. The 14
matched exact dense linear-adjoint comparisons are faster with lower peak RSS
(median 5.07×). The much larger speedups against finite differences describe a
different algorithm and are reported separately.

The 21 illumination cases compare full, selected dense and matrix-free paths.
Full and selected dense win all 19 fresh-setup upstream comparisons, with lower
forward peak RSS. Matrix-free is slower than upstream selected LU at 32 columns
(0.681×), despite lower memory. The 2,048- and 4,096-particle cases exceed the
explicit dense/oracle limit and carry no same-size upstream speed or memory claim.
Factor reuse and recorded reverse costs are separate; fresh-setup wins do not
imply faster reuse. Sixteen threads also slow the small cluster relative to one
thread, so more threads are not always useful.

The dense 256-particle forward-and-reverse peak RSS fell from 8,219.6 to
7,316.1 MiB (11.0%) against the preserved earlier build; the 128-particle case
fell from 2,107.8 to 1,880.7 MiB (10.8%). Forward-only peak RSS is essentially
unchanged. Baseline JSON is included in the new evidence archive. This comparison
spans builds/runs; it is not a same-run controlled experiment.

## Evidence and report

The [generated report](linux-qualification-report/index.html) includes all nine
accepted manifests, downloadable plots/PDF, raw JSON/gradient arrays and CSV
ledgers. The report is archived rather than checking in thousands of generated
files. Extract both archives from the repository root to inspect all new evidence:

```sh
tar -xzf benchmarks/results/linux-core-performance-20260919.tar.gz
cat benchmarks/results/linux-qualification-report-20260920.tar.gz.part-00 \
  benchmarks/results/linux-qualification-report-20260920.tar.gz.part-01 | tar -xz
```

Archive checksums and file counts are in the performance record. Each archived
file was compared byte-for-byte with its source. Forty report tests pass; all
2,660 local HTML links resolve. Representative runtime, memory, size/thread
scaling and illumination figures were visually inspected. Interactive browser
behavior was not checked because no browser was connected.

The accepted correctness evidence and the superseded first accuracy attempt are
preserved separately:

```sh
tar -xzf benchmarks/results/linux-core-qualification-20260919.tar.gz
```

The first accuracy run was superseded when the installed extension changed
mid-run. Accuracy was rerun against a copied package matching the binary already
used for gradient/boundary checks. The 18 overflowing reference inputs and five
raw upstream cluster disagreements remain explicit; independent cluster-reference
checks pass. The full thermal reproduction also passes.

To repeat the audits with the qualified package and numerical source installed:

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

Audits check source/build identity, case coverage, unchanged tolerances, gate
fields, timing consistency and archived array integrity. They do not rerun the
solver or recompute numerical arrays. The runner used unchanged per-case repeats,
timeouts and a 16 GiB process-group guard, with CPUs 8–11 for ordinary cases and
physical CPUs 0–15 for thread scaling. Sample intervals do not cover host/day
variation; fixed upstream-then-native ordering for larger cases does not measure
order-dependent drift.

## Scope left outside this qualification

There is no remaining Linux campaign case in the declared plan. Known performance
exceptions above are documented optimization opportunities, not discarded results.
No macOS, Python 3.12, other-host, browser/WASM or CUDA qualification was added.
Browser and GPU work remain on their separately pushed experimental branches.
Finite-case qualification does not certify arbitrary parameter ranges, multipole
convergence or a universal performance/memory advantage.

Historical Mac/Linux results and September 12 pause records remain unchanged.
Their old queue/renderer assumed a combined CPU/GPU campaign; do not run those
controllers unchanged on core-only `main`.
