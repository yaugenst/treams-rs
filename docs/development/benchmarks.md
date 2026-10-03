---
description: Run the benchmarks on an idle host, compare two builds and record new measurements next to the archived evidence.
---

# Benchmarks

The benchmark scripts compare treams-rs with the installed treams for speed, peak
memory and agreement. [`scripts/README.md`](../../scripts/README.md) lists every
script with its purpose and its test, and
[`benchmarks/README.md`](../../benchmarks/README.md) lists the plans, manifests,
summaries and raw results. The [performance](../performance/index.md) pages
summarize the results.

The locked development environment uses treams 0.4.7; historical records use
the version written in each record, often 0.4.5. Record the actual comparator
version for each new run rather than relabeling an archived measurement.

## Running the benchmarks

```sh
just bench               # every case of the reference grid
just bench-performance   # one group: also bench-geometry, bench-lattice, bench-api, bench-power
```

`just bench` builds the release extension, then reruns every case of
[`benchmarks/complete-qualification.json`](../../benchmarks/complete-qualification.json)
with its recorded arguments, thread count and pass criteria. It writes to
`benchmarks/results/local/`, which git ignores, continues after a failing case
and exits with an error if any case failed.

- **Release builds only.** Time only the extension of
  `just build-ext-release`. `just ci`, `just test-py` and `just docs` rebuild
  the development extension, so run `just build-ext-release` again after them.
- **An idle host.** Run on a machine with nothing else running, with both
  packages pinned to the same CPUs and given the same number of BLAS and treams-rs
  threads. Hosted CI shares its CPUs, so it never runs the benchmarks.

## Comparing two builds

To measure a change, compare release builds of the commit before and after it
on the same host:

- Alternate the two builds, run for run, so that a drift of the host affects
  both alike, and compare the median of the paired ratios.
- Repeat a comparison that shows a difference of a few percent before you
  report it; on a shared machine, runs of one build vary by that much.
- Report setup, data transfer and warm-up separately when they matter.
- Do not generalize a crossover measured at one size to every problem size.

`just bench-compare [ref]` does this for changes to the Python sources:
[`scripts/compare_builds.py`](../../scripts/compare_builds.py) runs the
Python package of `ref` (default `main`) and of the working tree on the same
release extension, alternating the two trees process by process. It first
requires every call to agree to `rtol=1e-12`, then fails a call whose median
paired ratio exceeds 1.05, or 1.10 below one millisecond. The calls cover the
NumPy API and the Advect, JAX and PyTorch namespaces at small and large sizes;
`--match` selects some, `--list` names them all, and `--memory` adds a
peak-memory comparison. A change to the native sources needs two built trees
instead: `--baseline <dir>` names the directory that holds the other build's
`treams_rs` package.

## Evidence

Everything under `benchmarks/results/` is archived evidence. Manifests,
summaries, documentation and the audit cite these files by path and by SHA-256
digest. Preserve numerical measurements and cited paths; privacy-only changes
follow the [privacy and provenance rules](../../benchmarks/README.md#privacy-and-provenance).
Write new runs to `benchmarks/results/local/`.

The summaries and manifests record where each result came from: the source
commit, the host, the thread counts, the arguments and the SHA-256 digests of
the native library, the Python sources and the benchmark scripts. A new run
that the documentation cites gets a row in the
[evidence provenance](../performance/evidence.md#evidence-provenance) table:
date, source commit, host, cases, outcome and summary file. Keep
[capabilities](../validation/capabilities.md) in line with what the evidence
shows.
