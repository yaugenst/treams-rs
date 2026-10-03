# Benchmarks and validation evidence

This directory holds the speed, memory and accuracy measurements of treams-rs
against treams 0.4.5: the plans that declare the cases of each run, the
manifests and summaries that record what ran, the raw results they cite, the
published-application reproductions and the API usability evaluation. The
scripts that produce and audit the evidence are listed in
[scripts/README.md](../scripts/README.md). The results are summarized on the
[performance](../docs/performance/index.md) page, the
[Mac and Linux comparison](../docs/performance/platform-comparison.md) and
[published applications](../docs/validation/published-applications.md). The
[evidence provenance](../docs/performance/evidence.md#evidence-provenance)
table gives the date, source commit, host and summary file of every run.

## Why nothing here moves

Manifests and summaries record result paths relative to the repository root, and
results record the sha256 digests of the scripts and sources that produced them.
`scripts/audit_qualification.py` and `scripts/replay_gated_benchmarks.py`
follow those paths, so moving or renaming a file, here or among the benchmark
scripts, breaks the audit trail. The audit also compares the recorded script
digests with the scripts of the checkout it is given as `root` and expects the
results under that root, so run it from a checkout that contains
`audit_qualification.py` and the evidence, and whose scripts match the recorded
digests. For the Linux core runs that is commit `0137eca`: it contains
the audit, both `linux-core-*.json` records and all three archives, and its
numerical source, benchmark scripts and report renderer match the recorded
digests. The runs themselves used `2470fbe`, the `source.commit` of
`linux-core-performance.json`; `0137eca` adds the performance record and
archives, updates the correctness record and the documentation to match, and
changes no source file other than the report renderer `plot_benchmarks.py`.

## Root files

| File | Kind | Contents |
| --- | --- | --- |
| [`complete-qualification.json`](complete-qualification.json) | verified replay manifest | The 527-case Linux reference grid: build fingerprints and, per case, the command, its pass criteria and the result path under `results/final/`. `scripts/replay_gated_benchmarks.py` replays it for `just bench*`; `run_benchmark_suite.py` and `audit_qualification.py` derive the broad phase from it. |
| [`mac-qualification.json`](mac-qualification.json) | summary | The 30 macOS regression cases, with results under `results/mac-final/`. |
| [`coefficient-qualification.json`](coefficient-qualification.json) | summary | Native digest and the 257 verified result paths of the coefficient-namespace run. |
| [`geometry-qualification.json`](geometry-qualification.json) | summary | Native digest and the 299 verified result paths of the lattice-geometry run, including its rechecks of the coefficient cases. |
| [`linux-core-qualification.json`](linux-core-qualification.json) | summary | Linux core correctness: 615 accuracy, gradient and boundary cases, with environment, source digests, phase manifests, audits and the evidence archive. |
| [`linux-core-performance.json`](linux-core-performance.json) | summary | Linux core runtime and peak RSS of the remaining 685 cases (769 with the retained gradient and boundary cases), with comparisons, the report and the evidence archives. |
| [`correction-plan.json`](correction-plan.json) | plan | The 1,300 cases behind the two Linux core records; the default plan of `audit_qualification.py --cohort corrected`. |
| [`comparison-plan.json`](comparison-plan.json) | plan | The 209 scaling, gradient and illumination cases of the Mac and Linux comparison beyond the reference grid; the default plan of `audit_qualification.py --cohort comparison`. |
| [`accuracy-plan.json`](accuracy-plan.json) | plan | Untimed accuracy evidence on both platforms: every upstream reference comparison of the grid plus independent physical, convergence and high-precision checks. |
| [`accuracy-boolean-correction-plan.json`](accuracy-boolean-correction-plan.json) | plan | Untimed replay of three Linux mode-selection checks with the corrected residual collector, kept apart from the timing evidence. |
| [`linux-core-qualification.md`](linux-core-qualification.md) | prose summary | The two Linux core records in prose, with the archive extraction and audit commands. |

Run the cases of one plan from the repository root with

```sh
uv run --no-sync python scripts/run_benchmark_suite.py --phase extra \
  --plan benchmarks/<plan>.json --output benchmarks/results/local/<dir>
```

`--phase all`, the default, also runs the broad phase derived from
`complete-qualification.json`.

## papers/

Published-application reproductions: the author data and their provenance, the
executed upstream notebook outputs, the recorded results and the figures of the
electron-beam, CPC periodic-array and thermal-emission comparisons.
`scripts/qualify_papers.py` writes `qualification.json` and its figures, and
`papers_thermal.py` writes `thermal-result.json` (and fetches
`thermal-source.json` from the pinned author commit when it is missing);
`qualify_paper_accuracy.py` and the paper tests read the fixtures.

## results/: immutable raw evidence

Everything under `results/` is raw evidence that the manifests, summaries,
documentation and audits cite by path and digest. Never edit, rename or delete
anything there. Write new runs to `results/local/`, which git ignores, by
passing a directory under it as `--output`; `just bench*` puts its results
there.

- `final/` and `mac-final/`: the results of `complete-qualification.json` and
  `mac-qualification.json`.
- The loose JSON files: per-kernel measurements and the coefficient and geometry
  results, cited from the [evidence](../docs/performance/evidence.md) page.
- `agent-api-20260920/`: the API usability evaluation report, summaries, figures
  and evidence archives.
- The archives of the Linux core runs: correctness, performance and the
  generated report (in two parts).

Extract the archives from the repository root to inspect them:

```sh
tar -xzf benchmarks/results/linux-core-performance-20260919.tar.gz
cat benchmarks/results/linux-qualification-report-20260920.tar.gz.part-00 \
  benchmarks/results/linux-qualification-report-20260920.tar.gz.part-01 | tar -xz
tar -xzf benchmarks/results/linux-core-qualification-20260919.tar.gz
```

They create `benchmarks/results/linux-core-performance-20260919/`,
`benchmarks/linux-qualification-report/` (open `index.html`),
`benchmarks/results/linux-core-correctness-20260919/` and
`benchmarks/results/linux-core-correctness-frozen-20260919/`. Git does not track
these extracted directories; delete them when you are done and keep the
archives. Archive digests and file counts are in the two `linux-core-*.json`
records.

Recorded paths in the results have user and machine identifiers removed, and
the manifest digests of the redacted JSON files match the files as
distributed. Numerical results, timing samples and the recorded source,
native-library and benchmark-script fingerprints are unchanged; the redaction
reran no measurement.

## agent-api/ and agent-usability/

- [`agent-api/`](agent-api/README.md): protocol, tasks, prompts, hidden references
  and grader amendments of the API usability evaluation, which measures how
  coding agents solve physics tasks adapted from the treams examples with only
  the installed wheel. Its evidence is in
  [`results/agent-api-20260920/`](results/agent-api-20260920/REPORT.md); the
  launchers and graders are `scripts/agent_eval/`.
- `agent-usability/`: the qualification record of the Python-only API discovery,
  documentation and diagnostics changes, with timings of the two public
  particle-cluster workflows that they touched.

No test or CI job uses either directory.
