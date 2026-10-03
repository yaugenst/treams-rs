# Scripts

Documentation, packaging, benchmark and validation scripts; the library never imports them.
The "Run by" column names the `just` recipe, CI job or script that invokes each
one. Scripts run "by hand" start from the repository root with the command in
their module docstring, which also describes their options.

The benchmark and validation scripts write evidence that is indexed in
[benchmarks/README.md](../benchmarks/README.md).

## Documentation

| Script | Purpose | Run by | Test |
| --- | --- | --- | --- |
| [`generate_docs.py`](generate_docs.py) | Generates the Python API reference under `docs/reference/python/` (one page per module), the generated regions of docs pages and `llms.txt` from the installed package's support catalog, or checks them with `--check`, and checks that every docs page is in the `mkdocs.yml` nav with a description. | `just docs`, `just docs-check` | [`tests/api/test_support_catalog.py`](../tests/api/test_support_catalog.py) |

## Packaging and floating point

| Script | Purpose | Run by | Test |
| --- | --- | --- | --- |
| [`check_wheel.py`](check_wheel.py) | Scans a built wheel for local paths, installs it with Advect into a clean environment without the upstream scientific stack and runs the core stage, then the optional HDF5 stage. | `just check-wheel` (CI) | none (CI runs it) |
| [`float_environment.py`](float_environment.py) | Requires native results for callers that flush subnormals (XLA, `torch.set_flush_denormal`) to equal those of IEEE callers bit for bit. | `check_wheel.py` (core stage) | [`tests/bindings/test_float_environment.py`](../tests/bindings/test_float_environment.py) |

## Benchmark scripts (evidence-bound)

| Script | Purpose | Run by | Test |
| --- | --- | --- | --- |
| [`_harness.py`](_harness.py) | Provenance digests and thread pinning shared by the benchmark and validation scripts. | imported by the scripts below | [`tests/scripts/test_audit_qualification.py`](../tests/scripts/test_audit_qualification.py), [`tests/scripts/test_qualify_references.py`](../tests/scripts/test_qualify_references.py) |
| [`benchmark_cluster.py`](benchmark_cluster.py) | Isolated-process, matched-thread comparisons with treams 0.4.5 of every checked workload (scattering, fields, special functions, waves, geometry, lattice sums, operator and power workflows), with agreement, speed and peak-RSS checks. | `run_benchmark_suite.py`, `replay_gated_benchmarks.py` | [`tests/scripts/test_benchmark_cluster.py`](../tests/scripts/test_benchmark_cluster.py) |
| [`benchmark_gradients.py`](benchmark_gradients.py) | Full real-coordinate gradients: upstream finite differences, and an exact linear adjoint for field coefficients, versus native pullbacks. | `run_benchmark_suite.py` | [`tests/scripts/test_benchmark_gradients.py`](../tests/scripts/test_benchmark_gradients.py) |
| [`benchmark_illumination.py`](benchmark_illumination.py) | Runtime and isolated-process peak memory of requested illuminations: full, selected dense and matrix-free. | `run_benchmark_suite.py` | [`tests/scripts/test_benchmark_metadata.py`](../tests/scripts/test_benchmark_metadata.py) |
| [`run_benchmark_suite.py`](run_benchmark_suite.py) | Runs a predeclared suite sequentially (the broad phase derived from `benchmarks/complete-qualification.json`, then the cases of a `--plan` file), with resumable raw evidence and a suite manifest. | by hand | [`tests/scripts/test_benchmark_suite.py`](../tests/scripts/test_benchmark_suite.py) |
| [`replay_gated_benchmarks.py`](replay_gated_benchmarks.py) | Reruns every verified entry of `benchmarks/complete-qualification.json` with its recorded arguments and pass criteria into `benchmarks/results/local/`. | `just bench`, `just bench-*` | [`tests/scripts/test_benchmark_suite.py`](../tests/scripts/test_benchmark_suite.py) |
| [`plot_benchmarks.py`](plot_benchmarks.py) | Renders reports (figures, tables, CSV ledgers) from suite manifests; runs no benchmark. | by hand | [`tests/scripts/test_benchmark_plots.py`](../tests/scripts/test_benchmark_plots.py) |
| [`audit_qualification.py`](audit_qualification.py) | Read-only audit of a finished benchmark run: source and build identity, case coverage, pass criteria, timing consistency and archive integrity. | by hand (commands in [`linux-core-qualification.md`](../benchmarks/linux-core-qualification.md)) | [`tests/scripts/test_audit_qualification.py`](../tests/scripts/test_audit_qualification.py) |

## Validation collectors

| Script | Purpose | Run by | Test |
| --- | --- | --- | --- |
| [`qualify_upstream.py`](qualify_upstream.py) | Records the numerical residuals of `benchmark_cluster.py`'s upstream reference check in a standalone process. | `run_benchmark_suite.py` (accuracy plans) | [`tests/scripts/test_qualify_upstream.py`](../tests/scripts/test_qualify_upstream.py) |
| [`qualify_physics.py`](qualify_physics.py) | Records physical identities and convergence, independently of treams. | `run_benchmark_suite.py` (accuracy plans) | [`tests/scripts/test_qualify_physics.py`](../tests/scripts/test_qualify_physics.py) |
| [`qualify_references.py`](qualify_references.py) | Independent high-precision references (Bessel functions, Ferrers functions, Mie coefficients, Wigner rotations), retaining failures and domain exclusions. | `run_benchmark_suite.py` (accuracy plans) | [`tests/scripts/test_qualify_references.py`](../tests/scripts/test_qualify_references.py) |
| [`qualify_cluster_conditioning.py`](qualify_cluster_conditioning.py) | Conditioning, residuals and cutoff convergence of the high-cutoff eight-sphere chain, with a certified reference. | `run_benchmark_suite.py` (accuracy plans) | [`tests/scripts/test_qualify_cluster_conditioning.py`](../tests/scripts/test_qualify_cluster_conditioning.py) |
| [`qualify_legendre.py`](qualify_legendre.py) | Checks the native real-degree Legendre functions against 70-digit hypergeometric values; its `ferrers` reference also serves `qualify_references.py`. | by hand | [`tests/scripts/test_qualify_references.py`](../tests/scripts/test_qualify_references.py) |
| [`qualify_ebcm_cancellation.py`](qualify_ebcm_cancellation.py) | Measures the degree-6, m = 0 EBCM cancellation floor of both implementations against an 80-digit integral. | by hand | none |

## Paper reproductions

| Script | Purpose | Run by | Test |
| --- | --- | --- | --- |
| [`qualify_papers.py`](qualify_papers.py) | Recomputes author-provided spectra (electron-beam and CPC periodic-array cases) and fails on the first disagreement. | by hand | [`tests/scripts/test_paper_qualification.py`](../tests/scripts/test_paper_qualification.py), [`tests/plane/test_diffraction_threshold.py`](../tests/plane/test_diffraction_threshold.py) |
| [`papers_thermal.py`](papers_thermal.py) | Reproduces PRB 112, 054307 (2025), Fig. 2, against the authors' data. | by hand | [`tests/scripts/test_paper_qualification.py`](../tests/scripts/test_paper_qualification.py) |
| [`qualify_paper_accuracy.py`](qualify_paper_accuracy.py) | Records the full paper spectra, source comparisons and physical residuals, keeping every failed comparison. | `run_benchmark_suite.py` (accuracy plans) | [`tests/scripts/test_qualify_paper_accuracy.py`](../tests/scripts/test_qualify_paper_accuracy.py) |

## Reference-data generators

| Script | Purpose | Run by | Test |
| --- | --- | --- | --- |
| [`generate_references.py`](generate_references.py) | Generates or checks (`--check`) every mpmath reference table in [`crates/treams-core/references/`](../crates/treams-core/references/README.md): `python scripts/generate_references.py {incgamma,kambe,kambe-lattice,lattice-sums,lattice-chain} [--check] [--output PATH]`. | by hand | none; the Rust tests listed in the [references README](../crates/treams-core/references/README.md) read the tables |

## API usability evaluation

| Script | Purpose | Run by | Test |
| --- | --- | --- | --- |
| [`agent_eval/`](agent_eval/) | Launchers, sandbox preflight, graders and summaries of the API usability evaluation; protocol and environment variables in [`benchmarks/agent-api/README.md`](../benchmarks/agent-api/README.md). | by hand | none (no test or CI uses it) |

## Rules

- The benchmark scripts are evidence-bound. Manifests and results record their
  paths and sha256 digests, and `audit_qualification.py` compares those digests
  with the files of the checkout it audits, so an audit of recorded evidence
  runs from a checkout that contains the audit and the evidence and whose
  scripts match the recorded digests
  ([benchmarks/README.md](../benchmarks/README.md) names that commit for the
  Linux core validation run). Do not move or rename the scripts. After an edit,
  rerun the cases of any new evidence that cites the edited script.
- Field names and labels that the scripts write into results (`ebcm_legacy`,
  `post_failure_diagnostic`, ...) are recorded in the evidence and must not change.
- Scripts import their siblings by module name (`from _harness import ...`), as
  `python scripts/<name>.py` allows; [`tests/_scripts.py`](../tests/_scripts.py)
  puts `scripts/` on `sys.path` for the tests.
- Two library tests import scripts:
  [`tests/plane/test_diffraction_threshold.py`](../tests/plane/test_diffraction_threshold.py)
  imports `qualify_papers`, and
  [`tests/bindings/test_float_environment.py`](../tests/bindings/test_float_environment.py) runs
  `float_environment.py`.
