# Paused core benchmark campaign

The campaign remains paused. `main` is the authoritative CPU core and Python
implementation. Browser/WASM work is preserved on
`experimental/browser`; CUDA work is preserved on
`experimental/gpu`. Each experimental branch excludes the other.

## Preserved checkpoint

The original Mac and Linux results remain under `results/comparison/`; targeted
correction evidence remains under `results/fixes/` and `results/corrections/`.
The September 12 pause records under `results/corrected/` describe the earlier
combined checkout. Keep those records immutable. Their source, plan and binary
hashes no longer describe the extracted branches. Existing raw evidence and
historical resume tools are local artifacts, not a portable release bundle.

The numerical corrections are saved in commit `082eeda`, before experiment
extraction. The complete corrected performance campaigns have not run. Do not
interpret passing functional tests as final performance or accuracy qualification.

## Resume the core campaign

1. Synchronize the core-only `main` revision to both hosts. Build fresh optimized
   extensions with `just build-ext-release` and record new source, plan, harness
   and native-library hashes. Keep timing hosts otherwise idle.
2. Run the Mac regression preflight before its full campaign: broad indices
   3, 15, 19, 25, 29, 39, 43, 52, 57, 66 and 105, plus
   `gradient-cylindrical-field-coefficients-n2-l2-s256-t1`. Preserve original
   commands, repeats and tolerances. Retain every outcome.
3. Use `scripts/run_benchmark_suite.py` with `benchmarks/correction-plan.json`.
   The core plan contains 1,300 cases: 736 original CPU performance, 531 accuracy,
   and 33 boundary cases. Mac excludes six unsupported 16-thread cases. Retain
   the 8 GiB Mac and 16 GiB Linux process-group guards. Use new output directories;
   never overwrite or relabel historical measurements.
4. The old `results/corrected/resume-tools/` queue and renderer assumed a combined
   CPU/GPU campaign. Update the execution and reporting selection for the nine
   CPU groups before reuse; do not invoke the old Linux GPU phase on `main`.
5. Audit provenance, completeness, numerical gates and raw timing/array evidence.
   Measure the proposed dense-memory improvement. Keep unresolved high-order
   reference disagreements, recording overhead and convergence tradeoffs visible.
6. Generate and visually inspect the CPU report, then update benchmark and status
   documentation with the actual results and remaining limits.

The GPU branch retains the original 20 CUDA cases and its separate hardware
qualification commands. Browser and GPU experiments are deferred; resuming the
core campaign does not require either lane or any deployment work.
