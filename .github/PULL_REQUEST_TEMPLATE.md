# Summary

<!-- What changes and why, in one or two sentences. Closes #… -->

## Checklist

- [ ] Tests at the right tier (Rust proptest for core invariants, Python Hypothesis
      or reference tests at the boundary); a bug fix adds a test that fails without
      it; `just ci` passes.
- [ ] New differentiable operations have analytic pullbacks; finite differences
      appear only in tests.
- [ ] Docstrings and `python/treams_rs/_native.pyi` match the code; `just docs` rerun.
- [ ] `CHANGELOG.md` bullet under "Unreleased" (renames as `old` → `new`); names that
      differ from treams are in `python/treams_rs/_upstream.py`.
- [ ] Numerical or hot-path change: parity evidence (treams 0.4.7 or an independent
      reference) and release-build before/after timings attached; archived benchmark
      measurements unchanged.

Details: [definition of done](https://github.com/yaugenst/treams-rs/blob/main/CONTRIBUTING.md#definition-of-done).
