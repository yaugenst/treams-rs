# Summary

<!-- What the change does and why. -->

## Linked issue

<!-- For example: Closes #123. -->

## Checklist

- [ ] Tests at the right tier: Rust proptest properties for invariants of the
      numerical core, Python Hypothesis or reference tests at the boundary.
- [ ] New differentiable operations have analytic pullbacks; finite differences
      appear only in tests.
- [ ] `just ci` passes.
- [ ] Docstrings and `python/treams_rs/_native.pyi` match the code, and `just docs`
      has been rerun.
- [ ] `CHANGELOG.md` has a bullet under "Unreleased".
- [ ] Renamed public names appear in the CHANGELOG as `old` → `new`, and names that
      differ from treams are listed in `python/treams_rs/_upstream.py`.
- [ ] Numerical or hot-path changes: parity evidence (treams 0.4.7 or an independent
      reference) and before/after performance from a release build are attached.
- [ ] Nothing under `benchmarks/results/` is modified.
