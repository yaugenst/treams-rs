# Contributing to treams-rs

treams-rs follows the numerical conventions of
[treams](https://github.com/tfp-photonics/treams), and contributions from treams
users are especially welcome: bug reports, numerical comparisons, documentation
and code.

## Reporting issues

Open an issue with one of the
[issue forms](https://github.com/yaugenst/treams-rs/issues/new/choose): bug
report, numerical discrepancy or feature request.

- Check [Differences from treams](https://yaugenst.github.io/treams-rs/latest/coming-from-treams/differences/)
  first. It records intentional differences and known reference defects,
  including the treams version of each comparison.
- A numerical discrepancy needs a minimal script, the treams-rs, treams, Python
  and NumPy versions, the reference result and the accuracy you expect.

## Development setup

The core supports CPython 3.12–3.15; use 3.12 or 3.13 for the complete
reference suite. You also need [uv](https://docs.astral.sh/uv/) 0.12.22,
[just](https://just.systems/) and Rust 1.94.0, pinned in
[`rust-toolchain.toml`](rust-toolchain.toml). From the repository root:

```sh
uv sync --locked --no-install-project --group dev
just build-ext
just ci
```

`just build-ext` builds the development extension; run Python with
`uv run --no-sync`. The [development setup](https://yaugenst.github.io/treams-rs/latest/development/#setup)
adds JAX and the CPU build of PyTorch for the optional adapter tests.
`just ci` runs the local Rust and Python checks. Hosted CI also checks native
wheels, dependency bounds and the supported Python versions; use
`HYPOTHESIS_PROFILE=ci` for its 100 examples per property.

The development pages on the site cover the rest:

- [Development](https://yaugenst.github.io/treams-rs/latest/development/): setup,
  checks, CI workflows and the clean-wheel check.
- [Source ownership](https://yaugenst.github.io/treams-rs/latest/development/architecture/):
  which files own a change, the rules that always hold, and how to add a
  binding or a physics feature.
- [Testing](https://yaugenst.github.io/treams-rs/latest/development/testing/): where a
  test goes, markers, Hypothesis profiles and helpers.
- [Documentation](https://yaugenst.github.io/treams-rs/latest/development/documentation/):
  the site, tested examples, the generated reference and the gallery.
- [Benchmarks](https://yaugenst.github.io/treams-rs/latest/development/benchmarks/)
  and [releasing](https://yaugenst.github.io/treams-rs/latest/development/releasing/).

## Definition of done

A pull request is ready for review when:

- each new behaviour is tested where it belongs: Rust proptest properties for
  the physics and the gradients of the numerical core, Python Hypothesis and
  reference tests for the Python API;
- every differentiable operation has an analytic pullback, the Rust function
  that turns the gradient with respect to the output into gradients with
  respect to the inputs; finite differences appear only in tests;
- docstrings and the native stub `python/treams_rs/_native.pyi` match the code,
  and `just docs` has regenerated the generated reference and `llms.txt`;
- `CHANGELOG.md` has a bullet under "Unreleased", and a public name that differs
  from its treams counterpart is listed in `python/treams_rs/_upstream.py`;
- a numerical or performance-critical change shows agreement with the pinned
  treams 0.4.7 reference implementation or an independent reference, and
  before/after performance measured with a release build on an otherwise idle host;
- archived measurements stay unchanged; privacy-only transformations follow
  [the evidence policy](benchmarks/README.md#privacy-and-provenance).

## License of contributions

treams-rs is released under the [MIT license](LICENSE). By opening a pull
request, you license your contribution under the same license. Code ported from
treams keeps its attribution in [LICENSE.treams](LICENSE.treams) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Add the source and license
attribution when you port further code.

## Conduct and security

Everyone taking part follows the [code of conduct](CODE_OF_CONDUCT.md). Report
security problems privately as described in the [security policy](SECURITY.md),
never in a public issue.
