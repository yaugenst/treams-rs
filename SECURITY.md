# Security policy

## Supported versions

Security fixes for the 0.1 series go to the `main` branch and are included in
subsequent 0.1.x releases. Pre-release checkouts should update to `main`.

## Reporting a vulnerability

Report a vulnerability privately through
[GitHub private vulnerability reporting](https://github.com/yaugenst/treams-rs/security/advisories/new),
not in a public issue or pull request. Only the maintainers can read the report.
Include:

- the affected versions or commits;
- a minimal reproducer;
- the impact: what an attacker can achieve, and under which conditions.

## Scope

- Memory safety of the native extension `treams_rs._native`. The workspace denies
  unsafe code except in `treams_core::fpenv` (floating-point environment control)
  and the NumPy ufunc loops of the treams-py crate; a crash, out-of-bounds access
  or other undefined behaviour reachable from Python is in scope wherever it
  originates.
- The packaging and CI supply chain: the build configuration, the locked
  dependencies (`Cargo.lock`, `uv.lock`) and the GitHub Actions workflows.

An inaccurate numerical result is a bug, not a vulnerability; report it with the
numerical discrepancy issue form.

## Response

Reports are handled on a best-effort basis. The maintainers reply in the private
advisory as soon as they can and publish the advisory once a fix is available.
