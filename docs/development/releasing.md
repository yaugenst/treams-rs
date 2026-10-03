---
description: Set the version, record the changes, check the wheel and turn on the documentation site.
---

# Releasing

## Version

`version` under `[workspace.package]` in the root `Cargo.toml` is the only
version number. `treams-py` inherits it, and maturin reads it into the Python
package because `pyproject.toml` declares the version as dynamic.

## Release steps

1. Rename `## [Unreleased]` in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD`.
2. Add an empty `## [Unreleased]` section above it, with the same
   subsections.
3. Set `X.Y.Z` in `Cargo.toml`, run `cargo update --workspace` to write it
   into `Cargo.lock`, and set `version: X.Y.Z` and
   `date-released: YYYY-MM-DD` in `CITATION.cff` (add them on the first
   release).
4. Run `just ci` and `just check-wheel`, and `just formal` if a Rust function
   with a Lean model changed.
5. Commit the four files together and tag the commit `vX.Y.Z`.

## Wheel check

`just check-wheel` builds an optimized wheel, installs it into a clean
environment without SciPy or treams, and checks the core and Advect behavior,
native results on threads that flush subnormals, optional HDF5 interchange and
the archive for local paths ([clean wheel](index.md#clean-wheel)).

## Documentation site

The Docs workflow builds the site on every pull request and push to `main`, and
publishes it only when publishing is turned on:

1. In the repository settings, open Pages and set Source to GitHub Actions.
2. Under Settings → Secrets and variables → Actions → Variables, add the
   repository variable `DOCS_DEPLOY` with the value `true`.

The next push to `main` publishes the site at
<https://yaugenst.github.io/treams-rs/>.
