---
description: Set up a development checkout, run the checks that hosted CI runs, and check a clean wheel.
---

# Development

## Setup

You need Python 3.12 or 3.13, [uv](https://docs.astral.sh/uv/) 0.12.3,
[just](https://just.systems/) and the Rust toolchain pinned in
[`rust-toolchain.toml`](../../rust-toolchain.toml). From the repository root:

```sh
uv sync --locked --group dev --extra jax --extra torch
just build-ext
uv run --no-sync pre-commit install
```

- `uv sync` installs the development tools and the test references: treams
  0.4.5, SciPy and mpmath. treams-rs itself needs only NumPy. The `jax` and
  `torch` extras are optional; without them the JAX and PyTorch tests skip.
- `just build-ext` compiles the extension into `.venv`. Run it again after
  every change under `crates/`.
- `pre-commit install` runs rustfmt, Clippy, Ruff and Pyrefly before each
  commit.

Run Python commands as `uv run --no-sync python ...` or
`uv run --no-sync pytest ...`. `--no-sync` keeps the environment and the
extension of `just build-ext` as they are.

## Checks

```sh
just ci
```

`just ci` runs the Rust and Python checks of hosted CI, through the same
recipes. Hosted CI also runs `just check-wheel` and 100 Hypothesis examples per
property (`HYPOTHESIS_PROFILE=ci`; the local default is 30). `just ci` has two
halves:

| Recipe | Checks | Hosted CI |
|---|---|---|
| `just ci-rust` | rustfmt, Clippy, the `treams-core` tests and rustdoc, with warnings as errors | once |
| `just ci-python` | file hygiene, lock files, Ruff format and lint, strict Pyrefly, and the Python tests against a fresh development extension | on Python 3.12 with `TREAMS_RS_NUM_THREADS=1` and on Python 3.13 with every CPU, with `HYPOTHESIS_PROFILE=ci` |
| `just check-wheel` | builds an optimized wheel and checks it in a clean environment ([clean wheel](#clean-wheel)); not part of `just ci` | after `just ci-python`, on Python 3.12 and 3.13 |

The Python tests include the check that the generated reference pages and
`llms.txt` match the code.

`just check` runs the lint checks without tests: file hygiene, lock files,
rustfmt, Clippy, Ruff, Pyrefly and `just docs-check`. Use it while you work and
`just ci` before you open a pull request. The [`justfile`](../../justfile)
holds the exact commands.

## Workflows

| Workflow | Runs on | Runs |
|---|---|---|
| [CI](../../.github/workflows/ci.yml) | every pull request and push to `main` | `just ci-rust`; `just ci-python` and `just check-wheel` on Python 3.12 (one thread) and 3.13 |
| [Docs](../../.github/workflows/docs.yml) | every pull request and push to `main`, and by hand | `just docs-build` and `just docs-rust`; publishes the site from `main` once [publishing is turned on](releasing.md#documentation-site) |
| [Formal](../../.github/workflows/formal.yml) | pull requests and pushes that change `formal/`, a Rust file with a Lean model, the `justfile` or the workflow | `just formal` |

## Development builds

`cargo test`, `just rust-test` and `just build-ext` use the dev profile. It
optimizes at level 1 (dependencies at level 2) and keeps debug assertions and
overflow checks. Its timings say nothing about performance; measure with a
release build as described in [benchmarks](benchmarks.md).

## Clean wheel

Run `just check-wheel` after a change to packaging or to public imports. It
builds an optimized wheel and installs it into a clean environment without
SciPy or treams. There it checks the core and Advect behavior, compares native
results on threads that flush subnormals with those of ordinary threads bit for
bit, and checks the optional HDF5 interchange. It also scans the archive for
local paths.

The `just` recipes rewrite the checkout, home, Cargo and Rustup paths in
compiled binaries to `/build`, so no local path ends up in a wheel or in a
panic message. Direct `cargo` or `maturin` calls skip this step, so build
distributable files only through the recipes. The wheel has no Rust SBOM,
because maturin writes absolute paths into its package identifiers;
`Cargo.lock` lists the Rust dependencies.

## Formal proofs

Lean models cover the diffraction-order enumeration, the lattice shells, the
translation selection rules, the LU equilibration and the gradients of
requested illuminations ([formal proofs](../design/formal-proofs.md)). After
changing one of those Rust functions, update its Lean definition and run
`just formal`. It needs [elan](https://github.com/leanprover/elan) and runs
in its own workflow, not in `just ci`. `cargo test` compares `cube`, `degrees`
and `harmonics` with the Lean output in
[`formal/golden/`](../../formal/golden/).
