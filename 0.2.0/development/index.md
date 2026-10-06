# Development

## Setup

You need CPython 3.12–3.15, [uv](https://docs.astral.sh/uv/) 0.12.18 or later
in the 0.12 series, [just](https://just.systems/) and Rust 1.94.0, pinned in
[`rust-toolchain.toml`](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/rust-toolchain.toml). Use Python 3.12 or 3.13
for the complete reference suite. From the repository root:

```sh
uv sync --locked --no-install-project --group dev --extra jax --extra autograd
just build-ext
uv run --no-sync pre-commit install
```

- `uv sync` installs the development tools and the test references: treams
  0.4.7, SciPy and mpmath. treams-rs itself needs only NumPy. Historical
  comparisons against treams 0.4.5 retain that version in their evidence.
- JAX, PyTorch and HIPS Autograd are optional; without them their adapter tests skip.
- `just build-ext` compiles the extension into `.venv`. Run it again after
  every change under `crates/`.

On Linux, add the locked CPU build of PyTorch as hosted CI does:

```sh
torch_version="$(
  uv export --locked --extra torch --no-emit-project --no-hashes \
    --no-header --no-annotate | sed -n 's/^torch==\([^ ;]*\).*/\1/p'
)"
uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==${torch_version}+cpu"
```

This CPU index covers Python 3.15 in the release dependency set. On macOS,
add `--extra torch` to the initial `uv sync` command on Python 3.12–3.14;
that installs PyTorch from PyPI. The standard PyPI `torch` extra on 3.15
is not tested for this release.

Run Python commands as `uv run --no-sync python ...` or
`uv run --no-sync pytest ...`. `--no-sync` keeps the environment and the
extension of `just build-ext` as they are.

The development dependency set omits the treams reference implementation on
Python 3.14–3.15 and h5py on 3.15. Tests that require an unavailable dependency skip
and appear in the test summary; independent tests still run. On 3.12 and
3.13, a missing treams installation is an error.

## Repository hooks

`uv run --no-sync pre-commit install` installs both the `pre-commit` and
`commit-msg` hooks from [`.pre-commit-config.yaml`](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/.pre-commit-config.yaml).
Git worktrees share these hooks. Their first run downloads the hook tools.

| Hooks | Checks |
| --- | --- |
| `pre-commit-hooks` | Whitespace, final newlines, line endings, YAML and JSON, conflict markers, symlinks, file sizes and private keys |
| `markdownlint-cli2` | Markdown formatting with [`.markdownlint-cli2.yaml`](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/.markdownlint-cli2.yaml); needs Node 22 or newer, which pre-commit downloads if Node is absent |
| `taplo-format` | TOML formatting |
| `typos` | Spelling, with exceptions in [`_typos.toml`](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/_typos.toml) |
| `commitizen` | Conventional Commit messages: `type(optional scope): subject` |
| Local `just` hooks | rustfmt, Clippy, Ruff and Pyrefly using the locked tools |

Review and stage any automatic formatting changes before committing again.
`just file-hygiene`, part of `just check` and CI, runs the file hooks over the
whole repository. Commitizen checks messages; it does not manage versions.
`Cargo.toml` sets the package version ([releasing](releasing.md)). Fix
generated documentation in its source or in
[`scripts/generate_docs.py`](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/scripts/generate_docs.py), then run
`just docs`.

## Checks

```sh
just ci
```

`just ci` runs the local Rust and Python checks through the same recipes
used by hosted CI. Use `HYPOTHESIS_PROFILE=ci` for 100 examples per property
(the local default is 30). `just ci` has two halves:

| Recipe | Checks |
|---|---|
| `just ci-rust` | rustfmt, Clippy, the `treams-core` tests and rustdoc, with warnings as errors |
| `just ci-python` | file hygiene, lock files, Ruff format and lint, strict Pyrefly, and the Python tests against a fresh development extension |
| `just check-wheel` | builds an optimized wheel and checks it in a clean environment ([clean wheel](#clean-wheel)); not part of `just ci` |

The Python tests include the check that the generated reference pages and
`llms.txt` match the code. Tests live under `tests/<domain>/`; the
[testing guide](testing.md) maps each directory to its subject.

`just check` runs the lint checks without tests: file hygiene, lock files,
rustfmt, Clippy, Ruff, Pyrefly and `just docs-check`. `just check-static` runs
all of them except `just docs-check`, which needs a built extension. Use
`just check` while you work and `just ci` before you open a pull request. The
[`justfile`](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/justfile) holds the exact commands.

## Workflows

These are the configured checks; a release needs successful runs for its
exact source revision, as described in [releasing](releasing.md).

| Workflow | Runs on | Runs |
|---|---|---|
| [CI](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/.github/workflows/ci.yml) | pull requests and pushes to `main` | Python 3.12–3.15, supported dependency bounds, Rust checks, coverage, repository checks, the host release wheel, native wheels, documentation and a history secret scan |
| [Native Wheels](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/.github/workflows/native-wheels.yml) | called by CI and Release Candidate, or by hand | All [supported wheels](../getting-started/install.md#install-from-pypi), installed-wheel checks and a source-distribution rebuild |
| [Docs](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/.github/workflows/docs.yml) | called by CI, publication and manual [Deploy docs](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/.github/workflows/deploy-docs.yml) | Strict Material site build and rustdoc; mike maintains `dev`, released versions and the `latest` alias, then the complete site is deployed through a GitHub Pages artifact |
| [Release Candidate](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/.github/workflows/release-candidate.yml) | called by Publish Release | Verify the source revision and build distribution files that cannot change during publication |
| [Publish Release](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/.github/workflows/publish-release.yml) | repository dispatch with event type `release` | Build and test the candidate, try it on TestPyPI (advisory), wait for the operator's tag and approval, then publish to PyPI and deploy release documentation |
| [Formal](https://github.com/yaugenst/treams-rs/blob/4720870afb93d4ff1ac1f564834580c783715849/.github/workflows/formal.yml) | pull requests and pushes that change `formal/`, a Rust file with a Lean model, the `justfile` or the workflow | `just formal` |

CI runs the complete reference suite on Python 3.12 and 3.13 and the available
tests on 3.14 and 3.15. Separate jobs cover NumPy 2.1–2.5, all declared
dependency floors together on 3.12, and current framework releases on 3.13
and 3.14. Rust checks cover 1.94.0 and stable. These checks, including the
coverage thresholds below, must pass for `CI Success`, which is required for
release.

The Coverage job checks the Python 3.12 report directly. Overall line coverage
may drop by at most **0.1 percentage points** from the exact base commit, and
at least **90% of changed executable lines** must be covered. The base is the
pull request's base SHA, or the previous `main` SHA for a push. CI reuses that
commit's cached report or its saved CI artifact, retained for 90 days. If both
are missing, the same Coverage job regenerates the report from that exact
commit using its locked dependencies and Python 3.12 tests, then caches it for
later runs of the pull request. Normal checks reuse the report without running
the base tests again. A regeneration failure fails the check; it never skips
the comparison.

To check two downloaded reports locally, run from the repository root:

```sh
uv run --script scripts/check_coverage.py coverage.xml base-coverage.xml --compare-branch BASE_SHA
```

Codecov receives a best-effort upload on `main` for the badge. Its PR comments
and status checks are disabled, and an upload failure cannot fail `CI Success`.

When a dependency range changes, update its CI job too. A change to a
Rust dependency linked into wheels also needs `just licenses` to refresh the
bundled license texts.

## Development builds

`cargo test`, `just rust-test` and `just build-ext` use the dev profile. It
optimizes at level 1 (dependencies at level 2) and keeps debug assertions and
overflow checks. Its timings do not represent release performance; measure
with a release build as described in [benchmarks](benchmarks.md).

## Clean wheel

Run `just check-wheel` after a change to packaging or to public imports. It
builds an optimized wheel and installs it into a clean environment without
SciPy or treams. There it checks the core, Advect and HIPS Autograd behavior, compares native
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
[`formal/golden/`](https://github.com/yaugenst/treams-rs/tree/4720870afb93d4ff1ac1f564834580c783715849/formal/golden).
