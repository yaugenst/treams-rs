# Development

Use Python 3.12 or 3.13, uv 0.12.3, `just`, and the Rust toolchain pinned in
[`rust-toolchain.toml`](../rust-toolchain.toml). From the repository root:

```sh
uv sync --locked --group dev --extra jax --extra torch
just build-ext
uv run --no-sync pre-commit install
```

The optional framework extras let the adapter tests run. The core package needs
only NumPy at runtime; upstream treams and SciPy are development oracles.
Run Python commands with `uv run --no-sync` after setup so an invocation does
not remove optional dependencies from the environment.

## Source and test ownership

Read the nearest scoped instructions before editing
[`crates/`](../crates/AGENTS.md) or [`python/`](../python/AGENTS.md).

| Change | Owning source | Focused check |
|---|---|---|
| Numerical kernels and native adjoints | `crates/treams-core/src/`; matching Rust module tests and `properties.rs` | `cargo test --locked -p treams-core` |
| Array conversion, GIL release, native contexts | `crates/treams-py/src/`; `python/treams_rs/_native.pyi` | `just build-ext`, then the matching Python suite |
| Physics objects, metadata, operator semantics | `python/treams_rs/_core.py`, `_array.py`, `_tmatrix.py`, `_smatrix.py`, `_operators.py` | `uv run --no-sync pytest tests/test_api.py tests/test_operator_objects.py tests/test_matrix_conveniences.py` |
| Recording and context ownership | `python/treams_rs/diff.py`; matching Rust/binding module | `uv run --no-sync pytest -m ad_contract` |
| Framework bridges | `python/treams_rs/advect.py`, `_adapters.py`, `jax.py`, `torch.py` | `uv run --no-sync pytest tests/test_advect.py tests/test_jax.py tests/test_torch.py` |
| Requested illuminations and iterative solves | `crates/treams-core/src/illumination.rs`, `iterative.rs`; matching bindings and `python/treams_rs/iterative.py` | `uv run --no-sync pytest tests/test_illumination.py tests/test_iterative_native.py` |
| Optional HDF5 interchange | `python/treams_rs/io.py` | `uv run --no-sync pytest tests/test_io.py` |
| CUDA execution | `crates/treams-cuda/`, `crates/treams-cuda-tile/`; native/Python `cuda` modules | `just rust-cuda-check`; `just gpu-check` on supported hardware |

Rebuild the extension after changing Rust or bindings. For a particular kernel,
run the corresponding `tests/test_*.py` suite as well as its native test: Python
tests cover broadcasting, noncontiguous arrays and the public workflow. See
[test strategy](test-strategy.md) for physical and algebraic invariants and
[public testing helpers](testing.md) for checking a composed objective or custom
recording function. Finite differences belong in qualification, never execution.

The `just` recipes remap local checkout, home, and configured Cargo/Rustup paths
in compiled binaries to neutral build paths.
Direct `cargo` or `maturin` invocations do not inherit this guarantee; use the
documented build entry points when preparing distributable artifacts. Wheel
qualification scans the complete archive for local paths. Maturin's generated
Rust SBOM is disabled because its workspace package identifiers contain absolute
paths; `Cargo.lock` remains the dependency source of truth.

Rust owns numerical work and analytic pullbacks. PyO3 owns the Python/native
boundary; Python owns physics metadata, user semantics and framework convention
conversion. Keep the native pairing `dL = Re(vdot(cotangent, direction))` and
one-use residual contract explicit. A Python docstring must state gradient order
when it differs from the forward argument order. See [architecture](architecture.md)
and [adapter contracts](adapters.md).

## Checks before completion

```sh
just verify
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

`just verify` runs file hygiene, dependency lock checks, Rust formatting/Clippy,
Python formatting/lint/type checks, native tests, and Python tests against a
fresh development extension. The recipes in [`justfile`](../justfile) and
hosted [CI](../.github/workflows/ci.yml) own the exact commands. For an API change,
update the source signature, docstring, native stub and capability entry where
applicable, then run `just docs` to regenerate the [API reference](api.md) and
`llms.txt`. `just docs-check`, included in `just check`, rejects stale generated
files. [`scripts/generate_agent_docs.py`](../scripts/generate_agent_docs.py) and
`treams_rs.support_catalog()` own this path. Do not hand-edit generated API/support
lists. The installed catalog includes signatures and docstrings even for optional
modules whose frameworks are absent; `python -m treams_rs --format markdown`
prints that reference offline.

Mark runnable documentation fences with `python exec`; `tests/test_docs.py`
executes them. Keep these examples self-contained, with valid physical inputs
and assertions that check the claimed result. Run
`uv run --no-sync pytest tests/test_docs.py` after changing an example.

Run `just check-wheel` for packaging or public import changes. It installs an
optimized wheel into a clean environment and checks core/Advect behavior plus
optional HDF5 interchange.

`just rust-cuda-check` compiles the dynamically loaded dense backend without a
toolkit. Executing `just gpu-check` requires the NVIDIA driver and CUDA 13.3
cuTile toolchain described in [GPU setup](gpu.md). It replaces the installed
extension with a CUDA-enabled release build. Neither a CPU check nor a WASM
check qualifies GPU execution. Framework adapters accept CPU arrays. The fused
CUDA field kernel is forward-only; a fixed sampling operator supports
coefficient pullbacks through its Hermitian matrix product. Geometry/wavevector
GPU pullbacks and automatic framework GPU routing are unsupported. Keep CUDA
dependencies out of default CPU/WASM builds.

For performance changes, build with `just build-ext-release` and run the affected
benchmark on an otherwise idle host. Correctness checks rebuild a development
extension, so rebuild release after `just verify` or `just test-py`. The
[benchmark guide](benchmarks.md) describes parity gates, thread budgets, raw
evidence and the existing memory exceptions. Report setup/transfers and warmup
separately when they matter. Do not generalize a measured crossover to every
problem size.

State checks run, checks not run and remaining gaps when handing off a change.
Keep [status](status.md) consistent with verified support. Report push, PR and
comment state separately.
