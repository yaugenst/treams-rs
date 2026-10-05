set shell := ["bash", "-euo", "pipefail", "-c"]

# Keep local build paths out of distributed binaries, including dependency panics.
export CARGO_ENCODED_RUSTFLAGS := `python3 scripts/wheel_build_paths.py rustflags`

build-ext:
    uv run --no-sync maturin develop -m crates/treams-py/Cargo.toml

build-ext-release:
    uv run --no-sync maturin develop --release -m crates/treams-py/Cargo.toml

# Install the locked test environment, including the CPU build of PyTorch.
ci-python-setup python:
    #!/usr/bin/env bash
    set -euo pipefail
    uv python install "{{ python }}"
    uv sync --locked --no-install-project --group dev --extra jax --extra autograd \
        --python "{{ python }}"
    torch_version="$(
        uv export --locked --extra torch --no-emit-project --no-hashes \
            --no-header --no-annotate | sed -n 's/^torch==\([^ ;]*\).*/\1/p'
    )"
    test -n "$torch_version"
    uv pip install \
        --python .venv/bin/python \
        --index-url "$TORCH_CPU_INDEX" \
        "torch==${torch_version}+cpu"

rust-fmt-check:
    cargo fmt --check --all

rust-lint:
    PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --locked --workspace --all-targets -- -D warnings

rust-test:
    PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --locked -p treams-core

rust-doc:
    PYO3_PYTHON="$PWD/.venv/bin/python" RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps

py-format-check:
    uv run --no-sync ruff format --check .

py-lint:
    uv run --no-sync ruff check .

py-types:
    uv run --no-sync pyrefly check --summarize-errors

dependency-lock-check:
    uv lock --check
    cargo metadata --locked --format-version 1 --no-deps > /dev/null

file-hygiene:
    uv run --no-sync pre-commit run --all-files --hook-stage manual --show-diff-on-failure

# The checks of `check` that need no built extension.
check-static: file-hygiene dependency-lock-check rust-fmt-check rust-lint py-format-check py-lint py-types

check: check-static docs-check

# Regenerate the generated documentation (API reference pages, generated regions, llms.txt) from a freshly built extension.
docs: build-ext
    uv run --no-sync python scripts/generate_docs.py

docs-check: build-ext
    uv run --no-sync python scripts/generate_docs.py --check

# Run the examples gallery and record what each script prints in docs/examples/output.
docs-examples: build-ext
    uv run --no-sync python scripts/generate_docs.py --examples

# Build the site strictly into site/ with the docs group in an isolated environment.
docs-build:
    NO_MKDOCS_2_WARNING=1 uv run --isolated --locked --only-group docs mkdocs build --strict

# Preview the site locally with live reload.
docs-serve:
    NO_MKDOCS_2_WARNING=1 uv run --isolated --locked --only-group docs mkdocs serve

# Add the treams-core rustdoc, private items included, to the built site under site/rust.
docs-rust:
    rm -rf target/doc
    cargo doc --locked -p treams-core --no-deps --document-private-items
    rm -rf site/rust
    mkdir -p site
    cp -a target/doc site/rust

test-py: build-ext
    uv run --no-sync pytest

test: rust-test test-py

formal:
    cd formal && lake exe cache get && lake build --wfail && lake env lean --run Golden.lean --check

# Checks independent of the Python version; hosted CI runs them once.
ci-rust: rust-fmt-check rust-lint rust-test rust-doc

# Checks that hosted CI runs for every supported Python version; test-py also
# checks that the generated documentation is current.
ci-python: file-hygiene dependency-lock-check py-format-check py-lint py-types test-py

ci: ci-rust ci-python

build-wheel:
    uv run --no-sync maturin build --release --locked --out dist

check-wheel: build-wheel
    #!/usr/bin/env bash
    set -euo pipefail
    wheel="$(ls -t dist/treams_rs-*.whl | head -n 1)"
    uv run --no-sync python scripts/wheel_build_paths.py check "$wheel"
    environments="$(mktemp -d "${TMPDIR:-/tmp}/treams-wheel-XXXXXX")"
    trap 'rm -rf "$environments"' EXIT
    profiles=(base advect autograd)
    if .venv/bin/python -c 'import sys; raise SystemExit(sys.version_info >= (3, 15))'; then
        profiles+=(io)
    else
        echo "Skipping io: h5py publishes no CPython 3.15 wheels"
    fi
    for profile in "${profiles[@]}"; do
        python="$environments/$profile/bin/python"
        uv venv --quiet --python .venv/bin/python "$environments/$profile"
        dependencies=(--only-binary :all: -r pyproject.toml)
        requirement="$wheel"
        if [[ "$profile" != "base" ]]; then
            dependencies+=(--extra "$profile")
            requirement="$wheel[$profile]"
        fi
        uv pip install --quiet --python "$python" "${dependencies[@]}"
        uv pip install --quiet --python "$python" --no-index "$requirement"
        "$python" scripts/smoke_wheel_install.py "$profile"
    done

# Hosted CI CPUs are shared; run on an otherwise idle performance host.
# Rerun recorded gated reference benchmarks into benchmarks/results/local.
bench group="all": build-ext-release
    uv run --no-sync python scripts/replay_gated_benchmarks.py --group "{{ group }}"

# Scattering, fields, special functions, coordinates, waves and namespaces.
bench-performance: (bench "performance")

# Native geometry and mode metadata helpers, including scalar and broadcast calls.
bench-geometry: (bench "geometry")

# All scalar Ewald components and direct shells, plus their recorded boundaries.
bench-lattice: (bench "lattice")

# Python operator workflows and native custom-table and real-degree boundaries.
bench-api: (bench "api")

# Public power workflows and the complete recorded native power boundary.
bench-power: (bench "power")

# Every recorded gated reference case, sequentially on the same idle CPU set.
bench-all: (bench "all")

# Compare this checkout's Python sources with those of a git ref on one native build.
bench-compare ref="main" *args: build-ext-release
    uv run --no-sync python scripts/compare_builds.py --baseline-ref "{{ ref }}" {{ args }}

# Regenerate the distribution license bundle after dependencies change.
licenses:
    cargo fetch --locked
    uv run --no-sync python scripts/third_party_licenses.py

licenses-check:
    cargo fetch --locked
    uv run --no-sync python scripts/third_party_licenses.py --check

deny:
    cargo deny check -W unmaintained
