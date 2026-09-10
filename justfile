set shell := ["bash", "-euo", "pipefail", "-c"]

build-ext:
    uv run --no-sync maturin develop -m crates/treams-py/Cargo.toml

build-ext-release:
    uv run --no-sync maturin develop --release -m crates/treams-py/Cargo.toml

rust-fmt-check:
    cargo fmt --check --all

rust-lint:
    PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --workspace --all-targets --all-features -- -D warnings

rust-test:
    PYO3_PYTHON="$PWD/.venv/bin/python" cargo test -p treams-core

py-format-check:
    uv run ruff format --check .

py-lint:
    uv run ruff check .

py-types:
    uv run pyrefly check --summarize-errors

dependency-lock-check:
    uv lock --check
    cargo metadata --locked --format-version 1 --no-deps > /dev/null

file-hygiene:
    uv run pre-commit run --all-files --hook-stage manual

check: file-hygiene dependency-lock-check rust-fmt-check rust-lint py-format-check py-lint py-types

test-py: build-ext
    uv run --no-sync pytest

test: rust-test test-py

verify: check test

ci: verify
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

build-wheel:
    uv run --no-sync maturin build --release --locked --out dist
