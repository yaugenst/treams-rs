# treams-rs

Personal Rust rewrite of tfp-photonics/treams. Follow Photonoodle's division:
Rust owns numerical execution and derivatives; Python owns user semantics and
framework adapters. Autodiff frameworks compose the native forward/pullback.

- Rust core: `crates/treams-core`; PyO3 bindings: `crates/treams-py`.
- Typed Python API: `python/treams_rs`; upstream is a development oracle only.
- Preserve treams conventions and supported Python workflows; track parity in
  `docs/status.md`. Unsupported behavior must not silently invoke upstream.
- Read `docs/architecture.md` before changing the numerical boundary.
- Use `uv`, `maturin`, and `just`. `just verify` is the authoritative gate.
- Rust: rustfmt and Clippy with warnings denied. Python: Ruff and strict Pyrefly.
- Implement complete numerical paths with reference, physical-invariant, and
  gradient checks. Never use finite differences as production pullbacks.
- Establish strong Rust tests using proptest for meaningful physical and algebraic
  invariants. Use Hypothesis through the Python/native boundary wherever a strong
  invariant can be stated. Preserve shrinking and replay of failures; avoid weak
  properties such as checking only that a result is finite.
- Keep dependencies and modules minimal; do not copy Photonoodle subsystems
  that have no use here. Preserve upstream license and scientific attribution.
- Build the optimized extension before reporting performance.
- Update `docs/status.md` with verified coverage and remaining gaps.
