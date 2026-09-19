# treams-rs

Rust implementation of tfp-photonics/treams numerical conventions with an
independently designed physics-first Python API.
Keep this repository private. Use its configured owner and credentials for Git
and GitHub operations; do not publish it under a different account or organization.
Rust owns numerical execution and derivatives; Python owns user semantics and
framework adapters. Autodiff frameworks compose the native forward/pullback.

## Route the task

- For using the package, start at [llms.txt](llms.txt) and
  [the agent guide](docs/agents.md). Discover current capabilities through
  `treams_rs.support_catalog()` and signatures through [the API reference](docs/api.md).
- For implementation, read [the development guide](docs/development.md) and the
  nearest scoped `AGENTS.md`: [Rust/bindings](crates/AGENTS.md) or
  [Python/adapters](python/AGENTS.md). Follow the relevant source and test route;
  do not survey unrelated subsystems.
- Read [architecture](docs/architecture.md) before changing the numerical boundary.
  The catalog and generated API documentation must come from their source data;
  do not introduce a second capability registry or edit generated lists by hand.

## Numerical and API contract

- Preserve numerical conventions and supported physical workflows; upstream
  Python API compatibility is not required. Track numerical parity in
  `docs/status.md`. Unsupported behavior must not silently invoke upstream.
- Implement complete numerical paths with reference, physical-invariant, and
  gradient checks. Never use finite differences as production pullbacks.
- Establish strong Rust tests using proptest for meaningful physical and algebraic
  invariants. Use Hypothesis through the Python/native boundary wherever a strong
  invariant can be stated. Preserve shrinking and replay of failures; avoid weak
  properties such as checking only that a result is finite.
- Keep dependencies and modules minimal. Preserve upstream license and
  scientific attribution.

## Verification

- Use `uv`, `maturin`, and `just`. Run focused checks while iterating;
  `just verify` is the authoritative CPU gate. Applicable wheel and
  performance lanes are listed in [the development guide](docs/development.md).
- Rust: rustfmt and Clippy with warnings denied. Python: Ruff and strict Pyrefly.
- Build the optimized extension before reporting performance.
- Update `docs/status.md` with verified coverage and remaining gaps.
- At completion, state what was verified, what was not, and remaining gaps.
  Report Git push, pull-request and comment state separately when changed.
