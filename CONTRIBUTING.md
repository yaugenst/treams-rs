# Contributing

This is a private personal project. Preserve its ownership, visibility and
scientific attribution; see [AGENTS.md](AGENTS.md).

Start with the [development guide](docs/development.md) for setup, source ownership
and runnable checks. Read [architecture](docs/architecture.md) before changing
the Rust/Python boundary and [test strategy](docs/test-strategy.md) for numerical
evidence. For package usage, use [the agent guide](docs/agents.md) and
[generated API reference](docs/api.md).

Keep a change coherent: update the owning implementation, its types/docstrings,
and the check that proves the intended behavior. Include native analytic
pullbacks when adding a differentiable operation. Reuse the existing capability
catalog and documentation generator instead of maintaining parallel support
tables. Record a limitation explicitly when a path is unsupported.
