# treams-rs

treams-rs follows treams. Rust (`crates/`) computes values and derivatives; `python/` holds the physics API.

- Using the package: [llms.txt](llms.txt), `treams_rs.support_catalog()` and
  [API discovery](docs/guide/api-discovery.md).
- Changing it: [CONTRIBUTING.md](CONTRIBUTING.md), [development](docs/development/index.md),
  [source ownership](docs/development/architecture.md) and [testing](docs/development/testing.md),
  then [crates/AGENTS.md](crates/AGENTS.md) or [python/AGENTS.md](python/AGENTS.md).

Rules ([details](docs/development/architecture.md#rules)):

- Derivatives are analytic and live in Rust; finite differences appear only in tests.
- A numerical change comes with a reference test, a physical identity and a gradient test;
  never loosen a tolerance or weaken a property to make a test pass.
- Generated files (`docs/reference/python/`, generated regions, `llms.txt`) come from `just docs`.
- Nothing under `benchmarks/results/` changes.
- A public rename gets a `CHANGELOG.md` bullet, and a treams name an `_upstream.py` entry.

Check: `just ci`, and `just check-wheel` after packaging changes.
