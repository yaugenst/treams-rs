# Testing

Both suites use the same seven domains: `special`, `linalg`, `lattice`,
`waves`, `plane`, `smatrix` and `tmatrix`. A Rust property file
`properties/<domain>.rs` and a Python directory `tests/<domain>/` cover the
same operations. [Validation](../validation/index.md) lists the kinds of
evidence the tests check.

## Rust

### Where a test goes

Rust tests cover implementation details and physical properties:

- A module's inline `tests` module checks how that module computes its
  results: fast paths against reference paths, tables and plans, dispatch
  thresholds, error paths, accuracy against reference tables, and private
  helpers. `smatrix::solve::tests` is an example, and so are the tests that
  compare Rust functions with the Lean models, such as
  `sw::coupling::tests::degrees_match_lean_model`.
- A file in `crates/treams-core/src/properties/` checks what the public
  operations of one domain promise: physical laws, analytic identities, and
  pullbacks against finite differences and adjoint pairings. (A pullback turns
  the gradient with respect to a result into gradients with respect to the
  inputs; see the [glossary](../reference/glossary.md#records-and-gradients).)

A test of one module's implementation goes inline; a physical, analytic or
adjoint identity goes into its domain file. Both groups use crate-private items,
so the crate has no `tests/` directory. `cargo test -p treams-core` runs both,
and `cargo test -p treams-core -- properties::waves` runs one domain.

### Property domains

| File | Domain |
|---|---|
| `special.rs` | Legendre, Wigner and Bessel identities and their adjoints |
| `linalg.rs` | Dense linear-algebra adjoints |
| `lattice/` | Lattice sums and periodic expansions: `checks.rs`, `ewald.rs`, `periodic.rs`, `strategies.rs` and the pinned sums of `tables.rs` |
| `waves.rs` | Spherical and cylindrical vector waves, translations, rotations and field evaluation |
| `plane.rs` | Plane waves, their expansions and diffraction channels |
| `smatrix.rs` | Planar S-matrices: interfaces, layers, composition, illumination and observables |
| `tmatrix.rs` | T-matrices of spheres, cylinders, EBCM particles and clusters |

### Writing a property

Each file starts with one `proptest!` block per case budget. A property there
only draws inputs and calls a documented
`fn check_*(..) -> Result<(), TestCaseError>` below the block, which states and
checks the identity. rustfmt does not format the brace-delimited body of a
`proptest!` block, so the logic stays in the `check_*` function. Plain
`#[test]` functions cover recorded inputs and error paths. Test names are
sentences of the form `<subject>_<verb>_<claim>`, such as
`regular_translations_compose`.

The case budgets are constants of `test_support`:

| Constant | Cases | Use |
|---|---|---|
| `ALGEBRA_CASES` | 256 | closed-form algebraic identities |
| `DEFAULT_CASES` | 64 | physical, scaling and adjoint identities for one numerical operation |
| `EXPENSIVE_CASES` | 24 | lattice sums, cluster solves and surface integrals, with their finite differences |

### Helpers

[`crates/treams-core/src/test_support.rs`](https://github.com/yaugenst/treams-rs/blob/4c6e4bc5fff6bf8aaabdbebe9ea4490cd74d75d0/crates/treams-core/src/test_support.rs)
holds the shared helpers:

- **Pairings.** `re_dot` is `Re Σ conj(a) b`, the pairing of a cotangent (the
  gradient of a loss with respect to a value) with a change of that value;
  `dot` is `Σ a b` for real values.
- **Assertions.** `prop_assert_close!(actual, expected, tolerance)` checks
  `distance < tolerance` and reports both expressions and values. `Distance`
  is the absolute difference of scalars and the Frobenius norm of the
  difference of arrays; arrays of different lengths are infinitely far apart.
  Add `f64::MIN_POSITIVE` to a relative tolerance whose scale can vanish.
- **Finite differences.** `central(h, f)`, normally with `h = 1e-5`, and the
  fourth-order `five_point(h, f)` for rapidly varying singular functions,
  normally with `h = 1e-4`. Bound the error of a difference by the derivative
  and by the rounding noise of the value.
- **Strategies.** `complex`, `log_polar`, `complex_vec`, `complex_matrix`,
  `log_uniform`, `radial`, `radial_kind`, `degree_order`, and the media of
  `material` with their `helicity_ks`.
- **Fixtures.** `patterned` (a deterministic matrix), `rotate` (a point rotated
  by Euler angles), `spherical_basis`, `cylindrical_basis` and
  `gauss_legendre` (quadrature nodes).
- **Thread counts.** `assert_same_bits_on_pools(run)` runs a computation on
  Rayon pools of one to four threads, where the crate's parallel regions run in
  place, and asserts that it returns the same `bits` (the bit patterns of the
  floats of a result) on each. A pullback that adds over many items checks with it
  that the thread count does not change the order of its additions; give it more
  items than `numerics::parallel::REDUCTION_CHUNKS`.
- **Tables.** `table` reads the `key: values` rows of `formal/golden/` and
  `crates/treams-core/references/`.

`clippy.toml` allows `unwrap`, `expect`, `panic` and indexing in tests, so a
failing case stops the test and proptest reduces it to a simpler failing input.

### Generators

- Draw dependent labels together with `prop_flat_map`: `degree_order(1..12)`
  draws `(l, m)` with `|m| <= l`. Reducing a seed modulo `2l + 1` skews the
  distribution and makes failing inputs harder to simplify.
- Draw quantities that span decades log-uniformly (`log_uniform`,
  `log_polar`).
- Use `patterned` only for deterministic fixtures, seeded from a strategy when
  the matrix should vary.

### Regressions

Proptest writes the seed of a failing case to
`crates/treams-core/proptest-regressions/`, in a file at the path of its source
file below `src/`: `properties/lattice/mod.txt` holds the seeds of
`properties/lattice/mod.rs`. Commit these files. Proptest replays every seed of
a file for every property of its source file, before the random cases.

- A seed replays only while its strategy keeps its shape. Before changing such
  a strategy, rewrite each recorded failure as an explicit case with its simplified
  values.
- When a source file splits, copy every seed line of its regression file into
  the regression file of each new source file.

### Pinned data

Tests that compare with fixed values read them from committed tables:

- [`crates/treams-core/references/`](https://github.com/yaugenst/treams-rs/blob/4c6e4bc5fff6bf8aaabdbebe9ea4490cd74d75d0/crates/treams-core/references/README.md)
  holds mpmath values at 30 to 100 digits. `scripts/generate_references.py`
  regenerates or checks each table, and CI only reads them. Never change a
  committed data line; add a new table or new rows.
- `formal/golden/` holds the outputs of the Lean models
  ([formal proofs](../design/formal-proofs.md)).
- Pinned inputs of one check share one table read by one test, such as the
  `SPLITS`, `ZEROS` and `RECORDED` rows in `properties/lattice/tables.rs`.

## Python

### Layout

The directory of a test gives the subsystem; its marker gives the kind of
evidence.

| Directory | Covers | Rust counterpart |
|---|---|---|
| `tests/special/` | special functions, coordinates, Wigner symbols, Ewald integrals | `properties/special.rs` |
| `tests/linalg/` | `diff.solve`, `eig`, `svdvals` | `properties/linalg.rs` |
| `tests/lattice/` | lattice sums, lattice geometry, periodic arrays | `properties/lattice/` |
| `tests/waves/` | spherical and cylindrical waves, translations, rotations, fields, polarization | `properties/waves.rs` |
| `tests/plane/` | plane waves, plane-wave bases, diffraction channels | `properties/plane.rs` |
| `tests/smatrix/` | S-matrices, layer stacks, bands, chirality density | `properties/smatrix.rs` |
| `tests/tmatrix/` | Mie coefficients, sphere and cylinder T-matrices, EBCM, metrics, clusters | `properties/tmatrix.rs` |
| `tests/api/` | the physics objects, operators, bases, namespaces, the API catalog, I/O, documentation and gallery examples | Python layer |
| `tests/bindings/` | the native module: the stub, contexts, ufuncs, the floating-point guard | bindings (`treams-py`) |
| `tests/autodiff/` | `treams_rs.testing`, adjoint identities, automatic dispatch and the four optional framework adapters | Python layer |
| `tests/scripts/` | the benchmark, validation and repository scripts under `scripts/` | none |

`tests/test_suite_rules.py` checks the rules of the suite itself and the
helpers of `tests/_support.py`. Two library tests import scripts, so a
renamed script needs their update too:
`tests/plane/test_diffraction_threshold.py` imports `qualify_papers`, and
`tests/bindings/test_float_environment.py` runs `float_environment.py`.

### Markers

Every test carries at least one marker, for a whole file with `pytestmark` or
per test; `tests/test_suite_rules.py` fails on a test without one. Select tests
with `-m`, for example `pytest -m gradients tests/tmatrix`. The descriptions are
those that `CATEGORIES` in [`tests/conftest.py`](https://github.com/yaugenst/treams-rs/blob/4c6e4bc5fff6bf8aaabdbebe9ea4490cd74d75d0/tests/conftest.py)
registers:

| Marker | Description |
|---|---|
| `physics` | analytic and conservation-law validation |
| `gradients` | native pullback correctness |
| `interface` | Python interface behavior |
| `workflows` | complete scattering workflows |
| `reference` | comparison to independent numerical references |

### Hypothesis profiles

`tests/conftest.py` loads the profile, so a property runs the same number of
examples alone or in the full suite. Never load a profile in a test module.

| Profile | Examples | Selected by |
|---|---|---|
| `dev` | 30 | default |
| `ci` | 100 | `HYPOTHESIS_PROFILE=ci` (hosted CI) |
| `thorough` | 300 | `HYPOTHESIS_PROFILE=thorough` or `--hypothesis-profile=thorough` |

No profile has a deadline, because debug-build timings vary. On CI, Hypothesis
also derandomizes every profile and runs without its example database. Every
health check stays enabled locally and on CI, including the check for slow
input generation. An explicit `@settings(max_examples=...)` bounds an expensive
property under every profile.

### Parallel runs

Hosted CI runs pytest with `-n auto`, one
[pytest-xdist](https://pytest-xdist.readthedocs.io/) worker per CPU; add it to
a local run to do the same. `tests/conftest.py` gathers the Rayon global-pool
check and the report of unavailable test dependencies from every worker, so a
parallel run fails and reports as a serial one does.

### Helpers and imports

[`tests/_support.py`](https://github.com/yaugenst/treams-rs/blob/4c6e4bc5fff6bf8aaabdbebe9ea4490cd74d75d0/tests/_support.py) holds what several test files
share:

- `assert_reusable_context`, which checks that repeated pullbacks agree and
  rejects wrong cotangents;
- `selecting` and `varying`, which adapt a native record for `check_pullback`;
- `LAYOUTS`, `arrange`, `layouts` and `strided_copies` for memory layouts, and
  `sum_to`, the adjoint of broadcasting;
- `complex_normal`, `finite_complex`, `complex_arrays` and the `degree_order`
  strategy;
- `assert_unitary_ports`, the `reciprocal` mode map, and `to_oracle` and
  `oracle_smatrix_array`, which build the matching treams objects.

[`tests/_scripts.py`](https://github.com/yaugenst/treams-rs/blob/4c6e4bc5fff6bf8aaabdbebe9ea4490cd74d75d0/tests/_scripts.py) imports a script of `scripts/`
as a module for the tests in `tests/scripts/`.

Test modules import `treams_rs` as `tr`, its namespaces by name
(`from treams_rs import special`), Advect as `ad`, treams as `treams` or
`treams.<ns> as upstream_<ns>`, and SciPy's special functions as
`scipy_special`.

### Adding a test

1. Put the test in the directory of its subsystem, next to the tests of the
   same operation.
2. Mark it: `physics`, `gradients`, `interface`, `workflows` or `reference`.
3. Keep its cost within the profile budget. Bound an expensive property with
   `@settings(max_examples=...)`.
4. Preserve a failure that Hypothesis found with `@example(...)` and the simplified
   values, so it runs on every profile.
5. Check derivatives with `treams_rs.testing.check_pullback` for a record, or
   `check_gradient` for a scalar objective
   ([gradient checks](../differentiation/gradient-checks.md)), instead of a
   hand-written finite difference.
6. Never loosen a tolerance to make a test pass. Find out which side is wrong,
   and state the reason for every tolerance that is not the default.
