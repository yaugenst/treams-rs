# Changelog

## 0.1.0 (unreleased)

First public release of treams-rs: electromagnetic T-matrix scattering with a
Rust numerical core, a typed Python API and analytic first-order derivatives.
This entry describes the release being prepared; it is not a publication notice.

### Capabilities

- Spherical, cylindrical and plane waves; helicity and parity bases; translations,
  rotations, polarization changes and finite or periodic basis expansions.
- Homogeneous and multilayer spheres and cylinders, including lossy, magnetic
  and chiral materials; finite particle clusters, periodic arrays and planar
  stacks. Dense and matrix-free sphere-cluster solvers support different problem
  sizes.
- Electric and magnetic fields, scattering cross sections and widths, power
  balance, circular dichroism, chirality and Bloch bands. Axisymmetric EBCM
  supports particles described by a sampled radial surface.
- The numerical namespaces `special`, `sw`, `cw`, `pw`, `lattice`, `coeffs`,
  `misc` and `ebcm` follow treams function names and conventions. The physics
  objects carry their basis, wavenumber and media explicitly.
- Native analytic pullbacks in `treams_rs.diff`, with optional Advect, JAX and
  PyTorch adapters for differentiating complete objectives. Gradient checks are
  available through `treams_rs.testing`.
- Optional HDF5 T-matrix input/output through `treams_rs.io`, including matrix
  sweeps and physical metadata.
- An owned CPU worker pool with `set_num_threads`, `get_num_threads`,
  `threads` and `thread_info`, process-pool support and optional threadpoolctl
  integration. Values and gradients repeat bit for bit across thread budgets
  within one build and machine; different builds or processors can differ.
- Real-degree Ferrers functions use an original implementation of DLMF
  formulas; upstream treams attribution and linked-library license notices
  accompany the distributions.
- Reference documentation, executable examples, treams comparisons, independent
  physical checks, numerical limits and benchmark provenance. Offline API help
  is available through `python -m treams_rs` and `support_catalog()`.

### Supported scope

- CPython 3.12–3.15. Optional frameworks and HDF5 support depend on their own
  package availability; see
  [installation](https://yaugenst.github.io/treams-rs/latest/getting-started/install/).
- CPU execution and first-order reverse-mode derivatives. Discrete mode labels,
  cutoffs and topology stay fixed during differentiation. GPU execution and
  higher-order derivatives are unsupported.
- The core Python package requires NumPy at run time; it does not call treams,
  SciPy or Cython. Framework dtype and transform support is listed in
  [framework adapters](https://yaugenst.github.io/treams-rs/latest/differentiation/frameworks/).
- Numerical agreement is subject to the documented conventions, deliberate
  differences and conditioning limits; see
  [validation](https://yaugenst.github.io/treams-rs/latest/validation/).

### Migrating from pre-release checkouts

The public API was consolidated before 0.1.0. The table below maps earlier
pre-release names to the release API; it does not promise aliases for old names.
For the complete mapping from upstream treams, including object members and
keywords, see the
[treams name map](https://yaugenst.github.io/treams-rs/latest/coming-from-treams/names/).

| Earlier name or call | Release API |
| --- | --- |
| `SphericalWaveBasis` | `SphericalBasis` |
| `CylindricalWaveBasis` | `CylindricalBasis` |
| `PlaneWaveBasisByUnitVector` | `PlaneWaveBasis` |
| `PlaneWaveBasisByComp` | `PlaneWavePorts` |
| `MultipoleWave` | `Wave` |
| `TMatrixC` | `CylindricalTMatrix` |
| `SMatrices` | `SMatrix`; a single port block is `ScatteringBlock` |
| Top-level operators and `PhysicsArray` | `treams_rs.operators` |
| `treams_rs.support` | `support_catalog()` |
| `coeffs.fresnel_with_context`, `coeffs.mie_with_context`, `coeffs.mie_cyl_with_context` | `diff.fresnel`, `diff.mie`, `diff.mie_cyl` |
| `lattice.expansion_with_context(destination, source, ks, a, kpar)` | `diff.lattice_expansion(destination, source, ks, kpar, a)` |
| `lattice.expansion(...)` | The value from `diff.lattice_expansion(...)[0]`, or `sw.translate_periodic` / `cw.translate_periodic` |
| `diff.interface`, `diff.propagation` | `diff.interface_coefficients`, `diff.propagation_matrix` |
| `diff.cluster`, `diff.cluster_factor` | `diff.sphere_cluster`, `diff.sphere_cluster_factor` |
| `diff.periodic_from_table`, `diff.periodic_conversion`, `diff.wigner` | `diff.lattice_expansion_from_table`, `diff.periodic_to_cw`, `diff.wignerd` |
| `advect.cluster`, `advect.periodic_from_table`, `advect.periodic_conversion`, `advect.wigner` | `advect.sphere_cluster`, `advect.lattice_expansion_from_table`, `advect.periodic_to_cw`, `advect.wignerd` |
| `advect.smatrix_cd` | `advect.smatrix_circular_dichroism` |
| `iterative.Pullback` | `iterative.IterativeContext` |
| `iterative.SphereCluster.solve_with_pullback(...)` | `iterative.SphereCluster.record(...)` |
| `plane_wave(..., polarization=<state>)` | `plane_wave(..., pol=<state>, k0=...)`; `polarization` selects `"helicity"` or `"parity"` |
| Multipole-wave `kind="outgoing"` | `kind="singular"`; framework wave `outgoing=` becomes `singular=` |
| Special-function selector `kind=` | `function=` in `bessel`, `angular`, `vector_wave`, `coordinates` and `vector_coordinates` records and adapters |
| `tmatrix_metric(kind=...)` | `tmatrix_metric(metric=...)` |
| `vector_wave(polarization=...)` | `vector_wave(pol=...)` |
| `PeriodicResponse.to_smatrix(orders=...)` in framework adapters | `diffraction_orders=...` |
| `ebcm.qmat(legacy=True)` and its differentiable variants | `radial_area_factor=False`; the default includes the radial surface-area factor |
| `diff.ebcm_qmat(out=..., in_=...)` and its Advect adapter | `destination=..., source=...` |
| `PlaneWave.amplitudes` | `PlaneWave.coefficients` |
| Catalog key `contract`; adapter key `contract` | `rules`; adapter key `summary` |

Further behavioral changes for pre-release callers:

- Wave constructors require an explicit `k0`. There is no global
  `config.POLTYPE`; pass `polarization=` or `poltype=` where appropriate.
- `TMatrix.translate` and `CylindricalTMatrix.translate` return a T-matrix;
  read `.array` for its NumPy matrix or use `operators.Translate` for the
  translation operator.
- `Cluster(particles, positions=...).solve()` constructs the coupled response.
  A `TMatrix` has the same behavior whether it came from one particle or a
  cluster.
- A record returns `(value, context)`. Its pullback consumes the context and
  returns gradients in forward-argument order. In particular,
  `diff.sphere_cluster` returns gradients `(k0, radii, epsilon, positions)`;
  `iterative.Gradient` adds `incident` and `convergence` after those fields.
- Framework field functions use `positions` and per-mode `kz`; their wave and
  T-matrix objects read axial wavenumbers from `basis.kz`. Framework port waves
  and S-matrices hold their port metadata in `ports`.
- Old class-name pickles are not compatible with the renamed classes. Use the
  public reference for native context types and Rust paths instead of relying
  on pre-release internal names.
- Removed the private-project and credentials wording, the repository-tooling
  credit in LICENSE and THIRD_PARTY_NOTICES.md, the references to unpublished
  experiment branches and the instructions to report push and pull-request state.
- Added the community files: a public CONTRIBUTING.md (issues, setup, definition of
  done, license of contributions), CODE_OF_CONDUCT.md (Contributor Covenant 2.1),
  SECURITY.md, CITATION.cff, GitHub issue forms (bug report, numerical
  discrepancy, feature request) and a pull request template.
- Added a documentation site built with MkDocs and the Material theme
  (`mkdocs.yml`, home page `docs/index.md`). A hook in `docs/_hooks/site.py`
  removes the test modes from python code fences and turns links to files
  outside `docs/` into GitHub links; a link to a missing file fails the build.
  The Docs workflow builds the site and the `treams-core` rustdoc (under
  `/rust/`) on every pull request and push to main, and publishes them to
  GitHub Pages only when the repository variable `DOCS_DEPLOY` is `true`.
- `tests/api/test_docs.py` also runs indented python fences (fences inside
  tabs), reads the fence modes from the site hook, rejects snippet includes in
  `exec` fences, and checks that every relative link in the repository's
  Markdown files and every link to a site page resolves.
- The documentation is reorganized into the site tree. Each page has a
  one-sentence `description` in its front matter, and the `mkdocs.yml` nav is
  the only list of pages: `llms.txt` lists the nav pages with their
  descriptions, and `just docs-check` fails when a page is missing from the nav
  or has no description. The Python API reference moves from `docs/api.md` to
  `docs/reference/python/index.md`. Moved pages (sections of a page can move
  to other pages of the new tree):

  | Old page | New page |
  | --- | --- |
  | `docs/user-guide.md` | `docs/guide/particles-and-waves.md`, with sections in `guide/index.md`, `clusters.md`, `periodic.md`, `planar.md`, `numerical-namespaces.md` and `io.md` |
  | `docs/agents.md` | `docs/guide/api-discovery.md` |
  | `docs/iterative.md` | `docs/guide/large-clusters.md` |
  | `docs/large-problems.md` | `docs/performance/large-problems.md`; usage in `docs/guide/large-clusters.md` |
  | `docs/api-physics-map.md` | `docs/coming-from-treams/index.md` |
  | `docs/upstream-findings.md` | `docs/coming-from-treams/differences.md` |
  | `docs/adapters.md` | `docs/differentiation/frameworks.md` |
  | `docs/api-autodiff-map.md` | `docs/differentiation/custom-records.md` and `docs/differentiation/frameworks.md` |
  | `docs/testing.md` | `docs/differentiation/gradient-checks.md` |
  | `docs/architecture.md` | `docs/design/index.md`, with sections in `design/pullbacks.md`, `design/adapters.md` and `differentiation/index.md` |
  | `docs/status.md` | `docs/validation/capabilities.md`, with sections in `validation/index.md`, `validation/numerical-limits.md`, `design/python-api.md`, `design/floating-point.md`, `differentiation/index.md` and `performance/index.md` |
  | `docs/upstream.md` | `docs/validation/capabilities.md#treams-inventory` |
  | `docs/test-strategy.md` | `docs/development/testing.md`; the kinds of evidence in `docs/validation/index.md` |
  | `docs/development.md` | `docs/development/index.md`, with sections in `development/architecture.md`, `documentation.md` and `benchmarks.md` |
  | `docs/benchmarks.md` | `docs/performance/evidence.md`; the summary in `docs/performance/index.md`, LU scheduling in `docs/design/numerics.md` |
  | `docs/benchmark-comparison.md` | `docs/performance/platform-comparison.md` |
  | `docs/paper-qualification.md` | `docs/validation/published-applications.md` |
  | `docs/api.md` | `docs/reference/python/index.md` |

  New pages: install and quickstart, the treams name map and conventions, formal
  proofs, releasing, the reference index, the Rust crate, the glossary and the
  changelog.
- The validation pages state what treams-rs covers and how accurate it is, with
  no dates or test counts. `validation/numerical-limits.md` splits the lattice-sum
  limits into subsections, with tables of the four failure messages and of the
  cases less accurate than treams. `validation/index.md` adds an accuracy-evidence
  table that links each kind of evidence to its tests, references and scripts.
  The pre-release timings of explicit small Ewald splits move to
  `performance/evidence.md`.
- `performance/index.md` opens with one table of headline results (cases,
  median speedup, median memory ratio and exceptions per set of measurements),
  followed by methodology, caveats and how to rerun the benchmarks.
  `performance/evidence.md` starts with an evidence provenance table that gives
  the date, source commit, host, cases, outcome and summary file of every set of
  measurements, and keeps every per-kernel table under "Archived per-kernel
  measurements". `RESUME.md` in `benchmarks/` is renamed
  `linux-core-qualification.md`, after the two records it summarizes.
  The large-problems reproduction commands write to `benchmarks/results/local/`.
- An examples gallery ports five treams examples: single sphere, cluster, chain,
  grid and photonic crystal. Each page shows the treams-rs script in
  `docs/examples/`, the treams 0.4.5 code with the same sizes and the printed
  results. `tests/api/test_examples.py` runs every script, compares its output
  with `docs/examples/output/` and, when treams is installed, compares the
  results with the treams code. `just docs-examples` rewrites the recorded
  output.
- The examples gallery adds three planar and periodic treams examples: a chiral
  slab, an array of spheres on a slab and the band structure of a periodic
  stack. The test compares the band wavenumbers with treams after sorting them,
  because the two codes return eigenvalues in different orders.
- The examples gallery adds six cylindrical treams examples: a single
  cylinder, a chain of spheres in cylindrical waves, a chain next to a
  cylinder, a cylinder crystal, a grating and an array of spheres built from
  cylindrical waves. The grating and the array agree with treams to 3e-8 and
  2e-8: treams sums a lattice of cylinders at an automatic Ewald split that
  leaves errors of up to 8e-9.
- The upstream differences page explains that `operators.expand` from
  cylindrical to spherical waves, and `operators.expandlattice` from spheres to
  cylinders, add the terms of all origin pairs, while treams pairs each sphere
  only with the axis of the same particle index. A tested example reproduces
  treams with a particle-index mask.
- The documentation guide describes the examples gallery: its files, its tests
  and the steps to add an example.
- The upstream differences page lists the accuracy of Legendre functions of
  fractional degree: treams uses SciPy's `lpmv`, which is off by up to 2.9e-7
  relative for abs(m) >= 7, so `special.lpmv` can differ from treams by about
  3e-7 there.
- The design pages give the reasons behind treams-rs: its goals and layers, the
  Rust module to Python namespace to treams crosswalk (generated from the
  treams-core crate docs), the explicit-object Python API and its naming,
  analytic pullbacks with one-use contexts, the shared framework adapters, the
  floating-point guard and the numerical choices of the Rust core (equilibrated
  LU, LU worker counts, Wigner 3j symbols, Ewald sums, complex branches, the EBCM
  surface element and parallel thresholds).
- The examples gallery adds two gradient examples that treams cannot run,
  listed under "Beyond treams": five gradient-ascent steps on the radius of a
  sphere with Advect, and `jax.grad` of the transmission of a sphere array on a
  slab with respect to the sphere radius and the slab thickness. Both compare
  their gradients with central differences.
- The Python reference lists module constants with their values and
  docstrings, for example `iterative.DEFAULT_RTOL = 1e-10`.
- The "Coming from treams" section shows seven treams workflows next to their
  treams-rs code, which runs as a test and checks the values treams prints. It
  adds a concept map, a list of pitfalls, a keyword table in the name map, a
  conventions page (units, time convention, pol indices, mode ordering, planar
  sides, field normalization, branches, label bounds and the treams commit
  treams-rs follows) and a list of deliberate differences from treams 0.4.5.
  The glossary gains the native names and the definition of records and
  pullbacks.
- The `treams_rs.diff` reference opens with the definition of records,
  contexts, pullbacks and cotangents, the pullback variants, the object
  records and how the signatures of the `advect` functions differ from the
  records. Every
  record lists its value shape and dtype under Returns, the inputs with a
  gradient in pullback order under Dynamic inputs, and the fixed labels and
  options under Static configuration.
- The physics classes name their treams counterparts, the planar
  constructors list their arguments with units, shapes and the
  negative-to-positive order of media, and the result tuples state their
  units. The package quickstart defines medium, the polarization convention
  against the pol state, and kind, and links the glossary.
- The reference documents `io`, `ebcm.qmat`, `iterative`, the operator
  functions and classes, `PhysicsArray`, the four bases, `Lattice`,
  `WaveVector` and `Material`: their arguments, their treams counterparts and
  their differences from treams. `ebcm.qmat` shows a sphere that recovers the
  Mie T-matrix, and the `operators` module shows how an operator reads the
  basis from a `PhysicsArray`.
- The differentiation section states the rules of records once, on its first
  page: the definition from the glossary, gradient order, the complex pairing,
  fixed inputs, one-use contexts and ownership. The framework adapters page
  compares Advect, JAX and PyTorch in one table and lists known framework
  issues, verified with advect 0.2.0. The custom records page explains `wrap`
  and maps every `diff` record to its Advect function, JAX and PyTorch helper
  and physics object. All examples run as tests.
- The reference documents every Python function of `special`, `coeffs`,
  `misc`, `sw`, `cw` and `pw`: its definition, its treams counterpart, its
  arguments with their destination and source roles, its return dtype and
  shape, and its differences from treams. Each of these modules lists the
  label conventions and the differences from treams (ValueError instead of
  NaN, complex128 results, one degree per `coeffs.mie` call, duplicate modes
  dropped by `translate_periodic`, no `config.POLTYPE`) and runs one example.
- The README has badges, a pip install from GitHub, the relationship to
  treams, a cluster example and an Advect gradient example, links to the site,
  and Citing and License sections; every link is absolute, so the page also
  reads correctly on PyPI. The home page, the install page (Python 3.12 and
  3.13, rustup, extras, `RAYON_NUM_THREADS`) and a quickstart with four checked
  examples (a sphere, a cluster, a periodic array and a slab stack) open the
  site. The user guide pages hold 28 runnable examples, one per numerical
  namespace among them, and the large-clusters page shows
  `iterative.SphereCluster.record`, `IterativeContext` and the fields of
  `Gradient`.
- The advect, jax and torch module docstrings open with the device, the
  derivative order, how long the stored forward data lives and the dtype
  rules, followed by a tested quickstart. The framework TMatrix, SMatrix,
  Wave, PlaneWave, PortWave, Cluster and PeriodicResponse say which
  constructor builds them, and `testing.check_pullback` and `check_gradient`
  document their arguments. The `_native` stub follows the binding files and
  names, for every context, the function that creates it and its pullback
  tuple.
- The formal proofs page lists each Lean-modeled Rust item with its file
  under `crates/treams-core/src/`, its Lean file and its main theorems, the
  three golden-file tests, the reason the diffraction-order code clamps its
  square-root argument at zero, and what the proofs do not cover.
  `formal/README.md` holds the requirements and `just formal`, and links to the
  page. The Lean doc comments name the Rust items by their module paths
  (`lattice::shells::Shells::sum`, `sw::coupling::tl_vsw_term`,
  `special::wigner3j`, `cluster::IlluminateResidual::pullback`, ...). The
  Formal workflow also runs when `special/wigner.rs`, `special/harmonics.rs` or
  the lattice shell files change, and `tests/scripts/test_repository.py` keeps
  its paths equal to the page's table and checks that the repository's YAML
  files parse.
- The API reference shows the call of a ufunc with the treams argument names, `jv(v, z, /, out=None, *, where=True)`, where it showed NumPy's `jv(x1, x2, /, out=None, *, where=True, casting='same_kind', ...)`.
- The development pages hold the contributor guidance.
  [Source ownership](https://yaugenst.github.io/treams-rs/development/architecture/)
  maps each Rust module, binding file and Python module to its tests and a
  focused check, lists the rules that hold across them, and shows how to add a
  binding or a physics feature.
  [Testing](https://yaugenst.github.io/treams-rs/development/testing/) says
  where a Rust or Python test goes and lists the seven test domains, the
  markers with the descriptions that `tests/conftest.py` registers, the
  Hypothesis profiles, the shared helpers and the steps to add a test. The
  [documentation](https://yaugenst.github.io/treams-rs/development/documentation/),
  [benchmarks](https://yaugenst.github.io/treams-rs/development/benchmarks/)
  and [releasing](https://yaugenst.github.io/treams-rs/development/releasing/)
  pages cover the site, the gallery, comparing two builds and the release
  steps. `CONTRIBUTING.md` links these pages, and the `AGENTS.md` files point
  to them with a short list of rules.
- The [documentation site](https://yaugenst.github.io/treams-rs/) has ten
  sections: Getting started (install and four examples), Coming from treams
  (workflows, the name map, conventions and the differences from treams 0.4.5),
  the User guide, an Examples gallery of the treams examples plus two gradient
  examples, Differentiation, Design (the reasons behind the layers, pullbacks,
  adapters, floating-point handling, numerics and the Lean proofs), Validation,
  Performance, the Python, Rust and glossary Reference, and Development
  (building, tests, the site and releases).
- `tests/api/test_docs.py` checks that the `docs/` paths and site links in
  comments and docstrings exist, that page descriptions have at most 140
  characters, and that the definition of records and gradients
  appears word for word in the glossary, the differentiation page, the `diff`
  docstring and both crate docs. Every `treams-core` module doc opens with a
  summary and names its treams namespace or says it is a treams-rs extension.
- The glossary lists the current native names (`singular`, `function`,
  `direction`, `smatrix_tr`, `sphere_cluster`, `lattice_expansion`, the
  `*_record` functions, ...), and the name map notes members that keep their
  treams name with "treams-rs keeps the treams name".

### Tooling
- tests/conftest.py registers the pytest category markers from one `CATEGORIES`
  table of names and descriptions; pyproject.toml no longer lists them.
- The default Hypothesis profile of the test suite is renamed `treams` → `dev`
  (`HYPOTHESIS_PROFILE=dev`); the `ci` and `thorough` profiles are unchanged.

- `crates/treams-core/examples/benchmark_lu.rs` → `cargo bench -p treams-core --bench lu_scheduling`
- The API usability evaluation scripts (`scripts/agent_eval/`) take the agent
  CLIs, the Codex state directory, extra sandbox paths and extra context markers
  from `TREAMS_EVAL_CODEX`, `TREAMS_EVAL_CLAUDE`, `CODEX_HOME`,
  `TREAMS_EVAL_READ_PATHS`, `TREAMS_EVAL_FORBIDDEN` and
  `TREAMS_EVAL_CONTEXT_MARKERS` instead of fixed home-directory paths and
  markers, and the sandbox preflight writes its report to
  `benchmarks/results/local/`.
- Added `scripts/README.md` and `benchmarks/README.md`: indexes of every script
  (purpose, invoker, covering test, rules for the evidence-bound scripts) and of
  the benchmark plans, manifests, summaries and immutable raw evidence.
- Tests named as regressions of upstream differences, or after the controller
  of a benchmark run, say what they check:
  `test_upstream_component_selection_alignment_regression` →
  `test_component_selection_keeps_alignment_unlike_upstream`,
  `test_upstream_interval_regression` →
  `test_chirality_interval_average_is_invariant_unlike_upstream` and
  `test_broad_plan_reconstruction_matches_the_campaign_controller` →
  `test_broad_plan_reconstruction_matches_the_benchmark_runner`.
- Test modules follow one import-alias convention (`tr` for treams_rs, `ad` and
  `tj` for its Advect and JAX adapters, upstream as `treams` or
  `upstream_<ns>`, SciPy as `scipy_special`), import public names from
  `treams_rs`, and open with their scope, oracle and layer. The JAX subnormal
  callback test is marked `interface` instead of `physics`.
- Python test modules move from the flat `tests/` directory into domain
  directories `tests/<domain>/` with unchanged module names: `special`,
  `linalg`, `lattice`, `waves`, `plane`, `smatrix` and `tmatrix`, named like the
  Rust property domains `properties/<domain>.rs`, plus `api`, `bindings`,
  `autodiff` and `scripts` (tests of the repository scripts). conftest.py,
  _support.py, _scripts.py and test_suite_rules.py stay in `tests/`.
- Python test modules are named after what they cover:
  `special/test_integral_adjoints.py` → `test_ewald_integrals.py`,
  `waves/test_polar_translation.py` → `test_translation_coefficients.py`,
  `waves/test_vectorwaves.py` → `test_vector_waves.py`,
  `waves/test_cylwaves.py` → `test_cylindrical_waves.py`,
  `plane/test_channels.py` → `test_diffraction_channels.py`,
  `smatrix/test_internal_bands.py` → `test_layer_fields_and_bands.py`,
  `tmatrix/test_api.py` → `test_sphere.py`,
  `tmatrix/test_iterative_native.py` → `test_iterative.py`,
  `api/test_operator_objects.py` → `test_operators.py`,
  `api/test_matrix_conveniences.py` → `test_matrix_methods.py` and
  `api/test_support.py` → `test_support_catalog.py`. `waves/test_invariants.py`
  is split into `waves/test_translation.py`, `tmatrix/test_particle_cluster.py`
  and `tmatrix/test_sphere.py`; `lattice/test_geometry_misc.py` into
  `lattice/test_lattice_geometry.py` and `api/test_misc.py`;
  `waves/test_config.py` merges into `waves/test_polarization.py` and
  `tmatrix/test_conditioning.py` into `tmatrix/test_particle_cluster.py`; the
  native incomplete gamma and Kambe references move from
  `lattice/test_lattice.py` to `special/test_ewald_integrals.py`. Test names,
  markers and checks are unchanged.
- The pytest category markers have plain names: `ad_contract` → `gradients`,
  `python_contract` → `interface`, `public_e2e` → `workflows` and
  `oracle_numerical` → `reference` (`physics` is unchanged);
  `tests/test_suite_contract.py` → `tests/test_suite_rules.py`. Select a
  category with, for example, `pytest -m gradients`.
- Added `just docs-build` (strict site build into `site/`), `just docs-serve`
  (live preview) and `just docs-rust` (`treams-core` rustdoc under
  `site/rust/`). The site recipes run the `docs` dependency group (mkdocs,
  mkdocs-material, pymdown-extensions) in an isolated environment; the `dev`
  group lists `pyyaml`.
- `just verify` is removed; use `just ci` for the complete check, the same as
  hosted CI, or `just check` for the fast lint checks. `just ci-python` drops
  its separate `docs-check`: its Python tests run the same check.
- `scripts/generate_lattice_references.py` → `scripts/generate_references.py lattice-chain`.
  The script has one subcommand per reference table in
  `crates/treams-core/references/` (`incgamma`, `kambe`, `kambe-lattice`,
  `lattice-sums`, `lattice-chain`); `--check` recomputes the committed rows and
  reports the worst relative error. `crates/treams-core/references/README.md`
  lists each table's quantity, precision, generator and test.
- `scripts/generate_agent_docs.py` → `scripts/generate_docs.py`. It writes one
  reference page per public module under `docs/reference/python/` (plus an
  index and a page of returned types), fills the generated tables of
  hand-written pages (the treams name map on
  [Name map](https://yaugenst.github.io/treams-rs/coming-from-treams/names/)),
  and writes `llms.txt`; `--check` compares all of them.
- `python -m treams_rs --format markdown` renders docstring sections (`Args:`,
  `Returns:`, `Raises:`, ...) as Markdown lists, `::` examples as Python code
  and `>>>` examples as console code, and shows signatures as Python code
  instead of plain text.
- The fractional Legendre property test compares `special.lpmv` with mpmath at
  30 digits instead of treams, whose SciPy values fail the 5e-10 tolerance at
  about 0.7% of random runs. A looser check against treams (1e-6) remains, and
  m=11, degree 20.0625, x=-0.875 is a fixed example.
- The lattice property tests use more accurate references. The complete
  derivative of an Ewald sum is compared with the Richardson extrapolation
  `(4 D(h) - D(2h)) / 3` of central differences, with smaller steps near an
  image, and a sum at the origin keeps its shift fixed. A sum moved up to 1e-20
  off its lattice plane is compared with the sum on it plus its first-order
  change, whose slope is also a Richardson extrapolation. At an explicit split
  that tolerance adds 64 units in the last place of the real and reciprocal
  parts, which can cancel by 2.7e3. Three recorded seeds, three cylindrical sums
  near a lattice point and two sums 1e-20 off their plane are fixed examples.
- The lattice direct-sum property test allows for the rounding of each image's
  unit vector: 4 units in the last place of the sum over the images of their
  distance from the shift times the modulus of their gradient. Next to a zero of
  its angular factor an image is far more sensitive than its size shows: for a
  degree-9 sum 0.16 from an image, one unit in the last place of the image's
  `cos theta` moves it by 1.4e-11, and the Ewald sum comes back 4.4e-12 off
  mpmath. That sum is a fixed example.
- At an explicit split, the lattice property test that moves a sum off its
  plane counts 64 units in the last place of the largest component of each
  Ewald part instead of the parts of the component it compares. The terms of
  one component can cancel within a part: those of `dS/da_0x` of a degree-9 sum
  add up to 9e3 in modulus in mpmath for a reciprocal part of 66, and the paths
  on and off the plane come back 5.6e-11 apart. That sum is a fixed example.
- `scripts/qualify_references.py` writes each error as the float of the
  high-precision string beside it, so `float()`, `json` and NumPy read the string
  as exactly the JSON number. Rounding the exact error directly can give the
  neighbouring float: an error of about 1e-145 that lies halfway between two
  floats came out as 1.0000000000000001e-145, while its string reads 1e-145.
  That error is a fixed example of the error-summary test.
- Dependabot opens one grouped pull request per week each for the Cargo
  dependencies, the Python dependencies in `uv.lock` and the GitHub Actions.

### Fixed
- Ewald sums reject gain wavenumbers (`Im k < 0`) and, for spherical waves,
  negative real wavenumbers (`Re k < 0`); finite direct shells retain their
  full complex-wavenumber domain.
- Spherical field values and analytic gradients use bounded internal length
  units, preventing intermediate solid-harmonic overflow when a finite field is
  expressed in very small or large length units.
- `PeriodicResponse.to_cylindrical` and `PeriodicWave.in_basis` gave wrong
  results for a cylindrical basis with several axes, because they counted
  every sphere once per axis. In the treams chain example with two spheres
  per cell, E_z at (200, 0, 50) nm came out as -2.21 - 0.20i instead of
  -0.66 + 0.56i. Each axis pairs only with its own particle, as in treams, and
  a basis with several axes needs one axis per particle position. Results for
  one axis are unchanged.
- Lattice sums whose split `eta` turns `k eta` far off the real axis failed
  with "Ewald sum did not converge within the shell limit", for example the
  derivatives of a degree-9 sum on a 2D lattice at a real split 3.7 times the
  automatic one with `k` 37 degrees off the real axis. Their reciprocal terms
  fall like `exp(-Re(1 / (k eta)^2) |q + G|^2 / 2)`: at 37 degrees 0.27 times as
  fast as the shell limit assumed. The limit reaches `sqrt(100 / Re(1 / (k eta)^2))`
  where that exceeds `|k| sqrt(1 + 100 |eta|^2)`. It only decides when a sum
  gives up, so every sum that converged stops at the same shell with the same
  value. 1D sums of spherical waves keep the old limit: where it fails they take
  their spectral series, which is more accurate there.
