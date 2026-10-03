# Changelog

All notable changes to treams-rs are documented here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). treams-rs has no
release yet, so "Unreleased" lists changes against the pre-release tree.

Entries use absolute links only, because this file is also rendered on the
documentation site. Every renamed public name is written as `old` → `new`, and
names that differ from treams also appear in the
[treams name map](https://yaugenst.github.io/treams-rs/coming-from-treams/names/).

## [Unreleased]

### Python API
- The private record module is renamed `treams_rs._adapters` →
  `treams_rs._records`, with `execute` → `run_record`, `gradients` →
  `apply_pullback` and `require_dtype` → `require_float64`; each documents the
  rule it enforces.
- `treams_rs.testing` no longer exposes the helpers it imports (`execute`,
  `input_array`, `pullback_gradients`); its public names are `check_pullback`
  and `check_gradient`.
- `treams_rs.torch` rejects a tensor dtype other than float64/complex128,
  including dtypes NumPy lacks such as bfloat16, with the `TypeError` message
  of the JAX adapter and `treams_rs.testing`, which names the received dtype and
  the remedy (an explicit cast).
- `treams_rs.advect` no longer exposes the internal class names
  `SphericalWaveBasis` and `CylindricalWaveBasis`; its signatures name the same
  classes by their exports `SphericalBasis` and `CylindricalBasis`.
- The `treams_rs.advect`, `treams_rs.jax` and `treams_rs.torch` module
  docstrings share one structure: rules, residual lifetime, dtypes and a link
  to the framework adapters guide. `treams_rs.advect` documents its dtype policy:
  it deliberately accepts float32/complex64 primals and returns their gradients
  in that dtype.
- The package root and the `special`, `sw`, `cw`, `pw`, `lattice`, `coeffs`,
  `misc`, `ebcm`, `io` and `iterative` namespaces declare a literal `__all__`
  and keep their imports private. The upstream mirrors export exactly the treams
  names plus `coeffs.*_with_context`, `lattice.expansion`,
  `lattice.expansion_with_context` and the documented type aliases
  `lattice.SumResult`, `ebcm.Modes` and `io.MatrixSet`. Imported names are no
  longer reachable: `treams_rs.import_module`, `treams_rs.ModuleType`,
  `sw.SphericalWaveBasis`, `cw.CylindricalWaveBasis`, `lattice.Lattice`,
  `lattice.WaveVector`, `lattice.SphericalWaveBasis`,
  `lattice.CylindricalWaveBasis`, `ebcm.SphericalWaveBasis`, `ebcm.lru_cache`,
  `io.Material`, `io.SphericalWaveBasis`, `io.TMatrix`, `io.Mapping`,
  `io.Sequence`, `io.Path`, `io.uuid4`, `io.version`, `iterative.Material`,
  `iterative.SphericalWaveBasis` and `iterative.MultipoleWave`. Use the root
  exports instead (`treams_rs.Lattice`, `treams_rs.WaveVector`,
  `treams_rs.SphericalBasis`, `treams_rs.CylindricalBasis`,
  `treams_rs.Material`, `treams_rs.TMatrix`, `treams_rs.Wave`).
- The `treams-rs` distribution metadata lists its authors, keywords,
  classifiers and project URLs: documentation site, repository, issues and
  changelog.
- `treams_rs.support` is private (`treams_rs.support` → `treams_rs._catalog`);
  use `treams_rs.support_catalog()`. `python -m treams_rs` is unchanged.
- `support_catalog()["adapters"][name]["contract"]` is the opening paragraph
  of the `treams_rs.advect`, `treams_rs.jax` or `treams_rs.torch` module
  docstring instead of a separately maintained summary.
- `python -m treams_rs --search` separates alternatives with a plain `|`
  (`--search 'a\|b'` → `--search 'a|b'`), as its help text states.
- The API reference and `support_catalog()` list the framework classes of
  `treams_rs.advect`, `treams_rs.jax` and `treams_rs.torch` (`Wave`,
  `PortWave`, `TMatrix`, `Cluster`, `PeriodicResponse`, `SMatrix`) without
  `__init__`, so the internal `Backend` argument is no longer documented as
  API.
- `treams_rs.config` removed: treams-rs has no global polarization setting;
  pass `poltype`/`polarization` explicitly (default helicity).
- `ChangePoltype.inv` and `ChangePoltype.get_kwargs` raise `ValueError` for an
  unknown polarization convention instead of falling back to helicity, like
  the forward `ChangePoltype` already does.
- `treams_rs.advect`, `treams_rs.jax` and `treams_rs.torch` return the result
  classes of the package root: `tr.advect.CrossSections is tr.CrossSections`, and likewise
  for `PowerBalance`, `BandModes` and `ScatteredPorts`, so `isinstance` checks
  work across NumPy and framework results. The field types are type
  parameters, for example `CrossSections[float]`.
- Classes carry their export names: `SphericalBasis`, `CylindricalBasis`,
  `PlaneWaveBasis`, `PlaneWavePorts`, `Wave` and `CylindricalTMatrix` were
  `SphericalWaveBasis`, `CylindricalWaveBasis`, `PlaneWaveBasisByUnitVector`,
  `PlaneWaveBasisByComp`, `MultipoleWave` and `TMatrixC` in reprs, `type(x).__name__`,
  error messages and signatures. Pickles that store the old class names do not load.
- treams top-level names that treams-rs spells differently raise an
  `AttributeError` naming the replacement, for example
  "treams_rs has no 'TMatrixC': treams.TMatrixC corresponds to
  treams_rs.CylindricalTMatrix." This covers the bases, `TMatrixC`, `SMatrices`,
  `PhysicsArray`, the operators (`efield`, `expand`, `translate`, ...), `config`
  and `util`.
- A missing optional dependency raises "treams_rs.jax needs jax; install it
  with pip install 'treams-rs[jax]'".
- `support_catalog()` has an `upstream` entry with the treams name map:
  `names` for the treams top level and `members` for members of treams objects.
- Catalog and reference signatures name concrete types: TMatrix lists
  `in_basis(basis: SphericalBasis) -> TMatrix` and CylindricalTMatrix lists
  `CylindricalBasis`, instead of the type parameters `B`, `M` and `Self` of
  shared base classes. `returned_python_types` lists the solver objects of
  `TMatrix.interaction` and `TMatrix.latticeinteraction` without `__init__`.
- `support_catalog()` names its entries plainly: `"contract"` → `"rules"` (how
  native calls run: precision, gradient pairing, derivative order) and
  `adapters[name]["contract"]` → `adapters[name]["summary"]`. The reference
  heading `Execution contracts` becomes "How native calls run".
- Functions that return a value and a context live in `diff` only:
  `coeffs.fresnel_with_context`, `coeffs.mie_with_context` and
  `coeffs.mie_cyl_with_context` → `diff.fresnel`, `diff.mie` and `diff.mie_cyl`.
  `diff.mie` takes `degree` and `diff.mie_cyl` takes `order` where the `coeffs`
  functions take `l` and `m`.
- `lattice.expansion_with_context(destination, source, ks, a, kpar)` →
  `diff.lattice_expansion(destination, source, ks, kpar, a)`. The arguments
  follow the order of the pullback outputs: (destination positions, source
  positions, ks, kpar, a). `lattice.expansion` is removed; use
  `diff.lattice_expansion(...)[0]`, `sw.translate_periodic` or
  `cw.translate_periodic`.
- `diff.interface` → `diff.interface_coefficients` and `diff.propagation` →
  `diff.propagation_matrix`, the names that `advect` already uses.
- `diff` lists its functions in `__all__`.
- `TMatrix.latticeinteraction` and `CylindricalTMatrix.latticeinteraction` name
  their lattice-vector argument `lattice` (was `a`), as in treams.
- The dense solver for homogeneous spheres in vacuum is named after what it
  solves: `diff.cluster` → `diff.sphere_cluster`, `diff.cluster_factor` →
  `diff.sphere_cluster_factor` and `advect.cluster` → `advect.sphere_cluster`.
  `Cluster` and `diff.particle_cluster` keep their names.
- `diff.periodic_from_table` → `diff.lattice_expansion_from_table` and
  `advect.periodic_from_table` → `advect.lattice_expansion_from_table`: they
  compute `diff.lattice_expansion` from a table of lattice sums that you supply.
- `diff.periodic_conversion` → `diff.periodic_to_cw` and
  `advect.periodic_conversion` → `advect.periodic_to_cw`, after
  `sw.periodic_to_cw`, whose coefficients they assemble into a matrix.
- `diff.wigner` → `diff.wignerd` and `advect.wigner` → `advect.wignerd`, after
  `special.wignerd`, the Wigner D-matrix element they compute.
- `diff.bessel`, `diff.angular`, `advect.bessel`, `advect.angular`,
  `jax.bessel` and `torch.bessel` select the special function with
  `function=` (was `kind=`), with the same values: `'j'`, `'y'`, `'h1'`,
  `'h2'` for Bessel functions and `'legendre'`, `'pi'`, `'tau'` for angular
  functions. `kind` names only the radial kind of a wave or the kind of a port.
- `diff.vector_wave` and `advect.vector_wave` select the vector wave with
  `function=` (was `kind=`), with the same values, for example `'vsw_rA'`,
  `'sph_harm'` or `'vpw_A'`.

- The singular radial kind is `kind='singular'` everywhere: on `Wave`,
  `Wave.in_basis`, `spherical_wave`, `cylindrical_wave`, `PeriodicWave.in_basis`
  and the framework `wave(kind=...)`. `Wave.kind` returns `'singular'` (was
  `'outgoing'`), and `kind='outgoing'` raises a `ValueError` that names
  `'singular'`. The framework `Wave` takes and stores `singular=` (was
  `outgoing=`), and its `in_basis` takes `singular=`.
- The plane-wave state is `pol`: `plane_wave(direction=..., pol=...)` takes
  `'positive_helicity'`, `'negative_helicity'`, an index 0 or 1, two amplitudes
  or three electric components (was `polarization=`). `polarization=` names
  the convention, `'helicity'` or `'parity'`, as on every other physics object.
  `plane_wave_angle` takes the same keywords, and the framework `PlaneWave` and
  `plane_wave` take `(direction, pol, *, k0, medium=1.0)`.
- `Wave`, `PlaneWave`, `plane_wave`, `plane_wave_angle`, `spherical_wave` and
  `cylindrical_wave` require the keyword `k0` (was `k0=1.0`), like every
  T-matrix and S-matrix constructor.
- Giving a physics keyword together with its treams name raises a `ValueError`,
  for example `medium=` with `material=`, `kind=` with `modetype=`,
  `polarization=` with `poltype=` or `direction=` with `kvec=` (was: the
  second one won).
- `PlaneWave.amplitudes` is private; use `PlaneWave.coefficients`.
- `CylindricalTMatrix.cross_sections(incident, *, flux=0.5)` returns the cross
  widths, so code that asks any T-matrix for `cross_sections` works for both
  wave families; `cross_widths` stays.
- `TMatrix`, `CylindricalTMatrix`, `SMatrix`, `Wave` and `PlaneWave` store
  their physics names (`medium`, `polarization`, `kind`) and list every treams
  name (`material`, `poltype`, `modetype`, `changepoltype`, `expand`, `xs`,
  `cd`, `add`, `illuminate`, `tr`, ...) last in the class, as a one-line
  delegation. `SMatrix.material` returns `(positive_medium, negative_medium)`.
  Looking up a treams member that treams-rs lacks, on the class or on an
  object, raises an `AttributeError` that names the replacement, for example
  `TMatrix.cluster` → `Cluster` and `SMatrix.from_array` →
  `solve_periodic(...).to_smatrix(ports)`.
- A wave, particle or layer with a different k0, medium or polarization
  convention raises one message: "illumination and T-matrix must have matching
  k0, medium and polarization convention", and likewise for cluster particles,
  cascaded S-matrices and the illuminated side of an S-matrix. Cascading
  S-matrices with different ports raises "cascaded S-matrices must have the
  same ports".
- `Cluster` keeps the particle blocks itself, so every `TMatrix` behaves the
  same way. The private `TMatrix._assemble` is removed: `Cluster(particles,
  positions=...).solve()` runs the same block solve, and `TMatrix.interaction`
  always solves the dense matrix it belongs to.
- The framework `Wave` and `TMatrix` classes of `treams_rs.advect`, `.jax` and
  `.torch` take no `kz=` argument and have no `kz` attribute; `basis.kz` holds
  the axial wavenumbers.
- Framework `material=`, `medium=`, `negative_medium=` and `positive_medium=`
  accept a root `treams_rs.Material` and an `(epsilon, mu, kappa)` tuple, as
  well as the namespace's own `Material`; any other value is epsilon (was:
  both were taken as epsilon).
- `treams_rs.advect` expert functions name the expansion centres `positions`
  (was `origins`) in `field`, `field_operator`, `hfield`, `gfield`, `ffield`,
  `plane_expansion` and `cylindrical_channels`, and
  `destination_positions`/`source_positions` (was
  `destination_origins`/`source_origins`) in `periodic_to_cw`.
- `treams_rs.advect.field`, `field_operator`, `hfield`, `gfield`, `ffield` and
  `periodic_to_cw` take `kz=`, one axial wavenumber per mode like `basis.kz`
  (was `kzs=`). `advect.expansion`, `lattice_expansion` and `cylinder` keep
  `kzs`, the sorted distinct values, like treams' `TMatrixC.cylinder(kzs)`.
- `TMatrix.translate(r)` and `CylindricalTMatrix.translate(r)` return a
  T-matrix of the same type, as `rotate` and `in_basis` do, instead of an
  ndarray. The values are unchanged: `translate(r) @ T @ translate(-r)` with
  `operators.translate`. Use `.array` for the plain matrix. `tm.translate`
  no longer has the operator methods `.eval`, `.eval_inv`, `.apply_left` and
  `.apply_right`, and takes no keyword overrides such as `k0=`. Use
  `operators.Translate(r)` for the operator alone: `operators.Translate(r) @ tm`
  gives the matrix product `translate(r) @ T`.
- `SMatrix.translate` and `permute` raise the constructor's error
  "S matrices require finite shape (2, 2, len(basis), len(basis))" when the
  result has infinite or NaN entries.
- `treams_rs.advect.PlaneWave`, `treams_rs.jax.PlaneWave` and
  `treams_rs.torch.PlaneWave` are one shared class, so `isinstance` checks hold
  across the three namespaces. Build plane waves with `plane_wave(direction, pol,
  k0=...)`; calling the class directly needs `backend=`.
- `treams_rs.advect` checks the gradients that a record's pullback returns: a
  wrong count or size raises `RuntimeError` with a message naming the problem,
  and an empty output tuple raises an error.
- Framework S-matrices and port waves (`treams_rs.advect`, `treams_rs.jax`,
  `treams_rs.torch`) carry their ports as one `ports` value. `basis`, `modes`
  and `transverse_wavevectors` stay as read-only properties; read `alignment` and
  `fixed_q` from `ports`.
- `treams_rs.advect.solve`, `interaction`, `illuminate`, `sphere` and `bessel`
  are the functions that `treams_rs.jax` and `treams_rs.torch` bind, with the
  same arguments and results. The reference lists them with the shared
  annotations and docstrings.
- The pullbacks of `diff.sphere_cluster` and
  `iterative.SphereCluster.solve_with_pullback` return gradients in the order of
  the arguments. The `diff.sphere_cluster` context returns
  `(k0, radii, epsilon, positions)` instead of `(radii, positions, epsilon, k0)`,
  and the fields of `iterative.Gradient` are
  `(k0, radii, epsilon, positions, incident, convergence)` instead of
  `(radii, positions, epsilon, k0, incident, convergence)`. Code that reads the
  fields by name keeps working.
- `sw.periodic_to_cw(kz, mu, pol, l, m, qol, k, a)` →
  `sw.periodic_to_cw(kz, m, pol, l, mu, qol, k, area)`, the argument names of
  treams: `m` is the order of the cylindrical mode, `mu` the order of the
  spherical mode and `area` the period along z. Positional calls are unchanged.
- `ebcm.qmat`, `diff.ebcm_qmat` and `advect.ebcm_qmat`: `legacy=True` →
  `radial_area_factor=False`. The default `radial_area_factor=True` keeps the
  factor r of the surface element; `False` reproduces `treams.ebcm.qmat`, which
  omits it.
- `diff.ebcm_qmat` and `advect.ebcm_qmat`: `out=`, `in_=` → `destination=`,
  `source=`. `ebcm.qmat` keeps the treams names `out` and `in_`.
- `iterative.Pullback` → `iterative.IterativeContext` and
  `iterative.SphereCluster.solve_with_pullback` → `iterative.SphereCluster.record`,
  so the matrix-free cluster records like every other solver object: `record`
  returns `(Solution, IterativeContext)`, and `IterativeContext.pullback`
  returns a `Gradient`.
- `iterative.DEFAULT_RTOL`, `DEFAULT_ATOL`, `DEFAULT_RESTART` and
  `DEFAULT_MAX_ITERATIONS` hold the GMRES defaults of `SphereCluster.solve`,
  `scatter` and `record` (1e-10, 0.0, 30 and 300, the defaults of the Rust
  solver).
- `coeffs.mie` with an array of degrees and `coeffs.mie_cyl` with an array of
  orders raise `ValueError: mie takes one integer degree; loop over degrees
  (treams broadcasts)` (and the `mie_cyl` equivalent) instead of a `TypeError`.
  treams broadcasts over these labels; loop over them instead.
- The constructors of `TMatrix`, `CylindricalTMatrix` and `PlaneWave` accept
  `medium=` and `polarization=`, and `SMatrix` accepts `positive_medium=`,
  `negative_medium=` and `polarization=`, like `Wave` and the factories. The
  treams keywords `material=` and `poltype=` keep working; `SMatrix(material=)`
  takes one medium or the pair (positive side, negative side). Giving a
  quantity under both names raises `ValueError`.
- `diff.tmatrix_metric(kind=)` → `metric=`, `diff.coordinates(kind=)` and
  `diff.vector_coordinates(kind=)` → `function=`, and
  `diff.vector_wave(polarization=)` → `pol=`, so `kind` names only the radial
  or port kind and `pol` the pol index. The old keywords keep working; giving
  both names raises `ValueError`.
- Error messages that change:
  - "selection must use the same wave family and origins" → "... wave family
    and positions".
  - "one finite nonnegative radius required per expansion origin" → "... per
    basis position".
  - "... or a single-origin response; ..." → "... or a response at one
    position; ...".
  - "polarization index must be 0 or 1" → "pol index must be 0 or 1".
  - "material requires the media above and below the S matrix" → "material
    requires two media: (positive side, negative side)".
  - "axial power forms currently require xy-aligned plane bases" → "axial
    power forms require xy-aligned plane bases".
  - "invalid polarization type for the medium" (`poynting_avg_z`) → "the
    parity convention requires an achiral medium".
- `Material.from_n` and `Material.from_nmp` raise an AttributeError that names
  `Material.from_refractive_index` and `Material.from_helicity_indices`, like
  the other treams members of the physics classes.
- Error messages: "pol labels must be 0 or 1" (was "polarizations must be 0
  or 1"), "plane components must be finite and pol 0 or 1" (was "... and
  polarization 0 or 1") and "T matrices must share mode labels, positions and
  polarization type" (was "... origins and ...").
- `advect.smatrix_cd` → `advect.smatrix_circular_dichroism`;
  `advect.tmatrix_metric(kind=)` → `metric=`, `advect.coordinates(kind=)` and
  `advect.vector_coordinates(kind=)` → `function=`,
  `advect.vector_wave(polarization=)` → `pol=`, and
  `PeriodicResponse.to_smatrix(orders=)` → `diffraction_orders=` in advect, jax
  and torch. The keywords match `treams_rs.diff`. The old names keep working;
  giving a keyword under both names raises `ValueError`.

### Native bindings

- Removed `plane_to_spherical` and `plane_to_cylindrical`, which nothing called.
- The binding sources mirror the core modules: shared helpers in `context.rs`, `broadcast.rs`, `convert.rs` and `args.rs`; `basis.rs` → `expansion.rs`, `rotation.rs`, `plane.rs` and `lattice.rs`; `cylinder.rs` → `coeffs.rs` and `tmatrix.rs`; `illumination.rs` and the cluster records of `tmatrix.rs` → `cluster.rs`; `polar.rs` → `translation.rs`; the incomplete gamma and Kambe records of `special.rs` → `integrals.rs`; the plane-wave records of `fields.rs` → `plane.rs`; the test hooks → `testing.rs`. `lib.rs` declares the modules and exports by core layer. Exported names are unchanged.
- `ufunc.rs` → the `ufunc/` directory: `ffi.rs` (the NumPy C interface and all raw-pointer operand access), `kinds.rs` (the loop kind constants, grouped by parameter), `loops.rs` (the inner loops), `registry.rs` (the ufunc tables) and `fast_paths.rs` (the Python-scalar fast paths, including `hankel_scalar`, `angular_value` and `wigner3j_scalar` from `special.rs` and the scalar `incgamma`, `intkambe` and `direct_cylindrical_*` from `lattice.rs`). Only `ffi.rs` and `loops.rs` allow unsafe code. Exported names are unchanged.
- `ClusterContext.pullback` returns `(k0, radii, epsilon, positions)` and `IterativeContext.pullback` returns `(k0, radii, epsilon, positions, incident, convergence)`, the order of the forward arguments, instead of `(radii, positions, epsilon, k0[, incident, convergence])`.
- `ebcm_qmat(..., singular, legacy)` → `ebcm_qmat(..., singular, radial_area_factor)` with the opposite meaning: `True` keeps the radial area factor r of the surface element, `False` omits it as `treams.ebcm.qmat` does.
- Each context class takes the name of the `treams_rs.diff` function that returns it, as `<DiffName>Context`; the return annotations of `diff` show these names. `CylinderContext` → `MieCylContext`, `CylinderMatrixContext` → `CylinderContext`, `MetricContext` → `TMatrixMetricContext`, `ArrayContext` → `SMatrixFromArrayContext`, `SMatrixContext` → `SMatrixAddContext`, `IlluminationContext` → `SMatrixIlluminateContext`, `TransmissionContext` → `SMatrixTrContext`, `BandContext` → `BandsContext`, `InterfaceContext` → `InterfaceCoefficientsContext`, `PropagationContext` → `PropagationMatrixContext`, `LayersContext` → `LayerStackContext`, `ChiralityContext` → `ChiralityDensityContext`, `ClusterContext` → `SphereClusterContext`, `InteractionIlluminationContext` → `IlluminateContext`, `PolarTranslationContext` → `SphericalTranslationContext`, `WaveContext` → `VectorWaveContext`, `PlanePhaseContext` → `PlanePhasesContext`, `PeriodicConversionContext` → `PeriodicToCwContext`, `PeriodicContext` → `LatticeExpansionContext`, `PeriodicTableContext` → `LatticeExpansionFromTableContext`, `GammaContext` → `IncgammaContext`, `KambeContext` → `IntkambeContext`, `CoordinateContext` → `CoordinatesContext`, `VectorCoordinateContext` → `VectorCoordinatesContext`, `WignerContext` → `WignerdContext`, `SingularContext` → `SvdvalsContext`, `EigenContext` → `EigContext`, `QContext` → `EbcmQmatContext`. The shared `ChannelsContext` splits into `SphericalChannelsContext` and `CylindricalChannelsContext`.
- `NativeSphereCluster` → `IterativeSphereCluster`, the name of the Rust type it wraps.
- Broadcast records take the name of their `treams_rs.diff` function and end in `_record`; their 0-d variants end in `_record_scalar`. A name ending in plain `_scalar` is always a fast path that records nothing. `bessel` → `bessel_record`, `bessel_scalar` → `bessel_record_scalar`, `angular` → `angular_record`, `angular_scalar` → `angular_record_scalar`, `wigner` → `wignerd_record`, `wigner_scalar` → `wignerd_record_scalar`, `gamma_record` → `incgamma_record`, `gamma_record_scalar` → `incgamma_record_scalar`, `kambe_record` → `intkambe_record`, `kambe_record_scalar` → `intkambe_record_scalar`, `lattice_record` → `lattice_sum_record`, `coordinates` → `coordinates_record`, `vector_coordinates` → `vector_coordinates_record`, `vector_wave` → `vector_wave_record`, `spherical_translation` → `spherical_translation_record`, `cylindrical_translation` → `cylindrical_translation_record`.
- The other records take the name of the `treams_rs.diff` function they serve; the cylindrical twin that a diff function calls for cylindrical bases starts with `cylindrical_`. `cyl_expansion` → `cylindrical_expansion`, `cyl_rotation` → `cylindrical_rotation`, `periodic_expansion` → `lattice_expansion`, `periodic_cyl_expansion` → `cylindrical_lattice_expansion`, `periodic_from_table` → `lattice_expansion_from_table`, `periodic_conversion` → `periodic_to_cw`, `linear_solve` → `solve`, `interact` → `interaction`, `cluster` → `sphere_cluster`, `cluster_factor` → `sphere_cluster_factor`, `interface` → `interface_coefficients`, `propagation` → `propagation_matrix`.
- The test hooks end in `_jet`, after the value-and-derivatives result they return: `translation` → `cartesian_translation_jet`, `spherical_wave` → `spherical_wave_jet`, `cyl_translation` → `cylindrical_cartesian_translation_jet`, `cylindrical` → `cylindrical_radial_jet`.
- The Python-scalar fast paths end in `_scalar` and take the name of the ufunc whose one-element call they replace: `incgamma` → `incgamma_scalar`, `intkambe` → `intkambe_scalar`, `legendre_real_value` → `lpmv_real_scalar`, `cylindrical_rotation_scalar` → `cw_rotate_scalar`, `cylindrical_translate_scalar` → `cw_translate_scalar`, `cylindrical_translation_scalar` → `tl_vcw_scalar`, `plane_permutation_scalar` → `pw_permute_xyz_scalar`, `direct_cylindrical_1d` → `dsumcw1d_scalar`, `direct_cylindrical_1d_shift` → `dsumcw1d_shift_scalar`, `direct_cylindrical_2d` → `dsumcw2d_scalar`. Two serve a family of ufuncs and take its name: `hankel_scalar` (`hankel1`, `hankel2`) keeps its name, and `angular_value` → `angular_scalar` (`lpmv`, `pi_fun`, `tau_fun`).
- The ufuncs take the upstream name, and those behind the `sw`, `cw` and `pw` functions start with the namespace and the function: `incgamma_ufunc` → `incgamma`, `intkambe_ufunc` → `intkambe`, `sw_to_pw_h`/`_p` → `sw_periodic_to_pw_h`/`_p`, `sw_to_cw_h`/`_p` → `sw_periodic_to_cw_h`/`_p`, `pw_permute_h`/`_p` → `pw_permute_xyz_h`/`_p`, `pw_inverse_h`/`_p` → `pw_permute_xyz_inverse_h`/`_p`, `cw_to_pw` → `cw_periodic_to_pw`.
- A ufunc or wrapper that a namespace exposes directly reports the public name as `__name__`, which NumPy error messages and reprs show: `cw.periodic_to_pw` is `'periodic_to_pw'` (was `'cw_to_pw'`), `pw.to_cw` is `'to_cw'` (was `'pw_to_cw'`), `misc.firstbrillouin1d` is `'firstbrillouin1d'` (was `'first_brillouin_1d'`), `pw.translate` is `'translate'` (was `'pw_translate'`), `lattice.volume` and `lattice.area` are `'volume'` (was `'cell_volume'`) and `lattice.reciprocal` is `'reciprocal'` (was `'cell_reciprocal'`). The `_native` attributes `pw_translate`, `cell_volume` and `cell_reciprocal` keep their names, and their `__module__` is the namespace that exposes them, so they pickle.
- Parameters take the names that the `treams_rs.diff` functions use: `outgoing` → `singular` (same meaning: `True` selects singular, outgoing Hankel waves) in `expansion`, `cylindrical_expansion`, `field`, `cylindrical_field`, `field_operator`, `cylindrical_field_operator` and the `*_jet` test hooks; `to` → `destination` and `to_positions` → `destination_positions` in `expansion`, `cylindrical_expansion`, `cw_to_sw`, `periodic_to_cw`, `rotation`, `cylindrical_rotation`, `lattice_expansion`, `cylindrical_lattice_expansion`, `lattice_expansion_from_table`, `ebcm_qmat` and the translation test hooks; `origins` → `positions` in the field and plane-expansion functions; `input_shapes` → `argument_shapes` in `vector_coordinates_record`.
- `fresnel(ks, kz, z)` → `fresnel(ks, kzs, zs)` and `interface_coefficients(ks, z, ...)` → `interface_coefficients(ks, zs, ...)`. `lattice_sum_record`, `lattice_expansion` and `cylindrical_lattice_expansion` take the Bloch vector as `kpar` (was `bloch`) and the lattice vectors as `a` (was `vectors`), and `lattice_sum_record` takes the shift as `r` (was `shift`), as treams' `lsum*` functions do. `smatrix_tr` and `smatrix_tr_value` take `direction` (was `transmission`): 0 is up, towards the positive side, and 1 is down.
- `kind` → `function` for the parameter that selects the special function of `bessel_record`, `bessel_record_scalar`, `angular_record`, `angular_record_scalar`, `angular_scalar` and `vector_wave_record`, as in `diff.bessel(..., function="h1")`.
- A function or method that records returns `(value, context)`, and its twin that computes the value only ends in `_value`: `IterativeSphereCluster.solve_with_pullback` → `IterativeSphereCluster.record`, as `InteractionFactor.record`; `smatrix_illuminate_forward` → `smatrix_illuminate_value`; `smatrix_transmittance(..., fixed_q, record=True)` → `smatrix_tr(..., fixed_q)` and `smatrix_transmittance(..., record=False)` → `smatrix_tr_value(...)`, which takes no `fixed_q`.
- Cotangent errors share one wording that names the expected shape: 'cotangent must be finite with shape (2, 2, 3, 3)'. It replaces about 25 messages, for example 'cotangent shape does not match forward output', 'cotangent must be finite and match output shape', 'S matrices must be finite' and 'array must be finite' for a matrix cotangent. Input errors name the argument: 'array must be finite' → 'operator must be finite', 'array must have shape (N, 3)' → 'points must have shape (N, 3)', 'S matrices require shape (2, 2, n, n) with n > 0' → 'lower must have shape (2, 2, n, n) with n > 0'. All stay ValueError.
- Every pullback takes a cotangent of any real or complex dtype and memory layout, or a nested list. The contexts of `bessel_record`, `vector_wave_record` and the other broadcast records raised TypeError for a float64 array. A pullback of a real output (`CoordinatesContext`, `SvdvalsContext`, `SMatrixTrContext`, `TMatrixMetricContext`) reads the real part of a complex cotangent.
- Every native class reports `__module__ == 'treams_rs._native'` (was `'builtins'`), so reprs and pickling errors show `treams_rs._native.SphereContext`.
- Every native function, solver method and ufunc has its own docstring; the ufuncs shared "Rust special function with NumPy broadcasting and output arrays.". A native function names the treams-core function it calls. A ufunc docstring opens with its call under the treams argument names, `jv(v, z, /, out=None, *, where=True)`, then gives the formula, the treams function it mirrors and the differences from it. The crate doc of `treams-py` describes the native module for maintainers: the record and gradient terms, the conventions, the naming rules and how to add a binding.

### Rust core

- `geometry::{volume, reciprocal, cube, diffraction_orders, first_brillouin, first_brillouin_1d}` → `lattice::{volume, reciprocal, cube, diffraction_orders, first_brillouin, first_brillouin_1d}`; the lattice cell helpers live in the private `lattice/geometry.rs` behind the `lattice` facade.
- The crate-root helpers `jet`, `broadcast`, `parallel`, `finite`, `ratio`, `complex_sqrt` and `EULER` → the internal L0 module `numerics` (`numerics::Jet`, `numerics::broadcast`, `numerics::parallel`, `numerics::{finite, ratio, complex_sqrt, EULER}`); `linalg` and `lattice` become directory modules with unchanged paths.
- Property-test domains: `properties/planar.rs` → `properties/smatrix.rs` and `properties/scattering.rs` → `properties/tmatrix.rs` (with `proptest-regressions/properties/scattering.txt` → `tmatrix.txt`); the numerics unit tests `helper_tests` → `numerics::tests`.
- Reference tables and Lean golden files are included through `CARGO_MANIFEST_DIR`, so test paths survive module moves.
- The special functions form the directory module `special` (≙ `treams.special`) with private files behind a flat facade: `special.rs` → `special/bessel.rs` (Bessel functions, radial jets) and `special/legendre.rs` (integer-degree Legendre, `pi`/`tau`), `legendre.rs` → `special/ferrers.rs`, `angular.rs` → `special/wigner.rs` (3j symbols) and `special/harmonics.rs` (solid harmonics), `coordinates.rs` → `special/coordinates.rs`, `integrals.rs` → `special/integrals.rs`. Paths: `angular::{wigner3j, ...}`, `integrals::{incgamma, intkambe, GammaResidual, KambeResidual, ...}` → `special::{...}`; `coordinates` → `special::coordinates`; `rotation::{wigner_small, wigner, wigner_array, WignerResidual}` → `special::{...}` and `rotation::wigner_d` → `special::wigner_d` (crate-internal); `vectorwaves::{radial_jet, polar_trig, select}` → `special::{...}` (in `special/bessel.rs`, `special/legendre.rs`, `special/polarization.rs`).
- Unit tests and regressions follow the special-function files: `special::tests` → `special::bessel::tests`, `angular::tests` → `special::{wigner, harmonics}::tests`, `integrals::tests` → `special::integrals::tests`, `coordinates::tests` → `special::coordinates::tests`; `proptest-regressions/angular.txt` → `special/wigner.txt` and `special/harmonics.txt` (same seeds), `integrals.txt` → `special/integrals.txt`.
- `special/integrals.rs` becomes the directory `special/integrals/`: `mod.rs` (`incgamma`, `intkambe`, `GammaResidual`, `KambeResidual`), `gamma.rs` (incomplete gamma function, `ScaledGammaLadder`), `kambe.rs` (Kambe integrals, `EvenKambe`, `SmallSplitKambe`) and `double.rs` (double-double arithmetic); paths through `special` are unchanged. Unit tests `special::integrals::tests` → `special::integrals::{gamma, kambe}::tests`; `proptest-regressions/special/integrals.txt` → `special/integrals/gamma.txt` and `special/integrals/kambe.txt` (same seeds).
- The spherical-wave machinery forms the directory module `sw` (≙ `treams.sw`) behind a flat facade: `waves.rs` → `sw/mod.rs` (`Mode`, `modes`), `sw/coupling.rs` (degree couplings) and `sw/cartesian.rs` (Cartesian single-pair translation jets); `translation_plan.rs` → `sw/plan.rs`; the spherical expansion matrices of `basis.rs` → `sw/expansion.rs`; the spherical half of `polar.rs` → `sw/polar.rs`; the periodic spherical-to-cylindrical half of `conversion.rs` → `sw/periodic_to_cw.rs`. `basis.rs` keeps `Basis`, the shared `TranslationGradient` and `PeriodicGradient`, and receives the conversion pullback helper `scatter` (crate-internal). Paths: `waves::{Mode, modes, Translation, translate}` → `sw::{...}`; `basis::{translation, periodic, periodic_from_table, TranslationResidual, PeriodicResidual, PeriodicTableResidual}` → `sw::{...}`; `polar::{SphericalTranslation, spherical, Residual}` → `sw::polar::{...}`; `conversion::{periodic_to_cylindrical, periodic_spherical_to_cylindrical, PeriodicConversionResidual, PeriodicConversionGradient}` → `sw::{...}`.
- Unit tests and regressions follow the sw files: `waves::tests` → `sw::coupling::tests`, `translation_plan::tests` → `sw::plan::tests`, `basis::tests` → `sw::expansion::tests`, the spherical `polar::tests` → `sw::polar::tests`; `proptest-regressions/polar.txt` is copied to `sw/polar.txt` (same seeds).
- The cylindrical-wave machinery forms the directory module `cw` (≙ `treams.cw`) behind a flat facade: `cylwaves.rs` → `cw/mod.rs` (`Mode`, `Basis`), `cw/cartesian.rs` (Cartesian single-pair translation jets) and `cw/expansion.rs` (expansion matrices between bases and over periodic arrays); `conversion.rs` → `cw/to_sw.rs`; `polar.rs` → `cw/polar.rs`. Paths: `cylwaves::{Mode, Basis, Translation, translate, expansion, ExpansionResidual, periodic, PeriodicResidual}` → `cw::{...}`; `conversion::{to_spherical, cylindrical_to_spherical, ConversionResidual}` → `cw::{...}`; `polar::{cylindrical, cylindrical_value, cylindrical_pullback, CylindricalResidual}` → `cw::polar::{...}`.
- Unit tests and regressions follow the cw files: `cylwaves::tests` → `cw::cartesian::tests` and `cw::expansion::tests`, the cylindrical `polar::tests` → `cw::polar::tests`; `proptest-regressions/polar.txt` → `cw/polar.txt` (the same seeds as `sw/polar.txt`).
- The plane-wave machinery forms the directory module `pw` (≙ `treams.pw` and `treams.misc.wave_vec_z`) behind a flat facade: `plane.rs` → `pw/polarization.rs` (directions, polarizations, `field_value`, `wave_vector_z`), `pw/expand.rs` (spherical and cylindrical expansions), `pw/permute.rs` (cyclic axis permutations) and `pw/field.rs` (translation phases and fields). Paths: `plane::X` → `pw::X` for every item; no item is renamed.
- The scattering code forms directory modules behind flat facades. `coeffs.rs` → `coeffs/mie.rs` (sphere Mie coefficients) and `coeffs/material.rs` (`Material`, crate-internal `validate_layers`); `cylinder.rs` → `coeffs/mie_cyl.rs` (cylinder Mie coefficients) and `tmatrix/cylinder.rs` (cylinder T-matrix); `tmatrix.rs` → `tmatrix/metric.rs`, `tmatrix/sphere.rs`, `cluster/particles.rs` and `cluster/spheres.rs`; `interaction.rs`, `illumination.rs` and `iterative.rs` → `cluster/`; the restarted GMRES solver of `iterative.rs` → `linalg/gmres.rs`; `layers.rs` → `smatrix/layers.rs`. The fixed-size matrix aliases `Matrix2`, `Matrix4` and `Matrix42` are declared once in `coeffs`. Paths: `cylinder::{mie_cyl, CylinderResidual, CylinderGradient}` → `coeffs::{...}`; `cylinder::{cylinder, CylinderMatrixResidual, CylinderMatrixGradient}` → `tmatrix::{...}`; `tmatrix::{particle_cluster, cylindrical_particle_cluster, ParticleClusterResidual, ParticleClusterGradient, cluster, cluster_factor, ClusterResidual, ClusterGradient}` → `cluster::{...}`; `interaction::X` → `cluster::interaction::X`; `illumination::X` → `cluster::illumination::X`; `iterative::{SphereCluster, IterativeResidual, IterativeGradient, IterativeSolution}` → `cluster::{...}`; `iterative::{GmresOptions, Convergence}` → `linalg::{...}`; `layers::{stack, LayersResidual, LayersGradient}` → `smatrix::{...}`. No item is renamed.
- Removed `illumination::operator` and `illumination::forward`, which nothing called.
- Unit tests follow the scattering files: `coeffs::tests` → `coeffs::mie::tests`, `illumination::tests` → `cluster::illumination::tests`, `iterative::tests` → `cluster::iterative::tests`, and its two GMRES tests → `linalg::gmres::tests`.
- The crate docs open with a module map: the `treams_rs` namespace and the treams counterpart of every module.
- The radial kind of multipole waves uses the treams word: `special::Radial::Outgoing` → `special::Radial::Singular` (rustdoc alias `outgoing`).
- The radial jets name what they compute: `special::spherical` → `special::spherical_radial`, `special::cylindrical` → `special::cylindrical_radial` and `special::spherical_sequence` → `special::spherical_radial_sequence`; the `spherical_kind` parameter of `special::bessel` → `spherical`.
- The Wigner functions follow `treams.special.wignersmalld` and `wignerd`: `special::wigner_small` → `special::wigner_small_d`, `special::wigner` → `special::wigner_d` (the D-matrix element), `special::wigner_array` → `special::wigner_d_array`, `special::WignerResidual` → `special::WignerDResidual`, and the crate-internal small-d matrix `special::wigner_d` → `special::wigner_small_d_matrix`.
- The real-degree Ferrers function `special::ferrers::factor` → `special::ferrers::ferrers_real_degree`, and the upper incomplete gamma function `special::gamma` → `special::upper_gamma` (both crate-internal).
- The integral residuals follow their functions: `special::GammaResidual` → `special::IncgammaResidual` and `special::KambeResidual` → `special::IntkambeResidual`.
- `special::bessel_values`, `special::angular_values`, `linalg::Equilibration` and `linalg::equilibrate` are crate-internal; nothing outside the crate used them.
- `numerics::Jet::map` → `numerics::Jet::chain`, the chain rule; the jet type and its methods are documented.
- The lattice-sum types: `lattice::Lattice` → `lattice::BlochLattice`, `lattice::Residual` → `lattice::SumResidual`, `lattice::Gradient` → `lattice::SumGradient` and `lattice::Wave` → `lattice::Family`. Their fields and parameters use the treams names: `Derivatives.position` and `SumGradient.position` → `shift` (rustdoc alias `r`), `Derivatives.bloch` and `SumGradient.bloch` → `kpar`, the `r` parameter of `lattice::sum`, `sum_part`, `derivatives` and `derivatives_part` → `shift`, the `bloch` parameter of `BlochLattice::new` and `BlochLattice::from_array` → `kpar`, and `basis::PeriodicGradient.bloch` → `kpar`. The `vectors` fields carry the rustdoc alias `a`. `BlochLattice::from_array` stores zero beyond the leading `dim` entries of `kpar`, as it already did for the lattice vectors.
- The spherical-harmonic normalization `lattice::normalization` → `special::harmonic_normalization`, and the crate-internal `basis::PeriodicGradient::new` → `zeros` and `basis::PeriodicGradient::lattice` → `add_lattice`.
- Tests named after renamed items: `properties::special::wigner_d_symmetries_and_unitarity` → `wigner_small_d_matrix_symmetries_and_unitarity`, `properties::special::wigner_small_matches_the_generator_exponential` → `wigner_small_d_matches_the_generator_exponential` and `special::bessel::tests::spherical_sequences_match_single_degrees` → `spherical_radial_sequences_match_single_degrees`.
- The supported label range has names: `MAX_DEGREE` = 128 at the crate root (the largest degree `l` and the largest cylindrical `|m|`), `special::MAX_LABEL` = 260 (the largest Kambe order or Wigner label; the bindings check against it), and the crate-internal `special::MAX_ORDER` = 256 (the largest order of the integer-order radial jets and of cylindrical order differences; `special::bessel` takes any finite order) and `special::SERIES_RADIUS` = 0.5 (where regular radial functions switch from power series to Bessel evaluations). Error messages keep their numbers.
- The parallel policy moves next to the parallel helpers: `numerics::broadcast::Parallel` → `numerics::parallel::Parallel`, with the variant `Parallel::From` → `Parallel::AtLeast`. The `numerics::parallel` docs list every parallel threshold and the pullbacks that reduce with Rayon directly.
- `special::select` → `special::polarized_wave` (crate-internal).
- Crate-internal helpers replace repeated expressions: `numerics::parity` (`(-1)^n`), `numerics::label_bits` (a hash key for real mode labels that merges -0.0 and +0.0), `special::helicity_sign` (`2 pol - 1`), `special::check_pol` (the "polarization must be 0 or 1" check) and `numerics::Jet::compose` (the chain rule of a polarization jet through three input jets).
- Error messages: `special-function evaluation failed: <message>` → `<message>` for non-finite results and convergence failures, for example `special-function evaluation failed: nonfinite propagation phase` → `non-finite propagation phase` and `special-function evaluation failed: Ewald split too small: …` → `Ewald split too small: …`. The prefix stays on failed AMOS Bessel, Legendre, Ferrers, Wigner, incomplete gamma and Kambe evaluations. `nonfinite` → `non-finite` in every message except the real-degree Ferrers one. `normal-incidence azimuth is undefined; use fixed_q=True for gradients at fixed incidence` → `normal-incidence azimuth is undefined; hold the transverse wavevector fixed (fixed_q) to differentiate at normal incidence`. The variants `Error::NonFinite` and `Error::NotConverged` carry these failures (GMRES non-convergence and Arnoldi breakdown: `InvalidInput` → `NotConverged`; overflowing plane-wave, lattice-table and chirality results: `InvalidInput` → `NonFinite`), and the crate-internal `Error::is_numerical` groups them with `SpecialFunction`. Python raises `ValueError` for every variant.
- The spherical and cylindrical basis matrices follow the treams `Expand` and `ExpandLattice` operators: `sw::translation` → `sw::expansion`, `sw::TranslationResidual` → `sw::ExpansionResidual`, `sw::periodic` → `sw::lattice_expansion`, `sw::PeriodicResidual` → `sw::LatticeExpansionResidual`, `sw::periodic_from_table` → `sw::lattice_expansion_from_table`, `sw::PeriodicTableResidual` → `sw::LatticeExpansionFromTableResidual`, `cw::periodic` → `cw::lattice_expansion` and `cw::PeriodicResidual` → `cw::LatticeExpansionResidual`. Their shared gradients: `basis::TranslationGradient` → `basis::ExpansionGradient` and `basis::PeriodicGradient` → `basis::LatticeExpansionGradient`.
- The polar translation coefficients of one mode pair: `sw::polar::{SphericalTranslation, spherical, Residual}` → `sw::{PolarTranslation, polar_translation_array, PolarTranslationResidual}` and `cw::polar::{cylindrical_value, cylindrical, CylindricalResidual}` → `cw::{polar_translation, polar_translation_array, PolarTranslationResidual}`. The `polar` files are private modules, and `cw::polar::cylindrical_pullback` → the crate-internal `polar_translation_pullback`.
- The Cartesian single-pair translation jets: `sw::Translation` → `sw::CartesianTranslation`, `sw::translate` → `sw::cartesian_translation`, `cw::Translation` → `cw::CartesianTranslation` and `cw::translate` → `cw::cartesian_translation`.
- Family changes take the treams names, and matrices between bases the suffix `_matrix`: `cw::to_spherical` → `cw::to_sw`, `cw::cylindrical_to_spherical` → `cw::to_sw_matrix`, `cw::ConversionResidual` → `cw::ToSwResidual`, `sw::periodic_to_cylindrical` → `sw::periodic_to_cw`, `sw::periodic_spherical_to_cylindrical` → `sw::periodic_to_cw_matrix`, `sw::PeriodicConversionResidual` → `sw::PeriodicToCwResidual`, `sw::PeriodicConversionGradient` → `sw::PeriodicToCwGradient`, `pw::to_spherical` → `pw::to_sw` and `pw::to_cylindrical` → `pw::to_cw`.
- `pw::translation` → `pw::translate` and `pw::permutation_coefficient` → `pw::permute_xyz`, as in treams. `pw::spherical` and `pw::cylindrical` → the test-only `pw::reference_spherical_expansion` and `pw::reference_cylindrical_expansion`.
- The basis rotations carry their family: `rotation::spherical` → `rotation::sw_rotation` and `rotation::cylindrical` → `rotation::cw_rotation`.
- Gradients name the expansion centres `positions`: `pw::ExpansionGradient.origins` → `positions` and `fields::FieldGradient.origins` → `positions`.
- The vector waves have their own label type: `sw::Mode` → `vectorwaves::WaveLabel` in `vectorwaves`, which documents the fields each `Family` reads and accepts the same labels. `vectorwaves::Residual` → `vectorwaves::VectorWaveResidual`, `vectorwaves::value` → `vectorwaves::vector_wave`, `vectorwaves::array` → `vectorwaves::vector_wave_array` and `vectorwaves::pullback` → `vectorwaves::vector_wave_pullback`.
- The plane-wave channels: `channels::spherical` → `channels::spherical_channels`, `channels::cylindrical` → `channels::cylindrical_channels`, `channels::spherical_to_plane` → `channels::sw_periodic_to_pw`, `channels::cylindrical_to_plane` → `channels::cw_periodic_to_pw`, `channels::ChannelsResidual` → `channels::SphericalChannelsResidual`, `channels::CylChannelsResidual` → `channels::CylindricalChannelsResidual` and `ChannelGradient.area` → `measure` (the area of a 2D array or the period of a 1D array).
- `ebcm::qmat(to, source, surface, ks, zs, singular: bool, legacy: bool)` → `ebcm::qmat(destination, source, surface, ks, zs, radial: Radial, radial_area_factor: bool)` with `radial_area_factor = !legacy`: `true` keeps the radial area factor `r` of the surface element, `false` omits it as `treams.ebcm.qmat` does. `ebcm::QResidual` → `ebcm::QmatResidual` and `ebcm::QGradient` → `ebcm::QmatGradient`. The native `ebcm_qmat(..., singular, legacy)` is unchanged.
- Crate-internal: `sw::coupling::helper` → `tl_vsw_term` (treams `special._tl_vsw_helper`) and `basis::scatter` → `basis::add_pair_gradient`. `TranslationPlan::new`, `evaluate` and `pullback`, which fixed `between(modes, modes, true)`, the singular radial kind and the real part of the wavenumber cotangent, are removed; the cluster solvers pass these choices explicitly, and `TranslationPlan::evaluate_with` → `evaluate` and `pullback_with` → `pullback`.
- The ufunc loops take their coefficient rules from core functions, named after their treams counterparts where one exists. Rotation: `rotation::sw_rotate` (`treams.sw.rotate`) and `rotation::cw_rotate` (`treams.cw.rotate`). Translation: `cw::tl_vcw` (`treams.special.tl_vcw` and `tl_vcw_r`), `cw::translate` (`treams.cw.translate`) and `sw::PolarTranslation::is_self_term`, the zero self term of `treams.sw.translate`. Special functions: `special::lpmv_real` (`treams.special.lpmv` with real arguments) and `vectorwaves::sph_harm` (`treams.special.sph_harm`), with `vectorwaves::sph_harm_from_vsh_z` and its pullback for arrays of `vsh_Z` waves. Media: `coeffs::real_refractive_indices` (`treams.misc.refractive_index` with real arguments). `special::pol_index` checks that an integer polarization label is 0 or 1 and returns it as an index. The bindings keep the dtype and label conversions and the label check at the zero self term of `sw.translate`. Values and error messages are unchanged.
- One generic basis type serves both multipole families: the spherical `basis::Basis` and `cw::Basis` → `basis::Basis<M>`, with the aliases `sw::Basis` (`Basis<sw::Mode>`) and `cw::Basis` (`Basis<cw::Mode>`). The trait `basis::ModeLabel` (`validate`, `pol`) replaces the methods `sw::Mode::validate` and `cw::Mode::validate`. `fields::FieldBasis` → `basis::MultipoleBasis`, with the crate-internal methods `origins` → `positions` and `origin_pol` → `position_pol`.
- Error messages: `basis particle index outside positions` and `basis origin index outside positions` → `basis position index outside positions`; `basis must be nonempty with finite origins` → `basis must be nonempty with finite positions`. The particle clusters: `particle modes must be grouped at distinct origins` → `particle modes must be grouped at distinct positions` and `cylinders require distinct transverse origins` → `cylinders require distinct transverse positions`. The native `periodic_from_table`: `table origin axes do not match the bases` → `table position axes do not match the bases` and `table axes must be destination origin, source origin, channel, harmonic` → `table axes must be destination position, source position, channel, harmonic`. The input checks of `sw::expansion`, `sw::lattice_expansion`, `cw::expansion`, `cw::lattice_expansion`, `cw::to_sw_matrix`, `sw::periodic_to_cw_matrix`, `fields::field`, `fields::operator` and `ebcm::qmat` give one message per failed condition: `medium wavenumbers must be finite` (spherical expansions, which accept k = 0), `medium wavenumbers must be finite and nonzero`, `parity polarization requires equal wavenumbers (an achiral medium)`, `field points must be finite`, `period must be positive and finite`, and in `cw::expansion` `medium wavenumber squared underflows to zero`. These replace `finite wave numbers required; parity requires an achiral medium`, `finite nonzero wavenumbers required`, `finite nonzero medium wavenumbers required`, `require nonzero finite medium wavenumbers; parity requires an achiral medium`, `require finite field points and nonzero wavenumbers; parity requires an achiral medium` and `require a positive finite period and nonzero finite medium wavenumbers; parity requires an achiral medium`. `cluster::cylindrical_particle_cluster` checks its wavenumbers like `cluster::particle_cluster`: `parity requires an achiral medium` → `parity polarization requires equal wavenumbers (an achiral medium)`. Every function accepts and rejects the same inputs.
- The sphere Mie coefficients take the treams name: `coeffs::mie_forward` → `coeffs::mie` (`treams.coeffs.mie`). The crate-internal `coeffs::treams_order` → `coeffs::to_mode_order`, which reorders a coefficient matrix from helicity order (negative, positive) to the basis order, where `pol = 1` comes first.
- Layer gradients name their radii: `coeffs::LayerGradient` (`radii`, `epsilon`, `mu`, `kappa`) holds the radius and material cotangents of the cylinder Mie gradient (`coeffs::MieCylGradient.layers`) and of the cylinder and sphere T-matrix gradients (`tmatrix::CylinderGradient.layers`, `tmatrix::SphereGradient.layers`), which carried the radius gradients in `MieGradient.sizes`. `SphereGradient.radii` and `SphereGradient.materials` → `SphereGradient.layers`. `MieGradient.sizes` keeps the size-parameter gradients of `coeffs::mie`.
- The cylinder Mie coefficients follow `coeffs::mie`: `coeffs::CylinderResidual` → `coeffs::MieCylResidual` and `coeffs::CylinderGradient` → `coeffs::MieCylGradient` (`treams.coeffs.mie_cyl`).
- The cylinder T-matrix types follow `tmatrix::cylinder`: `tmatrix::CylinderMatrixResidual` → `tmatrix::CylinderResidual` and `tmatrix::CylinderMatrixGradient` → `tmatrix::CylinderGradient` (`treams.TMatrixC.cylinder`).
- Error messages: `size parameters must be finite, positive and strictly increasing` → `radii and size parameters must be finite, positive and strictly increasing`, because `coeffs::mie_cyl`, `tmatrix::sphere` and `tmatrix::cylinder` take radii and only `coeffs::mie` takes size parameters.
- The dense solver for homogeneous spheres in vacuum names what it couples, so that it reads apart from `cluster::particle_cluster`, the counterpart of `treams.TMatrix.cluster` followed by `.interaction.solve()`: `cluster::cluster` → `cluster::sphere_cluster`, `cluster::ClusterResidual` → `cluster::SphereClusterResidual`, `cluster::ClusterGradient` → `cluster::SphereClusterGradient` and `cluster::cluster_factor` → `cluster::sphere_cluster_factor`. The fields of `SphereClusterGradient` follow the arguments of `sphere_cluster`: `k0`, `radii`, `epsilon`, `positions`.
- The interaction solve takes the operation's name: `cluster::interaction::forward` → `cluster::interaction` (`treams.TMatrix.interaction.solve`) and the crate-internal `cluster::interaction::forward_blocks` → `cluster::interaction_blocks`. `cluster::interaction::InteractionResidual` → `cluster::InteractionResidual`; the `interaction` file is a private module behind the `cluster` facade.
- The reusable interaction factor and its requested-illumination record match the native class `InteractionFactor` and the context of `diff.illuminate`: `cluster::illumination::Factor` → `cluster::InteractionFactor`, `cluster::illumination::Residual` → `cluster::IlluminateResidual` and `cluster::illumination::Gradient` → `cluster::IlluminateGradient`; the `illumination` file is a private module behind the `cluster` facade.
- The matrix-free sphere solver carries the `Iterative` prefix of its residual and gradient: `cluster::SphereCluster` → `cluster::IterativeSphereCluster`. Its convergence certificates use the Python name: `IterativeSolution.reports` → `IterativeSolution.convergence` and `IterativeGradient.reports` → `IterativeGradient.convergence`.
- Degree indices say degree: the crate-internal `tmatrix::block_orders` → `tmatrix::block_degrees`, the private Mie residual lists of `tmatrix::SphereResidual` and `cluster::IterativeSphereCluster` (`orders` → `degrees`) and the private per-mode degree index of `cluster::IterativeSphereCluster` (`mode_orders` → `mode_degrees`).
- The S-matrix illumination takes the treams name, and its variant without pullback data the suffix `_value`: `smatrix::illuminate_borrowed` → `smatrix::illuminate` (`treams.SMatrices.illuminate`), `smatrix::illuminate_forward` → `smatrix::illuminate_value` and `smatrix::IlluminationResidual` → `smatrix::IlluminateResidual`.
- The S-matrix composition residuals follow their functions: `smatrix::StackResidual` → `smatrix::AddResidual` (`smatrix::add`), `smatrix::BandResidual` → `smatrix::BandsResidual` (`smatrix::bands`) and `smatrix::ArrayResidual` → `smatrix::FromArrayResidual` (`smatrix::from_array`).
- The S-matrix transmittance takes the treams name `tr` (`treams.SMatrices.tr`, rustdoc aliases `transmittance` and `power`): `smatrix::transmittance` → `smatrix::tr`, `smatrix::transmittance_value` → `smatrix::tr_value`, `smatrix::TransmissionResidual` → `smatrix::TrResidual`, `smatrix::TransmissionGradient` → `smatrix::TrGradient` and `smatrix::PowerPorts` → `smatrix::TrPorts`. The field `PowerPorts.transmission` → `TrPorts.direction`: 0 means up (towards the positive side), 1 down.
- The chirality-density types say which chirality they belong to, apart from the T-matrix chirality metric: `smatrix::ChiralityResidual` → `smatrix::ChiralityDensityResidual` and `smatrix::ChiralityGradient` → `smatrix::ChiralityDensityGradient`. The `smatrix` chirality docs state what each of the two kernels, `chirality_density` and `oriented_chirality`, takes and differentiates.
- The compact layer stack follows the native `layer_stack`: `smatrix::stack` → `smatrix::layer_stack`, `smatrix::LayersResidual` → `smatrix::LayerStackResidual` and `smatrix::LayersGradient` → `smatrix::LayerStackGradient`.
- One return convention for forwards with a pullback: a forward returns `(value, XResidual)` when the pullback does not read the value, and otherwise the residual keeps the value behind a read-only accessor. `sw::expansion`, `sw::lattice_expansion`, `cw::expansion`, `cw::lattice_expansion`, `cw::to_sw_matrix`, `fields::field`, `tmatrix::sphere`, `tmatrix::cylinder`, `tmatrix::metric`, `channels::spherical_channels` and `channels::cylindrical_channels` return `(value, residual)`. The fields `value` of `coeffs::MieResidual`, `coeffs::MieCylResidual`, `cluster::InteractionResidual`, `cluster::IlluminateResidual`, `rotation::RotationResidual` and `linalg::SolveResidual`, `IterativeResidual.solution`, `SingularResidual.values`, `EigenResidual.values` and `.vectors` and `BandsResidual.wavenumbers` → the accessors `value()`, `solution()`, `values()`, `vectors()` and `wavenumbers()`. `smatrix::tr` returns `TrResidual`, whose `value()` holds the transmittance and reflectance.
- The residuals name their output-shape accessor `shape()`: `pw::PermutationResidual::modes` → `shape`, `smatrix::{AddResidual, PeriodicResidual, PropagationResidual}::dimension` → `shape`, `smatrix::FromArrayResidual::ports` → `shape`, and `sw::LatticeExpansionFromTableResidual::shape` → `table_shape` (the harmonic table), with a new `shape` for the output matrix. The Mie, Fresnel and interface outputs are always 2 × 2 and have no `shape()`; `smatrix::LayerStackResidual::channel_count()` gives the number of channels, each with four 2 × 2 blocks.
- Pullbacks with several inputs return named gradients with one field per input, in the order of the forward's arguments: the tuple aliases `smatrix::FresnelGradient` and `smatrix::InterfaceGradient` → structs, and the tuples of `smatrix::{AddResidual, FromArrayResidual, PropagationResidual, IlluminateResidual, BandsResidual}::pullback`, `tmatrix::MetricResidual::pullback`, `linalg::SolveResidual::pullback` and `cluster::InteractionResidual::{pullback, pullback_blocks}` → `AddGradient { lower, upper }`, `FromArrayGradient { response, channels }`, `PropagationGradient { vectors, distance }`, `IlluminateGradient { lower, upper, incoming }`, `BandsGradient { blocks, period }`, `MetricGradient { matrix, ks }`, `SolveGradient { operator, rhs }` and `InteractionGradient { local, coupling }`.
- The forwards take the choice to hold inputs fixed, and the pullbacks take only the cotangent: `fixed_q` moves from the pullbacks to `smatrix::interface`, `smatrix::layer_stack`, `smatrix::tr`, `channels::spherical_channels` and `channels::cylindrical_channels`, and `fixed_vectors` to `pw::field` and `pw::expansion`.
- The S-matrix files follow their operations: `smatrix/compose.rs` → `compose.rs` (`add`), `periodic.rs` (`periodic`, `bands`) and `array.rs` (`from_array`); `smatrix/internal.rs` → `solve.rs` (the internal-field solve) and `illuminate.rs` (`illuminate`, `illuminate_value`); `smatrix/power.rs` → `smatrix/tr.rs`. The new alias `smatrix::Channels` names the four channel arrays of `from_array`: incident up, incident down, emitted up, emitted down. Crate-internal: `InternalFactor::Iterative` → `InternalFactor::Krylov`, `InternalFactor::Direct` → `InternalFactor::Lu` and `InternalSolve::adjoint_rhs` → `InternalSolve::solve_adjoint`. Unit tests: `smatrix::internal::illumination_tests` → `smatrix::solve::tests`.
- Error messages: `singular scattering system` → `singular linear system`, because LU solves, 2 x 2 Mie inverses and S-matrix solves raise `Error::Singular`; `special-function evaluation failed: nonfinite real-degree Legendre result or derivative` → `special-function evaluation failed: non-finite real-degree Legendre result or derivative` (Ferrers). Every error message spells `non-finite`.
- One type holds the LU of `I - T C`: the crate-internal `cluster::interaction::Factorization` merges into `cluster::InteractionFactor`, which moves with `IlluminateResidual` and `IlluminateGradient` from `cluster/illumination.rs` into `cluster/interaction.rs`. The public methods `new`, `from_blocks`, `dimension`, `solve` and `record` keep their signatures; `InteractionFactor` gains `Clone`. `IterativeSphereCluster::record` borrows `self: &Arc<Self>`, as `InteractionFactor::record` does. Crate-internal: `interaction::local_dense` / `local_blocks` → `LocalMatrix::from_dense` / `from_blocks`, `LocalMatrix::dense` → `to_dense`. Unit test: `cluster::illumination::tests` → `cluster::interaction::tests`.
- The pullback argument follows the other residuals: `tmatrix::MetricResidual::pullback(weight)` → `pullback(cotangent)`. `coeffs::MaterialTangent` is visible only inside `coeffs`.
- The `coeffs`, `tmatrix` and `cluster` docs map each item to its treams counterpart with its differences, list the four cluster solvers and when to use each, and state for each pullback whether its sums run in a fixed order. The position and `k0` gradients of `IterativeSphereCluster` add with a Rayon `try_reduce`, so with more than one thread their last bits depend on the thread count.
- Crate-internal names in `sw` and `cw`: `TranslationPlan::normalize_lattice` → `weights_for_normalized_harmonics` (it divides the plan weights by the harmonic normalization, so that the plan reads tables of orthonormal-harmonic lattice sums); the degree bound of `TranslationPlan` and its harmonic table `order` → `lmax`, and `orders` → `degree_orders`; `sw::coupling::Coupling` takes a `Kinds { same, cross }` instead of `[bool; 2]`; the 5-tuple key of the cylindrical couplings → the struct `CouplingKey`; the private `translate` of `sw::expansion` → `expansion_block`. `cw::transverse`, `cw::transverse_wavenumber` and `cw::polar_translation_pullback` are private. Public pullbacks name their argument `cotangent`. Values are unchanged.
- Error messages: `lattice table contraction overflow` → `lattice table sum overflows` (`sw::lattice_expansion_from_table`, which `diff.lattice_expansion_from_table` calls).

### Documentation

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
