# treams workflows in treams-rs

!!! warning "SMatrix means something else"

    treams-rs `SMatrix` is the S-matrix of the whole two-port network, which
    treams calls `SMatrices`. One of its four blocks is a `ScatteringBlock`,
    which treams calls `SMatrix`. `tr.SMatrices` raises an `AttributeError`
    that names `SMatrix`.

treams-rs follows the numerical [conventions](conventions.md) of treams 0.4.7.
The Python API offers two ways to work:

- **Numerical namespaces.** `special`, `sw`, `cw`, `pw`, `lattice`, `coeffs`,
  `misc`, `ebcm` and `io` keep the treams function names and argument names.
  `operators` holds the treams operators and `PhysicsArray`. A call such as
  `treams.sw.translate(...)` runs as `treams_rs.sw.translate(...)`; the
  [differences page](differences.md) lists where results differ.
- **Physics objects.** `TMatrix`, `Wave`, `Cluster`, `SMatrix` and functions
  such as `sphere_tmatrix` take keyword arguments and store physical properties
  as attributes. They replace the annotated arrays of treams.

## Workflow translations

Each pair computes the same numbers. The treams-rs code runs as written; the
values in its checks come from the treams code next to it.

### A single sphere

<!-- fmt: off -->

=== "treams"

    ```python
    import treams

    tm = treams.TMatrix.sphere(3, 2.0, 0.5, [treams.Material(4), treams.Material()])
    print(tm.xs_ext_avg)  # 0.6258290233441384
    ```

=== "treams-rs"

    ```python
    import numpy as np
    import treams_rs as tr

    sphere = tr.sphere_tmatrix(k0=2.0, lmax=3, radius=0.5, material=4, medium=1)
    average = sphere.average_cross_sections
    np.testing.assert_allclose(average.extinction, 0.6258290233441384, rtol=1e-12)
    ```

<!-- fmt: on -->

`sphere_tmatrix` takes the particle `material` and the surrounding `medium`
separately; treams takes one list that ends with the embedding medium.
`average_cross_sections` holds both averages, `scattering` and `extinction`,
which treams returns as `xs_sca_avg` and `xs_ext_avg`.

### Scattering a plane wave

<!-- fmt: off -->

=== "treams"

    ```python
    import treams

    tm = treams.TMatrix.sphere(3, 2.0, 0.5, [treams.Material(4), treams.Material()])
    inc = treams.plane_wave([0, 0, 2.0], 1, k0=2.0, poltype="helicity")
    sca = tm @ inc.expand(tm.basis)
    print(tm.xs(inc.expand(tm.basis)))  # (0.6258290233441385, 0.6258290233441386)
    ```

=== "treams-rs"

    ```python
    import numpy as np
    import treams_rs as tr

    sphere = tr.sphere_tmatrix(k0=2.0, lmax=3, radius=0.5, material=4)
    incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=2.0)
    scattered = sphere.scatter(incident)
    assert scattered.kind == "singular"
    assert scattered.efield([[0, 0, 1.0]]).shape == (1, 3)
    cross = sphere.cross_sections(incident)
    np.testing.assert_allclose(cross.extinction, 0.6258290233441386, rtol=1e-12)
    ```

<!-- fmt: on -->

`scatter` expands the incident wave into the basis of the T-matrix and returns
a `Wave`. The wave stores its basis, `k0` and medium, and evaluates fields.

### Clusters

<!-- fmt: off -->

=== "treams"

    ```python
    import treams

    tm = treams.TMatrix.sphere(2, 2.0, 0.3, [treams.Material(4), treams.Material()])
    cluster = treams.TMatrix.cluster([tm, tm], [[0, 0, 0], [1, 0, 0]])
    solved = cluster.interaction.solve()
    inc = treams.plane_wave([0, 0, 2.0], 1, k0=2.0, poltype="helicity")
    sca = solved @ inc.expand(solved.basis)
    print(sca[0])  # (-0.0089964306-0.0048436211j)
    ```

=== "treams-rs"

    ```python
    import numpy as np
    import treams_rs as tr

    particle = tr.sphere_tmatrix(k0=2.0, lmax=2, radius=0.3, material=4)
    cluster = tr.Cluster([particle, particle], positions=[[0, 0, 0], [1, 0, 0]])
    incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=2.0)

    solved = cluster.solve()  # the coupled T-matrix
    scattered = solved.scatter(incident)
    np.testing.assert_allclose(
        scattered.coefficients[0], -0.0089964306 - 0.0048436211j, rtol=1e-8
    )
    # The same wave without the coupled T-matrix: one solve per call, or one
    # factorization for many illuminations.
    once = cluster.scatter(incident)
    reused = cluster.factor().scatter(incident)
    np.testing.assert_allclose(once.coefficients, scattered.coefficients, atol=1e-13)
    np.testing.assert_allclose(reused.coefficients, scattered.coefficients, atol=1e-13)
    ```

<!-- fmt: on -->

`Cluster` holds particles and positions; it has no matrix of its own.
`solve()` returns the coupled `TMatrix`; `scatter()` and `factor()` solve only
for the illuminations you pass. See [Clusters](../guide/clusters.md).

### Periodic arrays

<!-- fmt: off -->

=== "treams"

    ```python
    import treams

    tm = treams.TMatrix.sphere(3, 2.0, 0.5, [treams.Material(4), treams.Material()])
    array = tm.latticeinteraction.solve(treams.Lattice.square(2), [0, 0])
    ports = treams.PlaneWaveBasisByComp.default([0, 0])
    network = treams.SMatrices.from_array(array, ports)
    print(network.tr([1, 0]))  # (0.9854070565320263, 0.01459294346797362)
    ```

=== "treams-rs"

    ```python
    import numpy as np
    import treams_rs as tr

    sphere = tr.sphere_tmatrix(k0=2.0, lmax=3, radius=0.5, material=4)
    array = tr.solve_periodic(sphere, lattice=tr.Lattice.square(2), kpar=[0, 0])
    ports = tr.PlaneWavePorts.default([0, 0])
    network = array.to_smatrix(ports)
    power = network.power([1, 0])
    np.testing.assert_allclose(power.transmission, 0.9854070565320263, rtol=1e-12)
    np.testing.assert_allclose(power.reflection, 0.01459294346797362, rtol=1e-10)
    ```

<!-- fmt: on -->

`solve_periodic` returns a `PeriodicResponse`, which keeps the lattice and the
Bloch vector `kpar` (the wavevector component along the lattice). `to_smatrix`
and `to_cylindrical` convert it; `scatter` returns a `PeriodicWave`. See
[Periodic arrays](../guide/periodic.md).

### Slabs and stacks

<!-- fmt: off -->

=== "treams"

    ```python
    import treams

    ports = treams.PlaneWaveBasisByComp.default([0, 0.5])
    slab = treams.SMatrices.slab(0.5, ports, 2.0, [1, 4, 1])
    gap = treams.SMatrices.propagation([0, 0, 0.3], ports, 2.0, 1)
    total = treams.SMatrices.stack([slab, gap, slab])
    print(total.tr([1, 0]))  # (0.3636441759771435, 0.6363558240228586)
    ```

=== "treams-rs"

    ```python
    import numpy as np
    import treams_rs as tr

    ports = tr.PlaneWavePorts.default([0, 0.5])
    slab = tr.slab(basis=ports, k0=2.0, thickness=0.5, material=4)
    gap = tr.propagation(basis=ports, k0=2.0, distance=0.3)
    total = tr.stack([slab, gap, slab])
    power = total.power([1, 0])
    np.testing.assert_allclose(power.transmission, 0.3636441759771435, rtol=1e-12)
    ```

<!-- fmt: on -->

`slab` names its outer media `negative_medium` (below) and `positive_medium`
(above); both default to vacuum. `stack` runs from the negative side to the
positive side, as in treams. `power` is treams' `tr`. See
[Planar layers](../guide/planar.md).

### Polarization convention

<!-- fmt: off -->

=== "treams"

    ```python
    import treams

    treams.config.POLTYPE = "parity"
    tm = treams.TMatrix.sphere(2, 2.0, 0.3, [treams.Material(4), treams.Material()])
    treams.config.POLTYPE = "helicity"
    same = treams.TMatrix.sphere(
        2, 2.0, 0.3, [treams.Material(4), treams.Material()]
    ).changepoltype("parity")
    ```

=== "treams-rs"

    ```python
    import numpy as np
    import treams_rs as tr

    parity = tr.sphere_tmatrix(
        k0=2.0, lmax=2, radius=0.3, material=4, polarization="parity"
    )
    helicity = tr.sphere_tmatrix(k0=2.0, lmax=2, radius=0.3, material=4)
    converted = helicity.with_polarization("parity")
    np.testing.assert_allclose(converted.array, parity.array, atol=1e-15)
    ```

<!-- fmt: on -->

Each object stores its convention in `.polarization`: `"helicity"` unless you
pass `polarization="parity"`. No global setting exists.

### Plane waves

<!-- fmt: off -->

=== "treams"

    ```python
    import treams

    inc = treams.plane_wave([0, 0.5, 1.2], [1, 0, 0], k0=1.3, material=1)
    ```

=== "treams-rs"

    ```python
    import numpy as np
    import treams_rs as tr

    incident = tr.plane_wave(direction=[0, 0.5, 1.2], pol=[1, 0, 0], k0=1.3)
    np.testing.assert_allclose(incident.efield([[0, 0, 0]]), [[1, 0, 0]], atol=1e-15)
    ```

<!-- fmt: on -->

`direction` need not be normalized: `k0` and the medium set the length of the
wavevector. `pol` is the state of the wave: `"positive_helicity"`,
`"negative_helicity"`, an index 0 or 1, two amplitudes for pol 0 and pol 1, or
the three Cartesian components of the electric field.

## Concept map

| treams | treams-rs | Note |
| --- | --- | --- |
| `Material`, material lists `[..., embedding]` | `Material`; `material=` and `medium=` | `material` is the particle or layer; `medium` surrounds it. Planar objects have `negative_medium` and `positive_medium`. |
| `SphericalWaveBasis`, `CylindricalWaveBasis` | `SphericalBasis`, `CylindricalBasis` | Same mode labels, positions, selections and set operations. |
| `PlaneWaveBasisByUnitVector`, `PlaneWaveBasisByComp` | `PlaneWaveBasis`, `PlaneWavePorts` | `PlaneWaveBasis` holds unit directions. `PlaneWavePorts` holds two transverse wavevector components, so a port keeps them at every frequency. |
| `plane_wave(kvec, pol)`, `spherical_wave`, `cylindrical_wave` | `plane_wave(direction=..., pol=..., k0=...)`, `spherical_wave`, `cylindrical_wave` | `pol` is the state of the wave, `polarization` the convention. Multipole sources take the mode labels `l`, `m` or `kz`, and `pol`. |
| `PhysicsArray` of wave coefficients | `Wave`, `PlaneWave`, `PeriodicWave` | `.coefficients` is read-only; `.in_basis()` returns the wave in another basis. Coefficients of several illuminations have shape `(modes, illuminations)`, fields `(..., 3, illuminations)`. |
| `TMatrix.sphere`, `TMatrixC.cylinder` | `sphere_tmatrix`, `cylinder_tmatrix`, `multilayer_sphere_tmatrix`, `multilayer_cylinder_tmatrix` | Keyword geometry and a separate `medium`. `TMatrix(array, k0=..., basis=...)` wraps any matrix. |
| `TMatrixC` | `CylindricalTMatrix` | Same matrix. |
| `TMatrix.cluster(...).interaction.solve()` | `Cluster(...).solve()`, `.scatter()`, `.factor()` | A `Cluster` stays unsolved until you call one of the three. |
| `tm @ inc.expand(tm.basis)` | `tm.scatter(incident)` | Returns a `Wave` that evaluates fields. |
| `expand`, `changepoltype` | `in_basis`, `with_polarization` | Both return new objects. |
| `xs`, `xw`, `xs_ext_avg`, `xw_sca_avg`, `cd`, `db`, `chi` | `cross_sections`, `cross_widths`, `average_cross_sections`, `average_cross_widths`, `circular_dichroism`, `duality_breaking`, `electromagnetic_chirality` | Results are named tuples in the treams order, such as `CrossSections(scattering, extinction)`. Cross widths are lengths; cross sections are areas. |
| `latticeinteraction.solve(lattice, kpar)` | `solve_periodic(unit_cell, lattice=..., kpar=...)` | Returns a `PeriodicResponse`, which has no isolated-particle cross sections. |
| `SMatrices.from_array`, `TMatrixC.from_array` | `PeriodicResponse.to_smatrix(ports)`, `PeriodicResponse.to_cylindrical(basis)` | Both convert the solved array. |
| `SMatrices.interface`, `slab`, `propagation`, `stack` | `interface`, `slab`, `multilayer_slab`, `propagation`, `stack` | Keyword arguments; the outer media are `negative_medium` and `positive_medium`. |
| `SMatrices` (network), `SMatrix` (block) | `SMatrix` (network), `ScatteringBlock` (block) | See the warning at the top. |
| `SMatrices.add`, `illuminate`, `tr`, `cd`, `periodic`, `bands_kz` | `cascade`, `scatter`, `power`, `circular_dichroism`, `transfer_matrix`, `bands` | `scatter(negative=..., positive=...)` returns the outgoing waves by name. `bands` follows the basis normal, also when it is not z. |
| `config.POLTYPE` | `polarization=` on each object | The default is `"helicity"`. |
| Operators (`Rotate`, `Translate`, `Expand`, `EField`, ...) and `PhysicsArray` | `treams_rs.operators` | Physics objects have the common transforms as methods: `rotate`, `translate`, `in_basis`. |
| `Lattice`, `WaveVector` | `Lattice`, `WaveVector` | Periodic objects carry their lattice and Bloch vector. |
| `special`, `sw`, `cw`, `pw`, `lattice`, `coeffs`, `misc`, `ebcm`, `io` | the same module names | Same functions and arguments; see [differences](differences.md). |
| none | `iterative.SphereCluster` | Matrix-free solves for homogeneous nonmagnetic spheres in vacuum; see [large clusters](../guide/large-clusters.md). |
| none | the ordinary API with Advect, JAX, PyTorch or HIPS Autograd inputs; `diff` and the `advect`, `jax`, `torch` and `autograd` namespaces for records | Gradients of the numerical functions and physics objects; see [Differentiation](../differentiation/index.md). |

## Pitfalls

- **One material, one medium.** `sphere_tmatrix(material=4, medium=1.33)`
  replaces the list `[Material(4), Material(1.33)]`. Multilayer factories take
  `materials` from the inside out and a separate `medium`.
- **Arrays have no physical metadata.** `.array` is a read-only NumPy array
  without basis or `k0`. Array arithmetic returns plain arrays; wrap a result
  again with `TMatrix(array, k0=..., basis=...)`.
- **`scatter()`, not `@`.** `tm @ incident` returns bare coefficients;
  `tm.scatter(incident)` returns a `Wave` that evaluates fields.
- **Named results.** `cross_sections`, `power` and
  `SMatrix.circular_dichroism` return named tuples. They unpack in the treams
  order; use `.extinction` or `.transmission` to read one value.
- **No global `POLTYPE`.** `tr.config` raises an `AttributeError`. Pass
  `polarization="parity"` to the factory or call `with_polarization`.
- **`kind` is `"regular"` or `"singular"`.** Singular waves are outgoing: their
  radial part is the Hankel function of the first kind. `kind="outgoing"`
  raises a `ValueError` that names `"singular"`. Plane-wave ports use
  `kind="up"` or `"down"`.
- **`pol` is not `polarization`.** `pol` is the state of a wave or the index 0
  or 1 of a mode; `polarization` is the convention, `"helicity"` or `"parity"`.
- **`k0` is required.** Waves and factories need `k0=`; no object takes it from
  another.
- **treams member names still work on NumPy objects.** `xs`, `changepoltype`,
  `poltype`, `modetype`, `illuminate`, `add` and the other treams names call the
  treams-rs member. The [name map](names.md) lists each one; objects built from
  framework values lack some of them
  ([limits](../differentiation/frameworks.md#limits)).
