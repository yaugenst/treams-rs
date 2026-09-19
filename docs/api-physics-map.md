# Physics API redesign map

The primary API separates physical objects, unsolved geometry and solved
responses. Ordinary numerical work uses `.array` or `.coefficients`; physical
operations retain enough metadata to evaluate the resulting field.

This map records design decisions across the complete non-autodiff surface.
The installed `support_catalog()` and generated API reference own exact
signatures and available members; this document is not a capability registry.

| Family | Previous entry points | Primary workflow and decision |
| --- | --- | --- |
| Materials | `Material`, abbreviated index constructors and properties | Keep explicit dimensionless epsilon, mu and chirality. Exterior parameters are named `medium`, distinct from particle/layer `material`. |
| Spherical/cylindrical bases | `SphericalWaveBasis`, `CylindricalWaveBasis` | `SphericalBasis`, `CylindricalBasis`; preserve mode labels, origins, selections, set operations and diffraction-order construction. |
| Plane bases | `PlaneWaveBasisByUnitVector`, `PlaneWaveBasisByComp` | `PlaneWaveBasis` holds directions; `PlaneWavePorts` holds transverse wavevector components. Their frequency dependence is deliberately different. |
| Sources | Positional plane-wave direction/polarization and multipole constructors | `plane_wave(direction=..., polarization=..., k0=..., medium=...)`; named positive/negative helicity avoids integer-label ambiguity. Multipole sources retain explicit mathematical mode labels. |
| Wave amplitudes and fields | `MultipoleWave`, raw scattered coefficient arrays | `Wave`; `.coefficients` is read-only; `.in_basis()` returns a physical wave. E/H/D/B/G/F fields retain the established physical normalization. Batched amplitudes have shape `(modes, illuminations)` and fields have shape `(..., 3, illuminations)`. |
| Sphere and cylinder construction | `TMatrix.sphere`, `TMatrixC.cylinder`, interior/exterior materials in one list | `sphere_tmatrix`, `cylinder_tmatrix`, and explicit multilayer variants; keyword geometry and a separate exterior medium. Low-level arbitrary matrix construction remains available. |
| Finite clusters | Block-diagonal `TMatrix.cluster` with `.interaction` methods | `Cluster(particles, positions=...)` is unsolved and has no matrix multiplication. `.solve()` creates a complete response; `.scatter()` computes requested illuminations; `.factor().scatter()` reuses dense factorization. |
| Response application and transforms | Array `@`, implicit selection, `expand`, `changepoltype` | `.scatter()` returns a wave, `.select()` selects modes, `.in_basis()` changes representation and `.with_polarization()` changes convention. Expert coefficient operations remain explicit. |
| Particle observables | Tuple `xs/xw`, abbreviated averages and chirality metrics | Named `.cross_sections()` and `.cross_widths()` results expose scattering, extinction and absorption. Average results use the same names. Widths remain lengths; sphere cross sections remain areas. |
| Periodic interactions | Raw `.latticeinteraction.solve()` output | `solve_periodic()` returns `PeriodicResponse` with lattice and Bloch vector. It deliberately has no isolated-particle cross sections. |
| Periodic radiation and fields | Implicit interaction solves in array-to-S/cylindrical conversion | `.to_smatrix()` and `.to_cylindrical()` convert the existing solved response. `.scatter()` returns `PeriodicWave`; `.in_basis()` chooses plane ports, cylindrical radiation or local regular multipoles before field evaluation. |
| Planar interfaces and layers | `SMatrices.interface/slab/propagation` and ambiguous material ordering | Keyword `interface`, `slab`, `multilayer_slab`, `propagation`; exterior media are explicitly negative/positive relative to the basis normal. |
| Planar networks and blocks | `SMatrices` whole network, `SMatrix` one block | `SMatrix` means the complete network; `ScatteringBlock` means a port block. `stack()` and `.cascade()` proceed from negative to positive side. |
| Planar illumination and observables | Variable tuple `illuminate`, `tr`, `cd` | `.scatter(negative=..., positive=...)` returns named outgoing waves; `.power()` returns transmission/reflection/absorption. Circular dichroism names transmission and outgoing-power contrasts accurately. |
| Periodic planar bands | `periodic()` and z-specific `bands_kz()` | `.transfer_matrix()` and `.bands(period=...)`; wavenumbers follow the basis normal, including non-z orientations. |
| Matrix operators | Root-level functions, operator classes and bound descriptors | `operators` is the explicit numerical operator namespace. Familiar transforms on physical objects are ordinary methods; experts retain matrix builders and inverse operators. |
| Geometry | `Lattice`, partial `WaveVector`, transforms/set operations | Keep physical geometry types and lattice constructors. Periodic APIs explicitly carry lattice and Bloch wavevector. |
| Direct wave coefficients | `pw`, `cw`, `sw` | Preserve the complete plane/cylindrical/spherical coefficient and conversion families as numerical namespaces. These are expert operations, not a second physics-object API. |
| Material/scattering coefficients | `coeffs` | Preserve Fresnel, spherical Mie and cylindrical Mie kernels, including recording paths. |
| Special functions | `special` | Preserve Bessel/Hankel, Legendre, Wigner, gamma/Kambe, coordinate/vector transformations, vector harmonics/waves and translation coefficients. Mathematical names stay conventional. |
| Lattice numerics | `lattice` | Preserve full Ewald, real/reciprocal/direct sums, shifted geometries, supported dimensions, expansion kernels and cell/diffraction utilities. |
| Miscellaneous/EBCM numerics | `misc`, `ebcm` | Preserve index/wavevector, mode matching, basis conversion, Brillouin folding and EBCM quadrature/Q matrices. Corrected physics and explicitly selected legacy comparisons remain distinct. |
| Requested iterative solves | `iterative.SphereCluster` and convergence/gradient results | Preserve its restricted homogeneous nonmagnetic spheres in vacuum, requested-column execution and checked convergence. Dense-vs-iterative is an explicit computational choice. |
| Interchange/meshing | Optional `io` HDF5 and Gmsh helpers | Preserve optional imports, numerical round trips, units and scatterer metadata. File output does not require a separate object hierarchy. |
| Defaults | Mutable global `config.POLTYPE` | Deterministic helicity defaults; explicitly choose another convention. Existing objects retain their own convention. |
| Discovery/qualification | Catalog, offline CLI, public testing helpers | Keep source-derived signatures and derivative contracts. API presentation does not change numerical qualifications or derivative support. |

A complete finite-scattering workflow:

```python exec
import numpy as np
import treams_rs as tr

sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
incident = tr.plane_wave(direction=[0, 0, 1], polarization="positive_helicity", k0=1.3)
scattered = sphere.scatter(incident)
assert scattered.efield([[0.4, 0.2, 0.3]]).shape == (1, 3)
power = sphere.cross_sections(incident)
np.testing.assert_allclose(power.scattering, power.extinction, rtol=1e-11)
```

A cluster only solves what the caller requests:

```python exec
import numpy as np
import treams_rs as tr

particle = tr.sphere_tmatrix(k0=1.3, lmax=1, radius=0.1, material=3)
cluster = tr.Cluster([particle, particle], positions=[[0, 0, 0], [1, 0, 0]])
incident = tr.plane_wave(direction=[0, 0, 1], polarization="positive_helicity", k0=1.3)
factor = cluster.factor()
scattered = factor.scatter(incident)
np.testing.assert_allclose(
    scattered.coefficients,
    cluster.solve().scatter(incident).coefficients,
    rtol=1e-12,
    atol=1e-14,
)
```

Periodic radiation includes every translated cell. Its local coefficients must
not be evaluated as an isolated finite collection:

```python exec
import numpy as np
import treams_rs as tr

sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
response = tr.solve_periodic(sphere, lattice=np.eye(2) * 2, kpar=[0, 0])
ports = tr.PlaneWavePorts.default([0, 0])
incident = tr.plane_wave(direction=[0, 0, 1], polarization="positive_helicity", k0=1.3)
scattered = response.scatter(incident).in_basis(ports, kind="up")
assert scattered.efield([[0, 0, 1]]).shape == (1, 3)
network = response.to_smatrix(ports)
assert network.power(incident).transmission >= 0
```

For periodic near fields, choose a local spherical/cylindrical basis centered in
the region of interest and call `periodic_wave.in_basis(basis, kind="regular")`.
The local regular expansion is valid within its convergence region; plane-port
fields are valid on the corresponding exterior side. Direct `.efield()` on
reference-cell coefficients is intentionally absent.

Framework adapters use the same physical vocabulary. Their actual differentiated
parameter paths and execution limits are described in the adapter documentation;
ordinary NumPy objects do not silently trace framework values.
