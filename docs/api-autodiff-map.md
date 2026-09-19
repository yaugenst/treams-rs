# Framework physics and native differentiation

Import `treams_rs.jax`, `treams_rs.torch`, or `treams_rs.advect` to select an
explicit CPU execution framework. The shared physics implementation retains
framework arrays, frequency, media and moving origins. Ordinary `treams_rs`
objects use NumPy; passing a traced tensor to those constructors is not the
framework interface.

The primary constructors are `Material`, `sphere_tmatrix`,
`multilayer_sphere_tmatrix`, `cylinder_tmatrix`, `multilayer_cylinder_tmatrix`,
`plane_wave`, `wave`, `tmatrix`, `smatrix`, `Cluster`, `interface`, `slab`,
`multilayer_slab`, `propagation`, and `stack`. Objects implement scattering,
physical field evaluation, named cross sections and port power. A cluster
separates assembly from `solve()` and requested-illumination `scatter()`.
`solve_periodic()` returns a response whose `to_smatrix()` performs conversion
without another interaction solve.

```python
import jax
import treams_rs.jax as tr

jax.config.update("jax_enable_x64", True)


def loss(radius):
    tm = tr.sphere_tmatrix(
        radius=radius, material=tr.Material(3.0 + 0.1j), k0=1.2, lmax=2
    )
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    field = tm.scatter(incident).efield([[0.3, 0.2, 1.5]])
    return (abs(field) ** 2).sum()


value, gradient = jax.jit(jax.value_and_grad(loss))(0.2)
```

Compile scalar or array objectives, with physics objects created inside them.
The objects are not registered JAX PyTrees; returning a physical object directly
from `jit` is not supported. PyTorch uses the same constructors and ordinary
`backward()`. Advect uses the same constructors and ordinary `advect.grad`.
The bridges retain their existing residual lifetime and complex-gradient
conventions, documented in [adapters](adapters.md).

## Periodic geometry

A static `PlaneWavePorts` argument fixes dimensional transverse wavevectors.
For geometry optimization use integer diffraction orders instead:

```python
response = tr.solve_periodic(unit_cell, lattice=cell, kpar=bloch)
smatrix = response.to_smatrix(orders=[[0, 0], [1, 0], [-1, 0]])
power = smatrix.power(incident_coefficients)
```

Order ports follow `q = bloch + 2*pi*orders @ inverse(cell).T`; they differentiate
cell measure, coupling, incidence, radiation and port-power normalization.
They include helicities `(1, 0)` for each order. Their dimensional components
are available as `smatrix.transverse_wavevectors`; `basis` is `None` because a
static Python basis cannot contain traced wavevectors. Cylindrical order ports
use a vector of integer orders and one fixed axial sector. Distinct order labels
are static. At the polarization axis, direction derivatives remain undefined;
use a fixed port basis when only other parameters vary.

`smatrix.scatter(negative=incident)` returns named `positive` and `negative`
physical waves. Their E/H/D/B/G/F fields retain derivatives. `power()` infers the
incident side for a plane wave; explicit sides are `negative` and `positive`.
The same-side source must propagate toward the stack and match the represented
ports, exterior medium and frequency.

## Complete surface map

This table explains where complete existing numerical families live. Exact
signatures, derivative ordering, and exported members come from
`support_catalog()` and the generated [API reference](api.md), rather than a
second manually maintained symbol registry.

| Family | Native recording and expert access | Physical object path |
|---|---|---|
| Sphere and multilayer Mie coefficients | `diff.sphere`, `coeffs.mie_with_context` | Sphere constructors; scatter, cross sections and fields |
| Cylinders and cylindrical coefficients | `diff.cylinder`, `coeffs.mie_cyl_with_context` | Cylinder constructors with static axial labels |
| Finite heterogeneous particles | `diff.particle_cluster`, interaction and illumination records | `Cluster.solve()` and `.scatter()` |
| Reusable requested-illumination factors | `diff.factor_interaction`, block and cluster factors | Expert record/factor interface; physical cluster methods record fresh solves |
| Matrix-free sphere systems | `iterative.SphereCluster` and its named solution/gradient reports | Expert iterative interface, not an implicit dense fallback |
| E/H/D/B/G/F fields | Native multipole/plane field records and analytic framework composition | Typed waves and scattered ports |
| Basis conversion and geometry | Expansion, rotation, plane conversion/permutation and translation records | Wave basis conversion; expert records retain all specialized transforms |
| Periodic multipole systems | Lattice expansion/sums, table assembly and periodic conversion records | `solve_periodic(...).to_smatrix(...)` for supported planar radiation |
| Radiation channels | Spherical and cylindrical channel records | Static ports or moving integer order ports |
| Interfaces and stacks | Fresnel, interface, layer and propagation records | `interface`, `slab`, `multilayer_slab`, `propagation`, `stack` |
| Scattering networks and bands | S-matrix composition, illumination, transfer and bands records | `cascade`, `scatter`, `power`, `bands` |
| Chirality and response metrics | Chirality density, oriented chirality and T-matrix metric records | Expert metrics; ordinary framework objectives compose their outputs |
| Axisymmetric EBCM shapes | `diff.ebcm_qmat` | Expert shape/quadrature interface |
| Special functions and coordinates | Bessel, incomplete gamma, Kambe, angular/Wigner, vector waves, harmonics and coordinate records | Expert mathematical functions |
| Linear algebra | Native solve, eigenvalue/vector, singular-value records | Expert operations and internal physical solves |
| Derivative qualification | `testing.check_pullback`, `testing.check_gradient` | Independent directional checks of records/objectives |

Advect retains the complete direct expert primitive surface. Its former
numerical `interface` and `propagation` functions are now
`interface_coefficients` and `propagation_matrix`, freeing the physical names.
JAX and Torch retain their direct expert conveniences and universal `wrap(record)`
for every native family. `Wave`, `TMatrix` and `SMatrix` are shared returned
types; `wave`, `tmatrix` and `smatrix` bind their raw-array constructors to the
selected framework. Custom records must return gradients in dynamic input
order; several native contexts also contain embedded basis-coordinate or optional
material gradients. Nothing is silently approximated or delegated upstream.

## Explicit limits

All adapters provide analytic first-order reverse derivatives on CPU. JAX
requires x64, supports `jit` and sequential `vmap`, and recomputes native residuals
in reverse. Torch supports repeated backward by recomputation after the first
retained native residual. Advect native residuals remain one-use. GPU execution,
forward-mode and higher derivatives are not provided.

High-level plane direction is static; frequency, amplitudes and material are
dynamic. Fixed-direction angular factors and native translation phases are
separated so frequency derivatives work on the polarization axis. High-level
cylindrical axial labels remain static; plane illumination currently requires
`kz=0` and transverse incidence. Expert cylindrical primitives support additional
axial derivatives with fixed group correspondence.

Physical objects do not cover every specialized kernel operation with a method.
EBCM, custom lattice tables, sphere-to-cylinder periodic conversion, generalized
metrics, reusable factorizations and matrix-free solves retain their explicit
expert APIs. This is a deliberate surface boundary, not a claim that those
families lack native derivatives. Discrete cutoffs/topology are never traced.
Eigenvalue crossings, diffraction thresholds, conditioning, and polarization-axis
singularities retain their native restrictions.
