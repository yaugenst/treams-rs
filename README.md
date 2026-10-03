# treams-rs

[![CI](https://github.com/yaugenst/treams-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/yaugenst/treams-rs/actions/workflows/ci.yml)
[![Docs](https://github.com/yaugenst/treams-rs/actions/workflows/docs.yml/badge.svg)](https://yaugenst.github.io/treams-rs/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yaugenst/treams-rs/blob/main/LICENSE)
[![Python 3.12 | 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://yaugenst.github.io/treams-rs/getting-started/install/)

treams-rs computes electromagnetic scattering with T-matrices: spheres,
cylinders and layered or chiral particles, finite clusters, periodic arrays and
planar stacks, with their fields, cross sections and power. A T-matrix is the
linear map from the multipole coefficients of an incident wave to those of the
scattered wave. A Rust core does the numerical work and a typed Python API
describes the physics. The operations have analytic gradients, which Advect, JAX
and PyTorch use directly.

## Relationship to treams

treams-rs follows [treams](https://github.com/tfp-photonics/treams) by Dominik
Beutel and coworkers:

- **Same numbers.** treams-rs uses the units, polarization, mode ordering and
  normalization of treams 0.4.5, so arrays agree entry by entry.
- **Same numerical namespaces.** `special`, `sw`, `cw`, `pw`, `lattice`,
  `coeffs`, `misc`, `ebcm` and `io` keep the treams function and argument
  names.
- **Its own physics API.** Objects such as `TMatrix`, `Cluster`, `Wave` and
  `SMatrix` carry their basis, wavenumber and media as attributes, in place of
  annotated arrays.
- **A Rust core.** At run time it needs only NumPy: no treams, SciPy or Cython.

[Coming from treams](https://yaugenst.github.io/treams-rs/coming-from-treams/)
maps treams workflows and names to treams-rs.

## Install

treams-rs is not on PyPI yet. Install it from GitHub; pip compiles the Rust
core, so [rustup](https://rustup.rs) must be on the `PATH`:

```sh
pip install "treams-rs @ git+https://github.com/yaugenst/treams-rs"
```

Extras add optional packages: `[advect]`, `[jax]` and `[torch]` for gradients,
and `[io]` for HDF5 files. See
[Install](https://yaugenst.github.io/treams-rs/getting-started/install/) for
details.

## Example

Two absorbing dielectric spheres scatter a circularly polarized plane wave:

```python exec
import treams_rs as tr

spheres = [
    tr.sphere_tmatrix(k0=1.3, lmax=3, radius=r, material=4 + 0.1j) for r in (0.2, 0.25)
]
cluster = tr.Cluster(spheres, positions=[[0, 0, 0], [0, 0, 0.8]])
incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.3)
scattered = cluster.scatter(incident)
print(scattered.efield([[0.1, 0.2, 1.2]]))
```

## Differentiation

The same physics in `treams_rs.advect` gives the gradient of the scattering
cross section with respect to the sphere radius (install `treams-rs[advect]`):

```python exec
from advect import grad
from treams_rs import advect as tr


def scattering(radius):
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=3)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
    return sphere.cross_sections(incident).scattering


print(grad(scattering)(0.2))
```

`treams_rs.jax` and `treams_rs.torch` work the same way with `jax.grad` and
`torch.autograd`. Rust computes each derivative analytically; see
[Differentiation](https://yaugenst.github.io/treams-rs/differentiation/).

## Documentation

The documentation lives at <https://yaugenst.github.io/treams-rs/>:

- [Getting started](https://yaugenst.github.io/treams-rs/getting-started/quickstart/):
  four short examples.
- [Coming from treams](https://yaugenst.github.io/treams-rs/coming-from-treams/):
  workflows, names and conventions side by side.
- [User guide](https://yaugenst.github.io/treams-rs/guide/): particles, clusters,
  periodic arrays, planar stacks and the numerical namespaces.
- [Differentiation](https://yaugenst.github.io/treams-rs/differentiation/):
  gradients through Advect, JAX, PyTorch and the `diff` module.
- [Examples](https://yaugenst.github.io/treams-rs/examples/): the treams gallery
  in treams-rs, plus gradient-based design.
- [Validation](https://yaugenst.github.io/treams-rs/validation/): tests against
  treams, analytic results and published spectra.
- [Performance](https://yaugenst.github.io/treams-rs/performance/): CPU timings
  against treams 0.4.5.
- [Reference](https://yaugenst.github.io/treams-rs/reference/): every public
  function and class.
- [Development](https://yaugenst.github.io/treams-rs/development/): building
  from a checkout, tests and
  [contributing](https://github.com/yaugenst/treams-rs/blob/main/CONTRIBUTING.md).

## Citing

If you use treams-rs, cite it with the metadata in
[CITATION.cff](https://github.com/yaugenst/treams-rs/blob/main/CITATION.cff),
and cite the treams paper:

> D. Beutel, I. Fernandez-Corbaton and C. Rockstuhl, "treams – a T-matrix-based
> scattering code for nanophotonics", Computer Physics Communications 297,
> 109076 (2024), <https://doi.org/10.1016/j.cpc.2023.109076>.

## License

treams-rs is released under the
[MIT license](https://github.com/yaugenst/treams-rs/blob/main/LICENSE). It
ports numerical methods from treams
([LICENSE.treams](https://github.com/yaugenst/treams-rs/blob/main/LICENSE.treams))
and from the other projects listed in
[THIRD_PARTY_NOTICES.md](https://github.com/yaugenst/treams-rs/blob/main/THIRD_PARTY_NOTICES.md).
