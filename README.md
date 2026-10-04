<!-- markdownlint-disable MD033 -->
<!-- PyPI drops <source> and shows the light wordmark on its light pages. -->
<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/yaugenst/treams-rs/main/docs/assets/wordmark-dark.svg">
    <img src="https://raw.githubusercontent.com/yaugenst/treams-rs/main/docs/assets/wordmark.svg" alt="treams-rs">
  </picture>
</h1>

<p align="center">
  <a href="https://github.com/yaugenst/treams-rs/actions/workflows/ci.yml"><img src="https://github.com/yaugenst/treams-rs/actions/workflows/ci.yml/badge.svg?branch=main&amp;event=push" alt="CI"></a>
  <a href="https://app.codecov.io/gh/yaugenst/treams-rs"><img src="https://codecov.io/gh/yaugenst/treams-rs/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://pypi.org/project/treams-rs/"><img src="https://img.shields.io/pypi/v/treams-rs.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/treams-rs/"><img src="https://img.shields.io/pypi/pyversions/treams-rs.svg" alt="Python versions"></a>
</p>
<!-- markdownlint-enable MD033 -->

treams-rs is a Rust port of [treams](https://github.com/tfp-photonics/treams)
with a Python interface and analytic gradients.

**All credit for treams belongs to the original treams contributors.**
Their work is described in the
[treams paper](https://doi.org/10.1016/j.cpc.2023.109076); please cite it when
using this port.

treams-rs computes electromagnetic scattering with T-matrices: spheres,
cylinders and layered or chiral particles, finite clusters, periodic arrays and
planar stacks, with their fields, cross sections and power. A T-matrix is the
linear map from the multipole coefficients of an incident wave to those of the
scattered wave. Advect, JAX, PyTorch and HIPS Autograd use its analytic gradients
through the ordinary `treams_rs` API.

## Relationship to treams

- **Same conventions.** treams-rs uses treams' units, polarization, mode
  ordering and normalization. Reference tests use treams 0.4.7; historical
  comparisons and benchmarks identify their original treams version.
- **Same numerical modules.** `special`, `sw`, `cw`, `pw`, `lattice`,
  `coeffs`, `misc`, `ebcm` and `io` keep the treams function and argument
  names.
- **Its own Python interface.** Objects such as `TMatrix`, `Cluster`, `Wave` and
  `SMatrix` carry their basis, wavenumber and media as attributes, in place of
  annotated arrays.
- **A Rust core.** At run time it needs only NumPy: no treams, SciPy or Cython.

[Coming from treams](https://yaugenst.github.io/treams-rs/latest/coming-from-treams/)
maps treams workflows and names to treams-rs.

## Install

```sh
pip install treams-rs
```

Gradient frameworks and HDF5 files are optional extras:

```sh
pip install "treams-rs[advect]"   # or [jax], [torch], [autograd]
pip install "treams-rs[io]"       # HDF5 T-matrix files
```

Wheels cover CPython 3.12–3.15 on Linux (glibc 2.17 or newer, x86-64 and
arm64), macOS (Intel and Apple silicon) and Windows (x86-64), and need no Rust
toolchain. Standard PyPI installs of the `torch` and `io` extras on CPython
3.15 are outside the release qualification. NumPy 2.3 and newer have
Linux wheels only for glibc 2.27 or newer; with an older glibc, use CPython 3.12
or 3.13 and `pip install treams-rs "numpy<2.3"`. JAX and PyTorch have no Intel
macOS wheels; use the `advect` or `autograd` extra there. See
[Install](https://yaugenst.github.io/treams-rs/latest/getting-started/install/)
for optional dependencies and source builds.

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

Pass framework values to the same physics API: treams-rs selects the adapter
automatically. This gives the gradient of the scattering cross section with
respect to the sphere radius (`pip install "treams-rs[advect]"`):

```python exec
from advect import grad
import treams_rs as tr


def scattering(radius):
    sphere = tr.sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=3)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)
    return sphere.cross_sections(incident).scattering


print(grad(scattering)(0.2))
```

The same `scattering` function works with `jax.grad`, `torch.autograd` and
`autograd.grad`. Optional adapters load only when needed; Python and NumPy
inputs retain NumPy behavior. Rust computes each derivative analytically.
The adapters support first-order CPU gradients, with framework-specific dtype
and transform requirements; see
[Differentiation](https://yaugenst.github.io/treams-rs/latest/differentiation/).

## Documentation

The [documentation](https://yaugenst.github.io/treams-rs/latest/) covers:

- [Getting started](https://yaugenst.github.io/treams-rs/latest/getting-started/quickstart/):
  four short examples.
- [Coming from treams](https://yaugenst.github.io/treams-rs/latest/coming-from-treams/):
  workflows, names and conventions side by side.
- [User guide](https://yaugenst.github.io/treams-rs/latest/guide/): particles, clusters,
  periodic arrays, planar stacks and numerical functions.
- [Differentiation](https://yaugenst.github.io/treams-rs/latest/differentiation/):
  gradients through Advect, JAX, PyTorch, HIPS Autograd and the `diff` module.
- [Examples](https://yaugenst.github.io/treams-rs/latest/examples/): the treams gallery
  in treams-rs, plus gradient-based design.
- [Validation](https://yaugenst.github.io/treams-rs/latest/validation/): tests against
  treams, analytic results and published spectra.
- [Performance](https://yaugenst.github.io/treams-rs/latest/performance/): CPU timings
  against treams 0.4.5, with the measured builds and limitations recorded.
- [Reference](https://yaugenst.github.io/treams-rs/latest/reference/): every public
  function and class.
- [Development](https://yaugenst.github.io/treams-rs/latest/development/): building
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
