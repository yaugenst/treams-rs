# Examples

Each example in the first table repeats a calculation from the treams
documentation with treams-rs. Its page shows the treams-rs script, the same
calculation in treams 0.4.5 and the printed results.

| Example | What it computes | treams counterpart |
| --- | --- | --- |
| [Single sphere](sphere.md) | Averaged extinction and scattering spectra up to l = 4 and for dipoles only; the intensity around the sphere | [sphere.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/sphere.py) |
| [Cluster of spheres](cluster.md) | Cross sections and near fields of four coupled spheres, with local, global and rotated T-matrices | [cluster.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/cluster.py) |
| [Chain of spheres](chain.md) | The near field of a chain periodic along z | [chain.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/chain.py) |
| [Grid of spheres](grid.md) | The near field of a square array in the xy plane | [grid.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/grid.py) |
| [Photonic crystal](crystal.md) | The smallest singular value of the lattice interaction matrix of a cubic crystal | [crystal.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/crystal.py) |
| [Chiral slab](slab.md) | Transmission and reflection of a chiral slab for both helicities | [slab.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/slab.py) |
| [Array of spheres on a slab](array_spheres.md) | Transmission and reflection of a square array of chiral spheres on a slab at oblique incidence | [array_spheres.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/array_spheres.py) |
| [Band structure](band_structure.md) | Bloch wavenumbers of a periodic stack of sphere arrays and slabs | [band_structure.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/band_structure.py) |
| [Single cylinder](cylinder.md) | Averaged extinction and scattering cross widths up to m = 4 and for m = 0 only; the intensity around the cylinder | [cylinder_tmatrixc.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/cylinder_tmatrixc.py) |
| [Chain of spheres in cylindrical waves](cylindrical_chain.md) | The near field of a chain periodic along z, from its cylindrical T-matrix and from spherical waves | [chain_tmatrixc.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/chain_tmatrixc.py) |
| [Chain and cylinder](cylindrical_cluster.md) | The near field of a chain of spheres coupled to an infinite cylinder | [cluster_tmatrixc.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/cluster_tmatrixc.py) |
| [Cylinder crystal](cylindrical_crystal.md) | The smallest singular value of the lattice interaction matrix of a square lattice of cylinders | [crystal_tmatrixc.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/crystal_tmatrixc.py) |
| [Grating](grating.md) | The near field of chains of spheres and cylinders repeated along x | [grating_tmatrixc.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/grating_tmatrixc.py) |
| [Array of spheres from cylindrical waves](cylindrical_array.md) | Transmission and reflection of the sphere array on a slab, built from chains in cylindrical waves | [array_spheres_tmatrixc.py](https://github.com/tfp-photonics/treams/blob/1f5d0d6ebb007288f28bc9e16f6d266e8b55dc39/docs/examples/array_spheres_tmatrixc.py) |

## Beyond treams

These examples differentiate a result with respect to the geometry, which
treams cannot do. They have no treams counterpart; each checks its gradient
against central differences.

| Example | What it computes | Framework |
| --- | --- | --- |
| [Sphere radius optimization](sphere_radius_optimization.md) | Five gradient-ascent steps on the radius of a sphere to maximize its scattering efficiency | Advect |
| [Array transmission gradient](array_transmission_gradient.md) | The derivatives of the transmission of a sphere array on a slab with respect to the sphere radius and the slab thickness | JAX |

## Running the examples

Run a script from the repository root with treams-rs installed:

```bash
python docs/examples/sphere.py
```

The scripts print tables and draw no figures. Constants at the top, such as
`FREQUENCIES` and `GRID_POINTS`, set small sizes so that each script runs in
under a second. Enlarge them for full sweeps; the treams versions use 200
frequencies and grids of up to 101 x 101 points. The two gradient examples
need the `treams-rs[advect]` and `treams-rs[jax]` extras; the JAX example
takes about five seconds.

The test suite runs every script and compares what it prints with the recorded
output, to a relative tolerance of 1e-6. When treams is installed, it also runs
the treams versions and compares the results, to a relative tolerance of 1e-10.
Values near zero also get an absolute tolerance. The grating and the array of
spheres from cylindrical waves get a relative tolerance of 3e-8 and 2e-8,
because treams sums a lattice of cylinders less accurately. The test states the
reason for each tolerance.
