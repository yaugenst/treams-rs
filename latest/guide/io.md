# HDF5 and Gmsh

`treams_rs.io` needs the `[io]` extra, which installs h5py. It mirrors
`treams.io`.

## T-matrices in HDF5 files

`io.save_hdf5` writes spherical T-matrices to an open HDF5 group in the tmat.h5
layout used by treams, with units, positions, the embedding material and
optional metadata about the scatterers and the computation. `io.load_hdf5`
reads them back as a `TMatrix`, or as an array of them for a parameter sweep:

```python
import h5py
import numpy as np
import treams_rs as tr
from treams_rs import io

sphere = tr.sphere_tmatrix(k0=2 * np.pi / 700, lmax=2, radius=75, material=4)
with h5py.File("sphere.h5", "w", driver="core", backing_store=False) as file:
    io.save_hdf5(file, sphere, name="sphere", lunit="nm")
    loaded = io.load_hdf5(file, lunit="nm")
np.testing.assert_allclose(loaded.array, sphere.array)
np.testing.assert_allclose(loaded.k0, sphere.k0, rtol=1e-15)
```

`lunit` names the length unit of the positions and of `1 / k0`, here nm. A
loaded T-matrix supports `scatter` and `cross_sections` like any other;
evaluate fields on the scattered wave. `load_hdf5` also reads the older treams
names of the embedding chirality and particle indices.

## Meshes

`io.mesh_spheres(radii, positions, model)` adds spheres, their volume and
surface groups and mesh sizes to a Gmsh model. You keep control of Gmsh: your
code initializes it, generates and writes the mesh and finalizes it.
