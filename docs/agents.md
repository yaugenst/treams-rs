# Agent quickstart

Start at [`llms.txt`](../llms.txt); every link refers to this repository revision.
For development, `treams_rs` can coexist with the upstream reference package.
Local documentation works offline. The [physics-first guide](user-guide.md) is
the primary user workflow. Cluster assembly is separate from solving; scattering
returns typed waves; periodic responses convert without another solve.

With an installed wheel and no checkout, inspect the exact installed API offline:

```sh
python -m treams_rs --format markdown
python -m treams_rs > support.json
```

The JSON includes signatures, docstrings, returned pullback methods, installed
optional dependency versions. It reads the installed
Python source and native stubs; it does not import JAX, PyTorch or Advect, or infer runtime usability from an installed dependency. The generated
[API reference](api.md) contains the same signatures and contracts. `help()` and
`inspect.signature()` work directly on Python wrappers. Native NumPy ufuncs expose
their positional/broadcasting contract in `help()` rather than `inspect.signature()`.

```python exec
import json
import treams_rs as tr

catalog = tr.support_catalog()
assert catalog["backends"]["cpu"]["compiled"]
solve = next(row for row in catalog["api"] if row["path"] == "treams_rs.diff.solve")
assert solve["pullbacks"][0]["context"] == "SolveContext"
print(
    json.dumps(
        {"signature": solve["signature"], "pullbacks": solve["pullbacks"]}, indent=2
    )
)
```

Choose the path around the output you need:

| Need | Entry point | Contract |
| --- | --- | --- |
| Familiar scattering workflow | `sphere_tmatrix`, `Cluster`, typed waves and `SMatrix` | CPU; explicit physical metadata and read-only `.array` |
| Only a few incident waves | `Cluster.factor()` / `diff.cluster_factor` | Reuse a dense LU and solve only requested columns |
| Larger homogeneous sphere cluster | `iterative.SphereCluster` | Vacuum, nonmagnetic homogeneous spheres; matrix-free GMRES with explicit convergence checks |
| Native derivatives | `diff` and returned `.pullback` methods | First order; static mode counts/labels/topology; real Hermitian pairing |
| Composed objective derivatives | `advect`, `jax`, `torch` submodules | Optional CPU frameworks; read [adapter contracts](adapters.md) before wrapping |

A complete sphere calculation requires only NumPy and the installed extension:

```python exec
import numpy as np
import treams_rs as tr

sphere = tr.sphere_tmatrix(lmax=3, k0=2.0, radius=0.2, material=3.0)
assert sphere.array.shape == (30, 30)
np.testing.assert_allclose(
    sphere.average_cross_sections.extinction,
    sphere.average_cross_sections.scattering,
    rtol=1e-11,
)
print(
    "Lossless sphere scattering cross section:",
    sphere.average_cross_sections.scattering,
)
```

Lengths and inverse vacuum wavenumber must use consistent units. Material layers
run from inside to outside, followed by the embedding medium. Multipole cutoffs,
mode labels and topology are fixed during differentiation. For native VJPs,
`dL = Re(vdot(cotangent, perturbation))`; frameworks convert their own complex
conventions. Treat contexts as single-use and record again for another pullback.
Gradient order is operation-specific: inspect the recording docstring and the
returned context signature. [Testing helpers](testing.md) check a native boundary
or composed scalar objective without requiring an autodiff framework.

For numerical trust, consult [status](status.md), [paper qualification](paper-qualification.md)
and [upstream findings](upstream-findings.md). Recorded performance proofs retain
their measured source/binary hashes; source documentation changes do not relabel
those measurements. For code changes, follow [contributor routing](../CONTRIBUTING.md).
