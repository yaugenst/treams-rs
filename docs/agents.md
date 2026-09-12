# Agent quickstart

Start at [`llms.txt`](../llms.txt); every link refers to this repository revision.
For development, `treams_rs` can coexist with the upstream reference package.
Local documentation works offline.

With an installed wheel and no checkout, inspect the exact installed API offline:

```sh
python -m treams_rs --format markdown
python -m treams_rs > support.json
```

The JSON includes signatures, docstrings, returned pullback methods, installed
optional dependency versions and compiled CUDA flags. It reads the installed
Python source and native stubs; it does not import JAX, PyTorch or Advect, probe
the GPU, or infer runtime usability from an installed dependency. The generated
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
| Familiar scattering workflow | `TMatrix.sphere`, `TMatrix.cluster`, field and S-matrix objects | CPU; explicit physical metadata and read-only `.array` |
| Only a few incident waves | `diff.cluster_factor` / `.interaction.factor()` | Reuse a dense LU and solve only requested columns |
| Larger homogeneous sphere cluster | `iterative.SphereCluster` | Vacuum, nonmagnetic homogeneous spheres; matrix-free GMRES with explicit convergence checks |
| Native derivatives | `diff` and returned `.pullback` methods | First order; static mode counts/labels/topology; real Hermitian pairing |
| Composed objective derivatives | `advect`, `jax`, `torch` submodules | Optional CPU frameworks; read [adapter contracts](adapters.md) before wrapping |
| Browser execution | `crates/treams-wasm` JavaScript/TypeScript | Direct WASM; no Python or server; serial exported subset |
| Explicit NVIDIA execution | `cuda.Device`, `cuda.PlaneWaves` | Opt-in compilation and explicit transfers; dense solve and fixed-operator coefficient pullbacks; fused fields are forward-only |

A complete sphere calculation requires only NumPy and the installed extension:

```python exec
import numpy as np
import treams_rs as tr

sphere = tr.TMatrix.sphere(3, 2.0, 0.2, [3.0, 1.0])
assert sphere.array.shape == (30, 30)
np.testing.assert_allclose(sphere.xs_ext_avg, sphere.xs_sca_avg, rtol=1e-11)
print("Lossless sphere scattering cross section:", sphere.xs_sca_avg)
```

Lengths and inverse vacuum wavenumber must use consistent units. Material layers
run from inside to outside, followed by the embedding medium. Multipole cutoffs,
mode labels and topology are fixed during differentiation. For native VJPs,
`dL = Re(vdot(cotangent, perturbation))`; frameworks convert their own complex
conventions. Treat contexts as single-use and record again for another pullback.
Gradient order is operation-specific: inspect the recording docstring and the
returned context signature. [Testing helpers](testing.md) check a native boundary
or composed scalar objective without requiring an autodiff framework.

Ordinary Python calls run on the CPU. `compiled: true` for CUDA establishes build
support, not the presence of a compatible driver, CUDA libraries or free device
memory; construct `cuda.Device()` explicitly. Dense GPU results are mixed, and CPU
factor reuse wins the measured triangular solves. Read the [measured
GPU results](gpu.md), including warmup/transfers, before choosing a backend.

The [WASM qualification](wasm.md) exercises Node and Chrome against the same Rust
core. It exposes sphere and interacting/independent finite-cluster scattering,
repeated illumination, direct incident plane waves and exterior scattered electric
fields. `cluster_target_gradient` returns one fixed target's total intensity and
analytic radius/position derivatives; it is not a generic pullback API. The
regular `electric_field(..., false)` expansion sums all local origins: passing a cluster's plane-wave coefficients
counts the incident plane wave once per origin. It is not a physical total-field
constructor. Use `direct_plane_field` plus the scattered field for a cluster's
total field. Separate forward-only helpers expose bounded square-array
diffraction and lossless normal-incidence crystal spectra/internal Ex fields.
General periodic-system objects, particle-interior fields, and generic pullbacks
are not JS exports; CUDA support is separate from browser execution.

For numerical trust, consult [status](status.md), [paper qualification](paper-qualification.md)
and [upstream findings](upstream-findings.md). Recorded performance proofs retain
their measured source/binary hashes; source documentation changes do not relabel
those measurements. For code changes, follow [contributor routing](../CONTRIBUTING.md).
