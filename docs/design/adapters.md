---
description: Shared physics classes, native calls and gradients in Advect, JAX and PyTorch.
---

# Framework adapters

`treams_rs.advect`, `treams_rs.jax` and `treams_rs.torch` offer the physics objects
of `treams_rs` with framework arrays: waves, T-matrices, clusters, periodic
responses and S-matrices. All three use the same classes, constructors and
expert operations; an adapter adds only the step that turns a native call into
a framework operation. Advect also offers array functions, such as
`advect.field` and `advect.expansion`, that wrap `treams_rs.diff`; JAX and
PyTorch offer `wrap` for your own records. The
[framework guide](../differentiation/frameworks.md) shows how to use them.

## One entry point

The physics objects make every native call through `Backend.apply(record,
shape, *values, real=...)`. It runs `record(*values)`, which returns the output
and a context, and checks the declared output shape and dtype. Its pullback
returns one gradient per value, in order. Bases, mode labels and conventions
never appear in `values`: the record captures them.

Each adapter module supplies:

- its array namespace (`advect.numpy`, `jax.numpy`, `torch`);
- the function `_operation(record, *values, shape=..., real=...)`, which turns one
  record into one framework operation;
- optional `asarray` (PyTorch tensor conversion) and `validate` (JAX precision and
  device checks) functions.

The adapter then binds the shared constructors, `_framework.Constructors`, and
the shared expert operations, `_framework.Operations`: `solve`, `interaction`,
`illuminate`, `sphere` and `bessel`. These operations and Advect's array
functions call `_operation` directly, without the check of `Backend.apply`. Each
expert operation declares its output shape, which JAX needs to build its
callback; Advect and PyTorch read the output from the native forward.

## Guard records

Some checks compare metadata of two objects, such as the `k0` and medium of a
T-matrix and its illumination. Under `jax.jit` these values are traced, and their
numbers are available only inside the native callback. The check runs inside
a record, `Backend.guard`: it passes the value through, raises on a mismatch
and gives the compared metadata zero gradients. The `Backend.apply` docstring
lists every guard.

## How long a context lives

| Framework | Forward | Reverse pass |
|---|---|---|
| Advect | Records once and keeps the context | Consumes the context; another reverse pass needs another forward |
| JAX | Records inside `jax.pure_callback` and keeps only the inputs | Records again and consumes that context at once |
| PyTorch | Records once, keeps the context and snapshots of the inputs | The first backward consumes the context; a repeated backward records again from the snapshots |

JAX may skip or repeat a `pure_callback`, and the values it saves for the reverse
pass must be arrays. A native context cannot meet these requirements, so the JAX
adapter saves the inputs and repeats the native forward call for each reverse
pass. PyTorch keeps the context because a single backward is the common case;
the snapshots make `retain_graph=True` work without a context that can be used
twice.

## Static configuration

Bases, mode labels, polarization conventions, plane-wave directions and lattice
dimensions are Python values, not framework arrays. They decide array shapes, so
a traced value could not set them. Frequency, amplitudes, materials, positions,
radii and lattice vectors are framework arrays and carry gradients. An S-matrix
keeps its plane-wave ports in one `PortSet`: ports built from a basis have fixed
transverse wavevectors, while diffraction-order ports follow the lattice and the
Bloch vector and carry gradients.
