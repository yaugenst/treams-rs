# Framework adapters

The ordinary `treams_rs` API selects `treams_rs.advect`, `treams_rs.jax`,
`treams_rs.torch` or `treams_rs.autograd` from its inputs. The adapters offer
framework arrays in waves, T-matrices, clusters, periodic responses and
S-matrices. All four use the same classes, constructors and
expert operations; an adapter adds only the step that turns a native call into
a framework operation. Advect also offers array functions, such as
`advect.field` and `advect.expansion`, that wrap `treams_rs.diff`; JAX,
PyTorch and HIPS Autograd offer `wrap` for your own records. The
[framework guide](../differentiation/frameworks.md) shows how to use them.

## Selection before conversion

`_dispatch.backend_for` recognizes framework values without coercing them and
examines supported containers and materials. It imports only the selected
adapter; plain Python and NumPy inputs keep the existing NumPy implementation.
Different frameworks in one operation raise before conversion. No registry,
backend setting or ambient tracing context is involved.

Public constructors dispatch before validating numeric values. Existing
framework objects retain their backend. When a constant NumPy physics object
meets a framework value, `_promotion.promote` rebuilds its supported physical
representation using that backend. Numerical functions use
`_autodiff_functions.transparent_function` to select a native derivative record
while delegating NumPy calls, attributes and ufunc methods to the original
callable. It does not implement differentiation through ufunc methods or mutable
`out`/masked `where` outputs.

This boundary contains selection and conversion only. New physics still goes
in shared `_framework*.py` code; each adapter implements one record bridge.

## One entry point

The physics objects make every native call through `Backend.apply(record,
shape, *values, real=...)`. It runs `record(*values)`, which returns the output
and a context, and checks the declared output shape and dtype. Its pullback
returns one gradient per value, in order. Bases, mode labels and conventions
never appear in `values`: the record captures them.

Each adapter module supplies:

- its array namespace (`advect.numpy`, `jax.numpy`, `torch`, `autograd.numpy`);
- the function `_operation(record, *values, shape=..., real=...)`, which turns one
  record into one framework operation;
- optional `asarray` (PyTorch tensor conversion) and `validate` (JAX precision and
  device checks) functions.

The adapter then binds the shared constructors, `_framework.Constructors`, and
the shared expert operations, `_framework.Operations`: `solve`, `interaction`,
`illuminate`, `sphere` and `bessel`. These operations and Advect's array
functions call `_operation` directly, without the check of `Backend.apply`. Each
expert operation declares its output shape, which JAX needs to build its
callback; Advect, PyTorch and HIPS Autograd read the output from the native forward.

`_records.py` owns output and gradient normalization: outputs are a scalar,
array or flat tuple; a pullback returns one gradient per dynamic input, with
its shape and dtype. Real inputs receive real gradients. Native gradients use
`dL = Re(sum(conj(g) * dx))`; JAX and HIPS Autograd conjugate around the native
pullback to match their own complex pairing. Advect and PyTorch use the native
pairing directly.

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
| HIPS Autograd | Records once and keeps the context and inputs | The first reverse pass consumes the context; a repeated VJP records again |

JAX may skip or repeat a `pure_callback`, and the values it saves for the reverse
pass must be arrays. A native context cannot meet these requirements, so the JAX
adapter saves the inputs and repeats the native forward call for each reverse
pass. PyTorch keeps the context because a single backward is the common case;
the snapshots make `retain_graph=True` work without a context that can be used
twice.

## Static configuration

Bases, mode labels, polarization conventions and lattice
dimensions are Python values, not framework arrays. They decide array shapes, so
a traced value could not set them. Frequency, amplitudes, materials, positions,
radii, real plane-wave directions and lattice vectors can carry gradients. An S-matrix
keeps its plane-wave ports in one `PortSet`: ports built from a basis have fixed
transverse wavevectors, while diffraction-order ports follow the lattice and the
Bloch vector and carry gradients.
