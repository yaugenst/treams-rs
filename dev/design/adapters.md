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
returns one gradient per value, in order. Its pushforward takes one tangent
per value in that same order and returns the output tangent. Bases, mode labels and conventions
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

`_records.py` owns output and derivative normalization: outputs are a scalar,
array or flat tuple; a pullback returns one gradient per dynamic input, with
its shape and dtype. Real inputs receive real gradients. Native gradients use
`dL = Re(sum(conj(g) * dx))`; JAX and HIPS Autograd conjugate around the native
pullback to match their own complex pairing. Advect and PyTorch use the native
pairing directly. Pushforwards use input increments without conjugation. The
shared contract validates tangent shapes and real/complex domains and supplies
zeros for inactive dynamic inputs. Python records that select or reshape inputs
must carry both derivative methods; a custom pullback closure remains reverse-only.

The native residual is the source of both derivative directions. Adapters only
handle framework tracing, conversion and residual lifetime. Neither the shared
contract nor a framework bridge computes numerical derivatives or reconstructs
a Jacobian from repeated pullbacks.

## Guard records

Some checks compare metadata of two objects, such as the `k0` and medium of a
T-matrix and its illumination. Under `jax.jit` these values are traced, and their
numbers are available only inside the native callback. The check runs inside
a record, `Backend.guard`: it passes the value through, raises on a mismatch
and gives the compared metadata zero gradients. Its JVP passes through the
value's tangent, with no contribution from the checked metadata. The `Backend.apply` docstring
lists every guard.

## How long a context lives

| Framework | Primal and reverse mode | Forward mode |
|---|---|---|
| Advect | The tape owns the invocation's native context | Each tangent direction reuses that context |
| JAX | Built-ins retain numerical state or inputs under JAX ownership; unannotated custom records repeat the forward | Built-ins reuse computed state or reconstruct input-only contexts without reevaluating values |
| PyTorch | Keeps the context through the first backward and input snapshots for subsequent replay | `torch.func.jvp` and `forward_ad` borrow the context of the forward call |
| HIPS Autograd | Keeps and reuses the invocation's native context | `make_jvp` uses the context of the forward call |

JAX may skip or repeat a `pure_callback`, and its residuals must be arrays.
Built-in records retain numerical state or original inputs as arrays held by
JAX. Derivative callbacks reconstruct their context without repeating the
numerical forward. There is no global native-context registry.
Records without this support, including unannotated custom records, keep inputs
and repeat the native forward for every derivative application, including a
direct JVP.

The internal `SavedRecord` contract in `_saved.py` has one evaluation function
and three hooks: describe the state arrays from input shapes and fixed options,
save a context, and restore a context. `native_state` attaches this contract to
physics records; `native_record` and `primal_record` bind it to public `diff`
signatures, including `functools.partial`. Metadata belongs to the exact callable, so a wrapper that
changes its behavior cannot accidentally inherit the original derivative rules.
Records whose restore needs only saved state set `needs_primals=False`; JAX then
keeps input shapes and dtypes without retaining another copy of the inputs.
Shared output checks preserve this metadata lazily, so the other adapters do
not pay for resolving or copying a saved-state contract they do not use.

Contexts containing only inputs, such as Bessel functions, coordinates, lattice
sums and individual wave coefficients, retain those inputs and rebuild the derivative context without
evaluating the values again. Their ordinary forward and restoration paths share
the same argument preparation and native constructor.

Contexts containing computed numerical state implement the Rust `SavedState`
trait. Their byte layout is derived from dimensions and fixed options, with runtime branches normalized to
the same layout. The layout carries numerical data, never pointers or registry
handles. It is an internal transport format for one invocation, not a persistent
file format. Each record declares its restoration next to its definition:
either a context constructor for retained inputs, or a state codec and shape
rule for saved numerical work. Both share the same framework dispatch.

JAX's tangent boundary has an explicit transpose that calls the native VJP.
This preserves reverse mode without asking JAX to differentiate opaque host
callbacks. `jacfwd` evaluates tangent seeds sequentially; saved-state records
reuse one linearization across those seeds. `checkpoint` may deliberately
recompute it according to JAX's rematerialization policy.

PyTorch's JVP borrows its saved native context, so a backward through the
returned primal can reuse it. The first backward releases the context;
input snapshots let later backwards replay the record. HIPS Autograd keeps the
reusable context with the invocation. Advect's tape owns and releases its native
contexts. Advect supplies the same context to each JVP, so a direct JVP needs
one native forward and subsequent directions reuse it. The adapter requires
Advect 0.3.1 or later for this residual-aware JVP API.

These are first-order contracts. Differentiating a native tangent or adjoint
with respect to the original model inputs is unsupported. JAX can transpose a
linearization at fixed inputs; the [framework examples](../differentiation/frameworks.md#use-both-modes-together)
show how to combine both directions without taking second derivatives.
PyTorch supports `torch.func.jvp`; other `torch.func` transforms and
`torch.compile` remain unsupported.

## Costs of reusable contexts

Saved work still has a cost. JAX copies numerical state into arrays and decodes
it for each derivative callback. This saves expensive factorizations, but a tiny
operation can spend more time passing that state than recomputing it.

For example, a compiled gradient of a linear sum of a 2-by-2 solve takes two
callbacks here. The previous reverse-only bridge could discard its unused value
callback and perform the solve and pullback together in one callback. Asking for
the value too, or using a squared-output loss, keeps two callbacks in both
versions; the current bridge then performs one native solve instead of two.
These callback counts explain the tradeoff, not a universal timing prediction.
The [final regression measurements](https://github.com/yaugenst/treams-rs/blob/52a211c3b309cdcdb236c8791e84f114039251b2/benchmarks/forward-autodiff-final-20261005.json)
retain the small-operation slowdowns alongside the complete physics workflows.

Reusable contexts also keep their recorded matrices intact. Pullbacks allocate
gradient workspaces instead of overwriting those matrices. Thin cluster
illumination saves its useful response in place of the incident array, with no
increase in stored matrix count. S-matrix derivative solves that fall back from
iteration to a dense solve still refactor on each such call; that fallback is
not a cached-factor path.

## Static configuration

Bases, mode labels, polarization conventions and lattice
dimensions are Python values, not framework arrays. They decide array shapes, so
a traced value could not set them. Frequency, amplitudes, materials, positions,
radii, real plane-wave directions and lattice vectors can carry gradients. An S-matrix
keeps its plane-wave ports in one `PortSet`: ports built from a basis have fixed
transverse wavevectors, while diffraction-order ports follow the lattice and the
Bloch vector and carry gradients.
