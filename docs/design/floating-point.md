---
description: How native results stay independent of the flush-to-zero modes that JAX and PyTorch set, and how that is tested.
---

# Floating-point environment

Native results do not depend on the caller's floating-point mode. The rustdoc of
the Rust module
[`fpenv`](https://yaugenst.github.io/treams-rs/rust/treams_core/fpenv/index.html)
gives the implementation details.

## Flushing breaks native results

Nonzero numbers smaller than `2.2e-308` in magnitude are subnormal. Some callers
flush them to zero:

- XLA sets flush-to-zero (FTZ) and denormals-are-zero (DAZ) on the thread that runs
  a computation, also while a `jax.pure_callback` runs;
- `torch.set_flush_denormal(True)` sets them on the calling thread;
- Linux threads inherit the mode of the thread that creates them.

Algorithms that scale through the subnormal range then fail. faer computes the
complex reciprocal of an LU pivot `z` with `max(|Re z|, |Im z|) <= 1 < |z|` through
`f64::MIN_POSITIVE / |z|^2`, which is subnormal. A flushing caller gets zero for
the solve of the 1 × 1 system `(1 + 0.1i) x = 1`.

## The guard

`fpenv::ieee` runs a closure with subnormals on the calling thread and then
restores the caller's flushing bits exactly, also after an error or a panic. It
writes MXCSR on x86-64 (FTZ, bit 15, and DAZ, bit 6) and FPCR on AArch64 (FZ,
bit 24, and FIZ, bit 0). On other targets it only runs the closure. Every Python
entry point of the bindings, including each NumPy ufunc loop, runs its whole body
inside `fpenv::ieee`.

The compiler does not know about the control register, so it could move
arithmetic across the two register writes. Each write is an inline-assembly
block that takes a pointer as an operand: the clearing write a pointer to the
closure, the restoring write a pointer to the result. For the compiler, the first
write may change every captured input and the second may read the result. So all
arithmetic on the inputs starts after the first write and ends before the second.
The rustdoc of `fpenv` explains why this holds and what it does not cover.

## The Rayon pool starts lazily

The global Rayon pool starts in the first Rayon call of the process, and its
workers keep the mode of the thread that starts them for good. The bindings make
that first call inside the guard, so the workers keep subnormals. Starting the
pool eagerly would spawn threads in calls that never use Rayon, and a child
process forked after that would hang in its first parallel call.

## Limits

The guard covers native work only. What runs before it stays in the caller's
mode:

- PyO3 converts float64 and complex128 arguments as bit copies, but reads a
  float32 subnormal (a NumPy float32 scalar or cotangent) as zero on a flushing
  thread.
- The Python layer computes some float64 values with NumPy before the native call,
  and they flush. The wave vector of
  `tr.plane_wave([3e-310, 0, 1], [1, 0], k0=1)` is an example.
- Inside the guard, an operation whose operands are all compile-time constants may
  run outside the window. Guarded code does not rely on subnormal results of
  constant expressions.

## How it is tested

| Test | What it checks |
|---|---|
| `guarded_register_arithmetic_stays_inside` in `fpenv` | Fails at opt-level 1 when either pointer operand is removed |
| [`test_float_environment.py`](../../tests/bindings/test_float_environment.py) | Every binding body is one guard call; the first native call on a flushing thread keeps subnormals; forked children keep a working pool |
| [`float_environment.py`](../../scripts/float_environment.py), run by the test above and by `just check-wheel` | Every scalar binding, every ufunc loop, records with their pullbacks, solves and a slab give the same bits on a flushing thread as on an IEEE thread |
| `test_callbacks_keep_subnormals_and_restore_the_xla_mode` in [`test_jax.py`](../../tests/autodiff/test_jax.py) | Native calls inside JAX callbacks keep subnormals and leave XLA's mode as they found it |

The script draws operands until each binding and loop with a float or complex
operand has a call whose result flushing would change. Array bindings of
T-matrices, fields, EBCM, cylinders, illuminations and channels are not compared.
