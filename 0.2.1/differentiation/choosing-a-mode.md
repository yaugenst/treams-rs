# Choosing a differentiation mode

Start with the question you want to answer. If two physical inputs change,
and you want to know what happens across a whole field map, use forward mode.
If many particle dimensions or positions change, and you want to improve one
number such as a scattering cross section, use reverse mode.

Both compute the same first derivatives. They differ in how the work grows:
forward mode follows input changes through the calculation, while reverse
mode starts with an output of interest and works back to the inputs.

| Physical question | Result you need | Good starting choice |
| --- | --- | --- |
| How does every field value change when the common sphere radius or the illumination wavelength changes? | Two derivative maps, keeping amplitude and phase | Forward mode: `jax.jacfwd` |
| Which sphere radii and positions increase the scattering cross section? | One derivative for every radius and coordinate | Reverse mode: `jax.grad` or `jax.value_and_grad` |

## Two inputs, a whole field map

The [field sensitivity example](../examples/forward_field_sensitivity.md)
illuminates three spheres and samples their scattered electric field on a
5 × 5 grid. The complete map has 75 complex values: three components at
each point. Its two inputs are the common sphere radius and `k0`, which
sets the wavelength.

```python
derivatives = jax.jacfwd(field_map)(parameters)
```

This returns shape `(5, 5, 3, 2)`. The last axis selects the input parameter;
every other axis matches the field map. A matrix or array containing every
output derivative with respect to every input is called a **Jacobian**.
Forward mode needs only two input directions here. Reverse mode would need
to ask separately about all 150 real and imaginary field values to recover
the same information.

If you need only one particular combined change, such as all radii growing
by 1 nm while the frequency stays fixed, `jax.jvp` computes that one
direction. It can be cheaper than requesting every derivative. It answers
a smaller question, so the timing comparison below asks both modes for the
complete Jacobian.

## Many inputs, one objective

The [cluster gradient example](../examples/reverse_cluster_gradient.md)
illuminates eight spheres and asks for their total scattering cross section.
Each sphere has a radius and three center coordinates: 32 real inputs in
all, with one real output.

```python
cross_section, gradient = jax.value_and_grad(scattering)(parameters)
```

The gradient has the same `(8, 4)` shape as the parameters. A positive radius
entry says that a small radius increase raises the cross section, with the
other inputs held fixed. Position entries describe small translations along
x, y and z. Reverse mode obtains all these entries from one output; forward
mode would follow 32 input directions to produce the same full gradient.

The examples use the ordinary sphere, cluster and plane-wave APIs. JAX is
selected by the traced inputs. [Framework adapters](frameworks.md) shows the
corresponding forward- and reverse-mode operations for PyTorch, Advect and
HIPS Autograd.

## Measured examples

These are warm times for the complete calculation and its derivatives,
measured on 2026-10-04 on an AMD Ryzen 9 9950X with four CPU threads,
Python 3.13 and JAX 0.11.2 in double precision. The native extension used a
release build of the unreleased forward-mode implementation, before the
subsequent simplification pass. The archived source fingerprints identify
that measured version. Each entry is the median of seven samples; both
modes return the same complete derivative array.

| Calculation | Forward mode | Reverse mode | Faster choice |
| --- | ---: | ---: | --- |
| Three spheres, 5 × 5 field map: 2 inputs → 150 real outputs | 3.77 ms | 138.58 ms | Forward, 36.8× |
| Eight spheres, one cross section: 32 inputs → 1 output | 77.38 ms | 4.47 ms | Reverse, 17.3× |

A second run gave 3.78 vs 138.38 ms for the field map, and 76.91 vs 4.43 ms
for the cross section. The derivatives agreed between modes to better than
`5e-16` relative error; the independent finite-difference checks agreed to
better than `3e-10`.

Preparation was separate: tracing and compilation took 0.18 s forward and
0.19 s reverse for the field map, and 0.22 s forward and 0.12 s reverse for
the cross section. The warm times include a fresh scattering calculation on
every call. Numerical factors are shared within that call, not reused from
an earlier call.

The size of each derivative request explains the result: two input directions
are much cheaper than 150 output directions, while one output direction is
much cheaper than 32 input directions. These numbers describe these JAX
workflows on this machine; they are not fixed speedups for every problem or
framework. The following script lets you repeat or resize both comparisons.
The recorded samples and source fingerprints are in
`benchmarks/forward-reverse-20261004.json` in the repository.

## Compare equal results on your machine

After installing the checkout with its JAX extra, run:

```bash
uv run --script scripts/benchmark_forward_reverse.py
```

The script reuses the exact calculations from both examples. For the field
map it compares `jax.jacfwd` with `jax.jacrev`, representing each complex
field value by its real and imaginary parts so that both return the same
real array. For the cross section it compares `jax.jacfwd` with `jax.grad`.
It checks equality of the entire result and checks a joint parameter
perturbation against central differences before measuring anything.

For each mode it prints two separate times:

- **Prepare:** JAX traces the calculation and compiles it for these array shapes.
- **Warm median:** the median time for later calls, including the scattering
  calculation and its derivatives. The result is ready before the clock stops.

The default uses four CPU threads, seven samples of at least 40 ms each,
a 5 × 5 field grid and eight spheres. The two modes alternate within each
case. Run on an otherwise idle machine; to fix the CPU cores on Linux, prefix
the command with `taskset -c 8-11`, replacing the core numbers with suitable
ones for your machine. The JSON output under
`benchmarks/results/local/forward-reverse/` records the samples, compilation
times, dimensions, accuracy checks, versions and source fingerprints.

Use `--check-only` to check the two derivative results without measuring
time, or `--grid-points` and `--particles` to change the sizes. The comparison
includes the full derivative arrays; it does not compare a single direction
with a full gradient. Actual timing depends on problem size and framework
overhead, so use the measured results to choose for your calculation.

All these derivatives hold the mode cutoff, particle count and sampling
grid fixed. They are first-order sensitivities at the chosen parameters;
the examples also need ordinary physical convergence checks before being
used for a design decision.
