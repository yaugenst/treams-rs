# Numerical choices

The Rust core makes these choices for accuracy or speed.
[Numerical limits](../validation/numerical-limits.md) reports the measured accuracy.

## Equilibrated LU

Small particles at high multipole orders give interaction matrices with a unit
diagonal and off-diagonal entries above `1e20`. A plain LU of such a matrix loses
digits even when a rescaled system is well conditioned. The
[thermal-radiation reproduction](../validation/published-applications.md#thermal-radiation)
has such a system: four 250 nm SiC spheres at multipole order 10. Rescaling
makes its absorption at the lowest frequency stable.

`linalg::equilibrate` therefore rescales rows and columns before the LU, so the
LU factors `R A C` with positive diagonal scales `R` and `C`. It acts only when
the entries span more than a factor of 10, some diagonal entry is below a tenth
of the largest entry, and some row or column maximum is too; LAPACK uses the same
ratio of 0.1. Other matrices stay untouched. It scales the axis
with the larger spread first, so that normalizing large rows cannot underflow a
small column.

The forward solve and the adjoint solve use the same scales, in reverse order.
The scales change the coordinates of the solve, not its solution, so the pullback
differentiates the original system. The Lean model in
[formal proofs](formal-proofs.md) proves that both scaled solves are exact for any
nonzero scales.

## LU worker count

faer's recursive LU and triangular solves split the work into many narrow
panels. On a large Rayon pool, scheduling these panels can outweigh the benefit
of more workers. The private function `lu_threads` in `linalg` therefore picks the
number of faer workers from the matrix size:

```text
workers = max(1, min(budget, columns / 16, max(min(rows / 512, 4), rows / 2048)))
```

All divisions round down. `budget` is the treams-rs thread budget; pools of four
or fewer workers use the whole budget. The factorization passes
`columns = rows`; forward and adjoint solves pass the number of right-hand
sides. Fewer than 32 right-hand sides always run on one worker: faer's parallel
triangular solves split the inner dimension by the worker count, which would
change the last bits of the solution. Below 64 rows with at most 64 right-hand
sides, where faer takes no parallel branch, the call stays on the calling
thread. The worker count changes the scheduling only, not the factorization or
the pullback, so LU results are the same at every budget.

The [scheduling probe](https://github.com/yaugenst/treams-rs/blob/07843dcae969c29433886bade8c5b10b3a14e5de/benchmarks/results/cpu-parallelism.json) measured
these medians on a Ryzen 9950X with one 16-worker pool pinned to physical cores
0-15:

| Complex128 rows / right-hand sides | Workers (factor, solve) | All 16 workers | Bounded workers | Speedup |
|---|---|---:|---:|---:|
| 256 / 4 | 1, 1 | 47.56 ms | 0.57 ms | 83.9× |
| 1024 / 64 | 2, 2 | 527.15 ms | 15.85 ms | 33.3× |
| 2048 / 512 | 4, 4 | 974.03 ms | 152.22 ms | 6.4× |
| 4096 / 64 | 4, 4 | 2725.67 ms | 483.54 ms | 5.6× |
| 8192 / 64 | 4, 4 | 6528.76 ms | 3279.13 ms | 2.0× |

Times are factor plus solve, with one warm-up and three samples per case in
Cargo's default release profile. They measure scheduling only, not complete
calculations. At 8192 rows the factorization took 3.14 s with four workers,
3.28 s with eight and 5.03 s with sixteen. Sizes above 8192 rows were not timed;
there the worker count grows with the estimated panel work.

To repeat a measurement, run the benchmark with matrix rows, right-hand sides,
requested faer workers and repetitions, and compare 1, 2, 4, 8 and 16 workers on
the same CPU affinity. The benchmark calls faer directly on Rayon's global pool,
which `RAYON_NUM_THREADS` sizes:

```sh
RAYON_NUM_THREADS=16 cargo bench -p treams-core --bench lu_scheduling -- 1024 64 2 5
```

## Wigner 3j symbols

`special::wigner3j` uses the two-sided recurrence in `j3` of treams (Schulten and
Gordon). The upward recurrence is stable while the symbols grow from the lower
end; the downward recurrence is stable while they grow from the upper end. treams
switches direction at a fixed `j3`, a quarter of the way through the range. For
extreme orders a region of growing symbols can reach past that point, and
recurring into it loses all accuracy: errors reach 1e-2 below `j = 90` and a
factor of 1800 at `j = 260`.

treams-rs continues each direction for as long as the symbols grow, and turns at
the fixed point of treams otherwise. Elsewhere it equals treams up to rounding.
Whether a `j3` is reached upward or downward does not depend on which `j3` values
are requested, so a whole row of symbols and a single symbol agree bit for bit.

## Ewald sums

A lattice sum of outgoing waves converges slowly. The Ewald method splits it into
a real-space part and a reciprocal-space part that both converge fast; the split
parameter `eta` sets how the work divides. `lattice::sum` chooses and stops as
follows:

- `eta = 0` selects `eta = sqrt(2 pi) / (k L) max(|k L| / 8, 1)` with the cell
  length `L = measure^(1/dim)`. The exact sum does not depend on the split, and derivatives hold it
  fixed.
- Each part adds shells of integer cells until two consecutive shells add less than
  `2e-13 max(|P|, 1)` of the part `P`, and never before the far terms peak.
- A part fails after 200, 32 or 16 shells in one, two or three dimensions, or up
  to four times as many where a large `|k| L` or a small `|k eta|` needs them.
- A sum that would be inaccurate raises an error instead of returning a number.

[Lattice sums](../validation/numerical-limits.md#lattice-sums) lists every failure
message and its remedy; the rustdoc of
[`lattice::sum`](../rust/treams_core/lattice/fn.sum.html)
states the same rules.

## Complex branches

- **Normal wavenumber.** `pw::wave_vector_z` (Python `misc.wave_vec_z`) returns
  `kz = sqrt(k^2 - kx^2 - ky^2)` with `Im kz >= 0`, the wave that decays or
  propagates away, as treams does.
- **Square roots** take the principal branch and keep the sign of a zero
  imaginary part, so a value on the branch cut lands on the side its sign selects.
- **Lattice sums.** Spherical Ewald sums take the root of `k^2` with `Re k > 0`.
  Every diffraction order takes `k_q = sqrt(k^2 - q^2)` with `Im k_q >= 0`. At a
  lattice point, the self term takes the sheet that makes the sum the limit of
  the shifted sums. [Branches and sheets](../validation/numerical-limits.md#branches-and-sheets)
  gives the consequences for `Re k < 0` and `Im k < 0`.

## Subnormal Bessel orders

`special::bessel` evaluates a subnormal order, such as `1e-310`, as order 0. The
AMOS-based kernels lose accuracy at subnormal orders, and SciPy returns NaN for
some of them. At order 0 the result is the same to double precision: even at
extreme finite arguments, `order * log(z)` is far below the smallest correction
a double can hold.

## EBCM surface element

`ebcm::qmat` integrates over a body of revolution with radius `r(θ)`. Its
surface element is `r sin θ (r r̂ - r' θ̂) dθ dφ`. treams' `ebcm.qmat` uses
`sin θ (r r̂ - r' θ̂)` and omits the leading factor `r`. For a smooth lossless
shape this leaves a unitarity error of about 1e-3 at degrees 2 to 6, against
6e-6 to 1e-8 with the factor (see
[behavioral defects](../coming-from-treams/differences.md#behavioral-defects)).
`radial_area_factor=False` in `ebcm.qmat` and `diff.ebcm_qmat` reproduces treams.

## Parallel thresholds

Small inputs run on the calling thread because starting Rayon tasks costs more
than the work. A one-thread budget also keeps work on the calling thread. Each
kernel sets the size from which it runs in parallel:

| Rust item | Parallel from | Used by |
|---|---|---|
| `numerics::parallel::PARALLEL_ITEMS` | 1024 modes or columns | plane-wave permutations, chirality densities |
| `numerics::parallel::PARALLEL_ENTRIES` | 4096 matrix entries | plane-wave fields |
| `numerics::parallel::Parallel::AtLeast` | 8 to 1024 elements, per kernel | broadcast special functions, vector waves, translations, lattice-sum batches |
| `numerics::parallel::Parallel::Chunked` | 64 or 512 elements | cylindrical translation coefficients, in about four chunks per thread |
| `threads::product` | At least 65536 M·N·K per worker, capped by the configured budget | matrix products; fewer than two useful workers and all matrix-vector products stay sequential |
| `linalg::DECOMPOSITION` | never | eigen- and singular-value decompositions ([parallelism](parallelism.md#results-do-not-depend-on-the-thread-count)) |

Each element is computed the same way on either path and collected in index
order, so a vectorized ufunc call equals its per-element calls bit for bit.

Pullbacks that add gradients over many items, such as the columns of a
conversion matrix or the samples of a field, split the items into at most 64
chunks set by the number of items (`numerics::parallel::try_fold_ordered`) and
add the chunk sums in chunk order; the plane-wave field and phase pullbacks
split their longer axis into at most 32 blocks set by the shape. The thread
count decides how many chunks run at once, never the order of the additions,
so these gradients are the same bit for bit at every thread count.
