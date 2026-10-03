---
description: The treams-rs thread pool, reproducible results and remaining work.
---

# Parallelism

Every parallel step of treams-rs runs on one pool of worker threads that the
crate owns (`treams_core::threads`), sized by one budget.
[Threads and process pools](../guide/threads.md) shows how to use it.

## Why treams-rs owns its pool

Rayon's global pool, which treams-rs used before, has three problems for a
Python library:

- **Forked children hang.** A process forked after the global pool started
  inherits the pool without its worker threads, so the child's first parallel
  call waits forever. `multiprocessing` forks by default on Linux before Python
  3.14, and so do `ProcessPoolExecutor` and PyTorch `DataLoader` workers.
- **The size is fixed at the first parallel call** and reads only
  `RAYON_NUM_THREADS`, which silently means "every CPU" for values such as
  `two` or `-1`.
- **Each extension module has its own copy.** Another Rayon-based extension in
  the same process starts its own pool, which `RAYON_NUM_THREADS` sizes too.

## Design

- **Decision points.** A kernel decides whether a step runs in parallel; only
  then does it hand the step to the pool through `threads::install` (or
  `threads::join`, `threads::dense` and `threads::product` for faer). Steps that
  stay serial never touch the pool, so small calls avoid scheduling overhead.
  With a one-thread budget the helpers of `numerics::parallel` run everything
  on the calling thread. A step already on a worker runs in place.
- **Budget.** `threads::set_num_threads`, then `TREAMS_RS_NUM_THREADS`,
  `RAYON_NUM_THREADS` and `OMP_NUM_THREADS`, then
  `std::thread::available_parallelism`. Environment values are read once and
  validated; budgets are capped at Rayon's limit of 65535 threads.
- **Fork safety.** Pools live in per-process slots that record the process id.
  A forked child, which `os.register_at_fork` also reports, moves to a fresh
  slot and never locks, uses or drops the parent's pool, whose threads do not
  exist in the child. After 64 nested forks, every step builds a pool of its
  own.
- **Pools.** The current pool and the pool of the previous budget stay alive,
  so `with tr.threads(1):` does not restart threads. Pools are built outside the
  slot lock; threads are named `treams-<i>`. If threads cannot be spawned, the
  size halves down to one.
- **Floating point.** Each worker clears flush-to-zero and denormals-are-zero
  once, as it starts, so the workers keep subnormals whatever the mode of the
  thread that builds the pool ([floating-point environment](floating-point.md)).
- **faer.** treams-rs passes faer an explicit `Par` from the budget for every
  product, LU factorization and solve, `Par::Seq` for eigen- and singular-value
  decompositions, and never reads `faer::get_global_parallelism`.
- **threadpoolctl.** The extension exports the C symbol `treams_rs_num_threads`,
  by which a controller that `treams_rs.parallel` registers finds the module.

## Results do not depend on the thread count

Floating-point addition is not associative, so a sum depends on the order of its
terms. Every parallel step therefore fixes that order independently of the
number of threads:

- **Elementwise work** (special functions, fields, translation and expansion
  matrices) computes each output on its own and stores it in index order.
- **Reductions** (pullbacks that add contributions of many samples, modes or
  channels) use `numerics::parallel::try_fold_ordered`. It splits the items into
  at most 64 consecutive chunks whose boundaries depend on the item count only,
  folds each chunk in index order, and adds the chunk results in chunk order.
  The calling thread runs the same chunks with a one-thread budget, so one
  thread and many threads give the same bits. Per-item gradients are written in
  place, so a chunk carries only the shared sums.
- **Matrix products** split their output among workers and give the same bits
  at every worker count. A matrix-vector product would split its inner
  dimension, so it stays sequential.
- **LU factorizations** give the same bits at every worker count. Solves with
  fewer than 32 right-hand sides stay on one worker, because faer's parallel
  triangular solves split the inner dimension.
- **Eigen- and singular-value decompositions** run on one thread. faer's
  parallel Hessenberg and bidiagonal reductions split inner products by the
  worker count, which changed the last bits from n = 64 and the order of the
  eigenvalues. On four workers they gained at most 1.3× (eigenvalues) and 1.7×
  (singular values) at n = 768, and nothing below n = 384. Eigenvalues come in
  the solver's order, the same at every thread count.
- **NumPy ufunc loops** run serially when an input overlaps the output other
  than element for element, as in `reduce` and `accumulate`, which need each
  result before the next.

`tests/bindings/test_thread_pool.py` checks values and gradients bit for bit at
one, two and three threads. Different processors or builds can still differ in
the last bits: faer chooses AVX2 or AVX-512 kernels at run time, and libm comes
from the system.

## Rules for new code

- Run every Rayon parallel iterator, `rayon::join` and faer call with a parallel
  `Par` inside `threads::install`, `threads::join`, `threads::dense` or
  `threads::product`. `tests/bindings/test_thread_pool.py` rejects other
  parallel code in `crates/`, and `tests/conftest.py` fails the session if any
  test started Rayon's global pool. Operators on faer matrices read faer's
  global parallelism; use `linalg::product_into` instead.
- Never read `rayon::current_num_threads` outside a worker: it starts the global
  pool. `threads::current_num_threads` reads the budget.
- Add partial sums with `try_fold_ordered`, never with Rayon's `reduce`,
  `try_reduce`, `fold` or `sum`, whose splits follow the thread count.

## Audit and open work

An audit of the previous baseline (`5dadf1d`, a revision of the original
history; [`history-provenance.json`](../../benchmarks/history-provenance.json)
maps it to this repository) covered controls, process hazards, scheduling,
reproducibility, the bindings, hardware, validation and memory on a shared
4-vCPU Linux host. The shared host limits the timing evidence.
The following changes address the audit's findings:

- the owned, fork-safe pool and the controls of `treams_rs.parallel`;
- results that repeat bit for bit at every thread count (above);
- ufunc `reduce` and `accumulate`, which were wrong with more than one thread;
- small dense algebra on the calling thread, and no reads of faer's global
  parallelism;
- eigen- and singular-value decompositions on one thread, so `diff.eig` returns
  its eigenvalues in the same order at every thread count;
- the threadpoolctl controller;
- `MemoryError` for refused dense output matrices, decomposition vectors and
  LU, SVD and eigenvalue workspaces. Input copies, matrix products, gradient
  buffers and other allocations can still abort when refused;
- the module keeps the GIL; free-threaded CPython is outside the release scope.

### Further work

The timings and estimates below belong to that earlier audit. These follow-up
investigations are outside the 0.1.0 release scope.

| Item | Evidence | Estimate |
|---|---|---|
| Parallel singular-value decompositions from n ≈ 384 whose splits do not follow the budget, for example a fixed worker count by size. | At n = 768, 403 ms on four workers against 693 ms on one; the bits changed with the worker count. | 2–3 d |
| A cost model for parallel cutoffs: items times cost per item instead of fixed element counts. One-thread helpers now execute serially on the calling thread. | The audit found 32 regions without a size cutoff; a two-point field call cost 24–75 µs against 9 µs for one point. | 3–4 d |
| Calibrate the LU worker cap on more machines, at both ends. | Pools of four or fewer use every worker from 64 rows; larger pools keep 768 rows serial. Tuned on one Ryzen 9950X. | 1–2 d + machines |
| Ufunc parallel path without the collect-then-store copy, and parallel strided inputs. | Peak RSS 153 MiB at one thread against 306 MiB at four. | 1.5–2 d |
| Parallel cluster assembly and pullbacks in `basis`, and the EBCM forward. | The assembly behind `Cluster.solve` did not scale from one to four threads. | 1.5–2.5 d |
| Translation plans: memory and construction time. | About lmax⁵ terms; 2.07 GiB peak at lmax 30. | 2–3 d |
| Conversions that hold the GIL. | An N = 4000 conversion held the GIL 0.2–0.56 s. | 2 d |
| Native panics as a typed Python error instead of `PanicException`, a `BaseException`. | | 1 d |
| Interruptible long calls (Ctrl-C waits for the native call to return). | SIGINT took effect 2–24 s later. | 2–3 d |
| Measure scaling on macOS, ARM and Windows, including the default thread budget. | The configured wheel checks cover Linux x86-64 and ARM, macOS and Windows; the audit measurements came from Linux, with most of 1300 cases at four threads. | 3–5 d + machines |
| Audit free-threaded CPython before declaring `gil_used = false`. | Free-threaded interpreters are outside the 0.1.0 support matrix. | 1–2 d |
