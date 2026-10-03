---
description: How many CPU threads treams-rs uses, how to change it, and how it works with process pools, threadpoolctl and other thread pools.
---

# Threads and process pools

treams-rs runs its parallel kernels and its dense linear algebra on one pool of
threads that it owns. Nothing needs configuring: the pool uses the CPUs this
process may use, it starts with the first parallel call, process pools work,
and results do not depend on the number of threads.

```python exec
import numpy as np
import treams_rs as tr

info = tr.thread_info()
z = np.linspace(0.1, 5.0, 4096) + 0.05j
reference = tr.special.spherical_jn(2, z)
with tr.threads(1):  # process-wide, restored on exit
    assert tr.get_num_threads() == 1
    serial = tr.special.spherical_jn(2, z)
assert tr.get_num_threads() == info["threads"]
assert np.array_equal(serial, reference)  # bit for bit
```

## Changing the number of threads

| Control | Effect |
|---|---|
| `tr.set_num_threads(n)` | Budget of every later parallel call in this process. A parallel step that has started finishes with its threads. `None` restores the default. |
| `with tr.threads(n):` | The same, restored when the block ends. The budget is process-wide, not per Python thread. |
| `tr.get_num_threads()`, `tr.thread_info()` | The budget, where it came from, the CPUs available, the pool size, and settings that were ignored. Neither starts the pool. |
| `TREAMS_RS_NUM_THREADS` | Default budget for treams-rs alone. |
| `RAYON_NUM_THREADS` | Default when `TREAMS_RS_NUM_THREADS` is unset. |
| `OMP_NUM_THREADS` | Default when both are unset; the first entry of a list such as `8,4`. |

The environment is read once, when Python imports `treams_rs`. Later changes to
it have no effect; call `tr.set_num_threads` instead. Empty and zero values mean
"not set". A value that is not a positive integer is ignored with a
`tr.parallel.ThreadingWarning` at import. Without a setting, the budget is the
number of CPUs this process may use, which follows `taskset`, CPU affinity and
container CPU limits. `tr.set_num_threads` accepts more threads than CPUs with a
`ThreadingWarning`: dense linear algebra slows down when threads outnumber CPUs.

`thread_info()["diagnostics"]` lists ignored values, budgets above the CPU count,
and an `OMP_NUM_THREADS` that limits treams-rs. Launchers such as `torchrun` and
some cluster environments export `OMP_NUM_THREADS=1`, which makes treams-rs run
on one thread; set `TREAMS_RS_NUM_THREADS` to override it.

## Process pools

`multiprocessing` with any start method, `concurrent.futures.ProcessPoolExecutor`
and PyTorch `DataLoader` workers work, also after the parent has computed in
parallel: a forked child starts a pool of its own with the parent's budget. Each
of `p` worker processes should get about `cpus / p` threads:

```python no-exec
import os
from concurrent.futures import ProcessPoolExecutor

import treams_rs as tr

workers = 4
threads = max(1, len(os.sched_getaffinity(0)) // workers)
with ProcessPoolExecutor(
    workers, initializer=tr.set_num_threads, initargs=(threads,)
) as pool:
    results = list(pool.map(compute, cases))
```

joblib's process backend sets `OMP_NUM_THREADS` in its workers, which treams-rs
reads, so each worker uses its share. Native objects such as a factorized
cluster belong to the process that made them; compute them in each worker.

## Other thread pools

NumPy's BLAS, JAX and PyTorch run their own thread pools next to treams-rs. When
[threadpoolctl](https://github.com/joblib/threadpoolctl) is installed,
`threadpool_limits` limits treams-rs as well, and `threadpool_info()` lists it
with `user_api` `"treams"`:

```python no-exec
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=1):  # BLAS, OpenMP and treams-rs
    response = cluster.solve()
```

## Reproducibility

For a given build and processor, every value and gradient repeats bit for bit:
at any number of threads, from run to run, and in a forked child. A parallel step
splits its work at boundaries that depend on the problem size only and adds
partial results in a fixed order; [parallelism](../design/parallelism.md)
explains how. Different processors or builds can differ in the last bits,
because matrix products choose AVX2 or AVX-512 kernels when they run and libm
comes from the system.

## Memory

Each thread that multiplies matrices keeps a packing buffer of up to twice the
processor's last-level cache. treams-rs also keeps the pool of the previous
budget, so alternating between two budgets does not restart threads; the buffers
of both pools count.
