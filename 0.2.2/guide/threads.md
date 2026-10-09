# Threads and process pools

treams-rs uses one thread pool for parallel computations and dense linear
algebra. The pool starts with the first parallel call and, by default, uses
the CPUs available to the current process.

The configured count is a ceiling. Kernels use fewer workers when the work is
too small to repay scheduling costs; matrix-vector products stay sequential
to preserve the order of floating-point additions. Raising the limit therefore
does not guarantee a faster call. See the [CPU measurements](../performance/cpu-speedups.md)
for qualified workloads and [parallelism](../design/parallelism.md) for the policy.

```python
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
| `tr.set_num_threads(n)` | Thread limit for later parallel calls in this process. A step already running finishes with its previous limit. `None` restores the default. |
| `with tr.threads(n):` | The same, restored when the block ends. The limit applies to the whole process, not one Python thread. Blocks in different Python threads must not overlap: each restores the limit it saw on entry, in exit order. For a thread pool, call `tr.set_num_threads(k)` once before starting the workers. |
| `tr.get_num_threads()`, `tr.thread_info()` | The thread limit, where it came from, the CPUs available, the pool size, and settings that were ignored. Neither starts the pool. |
| `TREAMS_RS_NUM_THREADS` | Default thread limit for treams-rs alone. |
| `RAYON_NUM_THREADS` | Default when `TREAMS_RS_NUM_THREADS` is unset. |
| `OMP_NUM_THREADS` | Default when both are unset; the first entry of a list such as `8,4`. |

The environment is read once, when Python imports `treams_rs`. Later changes to
it have no effect; call `tr.set_num_threads` instead. Empty and zero values mean
"not set". A value that is not a positive integer is ignored with a
`tr.parallel.ThreadingWarning` at import. Without a setting, the limit is the
number of CPUs this process may use, which follows `taskset`, CPU affinity and
container CPU limits. `tr.set_num_threads` accepts more threads than CPUs with a
`ThreadingWarning`: dense linear algebra slows down when threads outnumber CPUs.

`thread_info()["diagnostics"]` lists ignored values, thread limits above the
CPU count, and an `OMP_NUM_THREADS` that limits treams-rs. Launchers such as
`torchrun` and some cluster environments export `OMP_NUM_THREADS=1`, which makes
treams-rs run on one thread; set `TREAMS_RS_NUM_THREADS` to override it.

## Process pools

`multiprocessing` with any start method, `concurrent.futures.ProcessPoolExecutor`
and PyTorch `DataLoader` workers work, also after the parent has computed in
parallel: a forked child starts a pool of its own with the parent's thread
limit. Each of `p` worker processes should get about `cpus / p` threads:

```python
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

```python
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
because matrix products use processor-specific instructions and some
mathematical functions come from the system.

## Memory

Each thread that multiplies matrices keeps a temporary buffer of up to twice the
processor's last-level cache. treams-rs also keeps the pool for the previous
thread limit, so alternating between two limits does not restart threads.
The buffers of both pools count towards memory use.
