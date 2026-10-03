//! Serial or Rayon-parallel evaluation of independent items. treams-rs extension.
//!
//! Four forms decide when work runs on the Rayon pool:
//!
//! - [`Parallel`] is the policy of the elementwise evaluations in
//!   [`broadcast`](super::broadcast): in parallel from a number of elements
//!   ([`Parallel::AtLeast`]), or in about four chunks per thread for cheap, uniform
//!   elements ([`Parallel::Chunked`]).
//! - [`try_map`] and [`try_fill_chunks`] take `parallel: bool`, which callers compute
//!   from a size and [`PARALLEL_ITEMS`] or [`PARALLEL_ENTRIES`].
//! - LU factorizations and solves pass faer a worker count computed from the matrix
//!   size (`linalg::lu_threads`).
//! - The S-matrix illumination in `smatrix::illuminate` compares a row count with a
//!   constant and calls Rayon directly.
//!
//! The first two keep every output in index order, so the thread count never changes
//! their results.
//!
//! | Module | Threshold | Value | Runs in parallel |
//! |---|---|---|---|
//! | `numerics::parallel` | `PARALLEL_ITEMS` | 1024 items | plane-wave permutations (`pw::permute`), chirality densities (`smatrix::chirality`) |
//! | `numerics::parallel` | `PARALLEL_ENTRIES` | 4096 entries | plane-wave phases and fields, and their pullbacks (`pw::field`) |
//! | `special::bessel` | `PARALLEL` | `AtLeast(64)` | Bessel arrays and their pullbacks |
//! | `special::legendre` | `PARALLEL` | `AtLeast(1024)` | angular-function arrays and their pullbacks |
//! | `special::wigner` | `PARALLEL` | `AtLeast(1024)` | Wigner D arrays and their pullbacks |
//! | `special::integrals` | `PARALLEL` | `AtLeast(1024)` | incomplete gamma and Kambe arrays and their pullbacks |
//! | `vectorwaves` | `PARALLEL` | `AtLeast(1024)` | vector-wave arrays and their pullbacks |
//! | `lattice::batch` | `PARALLEL` | `AtLeast(8)` | batched lattice sums and their pullbacks, per output |
//! | `sw::polar` | `PARALLEL` | `AtLeast(1024)` | spherical translation arrays and their pullbacks |
//! | `sw::polar` | `PLANS` | `AtLeast(64)` | coupling plans, per distinct mode pair |
//! | `cw::polar` | literal | `Chunked(512)` regular, `Chunked(64)` singular | cylindrical translation arrays |
//! | `cw::polar` | literal | `Chunked(64)` | their pullback |
//! | `cw::expansion` | `PARALLEL` | `AtLeast(64)` | expansion coefficients, per distinct coupling |
//! | `smatrix::illuminate` | `PARALLEL_SCAN_ROWS` | 256 rows | the finite check of illumination inputs |
//! | `smatrix::illuminate` | `PARALLEL_SNAPSHOT_ROWS` | 128 rows | copies of the recorded blocks, with more than one thread |
//! | `smatrix::illuminate` | `WORKER_SNAPSHOT_ROWS` | 512 rows | copies of the recorded blocks on the worker, with one thread |
//! | `linalg` | `lu_threads` | `max(min(rows / 512, 4), rows / 2048)` workers, at most `columns / 16`; the whole pool up to 4 threads | LU factorizations and solves (faer) |
//!
//! The ufunc loops of the bindings use their own tiers (`parallel_from` in the
//! `treams-py` file `ufunc/kinds.rs`). Other Rayon calls fill matrix columns or map
//! independent items at any size, also in index order.
//!
//! These pullbacks reduce with Rayon's `try_fold` and `try_reduce` instead of the
//! helpers:
//!
//! | Module | Pullback |
//! |---|---|
//! | `channels` | `SphericalChannelsResidual::pullback`, `CylindricalChannelsResidual::pullback` |
//! | `cluster::iterative` | `IterativeSphereCluster::pair_gradients` |
//! | `cw::to_sw` | `ToSwResidual::pullback` |
//! | `fields` | `WaveSet::pullback` |
//! | `pw::expand` | `ExpansionResidual::pullback` |
//! | `pw::field` | `FieldResidual::pullback` |
//! | `smatrix::layers` | `LayerStackResidual::pullback` |
//! | `sw::periodic_to_cw` | `PeriodicToCwResidual::pullback` |
//! | `sw::plan` | `TranslationPlan::pullback_periodic` |
//!
//! Rayon splits this work adaptively, by thread count and work stealing, and adds the
//! partial sums along those splits. The order of those additions sets the last bits of
//! the gradients. A shared helper would split differently, so these pullbacks keep
//! their own `try_fold` and `try_reduce`.

use rayon::prelude::*;

/// Items (modes, columns) from which independent per-item work runs in parallel.
pub(crate) const PARALLEL_ITEMS: usize = 1024;

/// Matrix entries from which independent per-entry work runs in parallel.
pub(crate) const PARALLEL_ENTRIES: usize = 4096;

/// How an elementwise evaluation uses the rayon pool.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Parallel {
    /// Split adaptively from this many elements.
    AtLeast(usize),
    /// For cheap, uniform elements: from this many elements and with more than one
    /// thread, split into about four chunks per thread to bound scheduling overhead.
    Chunked(usize),
}

impl Parallel {
    /// The minimum chunk length of a parallel evaluation of `size` elements, or
    /// `None` when they run in order on the calling thread.
    ///
    /// Only a chunked evaluation that reaches its threshold reads the thread
    /// count: reading it starts the global pool, whose worker threads would
    /// otherwise outlive an evaluation that never uses them.
    pub(super) fn chunk(self, size: usize) -> Option<usize> {
        match self {
            Self::AtLeast(threshold) if size >= threshold => Some(1),
            Self::Chunked(threshold) if size >= threshold => {
                let threads = rayon::current_num_threads();
                (threads > 1).then(|| (size / (4 * threads)).max(1))
            }
            Self::AtLeast(_) | Self::Chunked(_) => None,
        }
    }
}

/// Fill `data` chunk by chunk with `fill(index, chunk)`, stopping at the first error.
pub(crate) fn try_fill_chunks<T: Send, E: Send>(
    data: &mut [T],
    size: usize,
    parallel: bool,
    fill: impl Fn(usize, &mut [T]) -> Result<(), E> + Sync + Send,
) -> Result<(), E> {
    if parallel {
        data.par_chunks_mut(size)
            .enumerate()
            .try_for_each(|(i, chunk)| fill(i, chunk))
    } else {
        data.chunks_mut(size)
            .enumerate()
            .try_for_each(|(i, chunk)| fill(i, chunk))
    }
}

/// `map` over `0..count`, collected in index order and stopping at the first error.
pub(crate) fn try_map<T: Send, E: Send>(
    count: usize,
    parallel: bool,
    map: impl Fn(usize) -> Result<T, E> + Sync + Send,
) -> Result<Vec<T>, E> {
    if parallel {
        (0..count).into_par_iter().map(map).collect()
    } else {
        (0..count).map(map).collect()
    }
}
