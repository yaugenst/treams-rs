//! Serial or Rayon-parallel evaluation of independent items. treams-rs extension.
//!
//! Four forms decide when work runs on the Rayon pool:
//!
//! - [`Parallel`] is the policy of the elementwise evaluations in
//!   [`broadcast`](super::broadcast): in parallel from a number of elements
//!   ([`Parallel::AtLeast`]), or in about four chunks per thread for cheap, uniform
//!   elements ([`Parallel::Chunked`]).
//! - [`try_map`], [`try_fill_chunks`] and [`try_fold_ordered`] take `parallel: bool`,
//!   which callers compute from a size and [`PARALLEL_ITEMS`] or [`PARALLEL_ENTRIES`],
//!   or set for items that are expensive on their own.
//! - LU factorizations and solves pass faer a worker count computed from the matrix
//!   size (`linalg::lu_threads`).
//! - The S-matrix illumination in `smatrix::illuminate` compares a row count with a
//!   constant and calls Rayon directly.
//!
//! [`Parallel`], [`try_map`] and [`try_fill_chunks`] keep every output in index order,
//! and [`try_fold_ordered`] adds in an order set by the number of items, so the thread
//! count never changes their results.
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
//! # Reductions
//!
//! Pullbacks that add gradients over many items reduce with [`try_fold_ordered`]. It
//! splits the items into at most [`REDUCTION_CHUNKS`] consecutive chunks, with
//! boundaries set by the number of items alone, folds each chunk in index order and
//! adds the chunk results in chunk order, on the pool or on the calling thread alike,
//! so the thread count sets how many chunks run at once but never the order of the
//! additions. A gradient that belongs to one item, such as the `q` gradient of one
//! column, travels with that item as a mutable slot and is written in place; a
//! partial sum holds only the gradients that all items share.
//!
//! | Module | Pullback | Items | Written in place |
//! |---|---|---|---|
//! | `channels` | `SphericalChannelsResidual::pullback`, `CylindricalChannelsResidual::pullback` | plane-mode columns | `q` gradients |
//! | `cluster::iterative` | `IterativeSphereCluster::pair_gradients` | particles `i`, each with its pairs `(i, j)` | - |
//! | `cw::to_sw` | `ToSwResidual::pullback` | cylindrical source modes | - |
//! | `fields` | `WaveSet::pullback` | samples | point gradients |
//! | `pw::expand` | `ExpansionResidual::pullback` | plane modes | wavevector gradients |
//! | `pw::field` | `FieldResidual::pullback` | at most 32 blocks of points or of modes, whichever axis is longer | the gradients of that axis |
//! | `smatrix::layers` | `LayerStackResidual::pullback` | transverse wavevectors | `q` gradients |
//! | `sw::periodic_to_cw` | `PeriodicToCwResidual::pullback` | spherical source modes | - |
//! | `sw::plan` | `TranslationPlan::pullback_periodic` | harmonics, whose gradients are collected and then added in order | - |
//!
//! Each item is expensive on its own (a column of entries, the pair blocks of a
//! particle, a layer stack or a lattice sum), so these pullbacks pass `parallel: true`,
//! except `pw::field`, which runs its blocks in parallel from [`PARALLEL_ENTRIES`] as
//! its forward does. Its blocks are those of `pw::PhasesResidual::pullback`, which maps
//! them with [`try_map`] and adds the partial gradients in block order.

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
    /// `None` when they run in order on the calling thread: below the threshold,
    /// and with a one-thread budget. Reading the budget never starts the pool.
    pub(super) fn chunk(self, size: usize) -> Option<usize> {
        let threads = crate::threads::current_num_threads();
        match self {
            Self::AtLeast(threshold) if size >= threshold && threads > 1 => Some(1),
            Self::Chunked(threshold) if size >= threshold && threads > 1 => {
                Some((size / threads.saturating_mul(4)).max(1))
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
    if parallel && crate::threads::current_num_threads() > 1 {
        crate::threads::install(|| {
            data.par_chunks_mut(size)
                .enumerate()
                .try_for_each(|(i, chunk)| fill(i, chunk))
        })
    } else {
        data.chunks_mut(size)
            .enumerate()
            .try_for_each(|(i, chunk)| fill(i, chunk))
    }
}

/// Chunks of a deterministic reduction: [`try_fold_ordered`] splits its items into at
/// most this many consecutive chunks, which bounds both the parallelism of a
/// reduction and the partial results alive at once.
pub(crate) const REDUCTION_CHUNKS: usize = 64;

/// Fold `items` into one result whose bits do not depend on the thread count.
///
/// The items are split into at most [`REDUCTION_CHUNKS`] consecutive chunks, with
/// boundaries set by the number of items alone. Each chunk folds its items in index
/// order from `identity()`, with `fold(partial, index, item)`, and the chunk results
/// are combined in chunk order with `combine(left, right)`, starting from the first
/// chunk's result. The calling thread runs the same chunks when `parallel` is false
/// or the budget is one thread, so every schedule adds the same partial results in
/// the same order; one chunk reproduces a plain sequential fold.
///
/// The error is that of the first failing item in index order, as in a sequential
/// loop.
pub(crate) fn try_fold_ordered<T: Send, A: Send, E: Send>(
    items: Vec<T>,
    parallel: bool,
    identity: impl Fn() -> A + Sync + Send,
    fold: impl Fn(A, usize, T) -> Result<A, E> + Sync + Send,
    combine: impl Fn(A, A) -> A + Sync + Send,
) -> Result<A, E> {
    let chunk = items.len().div_ceil(REDUCTION_CHUNKS).max(1);
    let mut chunks: Vec<Vec<(usize, T)>> = Vec::with_capacity(items.len().div_ceil(chunk));
    let mut indexed = items.into_iter().enumerate().peekable();
    while indexed.peek().is_some() {
        chunks.push(indexed.by_ref().take(chunk).collect());
    }
    let fold_chunk = |chunk: Vec<(usize, T)>| {
        chunk
            .into_iter()
            .try_fold(identity(), |partial, (index, item)| {
                fold(partial, index, item)
            })
    };
    if parallel && chunks.len() > 1 && crate::threads::current_num_threads() > 1 {
        // ponytail: up to 64 partial gradients stay live; combine fixed batches
        // if their memory becomes the limiting cost.
        let partials: Vec<Result<A, E>> =
            crate::threads::install(|| chunks.into_par_iter().map(fold_chunk).collect());
        let mut partials = partials.into_iter();
        let first = partials.next().unwrap_or_else(|| Ok(identity()))?;
        partials.try_fold(first, |total, partial| Ok(combine(total, partial?)))
    } else {
        let mut chunks = chunks.into_iter();
        let first = chunks.next().map_or_else(|| Ok(identity()), fold_chunk)?;
        chunks.try_fold(first, |total, chunk| Ok(combine(total, fold_chunk(chunk)?)))
    }
}

/// `map` over `0..count`, collected in index order and stopping at the first error.
pub(crate) fn try_map<T: Send, E: Send>(
    count: usize,
    parallel: bool,
    map: impl Fn(usize) -> Result<T, E> + Sync + Send,
) -> Result<Vec<T>, E> {
    if parallel && crate::threads::current_num_threads() > 1 {
        crate::threads::install(|| (0..count).into_par_iter().map(map).collect())
    } else {
        (0..count).map(map).collect()
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::try_fold_ordered;
    use crate::test_support::DEFAULT_CASES;

    /// `items` summed with `try_fold_ordered` on a pool of `threads` workers.
    fn sum(items: &[f64], threads: usize, parallel: bool) -> Result<f64, usize> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|_| usize::MAX)?;
        pool.install(|| {
            try_fold_ordered(
                items.to_vec(),
                parallel,
                || 0.0,
                |total, index, item| {
                    if item.is_nan() {
                        Err(index)
                    } else {
                        Ok(total + item)
                    }
                },
                |left, right| left + right,
            )
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

        /// Sums whose rounding depends on the order of additions repeat bit for bit
        /// on every pool size and on the calling thread.
        #[test]
        fn ordered_folds_do_not_depend_on_the_thread_count(
            items in prop::collection::vec(
                (-1e3_f64..1e3, -20_i32..20).prop_map(|(x, e)| x * 10_f64.powi(e)),
                0..600,
            ),
        ) {
            let reference = sum(&items, 1, false).unwrap_or_default().to_bits();
            for threads in [1, 2, 3, 4] {
                for parallel in [false, true] {
                    let total = sum(&items, threads, parallel).unwrap_or_default();
                    prop_assert_eq!(total.to_bits(), reference, "{} threads", threads);
                }
            }
            // A single chunk is a plain sequential fold.
            if items.len() <= super::REDUCTION_CHUNKS {
                let sequential = items.iter().fold(0.0, |total, &item| total + item);
                prop_assert_eq!(reference, sequential.to_bits());
            }
        }

        /// The error is that of the first failing item, as in a sequential loop.
        #[test]
        fn ordered_folds_report_the_first_failing_item(
            len in 1_usize..600,
            failing in prop::collection::btree_set(0_usize..600, 1..4),
        ) {
            let failing: Vec<_> = failing.into_iter().filter(|&i| i < len).collect();
            let mut items = vec![1.0; len];
            for &i in &failing {
                items[i] = f64::NAN;
            }
            for threads in [1, 3] {
                let expected = failing
                    .first()
                    .map_or_else(|| Ok(f64::from(u16::try_from(len).unwrap_or_default())), |&i| Err(i));
                prop_assert_eq!(sum(&items, threads, true), expected);
            }
        }
    }
}
