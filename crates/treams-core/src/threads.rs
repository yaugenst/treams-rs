//! The thread budget and the Rayon pool that runs every parallel region. treams-rs
//! extension.
//!
//! Parallel regions and parallel dense linear algebra run inside [`install`], on a
//! pool owned by this module, never on Rayon's global pool. The global pool cannot
//! be resized once started, and a process forked after it started inherits the pool
//! without its worker threads, so its first parallel call would wait forever. The
//! owned pool records the process that built it: a forked child builds its own pool
//! on first use and never touches the parent's pool, locks or threads.
//!
//! Each worker clears the flushing bits of its floating-point control register when
//! it starts (`fpenv::keep_subnormals_on_worker`), so the workers keep subnormal
//! numbers whatever the mode of the thread that builds the pool; [`crate::fpenv`]
//! explains why the kernels need them.
//!
//! The thread budget comes from the first of [`set_num_threads`],
//! `TREAMS_RS_NUM_THREADS`, `RAYON_NUM_THREADS` and `OMP_NUM_THREADS` that
//! holds a positive integer, and otherwise from the CPUs this process may use
//! (affinity and cgroup quota). Environment variables are read once, on first
//! use. Empty and zero values mean "not set"; other unparsable values are
//! ignored and reported in [`Budget::diagnostics`].
//!
//! Calls already on a Rayon worker (a nested region, or a Rust caller's own
//! pool) run in place on that worker's pool.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, PoisonError, RwLock};

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Environment variables consulted for the default budget, in precedence order.
pub const ENVIRONMENT: [&str; 3] = [
    "TREAMS_RS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
];

/// Where the active thread budget came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// [`set_num_threads`].
    Requested,
    /// The named entry of [`ENVIRONMENT`].
    Environment(&'static str),
    /// [`std::thread::available_parallelism`].
    Available,
}

impl Source {
    /// Stable name reported to users.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Requested => "set_num_threads",
            Self::Environment(name) => name,
            Self::Available => "available_parallelism",
        }
    }
}

/// A resolved thread budget.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Budget {
    /// Number of worker threads parallel regions may use.
    pub threads: usize,
    /// Where `threads` came from.
    pub source: Source,
    /// CPUs this process may use.
    pub available: usize,
    /// Ignored environment values and oversubscription notes, in reading order.
    pub diagnostics: Vec<String>,
}

/// Live state of the pool, for diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    /// The budget the next parallel region will use.
    pub budget: Budget,
    /// Workers in the current pool; `None` until a parallel region has run in
    /// this process. Smaller than the budget only if threads could not be spawned.
    pub pool_threads: Option<usize>,
    /// Whether this process was forked after treams-rs was loaded; it then
    /// builds a pool of its own.
    pub forked: bool,
}

/// Run `op` on the treams-rs pool, or in place when already on a Rayon worker.
///
/// Every Rayon parallel iterator, `rayon::join`, and faer call with
/// `Par::rayon` in this crate runs inside `install`, so that none of them
/// starts Rayon's global pool.
pub fn install<R: Send>(op: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_some() {
        return op();
    }
    match pool() {
        Some(pool) => pool.install(op),
        // No owned worker could be spawned. Guard fallback work on this caller
        // and any workers Rayon can still start.
        None => crate::fpenv::ieee(op),
    }
}

/// [`rayon::join`] on the treams-rs pool.
pub fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    install(|| rayon::join(a, b))
}

/// Workers available to the calling code: the current Rayon pool's size on a
/// worker, otherwise the treams-rs budget. Never starts a pool.
#[must_use]
pub fn current_num_threads() -> usize {
    if rayon::current_thread_index().is_some() {
        rayon::current_num_threads()
    } else {
        num_threads()
    }
}

/// The budget parallel regions will use. Never starts a pool.
#[must_use]
pub fn num_threads() -> usize {
    match REQUESTED.load(Ordering::Acquire) {
        0 => default_budget().threads,
        threads => threads,
    }
}

/// Override the budget for this process.
///
/// `None` or `Some(0)` restores the environment/default resolution. A parallel
/// region that has started finishes on its pool; later regions use the new
/// budget. Budgets are capped at [`rayon::max_num_threads`].
pub fn set_num_threads(threads: Option<usize>) {
    let threads = threads.map_or(0, |threads| threads.min(rayon::max_num_threads()));
    REQUESTED.store(threads, Ordering::Release);
}

/// Record that this process was just forked; call it only in the child.
///
/// The pid check already detects a forked child; this also covers a reused pid
/// and a fork that interrupted another thread's first use of the pool.
pub fn after_fork() {
    GENERATION.fetch_add(1, Ordering::AcqRel);
}

/// The active budget with its source and diagnostics. Never starts a pool.
#[must_use]
pub fn budget() -> Budget {
    let default = default_budget();
    match REQUESTED.load(Ordering::Acquire) {
        0 => default.clone(),
        threads => {
            let mut diagnostics = default.diagnostics.clone();
            if threads > default.available {
                diagnostics.push(oversubscription(
                    "set_num_threads",
                    threads,
                    default.available,
                ));
            }
            Budget {
                threads,
                source: Source::Requested,
                available: default.available,
                diagnostics,
            }
        }
    }
}

/// Budget and pool state. Never starts a pool or claims a slot.
#[must_use]
pub fn info() -> Info {
    let pid = std::process::id();
    let generation = GENERATION.load(Ordering::Acquire);
    let existing = SLOTS.get(generation).and_then(OnceLock::get);
    let pool_threads = existing.filter(|slot| slot.pid == pid).and_then(|slot| {
        let pools = slot.pools.read().unwrap_or_else(PoisonError::into_inner);
        pools
            .current
            .as_ref()
            .map(|built| built.pool.current_num_threads())
    });
    Info {
        budget: budget(),
        pool_threads,
        forked: generation > 0 || existing.is_some_and(|slot| slot.pid != pid),
    }
}

/// Faer parallelism for a dense operation that may use `threads` workers.
/// The operation must run inside [`install`] unless this returns `Par::Seq`.
pub(crate) fn faer(threads: usize) -> faer::Par {
    match threads {
        0 | 1 => faer::Par::Seq,
        threads => faer::Par::rayon(threads),
    }
}

/// Run a dense operation with `threads` faer workers: inline when sequential,
/// otherwise on the treams-rs pool.
pub(crate) fn dense<R: Send>(threads: usize, op: impl FnOnce(faer::Par) -> R + Send) -> R {
    match faer(threads) {
        faer::Par::Seq => op(faer::Par::Seq),
        par @ faer::Par::Rayon(_) => install(|| op(par)),
    }
}

/// Faer multiplies complex128 matrices sequentially below this M·N·K
/// (`PAR_THRESHOLD_MNK` times the scalar size). Require at least this much
/// work per worker as well, so small matrices do not occupy the whole pool.
const SEQUENTIAL_PRODUCT: usize = 4096 * 16;

/// Run an `m × k` by `k × n` faer product with the treams-rs budget.
///
/// Matrix products split their output among workers and give the same bits at
/// every budget. A matrix-vector product (`m` or `n` equal to one) splits its inner
/// dimension instead, which changes the last bits with the worker count; it is
/// bound by memory bandwidth and stays sequential.
pub(crate) fn product<R: Send>(
    m: usize,
    n: usize,
    k: usize,
    op: impl FnOnce(faer::Par) -> R + Send,
) -> R {
    let work = m.saturating_mul(n).saturating_mul(k);
    let threads = if work < SEQUENTIAL_PRODUCT || m == 1 || n == 1 {
        1
    } else {
        current_num_threads().min(work / SEQUENTIAL_PRODUCT)
    };
    dense(threads, op)
}

// Forked children advance the generation, so this bounds nested forks. Past
// it, every parallel region builds a pool of its own.
const GENERATIONS: usize = 64;

/// Override from `set_num_threads`; zero means unset. Atomics stay readable
/// after fork, unlike a lock another thread may have held.
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
static GENERATION: AtomicUsize = AtomicUsize::new(0);
static DEFAULT: OnceLock<Budget> = OnceLock::new();
static SLOTS: [OnceLock<Slot>; GENERATIONS] = [const { OnceLock::new() }; GENERATIONS];

struct Slot {
    pid: u32,
    pools: RwLock<Pools>,
}

// The current pool plus one spare, so alternating between two budgets (for
// example a scoped one-thread section) does not respawn threads every time.
#[derive(Default)]
struct Pools {
    current: Option<Built>,
    spare: Option<Built>,
}

struct Built {
    requested: usize,
    pool: Arc<ThreadPool>,
}

fn default_budget() -> &'static Budget {
    if let Some(budget) = DEFAULT.get() {
        return budget;
    }
    // Resolve outside the cell so a fork can only interrupt a trivial store.
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    let budget = resolve(|name| std::env::var(name).ok(), available);
    DEFAULT.get_or_init(move || budget)
}

/// This process's slot. A forked child moves to a fresh slot instead of using
/// (or locking, or dropping) the parent's.
fn slot() -> Option<&'static Slot> {
    let pid = std::process::id();
    loop {
        let generation = GENERATION.load(Ordering::Acquire);
        let slot = SLOTS.get(generation)?.get_or_init(|| Slot {
            pid,
            pools: RwLock::new(Pools::default()),
        });
        if slot.pid == pid {
            return Some(slot);
        }
        let _ = GENERATION.compare_exchange(
            generation,
            generation + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

fn pool() -> Option<Arc<ThreadPool>> {
    #[cfg(test)]
    if UNAVAILABLE_POOL.get() {
        return None;
    }
    let requested = num_threads();
    let Some(slot) = slot() else {
        // Past GENERATIONS nested forks: a pool for this region only.
        return build(requested);
    };
    let current = slot
        .pools
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .get(requested);
    if current.is_some() {
        return current;
    }
    let promoted = slot
        .pools
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .promote(requested);
    if promoted.is_some() {
        return promoted;
    }
    // Spawn threads without holding the lock; the first pool installed wins.
    let pool = build(requested)?;
    Some(
        slot.pools
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(requested, pool),
    )
}

#[cfg(test)]
thread_local! {
    static UNAVAILABLE_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Exercise the failed-pool fallback without exhausting the host's threads.
#[cfg(test)]
pub(crate) fn with_unavailable_pool<T>(work: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            UNAVAILABLE_POOL.set(self.0);
        }
    }
    let _restore = Restore(UNAVAILABLE_POOL.replace(true));
    work()
}

impl Pools {
    fn get(&self, requested: usize) -> Option<Arc<ThreadPool>> {
        self.current
            .as_ref()
            .filter(|built| built.requested == requested)
            .map(|built| Arc::clone(&built.pool))
    }

    /// The pool for `requested`, swapping in the spare when it matches.
    fn promote(&mut self, requested: usize) -> Option<Arc<ThreadPool>> {
        if self
            .spare
            .as_ref()
            .is_some_and(|built| built.requested == requested)
        {
            std::mem::swap(&mut self.current, &mut self.spare);
        }
        self.get(requested)
    }

    /// Make `pool` current unless another thread installed one meanwhile.
    fn insert(&mut self, requested: usize, pool: Arc<ThreadPool>) -> Arc<ThreadPool> {
        if let Some(existing) = self.promote(requested) {
            return existing;
        }
        // Dropping the old spare ends its threads once running regions finish.
        self.spare = self.current.replace(Built {
            requested,
            pool: Arc::clone(&pool),
        });
        pool
    }
}

/// Build a pool, halving the size while threads cannot be spawned.
fn build(requested: usize) -> Option<Arc<ThreadPool>> {
    let mut threads = requested.max(1);
    loop {
        let result = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("treams-{index}"))
            .start_handler(|_| crate::fpenv::keep_subnormals_on_worker())
            .build();
        match result {
            Ok(pool) => return Some(Arc::new(pool)),
            Err(_) if threads > 1 => threads /= 2,
            Err(_) => return None,
        }
    }
}

/// Resolve the default budget from `lookup` (an environment) and the CPUs available.
pub(crate) fn resolve(lookup: impl Fn(&str) -> Option<String>, available: usize) -> Budget {
    let available = available.max(1);
    let mut diagnostics = Vec::new();
    for name in ENVIRONMENT {
        let Some(raw) = lookup(name) else {
            continue;
        };
        match parse(name, &raw) {
            Ok(None) => {}
            Ok(Some(threads)) => {
                if threads > available {
                    diagnostics.push(oversubscription(name, threads, available));
                } else if name == "OMP_NUM_THREADS" && threads < available {
                    diagnostics.push(format!(
                        "OMP_NUM_THREADS limits treams-rs to {threads} of {available} CPUs; \
                         set TREAMS_RS_NUM_THREADS or call set_num_threads to override"
                    ));
                }
                return Budget {
                    threads,
                    source: Source::Environment(name),
                    available,
                    diagnostics,
                };
            }
            Err(message) => diagnostics.push(message),
        }
    }
    Budget {
        threads: available,
        source: Source::Available,
        available,
        diagnostics,
    }
}

/// A positive thread count, `None` for empty or zero, or a diagnostic.
fn parse(name: &str, raw: &str) -> Result<Option<usize>, String> {
    let mut value = raw.trim();
    if name == "OMP_NUM_THREADS" {
        // OpenMP lists one count per nesting level; the outermost applies here.
        value = value.split(',').next().unwrap_or_default().trim();
    }
    if value.is_empty() {
        return Ok(None);
    }
    match value.parse::<usize>() {
        Ok(0) => Ok(None),
        Ok(threads) => Ok(Some(threads.min(rayon::max_num_threads()))),
        Err(_) => Err(format!(
            "ignored {name}={raw:?}: expected a positive integer"
        )),
    }
}

fn oversubscription(name: &str, threads: usize, available: usize) -> String {
    format!(
        "{name} requests {threads} threads but this process may use {available} CPUs; \
         oversubscription slows dense linear algebra"
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // proptest retains failing environments.
    use super::*;
    use proptest::prelude::*;
    use std::collections::HashMap;

    fn environment(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        move |name| map.get(name).cloned()
    }

    fn value() -> impl Strategy<Value = Option<String>> {
        prop_oneof![
            Just(None),
            (0_usize..300).prop_map(|n| Some(n.to_string())),
            (1_usize..300).prop_map(|n| Some(format!(" {n}\t"))),
            (1_usize..300, 1_usize..9).prop_map(|(n, m)| Some(format!("{n},{m}"))),
            "[a-z.+-]{0,4}".prop_map(Some),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(crate::test_support::DEFAULT_CASES))]

        #[test]
        fn resolution_takes_the_first_valid_source_in_precedence_order(
            values in proptest::collection::vec(value(), 3),
            available in 1_usize..512,
        ) {
            let pairs: Vec<(&str, &str)> = ENVIRONMENT
                .iter()
                .zip(&values)
                .filter_map(|(name, value)| value.as_deref().map(|v| (*name, v)))
                .collect();
            let budget = resolve(environment(&pairs), available);
            prop_assert!(budget.threads >= 1);
            prop_assert_eq!(budget.available, available);
            let first = ENVIRONMENT.iter().zip(&values).find_map(|(name, value)| {
                parse(name, value.as_deref()?).ok().flatten().map(|threads| (*name, threads))
            });
            if let Some((name, threads)) = first {
                prop_assert_eq!(budget.source, Source::Environment(name));
                prop_assert_eq!(budget.threads, threads);
                prop_assert_eq!(
                    budget.diagnostics.iter().any(|d| d.contains("oversubscription")),
                    threads > available
                );
            } else {
                prop_assert_eq!(budget.source, Source::Available);
                prop_assert_eq!(budget.threads, available);
            }
            // Every ignored, unparsable value before the chosen source is reported.
            for (name, value) in ENVIRONMENT.iter().zip(&values) {
                if let Some(value) = value
                    && parse(name, value).is_err()
                    && first.is_none_or(|(chosen, _)| {
                        ENVIRONMENT.iter().position(|n| n == name)
                            < ENVIRONMENT.iter().position(|n| *n == chosen)
                    })
                {
                    prop_assert!(budget.diagnostics.iter().any(|d| d.contains(name)));
                }
            }
        }
    }

    #[test]
    fn parsing_accepts_only_positive_integers() {
        assert_eq!(parse("RAYON_NUM_THREADS", " 3 "), Ok(Some(3)));
        assert_eq!(parse("RAYON_NUM_THREADS", "0"), Ok(None));
        assert_eq!(parse("RAYON_NUM_THREADS", ""), Ok(None));
        assert_eq!(parse("OMP_NUM_THREADS", "8,4"), Ok(Some(8)));
        for invalid in ["two", "-1", "2.0", "4,2"] {
            assert!(parse("RAYON_NUM_THREADS", invalid).is_err(), "{invalid}");
        }
        assert!(parse("OMP_NUM_THREADS", "x,4").is_err());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // Matrix indices are below 512.
    fn matrix_products_keep_their_bits_when_worker_counts_are_capped() {
        use crate::{Complex, linalg};
        use nalgebra::DMatrix;

        let shapes = [
            (32, 64, 32),
            (64, 64, 64),
            (128, 128, 128),
            (128, 4, 256),
            (4, 128, 512),
            (128, 1, 512),
            (1, 128, 512),
        ];
        let matrices = shapes.map(|(m, n, k)| {
            let left = DMatrix::from_fn(m, k, |i, j| {
                Complex::new(((i + 3 * j) as f64).sin(), ((2 * i + j) as f64).cos())
            });
            let right = DMatrix::from_fn(k, n, |i, j| {
                Complex::new(((5 * i + j) as f64).cos(), ((i + 2 * j) as f64).sin())
            });
            (left, right)
        });
        let mut reference = None;
        // Local pools leave the process-wide override to its existing test.
        for threads in [1, 4, 16, 32] {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .start_handler(|_| crate::fpenv::keep_subnormals_on_worker())
                .build()
                .unwrap();
            let bits = pool.install(|| {
                matrices
                    .iter()
                    .flat_map(|(left, right)| {
                        linalg::product(left, right)
                            .iter()
                            .map(|value| (value.re.to_bits(), value.im.to_bits()))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
            assert_eq!(&bits, reference.get_or_insert_with(|| bits.clone()));
        }
    }

    // Restores the process-wide override even if an assertion fails.
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            set_num_threads(None);
        }
    }

    // One test owns the process-wide override, so parallel tests elsewhere only
    // ever observe a valid budget. Other tests may swap the current pool
    // concurrently, so only what this test's own regions observe is asserted.
    #[test]
    fn install_runs_on_a_named_pool_sized_by_the_budget() {
        let _reset = Reset;
        for threads in [2, 3, 2, 1] {
            set_num_threads(Some(threads));
            assert_eq!(num_threads(), threads);
            let (size, name, nested) = install(|| {
                (
                    rayon::current_num_threads(),
                    std::thread::current().name().map(str::to_owned),
                    install(rayon::current_num_threads),
                )
            });
            assert_eq!((size, nested), (threads, threads));
            assert!(name.unwrap().starts_with("treams-"));
            assert_eq!(budget().source, Source::Requested);
            assert!(info().pool_threads.is_some());
        }
        set_num_threads(None);
        assert_eq!(num_threads(), default_budget().threads);
        let (a, b) = join(rayon::current_num_threads, rayon::current_num_threads);
        assert_eq!((a, b), (num_threads(), num_threads()));
    }
}
