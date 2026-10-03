//! The thread budget of the treams-rs pool, for `treams_rs.parallel`: `thread_info`,
//! `set_num_threads` and the fork hook `after_fork` (`treams_core::threads`).
use pyo3::{prelude::*, types::PyDict};
use treams_core::{fpenv::ieee, threads};

/// The budget, its source and the pool state: `threads::info`. Never starts the pool.
#[pyfunction]
pub(crate) fn thread_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    ieee(|| {
        let info = threads::info();
        let dict = PyDict::new(py);
        dict.set_item("threads", info.budget.threads)?;
        dict.set_item("source", info.budget.source.name())?;
        dict.set_item("available", info.budget.available)?;
        dict.set_item("pool_threads", info.pool_threads)?;
        dict.set_item("forked", info.forked)?;
        dict.set_item("diagnostics", info.budget.diagnostics)?;
        dict.set_item("environment", threads::ENVIRONMENT.to_vec())?;
        Ok(dict)
    })
}

/// Override the process budget; `None` restores the default: `threads::set_num_threads`.
#[pyfunction]
pub(crate) fn set_num_threads(threads: Option<usize>) {
    ieee(|| threads::set_num_threads(threads));
}

/// Move a forked child to a fresh pool slot: `threads::after_fork`. Python calls it
/// through `os.register_at_fork`.
#[pyfunction]
pub(crate) fn after_fork() {
    ieee(|| {
        threads::after_fork();
    });
}

/// The budget, under a C name that only this library exports. `treams_rs.parallel`
/// registers a threadpoolctl controller that recognizes the extension module by
/// this symbol, so `threadpool_limits` also limits treams-rs.
// An unmangled symbol is `unsafe_code` only because two such symbols could clash;
// the `treams_rs_` prefix keeps this one unique.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub(crate) extern "C" fn treams_rs_num_threads() -> usize {
    threads::num_threads()
}
