//! IEEE 754 subnormals for the numerical work, whatever the caller's floating-point mode.
//! treams-rs extension.
//!
//! The kernels assume gradual underflow. Some callers flush subnormals to
//! zero: XLA sets flush-to-zero and denormals-are-zero on the thread that runs a
//! computation, also while a `jax.pure_callback` runs,
//! `torch.set_flush_denormal(True)` sets them on the calling thread, and Linux
//! threads inherit the mode of their creator. Flushing breaks algorithms that
//! scale through the subnormal range: faer's complex reciprocal of an LU pivot
//! `z` with `max(|Re z|, |Im z|) <= 1 < |z|` forms `f64::MIN_POSITIVE / |z|^2`,
//! so a flushed scalar solve of `1 + 0.1i` returns zero.
//!
//! [`ieee`] runs work with subnormals on the calling thread and then restores
//! the caller's flushing bits; every Python entry point of the bindings runs its
//! whole body in it. `flushing` does the opposite, for tests. What precedes
//! the body stays in the caller's mode: `PyO3` reads a `float32` subnormal as
//! zero on a flushing thread, and `float64` values that the Python layer
//! computes from subnormal inputs flush before the call into Rust.
//!
//! [`ieee`] guards the calling thread only. The workers of the thread pool
//! ([`crate::threads`]) clear their flushing bits once, as each one starts
//! (`keep_subnormals_on_worker`), so they keep subnormals whatever the mode of
//! the thread that builds the pool; Linux threads would otherwise inherit their
//! creator's mode for good. treams-rs never uses Rayon's global pool, whose
//! workers would keep the mode of the thread that started it.
//!
//! This is the only module that reads or writes the control register: MXCSR
//! on x86-64 (flush-to-zero, bit 15, and denormals-are-zero, bit 6) and FPCR on
//! `AArch64` (FZ, bit 24, which XLA sets, and FIZ, bit 0, defined by
//! `FEAT_AFP`). On other targets both functions only run their work.
//!
//! # Soundness
//!
//! Rust assumes the default floating-point environment everywhere, and its
//! standard library documents changing flush-to-zero or denormals-are-zero as
//! undefined behaviour even when the change is undone (`_mm_setcsr` in
//! `core::arch::x86_64`): the compiler does not model the control register, so
//! it may move floating-point arithmetic across the instructions that write it.
//! A flushing caller has already left that assumption when it calls Rust
//! code. [`ieee`] cannot formally remove the violation, but it keeps the work
//! between its two register writes by data dependence rather than by code
//! layout. Each write is an `asm!` block without `nomem`, `readonly` or `pure`,
//! which the Rust Reference treats like a call to an unknown function that may
//! read and write memory, and it takes a pointer operand:
//!
//! - The clearing write takes a pointer to the closure, so for the compiler it
//!   may read and replace every captured value. Arithmetic on the inputs can
//!   only start after it, whether they live in registers or in memory.
//! - The restoring write takes a pointer to the result, so for the compiler it
//!   reads the result and may replace it. The result is complete before the
//!   write, and code after it reads the result back instead of recomputing it.
//! - Accesses to memory visible outside the function, such as `NumPy` buffers,
//!   and calls with side effects stay on their side of either write.
//!
//! This relies on the specified meaning of `asm!` operands, which no
//! optimization may see through, not on [`core::hint::black_box`], which is
//! documented as best effort only. Operations whose operands are all
//! compile-time constants are not anchored: LLVM usually folds them with IEEE
//! rules, but one that it leaves to run time may run outside the window, so
//! guarded work must not rely on subnormal results of constant expressions.
//! `guarded_register_arithmetic_stays_inside` fails at opt-level 1 when either
//! pointer operand is removed, and `tests/bindings/test_float_environment.py`
//! compares every scalar binding and ufunc loop on flushing threads bit for bit
//! with callers that keep subnormals.

// The control register is reachable only through inline assembly, one
// instruction per block below.
#![allow(unsafe_code)]

use std::{mem, ptr};

use control::{FLUSH, Word};

#[cfg(target_arch = "x86_64")]
mod control {
    use std::arch::asm;

    /// The SSE control and status register, MXCSR, which scalar and vector
    /// `f64` arithmetic uses on x86-64.
    pub(super) type Word = u32;

    /// MXCSR flush-to-zero (bit 15) and denormals-are-zero (bit 6).
    pub(super) const FLUSH: Word = 1 << 15 | 1 << 6;

    /// The flushing bits XLA sets on the thread that runs a computation.
    pub(super) const XLA: Word = FLUSH;

    /// The current thread's MXCSR.
    pub(super) fn read() -> Word {
        let mut word: Word = 0;
        // SAFETY: `stmxcsr` stores the 4-byte MXCSR at the address of `word`,
        // which is valid, aligned and writable, and has no other effect. Every
        // x86-64 processor implements SSE.
        unsafe {
            asm!(
                "stmxcsr dword ptr [{}]",
                in(reg) &raw mut word,
                options(nostack, preserves_flags),
            );
        }
        word
    }

    /// Load `word` into the current thread's MXCSR. For the compiler, the write
    /// may also read and write the memory `data` points to.
    ///
    /// # Safety
    ///
    /// `word` differs from the current MXCSR only in [`FLUSH`] bits that the
    /// processor implements, for example the bits it had before.
    pub(super) unsafe fn write(word: Word, data: *mut ()) {
        // SAFETY: `ldmxcsr` loads MXCSR from the address of `word`, which is
        // valid and aligned. By the safety requirement, no reserved bit is set,
        // which would fault, and rounding, exception masks and status flags stay
        // as the thread had them. `data` appears only in a comment, so the
        // instruction does not access it; without `nomem`, the compiler must
        // assume it may (see the module's soundness section).
        unsafe {
            asm!(
                "ldmxcsr dword ptr [{word}]",
                "/* {data} */",
                word = in(reg) &raw const word,
                data = in(reg) data,
                options(nostack, preserves_flags),
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod control {
    use std::arch::asm;

    /// The floating-point control register, FPCR.
    pub(super) type Word = u64;

    /// FPCR flush-to-zero (FZ, bit 24) and flush-inputs-to-zero (FIZ, bit 0).
    pub(super) const FLUSH: Word = 1 << 24 | 1;

    /// The flushing bit XLA sets on the thread that runs a computation. FZ is
    /// part of every FPCR; FIZ is reserved without `FEAT_AFP`.
    pub(super) const XLA: Word = 1 << 24;

    /// The current thread's FPCR.
    pub(super) fn read() -> Word {
        let word: Word;
        // SAFETY: `mrs` copies FPCR, which is readable at EL0, into a general
        // register and has no other effect.
        unsafe {
            asm!("mrs {}, fpcr", out(reg) word, options(nomem, nostack, preserves_flags));
        }
        word
    }

    /// Write `word` to the current thread's FPCR. For the compiler, the write
    /// may also read and write the memory `data` points to.
    ///
    /// # Safety
    ///
    /// `word` differs from the current FPCR only in [`FLUSH`] bits that the
    /// processor implements, for example the bits it had before.
    pub(super) unsafe fn write(word: Word, data: *mut ()) {
        // SAFETY: `msr` writes FPCR, which is writable at EL0. By the safety
        // requirement, reserved bits keep their values, and rounding, trap enables
        // and the other controls stay as the thread had them. `data` appears only
        // in a comment, so the instruction does not access it; without `nomem`,
        // the compiler must assume it may (see the module's soundness section).
        unsafe {
            asm!(
                "msr fpcr, {word}",
                "/* {data} */",
                word = in(reg) word,
                data = in(reg) data,
                options(nostack, preserves_flags),
            );
        }
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod control {
    /// No control register is managed on this target.
    pub(super) type Word = u8;

    /// No flushing bits are managed on this target.
    pub(super) const FLUSH: Word = 0;

    /// No flushing bits are set for tests either.
    pub(super) const XLA: Word = 0;

    /// No flushing bits are set.
    pub(super) fn read() -> Word {
        0
    }

    /// Nothing to write.
    ///
    /// # Safety
    ///
    /// Always safe; the requirement matches the other targets.
    pub(super) unsafe fn write(_: Word, _: *mut ()) {}
}

/// Set the current thread's flushing bits to `flush & FLUSH`, keeping every
/// other bit, and return the flushing bits it had. A write, when needed, may
/// read and write `*data` for the compiler.
///
/// # Safety
///
/// The processor implements every bit of `flush & FLUSH`, as it does the
/// flushing bits a thread had before; clearing (`flush == 0`) is always valid.
unsafe fn swap_flush(flush: Word, data: *mut ()) -> Word {
    let word = control::read();
    let wanted = (word & !FLUSH) | (flush & FLUSH);
    if wanted != word {
        // SAFETY: `wanted` differs from the register only in FLUSH bits, and
        // by the safety requirement the processor implements those it sets.
        unsafe { control::write(wanted, data) };
    }
    word & FLUSH
}

/// Restores the flushing bits a thread had before [`with_flush`] changed them.
struct Restore {
    /// The caller's flushing bits.
    caller: Word,
    /// Whether the register still holds other bits than `caller`.
    changed: bool,
}

impl Restore {
    /// Write back the caller's bits once. For the compiler, the write may read
    /// and write `*data`.
    fn restore(&mut self, data: *mut ()) {
        if mem::take(&mut self.changed) {
            // SAFETY: `caller` holds bits read from this thread's register, the
            // value never leaving `with_flush`, so the processor implements them.
            unsafe { swap_flush(self.caller, data) };
        }
    }
}

impl Drop for Restore {
    /// Restore the caller's bits while unwinding from a panic of the work.
    fn drop(&mut self) {
        self.restore(ptr::null_mut());
    }
}

/// Run `work` with the flushing bits `flush & FLUSH` on the current thread,
/// then restore the caller's flushing bits, also while unwinding a panic.
///
/// The writes take pointers to `work` and to its result, which keeps the work
/// between them (see the module's soundness section).
///
/// # Safety
///
/// The processor implements every bit of `flush & FLUSH`.
#[inline]
unsafe fn with_flush<T>(flush: Word, work: impl FnOnce() -> T) -> T {
    let mut work = work;
    // SAFETY: by this function's safety requirement, the processor implements the
    // bits set here.
    let caller = unsafe { swap_flush(flush, (&raw mut work).cast()) };
    let mut restore = Restore {
        caller,
        changed: caller != flush & FLUSH,
    };
    let mut value = work();
    restore.restore((&raw mut value).cast());
    value
}

/// Run `work` with IEEE 754 subnormals on the current thread.
///
/// Clears flush-to-zero and denormals-are-zero (FZ and FIZ on `AArch64`) for
/// `work`, then restores the caller's flushing bits exactly, also while
/// unwinding a panic; other control bits and the status flags keep their
/// values. A caller that keeps subnormals pays one register read, one that
/// flushes a read and a write on each side. Work on compile-time constants only
/// is not anchored (see the module's soundness section).
#[inline]
pub fn ieee<T>(work: impl FnOnce() -> T) -> T {
    // SAFETY: clearing flushing bits is always valid.
    unsafe { with_flush(0, work) }
}

/// Clear the current thread's flushing bits for the rest of its life.
///
/// The thread pool of [`crate::threads`] calls this as each worker starts, before
/// the worker runs any work.
pub(crate) fn keep_subnormals_on_worker() {
    // SAFETY: clearing flushing bits is always valid. The write returns the
    // worker to the default environment that Rust assumes, before the worker
    // runs any floating-point work that the write could be reordered against.
    unsafe { swap_flush(0, ptr::null_mut()) };
}

/// Run `work` with the flushing bits XLA sets on the current thread, then
/// restore the caller's flushing bits, also while unwinding a panic.
///
/// For tests only: `work` runs as in a JAX callback, in an environment that
/// Rust does not support outside [`ieee`] (see the module's soundness section).
/// Pool workers keep subnormals regardless. On other targets `work` runs
/// unchanged.
#[doc(hidden)]
pub fn flushing<T>(work: impl FnOnce() -> T) -> T {
    // SAFETY: XLA's flushing bits are implemented: FZ is part of every AArch64
    // FPCR, and every x86-64 processor has flush-to-zero and denormals-are-zero
    // (only early 32-bit SSE2 processors lack the latter).
    unsafe { with_flush(control::XLA, work) }
}

#[cfg(test)]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod tests {
    use std::{hint::black_box, panic, process::Command};

    use nalgebra::DMatrix;
    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::*;
    use crate::{
        Complex, Error, linalg,
        numerics::parallel,
        test_support::{DEFAULT_CASES, complex},
    };

    use control::XLA;

    /// Caller modes that flush: flush-to-zero and denormals-are-zero alone, too.
    #[cfg(target_arch = "x86_64")]
    const CALLERS: [Word; 3] = [1 << 15, 1 << 6, XLA];
    /// Caller modes that flush. FIZ alone is left out: a processor without
    /// `FEAT_AFP` reserves its bit.
    #[cfg(target_arch = "aarch64")]
    const CALLERS: [Word; 1] = [XLA];

    /// MXCSR status flags, which arithmetic sets; the control bits are the rest.
    #[cfg(target_arch = "x86_64")]
    const STATUS: Word = 0x3f;
    /// FPCR holds no status flags.
    #[cfg(target_arch = "aarch64")]
    const STATUS: Word = 0;

    /// Pivots in the band that faer's reciprocal flushes to zero:
    /// `max(|Re z|, |Im z|) <= 1 < |z|`.
    const BAND: [Complex; 5] = [
        Complex::new(1.0, 0.1),
        Complex::new(0.9, 0.6),
        Complex::new(-0.3, -0.99),
        Complex::new(1.0, 1.0),
        Complex::new(-1.0, 0.5),
    ];

    /// Names the child test that its parent test runs in a fresh process.
    const CHILD: &str = "TREAMS_FPENV_CHILD";

    fn control_bits() -> Word {
        control::read() & !STATUS
    }

    /// Whether subnormal results and operands keep their values on this thread:
    /// flush-to-zero zeroes `MIN_POSITIVE / 2`, denormals-are-zero reads the
    /// smallest subnormal as zero.
    fn keeps_subnormals() -> bool {
        let half = black_box(f64::MIN_POSITIVE) * black_box(0.5);
        half.to_bits() == 1 << 51 && black_box(f64::from_bits(1)) > black_box(0.0)
    }

    /// Run `work` on this thread with the flushing bits `flush`, some of XLA's,
    /// as [`flushing`] does with all of them, then restore the thread's bits.
    fn flushing_caller<T>(flush: Word, work: impl FnOnce() -> T) -> T {
        assert_eq!(flush & !XLA, 0, "{flush:#x} sets only XLA's flushing bits");
        // SAFETY: XLA's flushing bits are implemented, as `flushing` states.
        unsafe { with_flush(flush, work) }
    }

    /// `x * two * half`, which is `x` for `two * half == 1` when subnormals are
    /// kept. For the subnormal `x = MIN_POSITIVE / 2`, denormals-are-zero reads
    /// `x` as zero and flush-to-zero zeroes the final product.
    fn scaled(x: f64, two: f64, half: f64) -> f64 {
        x * two * half
    }

    /// `1 / z` by an LU solve of the `1 x 1` system `z x = 1`.
    fn reciprocal(z: Complex) -> crate::Result<Complex> {
        let one = DMatrix::from_element(1, 1, Complex::ONE);
        Ok(linalg::solve(&DMatrix::from_element(1, 1, z), one)?.value()[(0, 0)])
    }

    fn bits(values: &[Complex]) -> Vec<[u64; 2]> {
        values
            .iter()
            .map(|z| [z.re.to_bits(), z.im.to_bits()])
            .collect()
    }

    #[test]
    fn the_guard_keeps_subnormals_and_restores_the_caller_mode() {
        assert!(keeps_subnormals());
        for flush in CALLERS {
            let (flushed, set, inside, after) = flushing_caller(flush, || {
                let set = control_bits();
                let inside = ieee(|| (keeps_subnormals(), control::read() & FLUSH));
                (!keeps_subnormals(), set, inside, control_bits())
            });
            assert!(flushed, "{flush:#x} must flush subnormal arithmetic");
            assert_eq!(set & FLUSH, flush);
            assert_eq!(inside, (true, 0), "{flush:#x}");
            assert_eq!(after, set, "{flush:#x}");
        }
        // No flushing bits: the guard leaves the register as it is.
        let before = control_bits();
        let inside = ieee(control_bits);
        assert_eq!((inside, control_bits()), (before, before));
    }

    #[test]
    fn flushing_sets_the_bits_of_xla_and_restores_the_caller_mode() {
        let before = control_bits();
        let (set, flushed, guarded, nested) = flushing(|| {
            let nested = flushing(control_bits);
            (
                control_bits(),
                !keeps_subnormals(),
                ieee(keeps_subnormals),
                nested,
            )
        });
        assert_eq!(set, (before & !FLUSH) | XLA);
        assert!(flushed && guarded);
        assert_eq!(nested, set, "a nested call keeps the flushing bits");
        assert_eq!(control_bits(), before);
        let unwound = panic::catch_unwind(|| {
            flushing(|| panic::resume_unwind(Box::new("unwinding through flushing")))
        });
        assert!(unwound.is_err());
        assert_eq!(control_bits(), before);
    }

    /// `scaled` in `ieee`, like a binding that takes and returns register values.
    #[inline(never)]
    fn guarded_scaled(x: f64, two: f64, half: f64) -> f64 {
        ieee(|| scaled(x, two, half))
    }

    /// `guarded_scaled` in a loop, which makes `scaled` loop-invariant.
    #[inline(never)]
    fn guarded_scaled_loop(x: f64, two: f64, half: f64, values: &mut [f64]) {
        for value in values {
            *value = ieee(|| scaled(x, two, half));
        }
    }

    /// Arithmetic on register values stays between the two control-register writes
    /// of `ieee`, where the optimizer could otherwise sink it past the restoring write
    /// (`guarded_scaled`) or hoist it out of a loop (`guarded_scaled_loop`).
    #[test]
    fn guarded_register_arithmetic_stays_inside() {
        let (x, two, half) = (f64::MIN_POSITIVE / 2.0, black_box(2.0), black_box(0.5));
        let x = black_box(x);
        for flush in CALLERS {
            let (flushed, single, looped) = flushing_caller(flush, || {
                let mut looped = vec![0.0; black_box(4)];
                guarded_scaled_loop(x, two, half, &mut looped);
                let single = guarded_scaled(x, two, half);
                (scaled(black_box(x), two, half), single, looped)
            });
            assert_eq!(flushed.to_bits(), 0, "{flush:#x} must flush `scaled`");
            assert_eq!(single.to_bits(), x.to_bits(), "{flush:#x}: {single:e}");
            assert!(
                looped.iter().all(|value| value.to_bits() == x.to_bits()),
                "{flush:#x}: {looped:?}"
            );
        }
    }

    #[test]
    fn the_guard_restores_the_caller_mode_after_errors_panics_and_nesting() {
        flushing_caller(XLA, || {
            let caller = control_bits();
            assert!(ieee(|| Err::<(), _>(Error::Singular)).is_err());
            assert_eq!(control_bits(), caller);
            let unwound = panic::catch_unwind(|| {
                ieee(|| {
                    assert!(keeps_subnormals());
                    panic::resume_unwind(Box::new("unwinding through the guard"));
                })
            });
            assert!(unwound.is_err());
            assert_eq!(control_bits(), caller);
            ieee(|| {
                ieee(|| ());
                assert!(keeps_subnormals(), "an inner guard restores the outer one");
            });
            assert_eq!(control_bits(), caller);
        });
    }

    #[test]
    fn lu_pivots_in_the_flushed_band_solve_exactly_under_the_guard() -> crate::Result<()> {
        for z in BAND {
            let expected = reciprocal(z)?;
            assert!((expected - z.inv()).norm() <= 2.0 * f64::EPSILON * expected.norm());
            let (flushed, guarded) =
                flushing_caller(XLA, || (reciprocal(z), ieee(|| reciprocal(z))));
            // The defect the guard removes: the flushed reciprocal is zero.
            assert_eq!(flushed?, Complex::ZERO, "{z}");
            assert_eq!(bits(&[guarded?]), bits(&[expected]), "{z}");
        }
        Ok(())
    }

    /// Solves and their adjoints on a guarded flushing thread are bitwise those
    /// of a thread that keeps subnormals.
    fn check_guarded_solve(
        a: &DMatrix<Complex>,
        b: &DMatrix<Complex>,
    ) -> Result<(), TestCaseError> {
        let solve = || -> crate::Result<_> {
            let residual = linalg::solve(a, b.clone())?;
            let value = residual.value().clone();
            let gradient = residual.pullback(value.clone())?;
            Ok([value, gradient.operator, gradient.rhs])
        };
        let expected = solve().map_err(|e| TestCaseError::reject(e.to_string()))?;
        let actual =
            flushing_caller(XLA, || ieee(solve)).map_err(|e| TestCaseError::fail(e.to_string()))?;
        for (actual, expected) in actual.iter().zip(&expected) {
            prop_assert_eq!(bits(actual.as_slice()), bits(expected.as_slice()));
        }
        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

        /// Diagonals in the flushed band, which partial pivoting keeps as
        /// pivots up to the small off-diagonal updates.
        #[test]
        fn guarded_solves_on_flushing_threads_match_ieee_threads(
            (a, b) in (1_usize..=8, 1_usize..=3).prop_flat_map(|(n, k)| (
                (
                    prop::collection::vec(complex(0.1), n * n),
                    prop::collection::vec(
                        complex(1.0).prop_filter("flushed band", |z| z.norm_sqr() > 1.0),
                        n,
                    ),
                )
                    .prop_map(move |(entries, diagonal)| {
                        let mut a = DMatrix::from_vec(n, n, entries);
                        a.set_diagonal(&diagonal.into());
                        a
                    }),
                crate::test_support::complex_matrix(n, k, 1.0),
            )),
        ) {
            check_guarded_solve(&a, &b)?;
        }
    }

    /// Scalar solves in the flushed band on the pool's workers.
    fn reciprocals_on_workers() -> crate::Result<Vec<Complex>> {
        parallel::try_map(64, true, |i| reciprocal(BAND[i % BAND.len()]))
    }

    fn reciprocals_on_this_thread() -> crate::Result<Vec<Complex>> {
        (0..64).map(|i| reciprocal(BAND[i % BAND.len()])).collect()
    }

    /// Whether this process runs `name` as the child of
    /// `fresh_pools_keep_subnormals_whatever_their_first_caller`, which is its
    /// only purpose.
    fn is_child(name: &str) -> bool {
        std::env::var(CHILD).is_ok_and(|child| child == name)
    }

    #[test]
    fn fresh_pools_keep_subnormals_whatever_their_first_caller() {
        for name in ["pool_started_under_a_guard", "pool_started_without_a_guard"] {
            let output = Command::new(std::env::current_exe().unwrap())
                .args([&format!("fpenv::tests::{name}"), "--exact", "--ignored"])
                .args(["--test-threads=1"])
                .env(CHILD, name)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains(" 1 passed"),
                "{name}\n{stdout}\n{}",
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }

    /// The first call into Rust that uses Rayon, made inside a JAX callback: the
    /// pool starts under the guard of a flushing thread, and its workers keep
    /// subnormals after the caller flushes again.
    #[test]
    #[ignore = "run in a fresh process by its parent test"]
    fn pool_started_under_a_guard() -> crate::Result<()> {
        if !is_child("pool_started_under_a_guard") {
            return Ok(());
        }
        let (started, restored, workers, values) = flushing(|| {
            let started = ieee(reciprocals_on_workers);
            let restored = !keeps_subnormals();
            let workers = crate::threads::install(|| rayon::broadcast(|_| keeps_subnormals()));
            (started, restored, workers, reciprocals_on_workers())
        });
        let expected = bits(&reciprocals_on_this_thread()?);
        assert!(restored, "the guard restores the caller's mode");
        assert!(workers.iter().all(|&kept| kept), "{workers:?}");
        assert_eq!(bits(&started?), expected);
        assert_eq!(bits(&values?), expected);
        Ok(())
    }

    /// Rust code that starts the pool on a flushing thread without a guard still
    /// gets workers that keep subnormals, and neither path starts Rayon's global
    /// pool.
    #[test]
    #[ignore = "run in a fresh process by its parent test"]
    fn pool_started_without_a_guard() -> crate::Result<()> {
        if !is_child("pool_started_without_a_guard") {
            return Ok(());
        }
        ieee(|| ());
        let (workers, values) = flushing(|| {
            let workers = crate::threads::install(|| rayon::broadcast(|_| keeps_subnormals()));
            (workers, ieee(reciprocals_on_workers))
        });
        assert!(workers.iter().all(|&kept| kept), "{workers:?}");
        assert_eq!(bits(&values?), bits(&reciprocals_on_this_thread()?));
        let built = rayon::ThreadPoolBuilder::new().build_global();
        assert!(built.is_ok(), "the global pool stays unused");
        Ok(())
    }
}
