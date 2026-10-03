//! Kind parameters of the loops, and the inner-loop sizes from which loops run in parallel.
//!
//! Const generics take no enums on stable Rust, so each module below names the
//! values of one loop parameter. Loop bodies name a kind through its module, as
//! in `bessel::J`. A const generic argument must be a single identifier unless
//! it is braced, so the ufunc tables import the constants they pass.

/// Inner-loop sizes from which elementwise kernels run on the Rayon pool. A call
/// must amortize task dispatch and the collected output buffer, so the tiers
/// follow the per-element cost of each kernel family.
pub(super) mod parallel_from {
    /// A few arithmetic operations per element never amortize dispatch.
    pub(crate) const SERIAL: usize = usize::MAX;
    /// Ten microseconds or more per element: lattice sums (up to milliseconds)
    /// and odd-order Kambe integrals.
    pub(crate) const SLOW: usize = 8;
    /// About a microsecond per element: complex Bessel and Hankel functions, and
    /// spherical translations, whose Bessel series cost 0.4-3.3 us per element at
    /// degrees 1-3 and 3-7 us at degree 10 (regular ones are the cheaper).
    pub(crate) const BESSEL: usize = 64;
    /// One complex exponential, or a regular (series-dominated) Bessel function.
    pub(crate) const PHASE: usize = 512;
    /// Angular, Wigner, gamma, even-order Kambe, vector-wave, rotation and
    /// conversion kernels.
    pub(crate) const DEFAULT: usize = 1024;
}

macro_rules! kinds {
    ($($(#[$doc:meta])* $module:ident: $ty:ty { $($name:ident = $value:expr),+ })+) => {
        $($(#[$doc])* pub(super) mod $module {
            $(pub(crate) const $name: $ty = $value;)+
        })+
    };
}
kinds! {
    /// Bessel functions J, Y and Hankel functions.
    bessel: u8 { J = 0, Y = 1, H1 = 2, H2 = 3 }
    /// Spherical or cylindrical Bessel functions and lattice sums.
    family: bool { SPHERICAL = true, CYLINDRICAL = false }
    /// A Bessel function or its first derivative.
    derivative: u8 { VALUE = 0, DERIVATIVE = 1 }
    /// Angular functions.
    angular: u8 { LEGENDRE = 0, PI_FUN = 1, TAU_FUN = 2 }
    /// Vector spherical harmonics.
    vsh: u8 { HARMONIC_X = 0, HARMONIC_Y = 1, HARMONIC_Z = 2 }
    /// Regular or singular radial functions.
    radial: bool { REGULAR = true, SINGULAR = false }
    /// Parity polarizations; `A` reads a helicity label.
    pol: u8 { M = 0, N = 1, A = 2 }
    /// Polarization convention: helicity or parity.
    poltype: bool { HELICITY = true, PARITY = false }
    /// Cyclic xyz permutation turns.
    turns: usize { CYCLIC = 1, INVERSE = 2 }
    /// Ewald parts of lattice sums.
    ewald: u8 { FULL = 0, REAL = 1, RECIPROCAL = 2 }
    /// Lattice sums about a shifted origin.
    origin: bool { SHIFTED = true, UNSHIFTED = false }
}
