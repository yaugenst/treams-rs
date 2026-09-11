//! Electromagnetic T-matrix kernels and analytic pullbacks, independent of Python.

pub mod angular;
pub mod basis;
pub mod channels;
pub mod coeffs;
pub mod cylinder;
pub mod cylwaves;
pub mod fields;
pub mod integrals;
pub mod interaction;
mod jet;
pub mod lattice;
pub mod plane;
pub mod smatrix;
pub mod special;
pub mod tmatrix;
mod translation_plan;
pub mod waves;

#[cfg(test)]
mod properties;

/// Complex double precision used throughout the numerical core.
pub type Complex = num_complex::Complex64;

/// A numerical failure or invalid physical input.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The input does not satisfy the operation's contract.
    #[error("{0}")]
    InvalidInput(String),
    /// A special-function evaluation failed.
    #[error("special-function evaluation failed: {0}")]
    SpecialFunction(String),
    /// The scattering system is singular at the supplied parameters.
    #[error("singular scattering system")]
    Singular,
}

/// Numerical result with a typed error.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn finite(z: Complex) -> bool {
    z.re.is_finite() && z.im.is_finite()
}
