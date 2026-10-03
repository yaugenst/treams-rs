//! Hooks for tests and scripts: `build_profile`, `_run_flushing_for_tests`, which
//! calls Python while this thread flushes subnormals, and the `*_jet` hooks, which
//! return values with their derivatives.
use pyo3::prelude::*;
use treams_core::{
    Complex,
    fpenv::{flushing, ieee},
    special,
    sw::{self, Mode},
};

use crate::context::{error, radial};

/// The Cargo profile of this build: `"debug"` or `"release"`.
#[pyfunction]
pub(crate) fn build_profile() -> &'static str {
    ieee(|| {
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    })
}

/// Test support only: run `work()` while this thread flushes subnormals to zero
/// as XLA does, then restore the caller's mode, and return the result.
///
/// Native calls inside keep subnormals; Python arithmetic there flushes. On
/// targets other than x86-64 and `AArch64`, `work()` runs unchanged.
#[pyfunction(name = "_run_flushing_for_tests")]
pub(crate) fn run_flushing_for_tests<'py>(work: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    ieee(|| flushing(|| work.call0()))
}

/// A spherical translation coefficient with its derivatives: `sw::cartesian_translation`.
#[pyfunction]
pub(crate) fn cartesian_translation_jet(
    destination: (i32, i32, u8),
    source: (i32, i32, u8),
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    singular: bool,
) -> PyResult<(Complex, [Complex; 3], Complex)> {
    ieee(|| {
        let (l, m, pol) = destination;
        let destination = Mode { l, m, pol };
        let (l, m, pol) = source;
        let source = Mode { l, m, pol };
        let result =
            sw::cartesian_translation(destination, source, k, position, helicity, radial(singular))
                .map_err(error)?;
        Ok((result.value, result.position, result.k))
    })
}

/// A vector wave and its derivatives with respect to the position and to k.
type WaveJet = ([Complex; 3], [[Complex; 3]; 3], [Complex; 3]);

/// A spherical vector wave with its derivatives: `fields::spherical_wave`.
#[pyfunction]
pub(crate) fn spherical_wave_jet(
    mode: (i32, i32, u8),
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    singular: bool,
) -> PyResult<WaveJet> {
    ieee(|| {
        let (l, m, pol) = mode;
        let result = treams_core::fields::spherical_wave(
            Mode { l, m, pol },
            k,
            position,
            helicity,
            radial(singular),
        )
        .map_err(error)?;
        Ok((result.value, result.position, result.k))
    })
}

/// A coefficient and its derivatives with respect to the position, k and kz.
type CylJet = (Complex, [Complex; 3], Complex, Complex);

/// A cylindrical translation coefficient with its derivatives: `cw::cartesian_translation`.
#[pyfunction]
pub(crate) fn cylindrical_cartesian_translation_jet(
    destination: (f64, i32, u8),
    source: (f64, i32, u8),
    k: Complex,
    position: [f64; 3],
    singular: bool,
) -> PyResult<CylJet> {
    ieee(|| {
        let mode = |(kz, m, pol)| treams_core::cw::Mode { kz, m, pol };
        let result = treams_core::cw::cartesian_translation(
            mode(destination),
            mode(source),
            k,
            position,
            radial(singular),
        )
        .map_err(error)?;
        Ok((result.value, result.position, result.k, result.kz))
    })
}

/// A cylindrical radial function with its first two derivatives: `special::cylindrical_radial`.
#[pyfunction]
pub(crate) fn cylindrical_radial_jet(
    m: i32,
    z: Complex,
    singular: bool,
) -> PyResult<(Complex, Complex, Complex)> {
    ieee(|| {
        let jet = special::cylindrical_radial(m, z, radial(singular)).map_err(error)?;
        Ok((jet.value, jet.first, jet.second))
    })
}
