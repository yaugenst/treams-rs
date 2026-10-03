//! Plane-wave polarization vectors, their wavevector derivatives and the normal wavenumber.
//!
//! Upstream: `treams.special.vpw_M`, `vpw_N` and `vpw_A`, and `treams.misc.wave_vec_z`.
#![allow(clippy::indexing_slicing)] // Fixed three-component vectors and jet slots.

use crate::{
    Complex, Error, Result,
    numerics::{Jet, complex_sqrt, finite, ratio},
    special::{check_pol, polarized_wave},
};

/// The algebraic norm `sqrt(Σ v^2)` (principal root) of complex components.
///
/// Nearly axial directions may have transverse components small enough that their
/// squares underflow while their azimuth matters, so extreme magnitudes are scaled
/// before squaring.
#[inline]
fn algebraic_norm(values: &[Complex]) -> Complex {
    let scale = values
        .iter()
        .map(|v| v.re.abs().max(v.im.abs()))
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return Complex::default();
    }
    // Squares of magnitudes within 1e±150 stay well inside the normal f64 range
    // (1e±308) and need no scaling. Outside it, scaling keeps tiny near-axis
    // components from underflowing to zero and huge ones from overflowing.
    if (1e-150..=1e150).contains(&scale) {
        return complex_sqrt(values.iter().map(|v| v * v).sum::<Complex>());
    }
    complex_sqrt(values.iter().map(|v| (v / scale).powu(2)).sum::<Complex>()) * scale
}
/// The transverse norm `sqrt(kx^2 + ky^2)` (principal root) and the unit transverse
/// direction `(kx, ky) / norm`, which is `(cos phi, sin phi)` for a real wavevector.
///
/// Gives zeros for `kx = ky = 0`, and an error when a complex `(kx, ky)` is nonzero
/// but has a zero algebraic norm, where the direction is undefined.
#[inline]
pub(crate) fn transverse_values(vector: [Complex; 2]) -> Result<(Complex, [Complex; 2])> {
    let scale = vector
        .iter()
        .map(|v| v.re.abs().max(v.im.abs()))
        .fold(0.0, f64::max);
    Ok(if scale == 0.0 {
        (Complex::default(), [Complex::default(); 2])
    } else {
        let (scaled, factor) = if (1e-150..=1e150).contains(&scale) {
            ([vector[0], vector[1]], 1.0)
        } else {
            ([vector[0] / scale, vector[1] / scale], scale)
        };
        let norm = complex_sqrt(scaled[0] * scaled[0] + scaled[1] * scaled[1]);
        if norm == Complex::default() {
            return Err(Error::InvalidInput(
                "undefined polarization for a null transverse vector".into(),
            ));
        }
        (norm * factor, scaled.map(|v| ratio(v, norm)))
    })
}
/// The wavenumber `k`, the transverse norm and the unit transverse direction of a
/// finite wavevector with a nonzero algebraic norm.
pub(crate) fn wavenumbers(vector: [Complex; 3]) -> Result<(Complex, Complex, [Complex; 2])> {
    if vector.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput("wavevector must be finite".into()));
    }
    let (transverse, xy) = transverse_values([vector[0], vector[1]])?;
    let k = algebraic_norm(&vector);
    if k == Complex::default() || !finite(k) {
        return Err(Error::InvalidInput(
            "wavevector must have nonzero algebraic norm".into(),
        ));
    }
    Ok((k, transverse, xy))
}

/// The direction of a wavevector `(kx, ky, kz)` as jets in its components: slots 0, 1
/// and 2 hold the derivatives with respect to `kx`, `ky` and `kz` (only the first `N`
/// are kept).
pub(crate) struct Direction<const N: usize> {
    /// The wavenumber `k = sqrt(kx^2 + ky^2 + kz^2)`.
    pub(crate) k: Jet<N>,
    /// The transverse norm `sqrt(kx^2 + ky^2)`, which is `k sin(theta)`.
    pub(crate) transverse: Jet<N>,
    /// The unit transverse direction `(kx, ky) / transverse`, which is
    /// `(cos phi, sin phi)`; zero on the z axis.
    pub(crate) xy: [Jet<N>; 2],
}
impl<const N: usize> Direction<N> {
    /// The direction of `vector`; see [`wavenumbers`] for the errors.
    pub(crate) fn new(vector: [Complex; 3]) -> Result<Self> {
        let (k, transverse, xy) = wavenumbers(vector)?;
        Self::from_parts(vector, k, transverse, xy)
    }
    /// The direction from the values [`wavenumbers`] returns. On the z axis the
    /// transverse direction has no derivative, so `N > 0` gives an error there.
    #[inline]
    pub(crate) fn from_parts(
        vector: [Complex; 3],
        k: Complex,
        transverse: Complex,
        xy: [Complex; 2],
    ) -> Result<Self> {
        if N != 0 && transverse == Complex::default() {
            return Err(Error::InvalidInput("plane-wave direction derivative is undefined on the polarization axis; fix the wavevectors".into()));
        }
        let k_jet = Jet {
            value: k,
            derivative: std::array::from_fn(|a| ratio(vector[a], k)),
        };
        let transverse_jet = Jet {
            value: transverse,
            derivative: std::array::from_fn(|a| if a < 2 { xy[a] } else { Complex::default() }),
        };
        let xy: [Jet<N>; 2] = std::array::from_fn(|a| Jet {
            value: xy[a],
            derivative: std::array::from_fn(|b| {
                if b < 2 {
                    ratio(
                        Complex::new(if a == b { 1.0 } else { 0.0 }, 0.0) - xy[a] * xy[b],
                        transverse,
                    )
                } else {
                    Complex::default()
                }
            }),
        });
        Ok(Self {
            k: k_jet,
            transverse: transverse_jet,
            xy,
        })
    }
}

/// The polarization vector of a plane wave with wavevector `vector`, as jets in the
/// wavevector components.
///
/// M is `i (ky, -kx, 0) / transverse` and N is `(-kx kz, -ky kz, transverse^2) /
/// (k transverse)`, as upstream `vpw_M` and `vpw_N`; helicity combines them as
/// `polarized_wave` does. On the z axis M is `(0, -i, 0)` and N is
/// `(-kz / sqrt(kz^2), 0, 0)`, using the same principal norm as off the axis.
#[inline]
pub(crate) fn polarization_jet<const N: usize>(
    vector: [Complex; 3],
    pol: u8,
    helicity: bool,
) -> Result<[Jet<N>; 3]> {
    check_pol(pol)?;
    let Direction { k, transverse, xy } = Direction::new(vector)?;
    let z = vector[2];
    let m = if transverse.value == Complex::default() {
        [Jet::default(), Jet::constant(-Complex::i()), Jet::default()]
    } else {
        [Complex::i() * xy[1], -Complex::i() * xy[0], Jet::default()]
    };
    // Parity M (pol 0) needs no N part.
    if !helicity && pol == 0 {
        return Ok(m);
    }
    let n = if transverse.value == Complex::default() {
        [
            Jet::constant(-ratio(z, k.value)),
            Jet::default(),
            Jet::default(),
        ]
    } else {
        let longitudinal = Jet::variable(z, 2) / k;
        [-xy[0] * longitudinal, -xy[1] * longitudinal, transverse / k]
    };
    Ok(polarized_wave(m, n, pol, helicity))
}

/// Plane-wave electric vector at the origin in treams normalization.
///
/// Upstream: `treams.special.vpw_M` (`pol` 0) and `vpw_N` (`pol` 1) at the origin with
/// `helicity` false, and `vpw_A` with `helicity` true.
#[inline]
pub fn polarization(vector: [Complex; 3], pol: u8, helicity: bool) -> Result<[Complex; 3]> {
    Ok(polarization_jet::<0>(vector, pol, helicity)?.map(|p| p.value))
}

/// The field `polarization exp(i k·position)` of a plane wave from its polarization
/// vector, which [`polarization`] gives.
///
/// Upstream: `treams.special.vpw_M`, `vpw_N` and `vpw_A` at `position`.
#[inline]
pub fn field_value(
    polarization: [Complex; 3],
    k: [Complex; 3],
    position: [Complex; 3],
) -> Result<[Complex; 3]> {
    if position.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput("field position must be finite".into()));
    }
    let phase = (Complex::i() * (0..3).map(|a| k[a] * position[a]).sum::<Complex>()).exp();
    let values = polarization.map(|p| p * phase);
    if values.iter().any(|&v| !finite(v)) {
        return Err(Error::NonFinite("non-finite plane-wave field".into()));
    }
    Ok(values)
}

/// Chain the local Cartesian polarization derivative into solver parameters.
pub(crate) fn polarization_from_inputs<const N: usize>(
    vector: [Jet<N>; 3],
    pol: u8,
) -> Result<[Jet<N>; 3]> {
    let values = vector.map(|k| k.value);
    if N == 0
        || (values[0] == Complex::default()
            && values[1] == Complex::default()
            && vector[..2]
                .iter()
                .flat_map(|k| k.derivative)
                .all(|d| d == Complex::default()))
    {
        return Ok(polarization(values, pol, true)?.map(Jet::constant));
    }
    Ok(polarization_jet::<3>(values, pol, true)?.map(|e| e.compose(vector)))
}

/// The normal wavenumber `kz = sqrt(k^2 - kx^2 - ky^2)` on the root with a nonnegative
/// imaginary part, so the wave propagates or decays toward `+z`; zero at a cutoff.
///
/// Upstream: `treams.misc.wave_vec_z`.
#[must_use]
pub fn wave_vector_z(kx: Complex, ky: Complex, k: Complex) -> Complex {
    let root = complex_sqrt(k * k - kx * kx - ky * ky);
    if root.im < 0.0 { -root } else { root }
}
