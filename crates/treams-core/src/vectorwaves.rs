//! Vector waves one at a time: the vector spherical harmonics and the spherical,
//! cylindrical and plane vector waves in their local component frames, with pullbacks.
//!
//! Upstream: `treams.special.vsh_X`, `vsh_Y`, `vsh_Z`, `vsw_*`, `vcw_*`, `vpw_*` and
//! `sph_harm`.
//!
//! [`vector_wave`] evaluates one wave of a [`Family`] at six complex arguments,
//! [`vector_wave_pullback`] gives the gradients of all six, and [`vector_wave_array`]
//! broadcasts both over arrays of labels and arguments. Spherical waves are in spherical
//! components `(r, theta, phi)`, cylindrical waves in `(rho, phi, z)` and plane waves in
//! Cartesian components, as in treams.
//!
//! The inline `tests` module checks the waves against the Cartesian fields of
//! [`crate::fields`], their pullbacks against finite differences, the small-argument
//! branches and the normalization of the harmonics.
#![allow(clippy::indexing_slicing)] // Fixed wave argument and component arrays.

use crate::{
    Complex, Error, MAX_DEGREE, Result,
    numerics::{Jet, broadcast, finite, parallel::Parallel},
    special::{self, Radial, SERIES_RADIUS, polar_trig, polarized_wave, radial_jet},
};
use std::f64::consts::PI;

/// The kind of vector wave [`vector_wave`] evaluates.
///
/// Polarization 0/1 selects M/N (upstream `*_M`/`*_N`), or negative/positive helicity
/// (upstream `*_A`) when `helicity` is true. Harmonics ignore the polarization.
///
/// Upstream: `treams.special.vsh_*`, `vsw_*`, `vcw_*` and `vpw_*`; each variant names
/// its functions.
#[derive(Clone, Copy, Debug)]
pub enum Family {
    /// Tangential X harmonic in spherical components (upstream `vsh_X`).
    HarmonicX,
    /// Tangential Y harmonic in spherical components (upstream `vsh_Y`).
    HarmonicY,
    /// Radial Z harmonic, equal to i times the scalar spherical harmonic (upstream
    /// `vsh_Z`).
    HarmonicZ,
    /// Spherical vector wave at `(kr, theta, phi)`: upstream `vsw_rM`, `vsw_rN` and
    /// `vsw_rA` when regular, `vsw_M`, `vsw_N` and `vsw_A` when singular.
    Spherical(Radial),
    /// Cylindrical vector wave at `(kz, k_rho*rho, phi, z, k)`: upstream `vcw_rM`,
    /// `vcw_rN` and `vcw_rA` when regular, `vcw_M`, `vcw_N` and `vcw_A` when singular.
    Cylindrical(Radial),
    /// Plane vector wave at `(kx, ky, kz, x, y, z)`: upstream `vpw_M`, `vpw_N` and
    /// `vpw_A`.
    Plane,
}

/// The integer labels of one vector wave.
///
/// Each [`Family`] reads some of the fields:
///
/// | Family | Reads |
/// |---|---|
/// | `HarmonicX`, `HarmonicY`, `HarmonicZ` | `l`, `m` |
/// | `Spherical` | `l`, `m`, `pol` |
/// | `Cylindrical` | `m`, `pol` |
/// | `Plane` | `pol` |
///
/// Every evaluation accepts `0 <= l <= 128`, `|m| <= 128` and `pol` 0 or 1, whatever the
/// family. Every harmonic and spherical wave with `|m| > l` is zero. At `l = 0` the
/// spherical waves and the X and Y harmonics are zero, and the Z harmonic is `i Y_00`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaveLabel {
    /// Degree.
    pub l: i32,
    /// Order.
    pub m: i32,
    /// Polarization index: M/N (0/1), or negative/positive helicity.
    pub pol: u8,
}

/// The normalized scalar spherical harmonic `Y_lm(theta, phi)` and, if `tangential`
/// and `l > 0`, the angular functions `pi = m Y_lm / sin(theta)` and
/// `tau = dY_lm / dtheta`, both divided by `sqrt(l (l + 1))`: `[Y, pi, tau]`, or
/// `[Y, 0, 0]` otherwise.
///
/// The recurrence runs over normalized associated Legendre functions: it starts from
/// the normalized diagonal term of degree `|m|` and steps the degree up to `l`. Every
/// term stays of order one, which avoids the factorial overflow of the unnormalized
/// functions at high orders. For `m = 0` the formula for `tau` used at other orders
/// would divide by `sin(theta)`, so a second recurrence gives the derivative with
/// respect to `cos(theta)` instead.
fn angular<const N: usize>(
    l: i32,
    m: i32,
    theta: Jet<N>,
    phi: Jet<N>,
    tangential: bool,
) -> [Jet<N>; 3] {
    let [cosine, sine] = polar_trig(theta);
    let order = m.abs();
    // Normalize the Legendre recurrence before evaluating it. This removes
    // factorial overflow for high orders and repeated adjacent-order recurrences.
    let mut diagonal = (1.0 / (4.0 * PI)).sqrt();
    for j in 1..=order {
        diagonal *= -(f64::from(2 * j + 1) / f64::from(2 * j)).sqrt();
    }
    if m < 0 && order % 2 != 0 {
        diagonal = -diagonal;
    }
    let mut previous = Jet::default();
    let mut polynomial = Jet::constant(diagonal);
    let mut previous_derivative = Jet::default();
    let mut derivative = Jet::default();
    for j in order + 1..=l {
        let denominator = f64::from(j * j - order * order);
        let a = (f64::from(4 * j * j - 1) / denominator).sqrt();
        let b = if j == order + 1 {
            0.0
        } else {
            (f64::from((2 * j + 1) * ((j - 1) * (j - 1) - order * order))
                / (f64::from(2 * j - 3) * denominator))
                .sqrt()
        };
        if order == 0 && tangential {
            let next = a * (polynomial + cosine * derivative) - b * previous_derivative;
            previous_derivative = derivative;
            derivative = next;
        }
        let next = a * cosine * polynomial - b * previous;
        previous = polynomial;
        polynomial = next;
    }
    let phase = (Complex::i() * f64::from(m) * phi).exp();
    let power = sine.powi((order - 1).max(0));
    let harmonic = phase
        * polynomial
        * if order == 0 {
            Jet::constant(1.0)
        } else {
            power * sine
        };
    if l == 0 || !tangential {
        return [harmonic, Jet::default(), Jet::default()];
    }
    let norm = f64::from(l * (l + 1)).sqrt();
    if order == 0 {
        return [harmonic, Jet::default(), -phase * sine * derivative / norm];
    }
    let adjacent = (f64::from((l * l - order * order) * (2 * l + 1)) / f64::from(2 * l - 1)).sqrt();
    [
        harmonic,
        phase * f64::from(m) * power * polynomial / norm,
        phase * power * (f64::from(l) * cosine * polynomial - adjacent * previous) / norm,
    ]
}

/// The scalar harmonic `Y` and, if `tangential`, the angular functions `pi` and
/// `tau` of `label`; `None` when `|m| > l`.
fn harmonics<const N: usize>(
    label: WaveLabel,
    theta: Jet<N>,
    phi: Jet<N>,
    tangential: bool,
) -> Option<[Jet<N>; 3]> {
    (label.m.abs() <= label.l).then(|| angular(label.l, label.m, theta, phi, tangential))
}

/// The tangential X and Y vector harmonics from `[Y, pi, tau]`.
fn tangential<const N: usize>([_, pi, tau]: [Jet<N>; 3]) -> [[Jet<N>; 3]; 2] {
    [
        [Jet::default(), -pi, -Complex::i() * tau],
        [Jet::default(), Complex::i() * tau, -pi],
    ]
}

/// Spherical M and N waves at `(kr, theta, phi)`, or their helicity combinations.
fn spherical<const N: usize>(
    radial: Radial,
    label: WaveLabel,
    [kr, theta, phi]: [Jet<N>; 3],
    helicity: bool,
) -> Result<[Jet<N>; 3]> {
    if label.l == 0 {
        return Ok([Jet::default(); 3]);
    }
    let Some(harmonics) = harmonics(label, theta, phi, true) else {
        return Ok([Jet::default(); 3]);
    };
    let [x, tangent] = tangential(harmonics);
    if !helicity && label.pol == 0 {
        let value = radial_jet(label.l, kr, radial, true)?;
        return Ok(x.map(|v| v * value));
    }
    let l = label.l.unsigned_abs();
    let bessel = special::spherical_radial(l, kr.value, radial)?;
    let value = kr.chain(bessel.value, bessel.first);
    let first = kr.chain(bessel.first, bessel.second);
    let divided = if radial == Radial::Regular && kr.value.norm() < SERIES_RADIUS {
        let lower = special::spherical_radial(l - 1, kr.value, radial)?;
        let upper = special::spherical_radial(l + 1, kr.value, radial)?;
        kr.chain(
            (lower.value + upper.value) / f64::from(2 * l + 1),
            (lower.first + upper.first) / f64::from(2 * l + 1),
        )
    } else {
        value / kr
    };
    let mut n = tangent.map(|v| v * (divided + first));
    n[0] = Complex::i() * harmonics[0] * divided * f64::from(label.l * (label.l + 1)).sqrt();
    Ok(polarized_wave(x.map(|v| v * value), n, label.pol, helicity))
}

/// Cylindrical M and N waves at `(kz, krr, phi, z, k)`, or their helicity combinations;
/// `krr` is `k_rho rho`, as in upstream `vcw_*`.
fn cylindrical<const N: usize>(
    radial: Radial,
    label: WaveLabel,
    [kz, krr, phi, z, k, _]: [Jet<N>; 6],
    helicity: bool,
) -> Result<[Jet<N>; 3]> {
    let bessel = special::cylindrical_radial(label.m, krr.value, radial)?;
    let value = krr.chain(bessel.value, bessel.first);
    let first = krr.chain(bessel.first, bessel.second);
    let divided = if label.m == 0 {
        Jet::default()
    } else if radial == Radial::Regular && krr.value.norm() < SERIES_RADIUS {
        let lower = special::cylindrical_radial(label.m - 1, krr.value, radial)?;
        let upper = special::cylindrical_radial(label.m + 1, krr.value, radial)?;
        krr.chain(
            0.5 * (lower.value + upper.value),
            0.5 * (lower.first + upper.first),
        )
    } else {
        f64::from(label.m) * value / krr
    };
    let phase = (Complex::i() * (f64::from(label.m) * phi + kz * z)).exp();
    let m = [
        Complex::i() * divided * phase,
        -first * phase,
        Jet::default(),
    ];
    if !helicity && label.pol == 0 {
        return Ok(m);
    }
    if k.value == Complex::default() {
        return Err(Error::InvalidInput(
            "cylindrical N waves require nonzero k".into(),
        ));
    }
    // The principal root, as upstream `vcw_N` takes it. For a gain medium it can be the
    // negative of the root with a nonnegative imaginary part that `cw` uses.
    let transverse = (k * k - kz * kz).sqrt();
    let factor = phase / k;
    let n = [
        Complex::i() * kz * first * factor,
        -kz * divided * factor,
        transverse * value * factor,
    ];
    Ok(polarized_wave(m, n, label.pol, helicity))
}

/// Plane wave `p exp(i k·r)` at `(kx, ky, kz, x, y, z)`, with the chain rule through
/// the polarization `p(k)`.
fn plane<const N: usize>(
    label: WaveLabel,
    args: [Jet<N>; 6],
    helicity: bool,
) -> Result<[Jet<N>; 3]> {
    let vector = [args[0].value, args[1].value, args[2].value];
    if N == 0 {
        return Ok(crate::pw::field_value(
            crate::pw::polarization(vector, label.pol, helicity)?,
            vector,
            [args[3].value, args[4].value, args[5].value],
        )?
        .map(Jet::constant));
    }
    let polarization = crate::pw::polarization_jet::<3>(vector, label.pol, helicity)?
        .map(|p| p.compose([args[0], args[1], args[2]]));
    let phase = (Complex::i() * (0..3).map(|a| args[a] * args[a + 3]).sum::<Jet<N>>()).exp();
    Ok(polarization.map(|p| p * phase))
}

#[inline]
fn evaluate<const N: usize>(
    family: Family,
    label: WaveLabel,
    args: [Jet<N>; 6],
    helicity: bool,
) -> Result<[Jet<N>; 3]> {
    if label.pol > 1
        || label.l < 0
        || label.l > MAX_DEGREE
        || label.m.unsigned_abs() > MAX_DEGREE.unsigned_abs()
        || args.iter().any(|v| !finite(v.value))
    {
        return Err(Error::InvalidInput(
            // 128 is MAX_DEGREE.
            "require finite wave arguments, 0 <= l <= 128, |m| <= 128 and polarization 0 or 1"
                .into(),
        ));
    }
    let [theta, phi] = [args[0], args[1]];
    let zero = [Jet::default(); 3];
    Ok(match family {
        Family::HarmonicX => harmonics(label, theta, phi, true).map_or(zero, |h| tangential(h)[0]),
        Family::HarmonicY => harmonics(label, theta, phi, true).map_or(zero, |h| tangential(h)[1]),
        Family::HarmonicZ => harmonics(label, theta, phi, false).map_or(zero, |[y, _, _]| {
            [Complex::i() * y, Jet::default(), Jet::default()]
        }),
        Family::Spherical(radial) => {
            spherical(radial, label, [args[0], args[1], args[2]], helicity)?
        }
        Family::Cylindrical(radial) => cylindrical(radial, label, args, helicity)?,
        Family::Plane => plane(label, args, helicity)?,
    })
}
fn checked<const N: usize>(result: [Jet<N>; 3]) -> Result<[Jet<N>; 3]> {
    if result.iter().any(|v| !v.finite()) {
        return Err(Error::NonFinite(
            "non-finite vector wave or undefined derivative".into(),
        ));
    }
    Ok(result)
}

/// Evaluate a vector wave in its local component frame.
///
/// Harmonics use `(theta,phi)`, spherical waves `(kr,theta,phi)`, cylinders
/// `(kz,k_rho*rho,phi,z,k)`, and planes `(kx,ky,kz,x,y,z)`. Unused arguments are zero.
///
/// Upstream: `treams.special.vsh_X`, `vsh_Y`, `vsh_Z`, `vsw_*`, `vcw_*` and `vpw_*`.
#[inline]
pub fn vector_wave(
    family: Family,
    label: WaveLabel,
    args: [Complex; 6],
    helicity: bool,
) -> Result<[Complex; 3]> {
    Ok(checked(evaluate::<0>(
        family,
        label,
        args.map(Jet::constant),
        helicity,
    )?)?
    .map(|v| v.value))
}

/// The gradients of all six complex arguments of [`vector_wave`] for a `cotangent` of
/// the wave. A real physical input takes the real part of its gradient.
pub fn vector_wave_pullback(
    family: Family,
    label: WaveLabel,
    args: [Complex; 6],
    helicity: bool,
    cotangent: [Complex; 3],
) -> Result<[Complex; 6]> {
    if cotangent.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput(
            "vector-wave cotangent must be finite".into(),
        ));
    }
    if cotangent == [Complex::default(); 3] {
        vector_wave(family, label, args, helicity)?;
        return Ok([Complex::default(); 6]);
    }
    match family {
        Family::HarmonicX | Family::HarmonicY | Family::HarmonicZ => {
            pullback_impl::<2>(family, label, args, helicity, cotangent)
        }
        Family::Spherical(_) => pullback_impl::<3>(family, label, args, helicity, cotangent),
        Family::Cylindrical(_) => pullback_impl::<5>(family, label, args, helicity, cotangent),
        Family::Plane => pullback_impl::<6>(family, label, args, helicity, cotangent),
    }
}
/// [`vector_wave_pullback`] with derivatives in the first `N` arguments, the ones the
/// family reads; the others get zero.
fn pullback_impl<const N: usize>(
    family: Family,
    label: WaveLabel,
    args: [Complex; 6],
    helicity: bool,
    cotangent: [Complex; 3],
) -> Result<[Complex; 6]> {
    let values = checked(evaluate::<N>(
        family,
        label,
        std::array::from_fn(|a| Jet::variable(args[a], a)),
        helicity,
    )?)?;
    Ok(std::array::from_fn(|a| {
        values
            .iter()
            .zip(cotangent)
            .map(|(v, g)| v.derivative.get(a).copied().unwrap_or_default().conj() * g)
            .sum()
    }))
}

/// Vector-wave arrays evaluate in parallel from this many elements.
const PARALLEL: Parallel = Parallel::AtLeast(1024);

/// What [`vector_wave_array`] saves for its pullback: the family, the labels, the
/// convention and the arguments, each one value for all outputs or one per output.
///
/// The pullback computes each output's gradients on its own and adds the gradients of a
/// scalar argument in output order on the calling thread, so its result does not
/// depend on the thread count.
#[derive(Debug)]
pub struct VectorWaveResidual {
    family: Family,
    labels: Vec<WaveLabel>,
    arguments: [Vec<Complex>; 6],
    helicity: bool,
    size: usize,
}
impl VectorWaveResidual {
    fn element(&self, i: usize) -> (WaveLabel, [Complex; 6]) {
        (
            broadcast::element(&self.labels, i),
            self.arguments.each_ref().map(|a| broadcast::element(a, i)),
        )
    }
    /// The label and wave vector that several plane waves share, and hence their
    /// one polarization, when they differ only in their points.
    fn fixed_plane(&self) -> Option<(WaveLabel, [Complex; 3])> {
        let [kx, ky, kz, ..] = self.arguments.each_ref().map(Vec::as_slice);
        match (self.family, self.labels.as_slice(), kx, ky, kz) {
            (Family::Plane, &[label], &[kx], &[ky], &[kz]) if self.size > 1 => {
                Some((label, [kx, ky, kz]))
            }
            _ => None,
        }
    }
    /// The point `(x, y, z)` of plane wave `i`.
    fn point(&self, i: usize) -> [Complex; 3] {
        let [_, _, _, x, y, z] = &self.arguments;
        [x, y, z].map(|v| broadcast::element(v, i))
    }
    /// The gradients of the six arguments from one complex vector cotangent per output;
    /// an argument given as one value for all outputs gets the sum of its gradients.
    pub fn pullback(self, cotangent: &[[Complex; 3]]) -> Result<[Vec<Complex>; 6]> {
        // Waves of one direction share the polarization jet; each element then
        // reads only its point.
        let plane = match self.fixed_plane() {
            Some((label, k)) if cotangent.iter().flatten().any(|&v| v != Complex::default()) => {
                let p = crate::pw::polarization_jet::<3>(k, label.pol, self.helicity)?;
                Some((p, k))
            }
            _ => None,
        };
        broadcast::pullback(
            cotangent,
            self.size,
            "wave cotangent must be finite and match output",
            self.arguments.each_ref().map(Vec::len),
            PARALLEL,
            |i, &g| {
                if let Some((p, k)) = &plane {
                    fixed_plane_pullback(p, *k, self.point(i), g)
                } else {
                    let (m, a) = self.element(i);
                    vector_wave_pullback(self.family, m, a, self.helicity, g)
                }
            },
        )
    }
}

/// Pullback to `(k, r)` of a plane wave `p exp(i k·r)`, given the polarization jet
/// `p` in the wave vector.
fn fixed_plane_pullback(
    p: &[Jet<3>; 3],
    k: [Complex; 3],
    r: [Complex; 3],
    g: [Complex; 3],
) -> Result<[Complex; 6]> {
    let phase = (Complex::i() * (0..3).map(|j| k[j] * r[j]).sum::<Complex>()).exp();
    let g = g.map(|v| v * phase.conj());
    let amplitude = p
        .iter()
        .zip(g)
        .map(|(p, g)| p.value.conj() * g)
        .sum::<Complex>();
    let gradient = std::array::from_fn(|j| {
        if j < 3 {
            p.iter()
                .zip(g)
                .map(|(p, g)| p.derivative[j].conj() * g)
                .sum::<Complex>()
                + (Complex::i() * r[j]).conj() * amplitude
        } else {
            (Complex::i() * k[j - 3]).conj() * amplitude
        }
    });
    if gradient.iter().any(|&v| !finite(v)) {
        return Err(Error::NonFinite("non-finite plane-wave pullback".into()));
    }
    Ok(gradient)
}

/// [`vector_wave`] over arrays of labels and arguments, each one value for all outputs
/// or one per output; the residual keeps only the inputs.
///
/// Upstream: `treams.special.vsh_*`, `vsw_*`, `vcw_*` and `vpw_*` on broadcast arrays.
pub fn vector_wave_array(
    family: Family,
    labels: Vec<WaveLabel>,
    arguments: [Vec<Complex>; 6],
    helicity: bool,
) -> Result<(Vec<[Complex; 3]>, VectorWaveResidual)> {
    let [a, b, c, d, e, f] = arguments.each_ref().map(Vec::len);
    let size = broadcast::size(
        &[labels.len(), a, b, c, d, e, f],
        "wave arrays must have equal lengths or scalar inputs",
    )?;
    let residual = VectorWaveResidual {
        family,
        labels,
        arguments,
        helicity,
        size,
    };
    let plane = match residual.fixed_plane() {
        Some((label, k)) => {
            // Evaluating the first wave checks the shared label and wavevector once.
            vector_wave(family, label, residual.element(0).1, helicity)?;
            Some((crate::pw::polarization(k, label.pol, helicity)?, k))
        }
        None => None,
    };
    let values = broadcast::map(size, PARALLEL, |i| {
        if let Some((p, k)) = plane {
            crate::pw::field_value(p, k, residual.point(i))
        } else {
            let (m, a) = residual.element(i);
            vector_wave(family, m, a, helicity)
        }
    })?;
    Ok((values, residual))
}

/// Scalar spherical harmonic `Y_l^m(theta, phi)` at the polar angle `theta` and the
/// azimuth `phi`.
///
/// `Y` is `-i` times the radial component of the [`Family::HarmonicZ`] wave
/// ([`sph_harm_from_vsh_z`]).
///
/// Upstream: `treams.special.sph_harm`, which takes `(m, l, phi, theta)`.
/// Differences: degrees outside `[0, 128]`, orders with `|m| > 128` and non-finite angles
/// give an error, where treams returns a value or NaN; other orders with `|m| > l` give
/// zero, where treams returns NaN.
#[inline]
pub fn sph_harm(l: i32, m: i32, theta: Complex, phi: Complex) -> Result<Complex> {
    let zero = Complex::default();
    let label = WaveLabel { l, m, pol: 0 };
    let wave = vector_wave(
        Family::HarmonicZ,
        label,
        [theta, phi, zero, zero, zero, zero],
        false,
    )?;
    Ok(sph_harm_from_vsh_z(wave))
}

/// The scalar spherical harmonic `Y = -i Z_r` of a [`Family::HarmonicZ`] wave `Z`
/// (`treams.special.vsh_Z`), whose only nonzero component is the radial one, `Z_r = i Y`.
#[inline]
#[must_use]
pub fn sph_harm_from_vsh_z([z_r, _, _]: [Complex; 3]) -> Complex {
    -Complex::i() * z_r
}

/// Pullback of [`sph_harm_from_vsh_z`]: the [`Family::HarmonicZ`] wave cotangent
/// `[i cotangent, 0, 0]` for the scalar-harmonic `cotangent`.
#[inline]
#[must_use]
pub fn sph_harm_from_vsh_z_pullback(cotangent: Complex) -> [Complex; 3] {
    [
        Complex::i() * cotangent,
        Complex::default(),
        Complex::default(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        Complex, Family, PI, Radial, SERIES_RADIUS, WaveLabel, vector_wave, vector_wave_pullback,
    };
    use crate::{
        cw, fields,
        special::coordinates::{self, Transform},
        sw,
        test_support::{EXPENSIVE_CASES, complex, degree_order, prop_assert_close, radial},
    };
    use proptest::{prelude::*, test_runner::TestCaseError};

    const ZERO: Complex = Complex::new(0.0, 0.0);

    // Fewer cases than DEFAULT_CASES: each case checks several families and
    // polarizations, and the adjoint check takes central differences in all six
    // arguments of eight families.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(40))]

        #[test]
        fn spherical_and_cylindrical_cartesian_reconstruction(
            (l, m) in degree_order(1..8),
            r in 0.05_f64..1.5,
            theta in 0.3_f64..2.8,
            phi in -3.0_f64..3.0,
            helicity in any::<bool>(),
            radial in radial(),
        ) {
            check_reconstruction(l, m, [r, theta, phi], helicity, radial)?;
        }

        #[test]
        fn vector_wave_all_argument_adjoints(
            (l, m) in degree_order(1..6),
            pol in 0_u8..2,
            magnitude in prop_oneof![0.05_f64..0.45, 0.6_f64..2.0],
            phase in -0.5_f64..0.5,
            theta in 0.3_f64..2.8,
            phi in -2.0_f64..2.0,
            helicity in any::<bool>(),
            g in prop::array::uniform3(complex(0.5)),
        ) {
            let z = Complex::from_polar(magnitude, phase);
            check_adjoints(WaveLabel { l, m, pol }, z, [theta, phi], helicity, g)?;
        }

        #[test]
        fn regular_small_argument_branches_are_continuous(
            (l, m) in degree_order(1..9),
            phase in -PI..PI,
            theta in 0.3_f64..2.8,
            phi in -3.0_f64..3.0,
            g in prop::array::uniform3(complex(0.5)),
        ) {
            check_small_argument_continuity(l, m, phase, [theta, phi], g)?;
        }
    }

    // Each case sums 2l + 1 harmonics of degree up to 128 for three families.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(EXPENSIVE_CASES))]

        #[test]
        fn vector_harmonics_obey_unsolds_theorem(
            l in 20_i32..129,
            theta in prop_oneof![Just(0.0), Just(1e-10), Just(PI), 0.0..PI],
            phi in -3.0_f64..3.0,
        ) {
            check_unsolds_theorem(l, theta, phi)?;
        }
    }

    /// The six wave arguments with `leading` in front and zeros after.
    fn arguments<const N: usize>(leading: [Complex; N]) -> [Complex; 6] {
        std::array::from_fn(|i| leading.get(i).copied().unwrap_or_default())
    }

    /// Spherical and cylindrical waves in their local frames, rotated to Cartesian
    /// components, equal the Cartesian fields of `fields`, on both sides of the
    /// small-argument series switch `|z| = SERIES_RADIUS` (0.5).
    fn check_reconstruction(
        l: i32,
        m: i32,
        [r, theta, phi]: [f64; 3],
        helicity: bool,
        radial: Radial,
    ) -> Result<(), TestCaseError> {
        let k = Complex::new(1.3, 0.05);
        let kz = 0.2;
        let krho = (k * k - kz * kz).sqrt();
        for pol in [0, 1] {
            let label = WaveLabel { l, m, pol };
            let spherical = [r, theta, phi];
            let args = arguments([k * r, theta.into(), phi.into()]);
            let local = vector_wave(Family::Spherical(radial), label, args, helicity).unwrap();
            let actual = coordinates::vector(local, spherical, Transform::SphToCar).unwrap();
            let position = coordinates::point(spherical, Transform::SphToCar).unwrap();
            let mode = sw::Mode { l, m, pol };
            let expected = fields::spherical_wave(mode, k, position, helicity, radial)
                .unwrap()
                .value;
            let scale = 1e-11 * (1.0 + expected.map(Complex::norm).iter().sum::<f64>());
            prop_assert_close!(actual, expected, scale, "spherical {label:?}");
            let cylindrical = [r, phi, 0.3];
            let args = arguments([kz.into(), krho * r, phi.into(), 0.3.into(), k]);
            let local = vector_wave(Family::Cylindrical(radial), label, args, helicity).unwrap();
            let actual = coordinates::vector(local, cylindrical, Transform::CylToCar).unwrap();
            let position = coordinates::point(cylindrical, Transform::CylToCar).unwrap();
            let expected =
                fields::cylindrical_wave(cw::Mode { kz, m, pol }, k, position, helicity, radial)
                    .unwrap()
                    .value;
            let scale = 1e-11 * (1.0 + expected.map(Complex::norm).iter().sum::<f64>());
            prop_assert_close!(actual, expected, scale, "cylindrical {label:?}");
        }
        Ok(())
    }

    /// Central differences of `vector_wave` along a complex `direction` in argument `a`,
    /// paired with the cotangent `g`, and the scale of the paired terms.
    fn paired_difference(
        family: Family,
        label: WaveLabel,
        args: [Complex; 6],
        helicity: bool,
        g: [Complex; 3],
        a: usize,
        direction: Complex,
    ) -> (Complex, f64) {
        let h = 1e-6;
        let step = |t: f64| {
            let mut shifted = args;
            shifted[a] += t * h * direction;
            vector_wave(family, label, shifted, helicity).unwrap()
        };
        let (plus, minus) = (step(1.0), step(-1.0));
        let terms: Vec<Complex> = (0..3)
            .map(|c| g[c].conj() * (plus[c] - minus[c]) / (2.0 * h))
            .collect();
        (terms.iter().sum(), terms.iter().map(|t| t.norm()).sum())
    }

    /// Every family's pullback, paired with a complex direction in each of the six
    /// arguments, is the holomorphic directional derivative of `vector_wave`; the
    /// radial argument lies on either side of the small-argument threshold.
    fn check_adjoints(
        label: WaveLabel,
        z: Complex,
        [theta, phi]: [f64; 2],
        helicity: bool,
        g: [Complex; 3],
    ) -> Result<(), TestCaseError> {
        let k = Complex::new(1.3, 0.05);
        let families = [
            (Family::HarmonicX, arguments([theta.into(), phi.into()])),
            (Family::HarmonicY, arguments([theta.into(), phi.into()])),
            (Family::HarmonicZ, arguments([theta.into(), phi.into()])),
            (
                Family::Spherical(Radial::Regular),
                arguments([z, theta.into(), phi.into()]),
            ),
            (
                Family::Spherical(Radial::Singular),
                arguments([z, theta.into(), phi.into()]),
            ),
            (
                Family::Cylindrical(Radial::Regular),
                arguments([0.2.into(), z, phi.into(), 0.3.into(), k]),
            ),
            (
                Family::Cylindrical(Radial::Singular),
                arguments([0.2.into(), z, phi.into(), 0.3.into(), k]),
            ),
            (
                Family::Plane,
                [
                    Complex::new(0.3, 0.1),
                    Complex::new(0.4, -0.05),
                    z,
                    0.2.into(),
                    0.3.into(),
                    0.4.into(),
                ],
            ),
        ];
        let direction = Complex::new(0.2, 0.1);
        for (family, args) in families {
            let gradient = vector_wave_pullback(family, label, args, helicity, g).unwrap();
            for (a, gradient) in gradient.into_iter().enumerate() {
                let (numeric, scale) =
                    paired_difference(family, label, args, helicity, g, a, direction);
                prop_assert_close!(
                    gradient.conj() * direction,
                    numeric,
                    2e-7 * (1.0 + scale),
                    "{family:?} argument {a}"
                );
            }
        }
        Ok(())
    }

    /// Regular spherical and cylindrical waves and their pullbacks agree just below
    /// and just above `|z| = SERIES_RADIUS` (0.5), where their radial functions switch
    /// from power series to Bessel evaluations.
    fn check_small_argument_continuity(
        l: i32,
        m: i32,
        phase: f64,
        [theta, phi]: [f64; 2],
        g: [Complex; 3],
    ) -> Result<(), TestCaseError> {
        let at = |family: Family, label: WaveLabel, helicity: bool, magnitude: f64| {
            let z = Complex::from_polar(magnitude, phase);
            assert_eq!(
                z.norm() < SERIES_RADIUS,
                magnitude < SERIES_RADIUS,
                "{z} is on the wrong branch"
            );
            let args = match family {
                Family::Cylindrical(_) => arguments([
                    0.2.into(),
                    z,
                    phi.into(),
                    0.3.into(),
                    Complex::new(1.3, 0.05),
                ]),
                _ => arguments([z, theta.into(), phi.into()]),
            };
            (
                vector_wave(family, label, args, helicity).unwrap(),
                vector_wave_pullback(family, label, args, helicity, g).unwrap(),
            )
        };
        for family in [
            Family::Spherical(Radial::Regular),
            Family::Cylindrical(Radial::Regular),
        ] {
            for (pol, helicity) in [(0, false), (1, false), (0, true), (1, true)] {
                let label = WaveLabel { l, m, pol };
                // Nine and four ulps from the threshold, respectively.
                let (below, below_gradient) =
                    at(family, label, helicity, SERIES_RADIUS * (1.0 - 1e-15));
                let (above, above_gradient) =
                    at(family, label, helicity, SERIES_RADIUS * (1.0 + 1e-15));
                let scale = below.iter().map(|v| v.norm()).sum::<f64>();
                prop_assert_close!(
                    below,
                    above,
                    1e-12 * scale + f64::MIN_POSITIVE,
                    "{family:?} {label:?} {helicity}"
                );
                let scale = below_gradient.iter().map(|v| v.norm()).sum::<f64>();
                prop_assert_close!(
                    below_gradient,
                    above_gradient,
                    1e-12 * scale + f64::MIN_POSITIVE,
                    "{family:?} {label:?} {helicity} pullback"
                );
            }
        }
        Ok(())
    }

    /// Unsöld's theorem `Σ_m |X_lm|² = (2l + 1) / 4π` for the normalized vector
    /// spherical harmonics at high degree, including both poles and a polar angle
    /// of `1e-10`.
    fn check_unsolds_theorem(l: i32, theta: f64, phi: f64) -> Result<(), TestCaseError> {
        let args = arguments([theta.into(), phi.into()]);
        for family in [Family::HarmonicX, Family::HarmonicY, Family::HarmonicZ] {
            let total = (-l..=l)
                .map(|m| {
                    vector_wave(family, WaveLabel { l, m, pol: 0 }, args, false)
                        .unwrap()
                        .iter()
                        .map(Complex::norm_sqr)
                        .sum::<f64>()
                })
                .sum::<f64>();
            let expected = f64::from(2 * l + 1) / (4.0 * PI);
            prop_assert_close!(total, expected, 2e-11 * expected, "{family:?}");
        }
        Ok(())
    }

    #[test]
    fn regular_spherical_origin_matches_the_cartesian_field() {
        for (l, m) in (1..=3).flat_map(|l| (-l..=l).map(move |m| (l, m))) {
            let label = WaveLabel { l, m, pol: 1 };
            let mode = sw::Mode { l, m, pol: 1 };
            for theta in [0.0, 0.7, PI] {
                let local = vector_wave(
                    Family::Spherical(Radial::Regular),
                    label,
                    arguments([ZERO, theta.into(), 0.3.into()]),
                    false,
                )
                .unwrap();
                let actual =
                    coordinates::vector(local, [0.0, theta, 0.3], Transform::SphToCar).unwrap();
                let expected =
                    fields::spherical_wave(mode, 1.3.into(), [0.0; 3], false, Radial::Regular)
                        .unwrap()
                        .value;
                for (a, b) in actual.into_iter().zip(expected) {
                    assert!((a - b).norm() < 1e-13, "{label:?} {theta}");
                }
            }
        }
    }

    /// At the poles the azimuthal cotangent is exactly that of `∂_φ X = i m X`, and
    /// the polar cotangent is the one-sided derivative from inside `(0, π)`.
    #[test]
    fn pole_pullbacks_are_one_sided_derivatives() {
        let g = [
            Complex::new(0.3, 0.2),
            Complex::new(-0.1, 0.4),
            Complex::new(0.2, -0.3),
        ];
        let h = 1e-5;
        let kr = Complex::new(1.2, 0.1);
        for (l, m) in (1..=3).flat_map(|l| (-l..=l).map(move |m| (l, m))) {
            for (pol, helicity) in [(0, false), (1, false), (1, true)] {
                let label = WaveLabel { l, m, pol };
                for (family, offset) in [
                    (Family::HarmonicX, 0),
                    (Family::HarmonicY, 0),
                    (Family::HarmonicZ, 0),
                    (Family::Spherical(Radial::Regular), 1),
                    (Family::Spherical(Radial::Singular), 1),
                ] {
                    for (pole, inward) in [(0.0, 1.0), (PI, -1.0)] {
                        let args = |theta: f64| {
                            let mut args = arguments([kr, theta.into(), 0.3.into()]);
                            if offset == 0 {
                                args.rotate_left(1);
                            }
                            args
                        };
                        let at =
                            |t: f64| vector_wave(family, label, args(pole + t), helicity).unwrap();
                        let pole_value = at(0.0);
                        let gradient =
                            vector_wave_pullback(family, label, args(pole), helicity, g).unwrap();
                        let azimuthal: Complex = pole_value
                            .iter()
                            .zip(g)
                            .map(|(x, g)| (Complex::i() * f64::from(m) * x).conj() * g)
                            .sum();
                        let scale = 1.0 + pole_value.iter().map(|x| x.norm()).sum::<f64>();
                        let context = format!("{family:?} {label:?} {helicity} at {pole}");
                        assert!(
                            (gradient[offset + 1] - azimuthal).norm() < 1e-14 * scale,
                            "{context}: {} != {azimuthal}",
                            gradient[offset + 1]
                        );
                        let (one, two) = (at(inward * h), at(2.0 * inward * h));
                        let polar: Complex = (0..3)
                            .map(|c| {
                                let derivative = inward
                                    * (-3.0 * pole_value[c] + 4.0 * one[c] - two[c])
                                    / (2.0 * h);
                                derivative.conj() * g[c]
                            })
                            .sum();
                        assert!(
                            (gradient[offset] - polar).norm() < 1e-7 * (1.0 + polar.norm()),
                            "{context}: {} != {polar}",
                            gradient[offset]
                        );
                    }
                }
            }
        }
    }
}
