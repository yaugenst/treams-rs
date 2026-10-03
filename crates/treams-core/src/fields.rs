//! Cartesian spherical and cylindrical vector waves, field sums and field operators.
//!
//! [`field`] sums a mode expansion at sample points, and [`operator`] builds the
//! matrix that maps mode coefficients to those fields; both have analytic gradients.
//!
//! The waves use Cartesian components built from solid harmonics, so they stay finite
//! on the z axis and at the expansion centre, where polar coordinates are singular.
//!
//! Upstream: `treams.efield` and the vector waves `treams.special.vsw_*` and `vcw_*`.
#![allow(clippy::indexing_slicing)] // Fixed Cartesian vectors and Hessians.

use std::{
    array::from_fn,
    f64::consts::{FRAC_1_SQRT_2, PI},
};

use crate::{
    Complex, Error, Result,
    basis::{ModeLabel, MultipoleBasis, validate_wavenumbers},
    cw,
    numerics::{
        self, Jet, finite,
        parallel::{try_fill_chunks, try_fold_ordered, try_map},
    },
    special::{Radial, SERIES_RADIUS, Solid, bessel, helicity_sign, polarized_wave, solid},
    sw::Mode,
};

/// A spherical or cylindrical electric vector wave with its analytic derivatives with
/// respect to the evaluation point and the medium wavenumber `k`.
///
/// Evaluations without derivatives leave `position` and `k` zero.
#[derive(Clone, Copy, Debug, Default)]
pub struct VectorWave {
    /// Cartesian electric field.
    pub value: [Complex; 3],
    /// `position[component][axis]` is the field's Cartesian Jacobian.
    pub position: [[Complex; 3]; 3],
    /// Derivative of each component with respect to the complex wavenumber `k`.
    pub k: [Complex; 3],
}

/// Cartesian cylindrical vector wave and its point and wavenumber derivatives.
/// Adjacent scalar harmonics remove all polar-coordinate divisions at the axis.
///
/// Upstream: `treams.special.vcw_rA`, `vcw_A`, `vcw_rM`, `vcw_M`, `vcw_rN` and `vcw_N`,
/// converted to Cartesian components as by `treams.special.vcyl2car`.
pub fn cylindrical_wave(
    mode: cw::Mode,
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    radial: Radial,
) -> Result<VectorWave> {
    let [m, n] = cylindrical_components::<SPATIAL_AND_K>(mode, k, position, radial)?;
    Ok(pack_cylindrical(polarized_wave(m, n, mode.pol, helicity)))
}

/// The value, point derivatives (jet variables 0 to 2) and wavenumber derivative
/// (variable 3) of a cylindrical wave; jets without derivatives give zeros.
fn pack_cylindrical<const N: usize>(fields: [Jet<N>; 3]) -> VectorWave {
    VectorWave {
        value: fields.map(|v| v.value),
        position: fields.map(|v| from_fn(|i| v.derivative.get(i).copied().unwrap_or_default())),
        k: fields.map(|v| v.derivative.get(3).copied().unwrap_or_default()),
    }
}

/// The M and N components of a cylindrical wave, which do not depend on its
/// polarization. Jet variables are the position (0..3), `k` (3) and `kz` (4).
fn cylindrical_components<const N: usize>(
    mode: cw::Mode,
    k: Complex,
    position: [f64; 3],
    radial: Radial,
) -> Result<[[Jet<N>; 3]; 2]> {
    mode.validate()?;
    if !finite(k) || k == Complex::default() || position.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidInput(
            "require a finite nonzero wavenumber and finite field position".into(),
        ));
    }
    let r: [Jet<N>; 3] = from_fn(|i| Jet::variable(position[i], i));
    let k = Jet::<N>::variable(k, 3);
    let kz = Jet::<N>::variable(mode.kz, 4);
    let transverse = cw::transverse(k, kz);
    if transverse.value == Complex::default() {
        return Err(Error::InvalidInput(
            "cylindrical cutoff requires a limiting formulation".into(),
        ));
    }
    let rho = position[0].hypot(position[1]);
    let [lower, center, upper] =
        if radial == Radial::Regular && (transverse.value * rho).norm() < SERIES_RADIUS {
            [mode.m - 1, mode.m, mode.m + 1].map(|m| cw::regular_harmonic(m, transverse, r, kz))
        } else {
            if rho == 0.0 {
                return Err(Error::NonFinite(
                    "outgoing cylindrical wave is singular on the axis".into(),
                ));
            }
            let radius = (r[0] * r[0] + r[1] * r[1]).sqrt();
            let argument = transverse * radius;
            let radial = crate::special::cylindrical_radial(mode.m, argument.value, radial)?;
            let value = argument.chain(radial.value, radial.first);
            let derivative = argument.chain(radial.first, radial.second);
            let azimuth = (r[0] + Complex::i() * r[1]) / radius;
            let phase = (Complex::i() * kz * r[2]).exp() * azimuth.powi(mode.m);
            let adjacent = f64::from(mode.m) * value / argument;
            [
                (adjacent + derivative) * phase / azimuth,
                value * phase,
                (adjacent - derivative) * phase * azimuth,
            ]
        };
    let m = [
        0.5 * Complex::i() * (lower + upper),
        0.5 * (upper - lower),
        Jet::default(),
    ];
    let n = [
        0.5 * Complex::i() * kz / k * (lower - upper),
        -0.5 * kz / k * (lower + upper),
        transverse / k * center,
    ];
    Ok([m, n])
}

fn cross(a: [Complex; 3], b: [Complex; 3]) -> [Complex; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// z_l(kr)/r^l has a finite regular limit. Evaluating its series directly
// avoids divisions by zero and cancellation in the near-origin field gradients.
// Elsewhere only the value of z_l is needed: one complex Bessel evaluation.
fn scaled_radial(l: u32, k: Complex, r: f64, radial: Radial) -> Result<Complex> {
    let x = k * r;
    if radial == Radial::Regular && x.norm() < SERIES_RADIUS {
        let mut denominator = 1.0;
        for i in 0..=l {
            denominator *= f64::from(2 * i + 1);
        }
        let mut coefficient = k.powu(l) / denominator;
        let mut sum = Complex::default();
        for q in 0..32 {
            sum += coefficient * x.powu(2 * q);
            coefficient /= -2.0 * f64::from(q + 1) * f64::from(2 * l + 2 * q + 3);
        }
        return Ok(sum);
    }
    if x.norm_sqr() == 0.0 {
        return Err(Error::InvalidInput(
            "outgoing spherical waves are singular at zero".into(),
        ));
    }
    Ok(bessel(f64::from(l), x, radial.into(), true, 0)? / r.powf(f64::from(l)))
}

/// A common length unit bounds solid-harmonic coordinates by one and bounds the
/// wavenumber in the regular origin series. Hold this scale fixed when taking
/// derivatives: the wave is unchanged by any common change of length units.
fn spherical_arguments(k: Complex, position: [f64; 3], radial: Radial) -> (Complex, [f64; 3], f64) {
    let mut scale = position.iter().map(|x| x.abs()).fold(0.0, f64::max);
    // Only regular waves need an origin series. Singular waves keep unit-sized
    // coordinates even at small kr, so dividing a large Hankel value by r^l
    // cannot overflow before multiplication by its solid harmonic.
    if radial == Radial::Regular || scale == 0.0 {
        scale = scale.max(1.0 / k.re.abs().max(k.im.abs()));
    }
    (k * scale, position.map(|x| x / scale), scale)
}

/// Evaluate a vector spherical wave in treams normalization.
/// Parity polarization 0 is M, 1 is N; helicity is `(N + (2pol-1) M)/sqrt(2)`.
///
/// Upstream: `treams.special.vsw_rA`, `vsw_A`, `vsw_rM`, `vsw_M`, `vsw_rN` and `vsw_N`,
/// converted to Cartesian components as by `treams.special.vsph2car`.
pub fn spherical_wave(
    mode: Mode,
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    radial: Radial,
) -> Result<VectorWave> {
    spherical_wave_impl::<true>(mode, k, position, helicity, radial)
}

/// [`spherical_wave`] with the point and wavenumber derivatives only if `DERIVATIVES`.
fn spherical_wave_impl<const DERIVATIVES: bool>(
    mode: Mode,
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    radial: Radial,
) -> Result<VectorWave> {
    mode.validate()?;
    if !finite(k) || k.norm_sqr() == 0.0 || position.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "require a finite nonzero wavenumber and finite position".into(),
        ));
    }
    let (k, position, scale) = spherical_arguments(k, position, radial);
    let r2 = position.iter().map(|v| v * v).sum::<f64>();
    let l = mode.l.unsigned_abs();
    let radials = [
        scaled_radial(l, k, r2.sqrt(), radial)?,
        scaled_radial(l + 1, k, r2.sqrt(), radial)?,
        if DERIVATIVES {
            scaled_radial(l + 2, k, r2.sqrt(), radial)?
        } else {
            Complex::default()
        },
    ];
    let solid = solid::<DERIVATIVES>(mode.l, mode.m, position);
    let parts = spherical_parts::<DERIVATIVES>(mode.l, k, position, r2, &solid, radials, scale);
    Ok(combine::<DERIVATIVES>(
        &parts,
        normalization(mode.l, mode.m),
        weights(mode.pol, helicity),
    ))
}

/// The unnormalized parts `[n, m]` of a spherical wave of degree `l`, which every
/// polarization combines with its own [`weights`].
///
/// With the scaled radial function `c = z_l(kr) / r^l` and the solid harmonic
/// `S = r^l P_l^m(cos theta) e^(i m phi)`, the scalar wave is `psi = c S`. Then `m = r x grad psi`,
/// the negative of the unnormalized `M = curl(r psi) = grad psi x r`, and `n` is the
/// unnormalized `N`. The inputs are the scaled radial functions of orders `l`, `l + 1` and
/// `l + 2`; the last one enters only the derivatives. `scale` returns the derivatives
/// from the internal length unit to the caller's unit.
fn spherical_parts<const DERIVATIVES: bool>(
    l: i32,
    k: Complex,
    position: [f64; 3],
    r2: f64,
    solid: &Solid,
    [c, e, f]: [Complex; 3],
    scale: f64,
) -> [VectorWave; 2] {
    let degree = f64::from(l);
    let d = (degree + 1.0) / k * c - r2 * e;
    let vector = position.map(|v| Complex::new(v, 0.0));
    let rotation = cross(vector, solid.gradient);
    let mut n = VectorWave {
        value: from_fn(|i| d * solid.gradient[i] + degree * e * solid.value * position[i]),
        ..VectorWave::default()
    };
    let mut m = VectorWave {
        value: from_fn(|i| c * rotation[i]),
        ..VectorWave::default()
    };
    if DERIVATIVES {
        let ck = degree / k * c - r2 * e;
        let ek = (degree + 1.0) / k * e - r2 * f;
        let dk = (degree + 1.0) / k * ck - (degree + 1.0) / k.powu(2) * c - r2 * ek;
        n.k =
            from_fn(|i| scale * (dk * solid.gradient[i] + degree * ek * solid.value * position[i]));
        m.k = from_fn(|i| scale * ck * rotation[i]);
        for axis in 0..3 {
            let ca = -k * e * position[axis];
            let ea = -k * f * position[axis];
            let da = (degree + 1.0) / k * ca - 2.0 * position[axis] * e - r2 * ea;
            let mut unit = [Complex::default(); 3];
            unit[axis] = Complex::new(1.0, 0.0);
            let first = cross(unit, solid.gradient);
            let second = cross(vector, from_fn(|i| solid.hessian[i][axis]));
            for i in 0..3 {
                n.position[i][axis] = (da * solid.gradient[i]
                    + d * solid.hessian[i][axis]
                    + degree
                        * (ea * solid.value * position[i]
                            + e * solid.gradient[axis] * position[i]
                            + if i == axis {
                                e * solid.value
                            } else {
                                Complex::default()
                            }))
                    / scale;
                m.position[i][axis] = (ca * rotation[i] + c * (first[i] + second[i])) / scale;
            }
        }
    }
    [n, m]
}

/// Normalization `i sqrt((2l + 1) / (4 pi l (l + 1))) sqrt((l - m)! / (l + m)!)` of the
/// spherical wave `(l, m)`; it depends on neither point nor `k`.
fn normalization(l: i32, m: i32) -> Complex {
    let degree = f64::from(l);
    Complex::i()
        * ((2.0 * degree + 1.0) / (4.0 * PI * degree * (degree + 1.0))).sqrt()
        * crate::special::factorial_ratio_sqrt(l, m)
}

/// Weights `[weight_n, weight_m]` of the N and M parts of polarization `pol`: parity
/// polarization 1 is N and 0 is M; helicity polarization `pol` is
/// `(N + (2 pol - 1) M) / sqrt(2)`, as in `treams.special.vsw_A`.
fn weights(pol: u8, helicity: bool) -> [f64; 2] {
    if helicity {
        [FRAC_1_SQRT_2, helicity_sign(pol) * FRAC_1_SQRT_2]
    } else if pol == 1 {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    }
}

/// The wave `normalization (weight_n N + weight_m M)` of the value and requested
/// derivatives. The part `m` from [`spherical_parts`] is `-M`, so it enters with a
/// minus sign.
fn combine<const DERIVATIVES: bool>(
    [n, m]: &[VectorWave; 2],
    normalization: Complex,
    [weight_n, weight_m]: [f64; 2],
) -> VectorWave {
    let combine = |n: Complex, m: Complex| normalization * (weight_n * n - weight_m * m);
    let mut wave = VectorWave {
        value: from_fn(|i| combine(n.value[i], m.value[i])),
        ..VectorWave::default()
    };
    if DERIVATIVES {
        wave.k = from_fn(|i| combine(n.k[i], m.k[i]));
        wave.position = from_fn(|i| from_fn(|a| combine(n.position[i][a], m.position[i][a])));
    }
    wave
}

/// What [`field`] saves for its pullback: the wave set, the points and the
/// coefficients. Its size grows with the samples plus the modes, not their product.
#[derive(Debug)]
pub struct FieldResidual {
    waves: WaveSet,
    points: Vec<[f64; 3]>,
    coefficients: Vec<Complex>,
}

/// Derivative level of a wave evaluation: values only. Each level is the jet size of
/// the cylindrical path; the spherical path evaluates derivatives above this level.
pub(crate) const VALUES: usize = 0;
/// Derivative level: values, point derivatives (jet variables 0 to 2) and the
/// wavenumber derivative (variable 3).
pub(crate) const SPATIAL_AND_K: usize = 4;
/// Derivative level: [`SPATIAL_AND_K`] plus the axial-wavenumber derivative of
/// cylindrical waves (variable 4).
pub(crate) const WITH_AXIAL: usize = 5;

/// Evaluates the waves of a multipole basis at any point, for one pair of medium
/// wavenumbers, polarization convention and radial kind.
#[derive(Debug)]
pub(crate) struct WaveSet {
    basis: MultipoleBasis,
    ks: [Complex; 2],
    helicity: bool,
    radial: Radial,
    /// Point-independent normalization of each spherical mode; empty for cylinders.
    normalizations: Vec<Complex>,
    /// Scaled radial functions cached per position and wavenumber: the largest
    /// spherical degree plus three, or zero for cylinders.
    radial_table_len: usize,
}

/// Work shared by the waves of one sample point: scaled radial functions per
/// (position, wavenumber, order), and the last spherical parts or cylindrical
/// components, which the adjacent polarization of the same mode reuses.
/// Every entry is keyed by all of its inputs, so any mode order is
/// evaluated exactly as without the cache. `N` is the derivative level
/// ([`VALUES`], [`SPATIAL_AND_K`] or [`WITH_AXIAL`]).
pub(crate) struct SampleCache<const N: usize> {
    radials: Vec<Option<Complex>>,
    spherical: Option<(PartsKey, [VectorWave; 2])>,
    cylindrical: Option<(ComponentsKey, [[Jet<N>; 3]; 2])>,
}

/// Position index, wavenumber index, degree and order of spherical parts.
type PartsKey = (usize, usize, i32, i32);
/// Position index, wavenumber index, order and axial-wavenumber bits of cylindrical components.
type ComponentsKey = (usize, usize, i32, u64);

/// The waves of `basis` and the sample points, checked in this order: basis, points,
/// wavenumbers.
fn sampled_waves(
    basis: MultipoleBasis,
    points: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
    radial: Radial,
) -> Result<(WaveSet, Vec<[f64; 3]>)> {
    basis.validate()?;
    if points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput("field points must be finite".into()));
    }
    Ok((
        WaveSet::of_valid_basis(basis, ks, helicity, radial)?,
        points,
    ))
}

impl WaveSet {
    pub(crate) fn new(
        basis: MultipoleBasis,
        ks: [Complex; 2],
        helicity: bool,
        radial: Radial,
    ) -> Result<Self> {
        basis.validate()?;
        Self::of_valid_basis(basis, ks, helicity, radial)
    }

    /// [`Self::new`] for a basis that has passed `validate`.
    fn of_valid_basis(
        basis: MultipoleBasis,
        ks: [Complex; 2],
        helicity: bool,
        radial: Radial,
    ) -> Result<Self> {
        validate_wavenumbers(ks, helicity, false)?;
        let (normalizations, radial_table_len) = match &basis {
            MultipoleBasis::Spherical(b) => (
                b.modes
                    .iter()
                    .map(|&(_, mode)| normalization(mode.l, mode.m))
                    .collect(),
                b.modes
                    .iter()
                    .map(|&(_, mode)| mode.l.unsigned_abs() as usize + 3)
                    .max()
                    .unwrap_or_default(),
            ),
            MultipoleBasis::Cylindrical(_) => (Vec::new(), 0),
        };
        Ok(Self {
            basis,
            ks,
            helicity,
            radial,
            normalizations,
            radial_table_len,
        })
    }

    pub(crate) fn cache<const N: usize>(&self) -> SampleCache<N> {
        SampleCache {
            radials: vec![None; 2 * self.basis.positions().len() * self.radial_table_len],
            spherical: None,
            cylindrical: None,
        }
    }

    /// Index of the wavenumber of polarization `pol`; equal wavenumbers share one.
    fn wavenumber(&self, pol: u8) -> usize {
        if self.ks[0] == self.ks[1] {
            0
        } else {
            usize::from(pol)
        }
    }

    /// Wave `i` at `point` and, for cylindrical waves at level [`WITH_AXIAL`], its
    /// axial-wavenumber derivative. Derivatives are evaluated only above [`VALUES`].
    pub(crate) fn wave<const N: usize>(
        &self,
        i: usize,
        point: [f64; 3],
        cache: &mut SampleCache<N>,
    ) -> Result<(VectorWave, [Complex; 3])> {
        match &self.basis {
            MultipoleBasis::Spherical(basis) => {
                let (pidx, mode) = basis.modes[i];
                let wavenumber = self.wavenumber(mode.pol);
                let key = (pidx, wavenumber, mode.l, mode.m);
                let parts = match cache.spherical {
                    Some((cached, parts)) if cached == key => parts,
                    _ => {
                        let parts = self.cached_parts(pidx, wavenumber, mode, point, cache)?;
                        cache.spherical = Some((key, parts));
                        parts
                    }
                };
                let normalization = self.normalizations[i];
                let weights = weights(mode.pol, self.helicity);
                let wave = if N > VALUES {
                    combine::<true>(&parts, normalization, weights)
                } else {
                    combine::<false>(&parts, normalization, weights)
                };
                Ok((wave, [Complex::default(); 3]))
            }
            MultipoleBasis::Cylindrical(basis) => {
                let (pidx, mode) = basis.modes[i];
                let wavenumber = self.wavenumber(mode.pol);
                let key = (pidx, wavenumber, mode.m, mode.kz.to_bits());
                let components = match cache.cylindrical {
                    Some((cached, components)) if cached == key => components,
                    _ => {
                        let position = from_fn(|a| point[a] - basis.positions[pidx][a]);
                        let k = self.ks[wavenumber];
                        let components = cylindrical_components(mode, k, position, self.radial)?;
                        cache.cylindrical = Some((key, components));
                        components
                    }
                };
                let [m, n] = components;
                let fields = polarized_wave(m, n, mode.pol, self.helicity);
                let axial = fields.map(|v| v.derivative.get(4).copied().unwrap_or_default());
                Ok((pack_cylindrical(fields), axial))
            }
        }
    }

    /// The spherical parts of `mode` at `point` from the cached radial functions and
    /// a scaled solid harmonic. It rejects what [`spherical_wave`] rejects: a wavenumber whose
    /// square underflows and a relative position that overflows.
    fn cached_parts<const N: usize>(
        &self,
        pidx: usize,
        wavenumber: usize,
        mode: Mode,
        point: [f64; 3],
        cache: &mut SampleCache<N>,
    ) -> Result<[VectorWave; 2]> {
        let position: [f64; 3] = from_fn(|a| point[a] - self.basis.positions()[pidx][a]);
        let k = self.ks[wavenumber];
        if k.norm_sqr() == 0.0 || position.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidInput(
                "require a finite nonzero wavenumber and finite position".into(),
            ));
        }
        let (k, position, scale) = spherical_arguments(k, position, self.radial);
        let r2 = position.iter().map(|v| v * v).sum::<f64>();
        let table = (2 * pidx + wavenumber) * self.radial_table_len;
        let mut radial = |order: u32| -> Result<Complex> {
            let entry = &mut cache.radials[table + order as usize];
            if let Some(value) = *entry {
                return Ok(value);
            }
            let value = scaled_radial(order, k, r2.sqrt(), self.radial)?;
            *entry = Some(value);
            Ok(value)
        };
        let l = mode.l.unsigned_abs();
        let radials = [
            radial(l)?,
            radial(l + 1)?,
            if N > VALUES {
                radial(l + 2)?
            } else {
                Complex::default()
            },
        ];
        let solid = if N > VALUES {
            solid::<true>(mode.l, mode.m, position)
        } else {
            solid::<false>(mode.l, mode.m, position)
        };
        Ok(if N > VALUES {
            spherical_parts::<true>(mode.l, k, position, r2, &solid, radials, scale)
        } else {
            spherical_parts::<false>(mode.l, k, position, r2, &solid, radials, scale)
        })
    }

    /// The pullback over all `points`; level [`WITH_AXIAL`] adds per-mode axial
    /// gradients.
    fn pullback<const N: usize>(
        &self,
        points: &[[f64; 3]],
        coefficients: Option<&[Complex]>,
        cotangent: impl Fn(usize, usize) -> [Complex; 3] + Sync,
    ) -> Result<(FieldGradient, Vec<f64>)> {
        let with_axial = N == WITH_AXIAL;
        if with_axial && matches!(self.basis, MultipoleBasis::Spherical(_)) {
            return Err(Error::InvalidInput(
                "axial field derivatives require a cylindrical basis".into(),
            ));
        }
        let zero = || Accumulator {
            gradient: FieldGradient {
                coefficients: vec![Complex::default(); coefficients.map_or(0, <[Complex]>::len)],
                points: Vec::new(),
                positions: vec![[0.0; 3]; self.basis.positions().len()],
                ks: [Complex::default(); 2],
            },
            axial: vec![0.0; if with_axial { self.basis.len() } else { 0 }],
        };
        let mut point_gradients = vec![[0.0; 3]; points.len()];
        let samples: Vec<_> = point_gradients.iter_mut().zip(points).collect();
        let Accumulator {
            mut gradient,
            axial,
        } = try_fold_ordered(
            samples,
            true,
            zero,
            |mut sum, sample, (point_gradient, point)| {
                let mut cache = self.cache::<N>();
                for i in 0..self.basis.len() {
                    let (pidx, pol) = self.basis.position_pol(i);
                    let (wave, axial_derivative) = self.wave(i, *point, &mut cache)?;
                    let amplitude = coefficients.map_or(Complex::new(1.0, 0.0), |c| c[i]);
                    let g = cotangent(sample, i);
                    for (component, &cot) in g.iter().enumerate() {
                        if coefficients.is_some() {
                            sum.gradient.coefficients[i] += wave.value[component].conj() * cot;
                        }
                        sum.gradient.ks[pol] += (amplitude * wave.k[component]).conj() * cot;
                        if with_axial {
                            sum.axial[i] +=
                                (cot.conj() * amplitude * axial_derivative[component]).re;
                        }
                        for (axis, point_derivative) in point_gradient.iter_mut().enumerate() {
                            let derivative =
                                (cot.conj() * amplitude * wave.position[component][axis]).re;
                            *point_derivative += derivative;
                            sum.gradient.positions[pidx][axis] -= derivative;
                        }
                    }
                }
                Ok(sum)
            },
            Accumulator::merge,
        )?;
        gradient.points = point_gradients;
        Ok((gradient, axial))
    }
}

/// Partial sums of a field pullback over a chunk of consecutive samples. Point
/// gradients are written in place, one per sample.
struct Accumulator {
    gradient: FieldGradient,
    axial: Vec<f64>,
}

impl Accumulator {
    fn merge(mut self, other: Self) -> Self {
        let (gradient, sum) = (other.gradient, &mut self.gradient);
        for (a, b) in sum.coefficients.iter_mut().zip(gradient.coefficients) {
            *a += b;
        }
        for (a, b) in sum
            .positions
            .iter_mut()
            .flatten()
            .zip(gradient.positions.iter().flatten())
        {
            *a += b;
        }
        for (a, b) in sum.ks.iter_mut().zip(gradient.ks) {
            *a += b;
        }
        for (a, b) in self.axial.iter_mut().zip(other.axial) {
            *a += b;
        }
        self
    }
}

/// Input gradients of a field or operator pullback, under the convention
/// `dL = Re sum(conj(g) dx)`.
#[derive(Debug)]
pub struct FieldGradient {
    /// Gradients of the complex mode coefficients; empty for [`OperatorResidual`].
    pub coefficients: Vec<Complex>,
    /// Gradients of the sample points.
    pub points: Vec<[f64; 3]>,
    /// Gradients of the expansion centres of the basis.
    pub positions: Vec<[f64; 3]>,
    /// Gradients of the wavenumbers of polarizations 0 and 1 (negative and positive
    /// helicity).
    pub ks: [Complex; 2],
}

/// Evaluate weighted electric fields in parallel over samples: one Cartesian field
/// vector per point.
///
/// Upstream: `treams.efield(r, basis=...) @ coefficients`.
pub fn field(
    basis: impl Into<MultipoleBasis>,
    coefficients: Vec<Complex>,
    points: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
    radial: Radial,
) -> Result<(Vec<[Complex; 3]>, FieldResidual)> {
    let basis = basis.into();
    if coefficients.len() != basis.len() || coefficients.iter().any(|&v| !finite(v)) {
        return Err(Error::InvalidInput(
            "require one finite field coefficient per mode".into(),
        ));
    }
    let (waves, points) = sampled_waves(basis, points, ks, helicity, radial)?;
    let value = try_map(points.len(), points.len() > 1, |index| -> Result<_> {
        let mut cache = waves.cache::<VALUES>();
        let mut value = [Complex::default(); 3];
        for (i, &amplitude) in coefficients.iter().enumerate() {
            let (wave, _) = waves.wave(i, points[index], &mut cache)?;
            for (v, f) in value.iter_mut().zip(wave.value) {
                *v += amplitude * f;
            }
        }
        Ok(value)
    })?;
    Ok((
        value,
        FieldResidual {
            waves,
            points,
            coefficients,
        },
    ))
}

impl FieldResidual {
    /// Samples and Cartesian components: the shape of the field array.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.points.len(), 3)
    }

    /// Input gradients from `cotangent`, the gradient of a real loss with respect to the
    /// field, one Cartesian vector per sample. The waves are evaluated again per sample;
    /// no dense Jacobian is stored.
    ///
    /// Each point gradient comes from its own sample only. The coefficient, position
    /// and wavenumber gradients add in chunks of consecutive samples fixed by the
    /// sample count, and the chunk sums in chunk order
    /// (`numerics::parallel::try_fold_ordered`), so the thread count does not change
    /// them.
    pub fn pullback(self, cotangent: &[[Complex; 3]]) -> Result<FieldGradient> {
        self.pullback_impl::<SPATIAL_AND_K>(cotangent)
            .map(|(gradient, _)| gradient)
    }

    /// [`Self::pullback`] plus a real axial-wavenumber gradient for each mode of a
    /// cylindrical basis; the axial gradients add their chunk sums the same way.
    pub fn pullback_axial(self, cotangent: &[[Complex; 3]]) -> Result<(FieldGradient, Vec<f64>)> {
        self.pullback_impl::<WITH_AXIAL>(cotangent)
    }

    fn pullback_impl<const N: usize>(
        self,
        cotangent: &[[Complex; 3]],
    ) -> Result<(FieldGradient, Vec<f64>)> {
        if cotangent.len() != self.points.len() || cotangent.iter().flatten().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput("invalid field cotangent".into()));
        }
        self.waves
            .pullback::<N>(&self.points, Some(&self.coefficients), |sample, _| {
                cotangent[sample]
            })
    }
}

/// What [`operator`] saves for its pullback: the wave set and the points, without the
/// matrix.
#[derive(Debug)]
pub struct OperatorResidual {
    waves: WaveSet,
    points: Vec<[f64; 3]>,
}

/// Matrix mapping multipole amplitudes to Cartesian samples. Rows pack (sample, component).
/// The residual keeps the basis, wavenumbers and points, without the output or any
/// derivative matrix.
///
/// Upstream: `treams.efield`, which returns the matrix with shape `(..., 3, modes)`.
pub fn operator(
    basis: impl Into<MultipoleBasis>,
    points: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
    radial: Radial,
) -> Result<(nalgebra::DMatrix<Complex>, OperatorResidual)> {
    let (waves, points) = sampled_waves(basis.into(), points, ks, helicity, radial)?;
    let samples = points.len();
    let mut value = numerics::zeros(3 * samples, waves.basis.len())?;
    if samples > 0 {
        // Each task evaluates all modes at a block of consecutive samples, so the
        // waves of one sample share their radial functions. It writes the rows of
        // its samples in every column.
        let block = samples
            .div_ceil(crate::threads::current_num_threads().saturating_mul(8))
            .clamp(1, 64);
        // One column slice per block and mode: a third of the matrix for one-sample
        // blocks.
        let mut blocks: Vec<Vec<&mut [Complex]>> = (0..samples.div_ceil(block))
            .map(|_| {
                let mut columns = Vec::new();
                numerics::reserve(&mut columns, waves.basis.len()).map(|()| columns)
            })
            .collect::<Result<_>>()?;
        for column in value.as_mut_slice().chunks_exact_mut(3 * samples) {
            for (rows, columns) in column.chunks_mut(3 * block).zip(&mut blocks) {
                columns.push(rows);
            }
        }
        try_fill_chunks(
            &mut blocks,
            1,
            samples > block,
            |index, chunk| -> Result<()> {
                let columns = &mut chunk[0];
                let block_points = points.iter().skip(index * block).take(block);
                for (sample, &point) in block_points.enumerate() {
                    let mut cache = waves.cache::<VALUES>();
                    for (i, column) in columns.iter_mut().enumerate() {
                        let (wave, _) = waves.wave(i, point, &mut cache)?;
                        column[3 * sample..3 * sample + 3].copy_from_slice(&wave.value);
                    }
                }
                Ok(())
            },
        )?;
    }
    Ok((value, OperatorResidual { waves, points }))
}

impl OperatorResidual {
    /// Flattened operator shape (three times samples, modes).
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (3 * self.points.len(), self.waves.basis.len())
    }

    /// Point, position and wavenumber gradients from `cotangent`, the gradient of a real
    /// loss with respect to the matrix; the coefficient gradient is empty.
    ///
    /// Each point gradient comes from its own sample only. The position and wavenumber
    /// gradients add as in [`FieldResidual::pullback`], so the thread count does not
    /// change them.
    pub fn pullback(self, cotangent: &nalgebra::DMatrix<Complex>) -> Result<FieldGradient> {
        self.pullback_impl::<SPATIAL_AND_K>(cotangent)
            .map(|(gradient, _)| gradient)
    }

    /// [`Self::pullback`] plus a real axial-wavenumber gradient for each mode of a
    /// cylindrical basis; the axial gradients add their chunk sums the same way.
    pub fn pullback_axial(
        self,
        cotangent: &nalgebra::DMatrix<Complex>,
    ) -> Result<(FieldGradient, Vec<f64>)> {
        self.pullback_impl::<WITH_AXIAL>(cotangent)
    }

    fn pullback_impl<const N: usize>(
        self,
        cotangent: &nalgebra::DMatrix<Complex>,
    ) -> Result<(FieldGradient, Vec<f64>)> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "invalid field operator cotangent".into(),
            ));
        }
        self.waves
            .pullback::<N>(&self.points, None, |sample, mode| {
                from_fn(|i| cotangent[(3 * sample + i, mode)])
            })
    }
}

#[cfg(test)]
mod tests {
    use std::array::from_fn;

    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::{field, operator, spherical_wave, spherical_wave_impl};
    use crate::{
        Complex, Error,
        special::Radial,
        sw::Mode,
        test_support::{
            assert_same_bits_on_pools, bits, degree_order, patterned, radial, spherical_basis,
        },
    };

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn forward_waves_carry_no_derivatives(
            (l, m) in degree_order(1..6),
            pol in 0_u8..2,
            position in prop::array::uniform3(-1.5_f64..1.5),
            helicity in any::<bool>(),
            radial in radial(),
        ) {
            check_forward_wave(Mode { l, m, pol }, position, helicity, radial)?;
        }
    }

    /// Spherical fields and operators reject what a single spherical wave rejects,
    /// a wavenumber whose square underflows and a sample whose offset from its basis position
    /// overflows, with the same message instead of NaN values. The pullbacks evaluate
    /// the same waves and are reached only through a successful forward pass.
    #[test]
    fn spherical_fields_reject_what_a_single_wave_rejects() {
        let message = "require a finite nonzero wavenumber and finite position";
        let rejected = |result: crate::Result<()>| matches!(result, Err(Error::InvalidInput(ref text)) if text == message);
        let tiny = [Complex::new(1e-170, 1e-170); 2];
        let unit = [Complex::new(1.0, 0.0); 2];
        let cases = [
            (tiny, [0.0; 3], [0.3, 0.2, 0.1]),
            (unit, [-1e308, 0.0, 0.0], [1e308, 0.0, 0.0]),
        ];
        for (ks, basis_position, point) in cases {
            for radial in [Radial::Regular, Radial::Singular] {
                for helicity in [true, false] {
                    let basis = spherical_basis(1, basis_position);
                    let position = from_fn(|a| point[a] - basis_position[a]);
                    let mode = basis.modes[0].1;
                    let wave = spherical_wave(mode, ks[0], position, helicity, radial);
                    assert!(rejected(wave.map(drop)));
                    let coefficients = vec![Complex::new(1.0, 0.0); basis.modes.len()];
                    let values = field(
                        basis.clone(),
                        coefficients,
                        vec![point],
                        ks,
                        helicity,
                        radial,
                    );
                    assert!(rejected(values.map(drop)), "{radial:?}");
                    let matrix = operator(basis, vec![point], ks, helicity, radial);
                    assert!(rejected(matrix.map(drop)), "{radial:?}");
                }
            }
        }
    }

    /// Field and operator pullbacks add their shared gradients in chunks fixed by the
    /// sample count: with more samples than chunks, every gradient of a weighted
    /// spherical field and of the axial pullback of a cylindrical operator repeats bit
    /// for bit on every pool size.
    #[test]
    fn pullbacks_do_not_depend_on_the_thread_count() {
        let points: Vec<[f64; 3]> = (0..120_u32)
            .map(|i| {
                let t = f64::from(i);
                [
                    1.5 * (0.37 * t).sin(),
                    1.2 * (0.53 * t).cos(),
                    0.9 * (0.71 * t).sin() + 0.1,
                ]
            })
            .collect();
        let ks = [Complex::new(1.1, 0.02), Complex::new(1.3, 0.01)];
        let modes = crate::sw::modes(3).unwrap();
        let spheres = crate::sw::Basis {
            modes: (0..2)
                .flat_map(|p| modes.iter().map(move |&mode| (p, mode)))
                .collect(),
            positions: vec![[0.1, 0.2, -0.3], [-0.2, 0.0, 0.4]],
        };
        let coefficients = patterned(spheres.modes.len(), 1, 0.2).as_slice().to_vec();
        let g = patterned(3, points.len(), 0.9);
        let g = g.as_slice().as_chunks::<3>().0;
        assert_same_bits_on_pools(|| {
            let (_, residual) = field(
                spheres.clone(),
                coefficients.clone(),
                points.clone(),
                ks,
                true,
                Radial::Regular,
            )
            .unwrap();
            let g = residual.pullback(g).unwrap();
            bits(&[&g.coefficients, &g.points, &g.positions, &g.ks])
        });
        let cylinders = crate::cw::Basis {
            modes: (0..2)
                .flat_map(|p| {
                    [0.2, -0.35].into_iter().flat_map(move |kz| {
                        (-3..=3).flat_map(move |m| {
                            [1, 0].map(|pol| (p, crate::cw::Mode { kz, m, pol }))
                        })
                    })
                })
                .collect(),
            positions: vec![[0.1, 0.2, 0.0], [-0.2, 0.0, 0.0]],
        };
        let g = patterned(3 * points.len(), cylinders.modes.len(), 0.4);
        assert_same_bits_on_pools(|| {
            let (_, residual) =
                operator(cylinders.clone(), points.clone(), ks, true, Radial::Regular).unwrap();
            let (g, axial) = residual.pullback_axial(&g).unwrap();
            bits(&[&g.points, &g.positions, &g.ks, &axial])
        });
    }

    /// A forward-only spherical wave has exactly the value of the differentiated one
    /// and zero position and wavenumber derivatives.
    fn check_forward_wave(
        mode: Mode,
        position: [f64; 3],
        helicity: bool,
        radial: Radial,
    ) -> Result<(), TestCaseError> {
        let distance = position[0].hypot(position[1]).hypot(position[2]);
        prop_assume!(radial == Radial::Regular || distance > 0.1);
        let k = Complex::new(1.3, 0.1);
        let full = spherical_wave_impl::<true>(mode, k, position, helicity, radial).unwrap();
        let forward = spherical_wave_impl::<false>(mode, k, position, helicity, radial).unwrap();
        prop_assert_eq!(forward.value, full.value);
        prop_assert_eq!(forward.position, [[Complex::default(); 3]; 3]);
        prop_assert_eq!(forward.k, [Complex::default(); 3]);
        Ok(())
    }
}
