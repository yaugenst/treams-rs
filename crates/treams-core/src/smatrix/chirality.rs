//! Chirality-density forms of plane-wave modes, averaged over an interval.
//!
//! Two kernels compute these forms from different inputs, and each differentiates its
//! own. [`chirality_density`] takes the full and normal wavenumbers of modes with real
//! transverse wavevectors in the xy plane and returns the forms without the helicity
//! sign. [`oriented_chirality`] takes the real transverse components and the normal
//! wavenumber along any Cartesian axis and includes the helicity sign of each mode.
//!
//! Both kernels exist because their gradients answer different questions: the first
//! differentiates with respect to the medium (`k`, `kz`), the second with respect to
//! the direction of incidence (`q`, `kz`). On modes with `q.q = k^2 - kz^2` their
//! forms agree up to the helicity sign.
//!
//! Upstream: `treams.chirality_density`. Its interval average decays with `Re(kz)`
//! instead of `Im(kz)` and drops the phase of the cross form; these kernels keep both.

mod saved;

use nalgebra::DMatrix;

use crate::{
    Complex, Error, Result,
    numerics::{
        Jet, finite,
        parallel::{PARALLEL_ITEMS, try_fill_chunks, try_map},
    },
    special::helicity_sign,
};

/// Jet seed offsets of the six real local parameters of one mode: the full wavenumber
/// `k` (real, imaginary) of [`chirality_density`] or the transverse components `q` of
/// [`oriented_chirality`] at `KS` or `Q`, the normal wavenumber (real, imaginary) at
/// `KZS` and the interval endpoints at `INTERVAL`.
const KS: usize = 0;
/// See [`KS`].
const Q: usize = 0;
/// See [`KS`].
const KZS: usize = 2;
/// See [`KS`].
const INTERVAL: usize = 4;

/// What [`chirality_density`] saves for its pullback: the full and normal wavenumbers
/// and the interval.
#[derive(Debug)]
pub struct ChiralityDensityResidual {
    ks: Vec<Complex>,
    normal: Vec<Complex>,
    interval: [f64; 2],
}

/// Complex full/normal wavenumber and real interval-endpoint cotangents.
#[derive(Debug)]
pub struct ChiralityDensityGradient {
    /// Cotangents of the full wavenumbers.
    pub ks: Vec<Complex>,
    /// Cotangents of the normal wavenumbers.
    pub normal: Vec<Complex>,
    /// Cotangents of the start and end of the averaging interval.
    pub interval: [f64; 2],
}

/// The mean of `exp(slope z)` over `z` in `interval`, from a six-term series when the
/// exponent changes by less than 0.1 across the interval.
fn mean_exp<const N: usize>(slope: Jet<N>, interval: [Jet<N>; 2]) -> Jet<N> {
    let width = slope * (interval[1] - interval[0]);
    if width.value.norm() < 0.1 {
        let square = (width * 0.5).powi(2);
        let mut term = Jet::constant(1.0);
        let mut sum = term;
        for j in 1..=6 {
            term *= square / f64::from(2 * j * (2 * j + 1));
            sum += term;
        }
        (slope * (interval[0] + interval[1]) * 0.5).exp() * sum
    } else {
        ((slope * interval[1]).exp() - (slope * interval[0]).exp()) / width
    }
}

/// The up, down and cross forms of one mode of [`chirality_density`].
fn chirality_mode<const N: usize>(k: Complex, normal: Complex, z: [f64; 2]) -> [Jet<N>; 3] {
    chirality_forms(
        [Jet::variable(k.re, KS), Jet::variable(k.im, KS + 1)],
        [
            Jet::variable(normal.re, KZS),
            Jet::variable(normal.im, KZS + 1),
        ],
        [
            Jet::variable(z[0], INTERVAL),
            Jet::variable(z[1], INTERVAL + 1),
        ],
    )
}

/// The same scaled formulas for value, coordinate derivatives and directional jets.
fn chirality_forms<const N: usize>(
    k: [Jet<N>; 2],
    normal: [Jet<N>; 2],
    z: [Jet<N>; 2],
) -> [Jet<N>; 3] {
    // The common scale cancels algebraically; holding it fixed also avoids taking
    // a derivative of the normalization rather than of the physical observable.
    let scale = k[0].value.re.hypot(k[1].value.re);
    let [kr, ki] = k.map(|v| v / scale);
    let [nr, ni] = normal;
    let denominator = kr.powi(2) + ki.powi(2);
    let same = 2.0 * (kr.powi(2) + (ni / scale).powi(2)) / denominator;
    let cross = 2.0 * (kr.powi(2) - (nr / scale).powi(2)) / denominator;
    [
        same * mean_exp(-2.0 * ni, z),
        same * mean_exp(2.0 * ni, z),
        2.0 * cross * mean_exp(2.0 * Complex::i() * nr, z),
    ]
}

/// Up, down and coherent cross coefficients, shape (3, modes), before polarization.
///
/// The density is 2 Re(E* . i Z H). The cross form X enters it as Re(d* X u), with the
/// down and up amplitudes d and u. The modes have real transverse wavevectors in the xy
/// plane; the gradients cover the full and normal wavenumbers.
///
/// Upstream: `treams.chirality_density`, with the differences of the module docs.
pub fn chirality_density(
    ks: Vec<Complex>,
    normal: Vec<Complex>,
    interval: [f64; 2],
) -> Result<(DMatrix<Complex>, ChiralityDensityResidual)> {
    if ks.is_empty()
        || normal.len() != ks.len()
        || ks.iter().any(|&k| !finite(k) || k == Complex::default())
        || normal.iter().any(|&k| !finite(k))
        || interval.iter().any(|z| !z.is_finite())
    {
        return Err(Error::InvalidInput(
            "chirality requires matching finite wavenumbers, nonzero full k and finite interval"
                .into(),
        ));
    }
    let value = mode_forms(
        ks.len(),
        |j| Ok(chirality_mode::<0>(ks[j], normal[j], interval)),
        |v| v.value,
    )?;
    Ok((
        value,
        ChiralityDensityResidual {
            ks,
            normal,
            interval,
        },
    ))
}

/// The three forms of every mode as a (3, modes) matrix, in parallel from 1024 modes.
fn mode_forms<const N: usize>(
    modes: usize,
    forms: impl Fn(usize) -> Result<[Jet<N>; 3]> + Sync,
    component: impl Fn(Jet<N>) -> Complex + Sync,
) -> Result<DMatrix<Complex>> {
    let mut value = DMatrix::zeros(3, modes);
    try_fill_chunks(
        value.as_mut_slice(),
        3,
        modes >= PARALLEL_ITEMS,
        |j, column| {
            for (out, form) in column.iter_mut().zip(forms(j)?) {
                *out = component(form);
                if !finite(*out) {
                    return Err(Error::NonFinite("chirality density overflow".into()));
                }
            }
            Ok(())
        },
    )?;
    Ok(value)
}

/// Pair the six local real derivatives of every mode's forms with its column of the
/// (3, modes) cotangent, in parallel from 1024 modes.
fn mode_cotangents(
    cotangent: &DMatrix<Complex>,
    modes: usize,
    forms: impl Fn(usize) -> Result<[Jet<6>; 3]> + Sync,
) -> Result<Vec<[f64; 6]>> {
    if cotangent.shape() != (3, modes) || cotangent.iter().any(|&g| !finite(g)) {
        return Err(Error::InvalidInput("invalid chirality cotangent".into()));
    }
    try_map(modes, modes >= PARALLEL_ITEMS, |j| {
        let forms = forms(j)?;
        let g: [f64; 6] = std::array::from_fn(|p| {
            forms
                .iter()
                .enumerate()
                .map(|(i, c)| (cotangent[(i, j)].conj() * c.derivative[p]).re)
                .sum()
        });
        if g.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite("chirality derivative overflow".into()));
        }
        Ok(g)
    })
}

impl ChiralityDensityResidual {
    /// Compact output dimensions.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (3, self.ks.len())
    }

    /// Propagate one real direction in the complex wavenumbers and interval.
    pub fn pushforward(
        &self,
        ks: &[Complex],
        normal: &[Complex],
        interval: [f64; 2],
    ) -> Result<DMatrix<Complex>> {
        if ks.len() != self.ks.len()
            || normal.len() != self.normal.len()
            || ks.iter().chain(normal).any(|&v| !finite(v))
            || interval.iter().any(|v| !v.is_finite())
        {
            return Err(Error::InvalidInput(
                "chirality tangents must be finite and match inputs".into(),
            ));
        }
        let interval = std::array::from_fn(|j| Jet {
            value: self.interval[j].into(),
            derivative: [interval[j].into()],
        });
        mode_forms(
            self.ks.len(),
            |j| {
                Ok(chirality_forms(
                    real_direction(self.ks[j], ks[j]),
                    real_direction(self.normal[j], normal[j]),
                    interval,
                ))
            },
            |v| v.derivative[0],
        )
    }

    /// Recompute six local derivatives per mode instead of keeping a dense Jacobian.
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<ChiralityDensityGradient> {
        let local = mode_cotangents(cotangent, self.ks.len(), |j| {
            Ok(chirality_mode(self.ks[j], self.normal[j], self.interval))
        })?;
        let mut gradient = ChiralityDensityGradient {
            ks: Vec::with_capacity(local.len()),
            normal: Vec::with_capacity(local.len()),
            interval: [0.0; 2],
        };
        for g in local {
            gradient.ks.push(Complex::new(g[KS], g[KS + 1]));
            gradient.normal.push(Complex::new(g[KZS], g[KZS + 1]));
            gradient.interval[0] += g[INTERVAL];
            gradient.interval[1] += g[INTERVAL + 1];
        }
        Ok(gradient)
    }
}

/// What [`oriented_chirality`] saves for its pullback: the real transverse components,
/// the normal wavenumbers, the polarizations, the axis and the interval.
#[derive(Debug)]
pub struct OrientedChiralityResidual {
    transverse: Vec<[f64; 2]>,
    normal: Vec<Complex>,
    polarizations: Vec<u8>,
    axis: usize,
    interval: [f64; 2],
}

/// Real transverse, complex normal and real interval-endpoint cotangents.
#[derive(Debug)]
pub struct OrientedChiralityGradient {
    /// Cotangents of the two transverse components, in cyclic Cartesian order.
    pub transverse: Vec<[f64; 2]>,
    /// Cotangents of the normal wavenumbers.
    pub normal: Vec<Complex>,
    /// Cotangents of the start and end of the averaging interval.
    pub interval: [f64; 2],
}

/// The helicity-signed up, down and cross forms of one mode of [`oriented_chirality`].
fn oriented_chirality_mode<const N: usize>(
    transverse: [f64; 2],
    normal: Complex,
    pol: u8,
    axis: usize,
    interval: [f64; 2],
) -> Result<[Jet<N>; 3]> {
    oriented_chirality_forms(
        [
            Jet::variable(transverse[0], Q),
            Jet::variable(transverse[1], Q + 1),
        ],
        [
            Jet::variable(normal.re, KZS),
            Jet::variable(normal.im, KZS + 1),
        ],
        pol,
        axis,
        [
            Jet::variable(interval[0], INTERVAL),
            Jet::variable(interval[1], INTERVAL + 1),
        ],
    )
}

/// Jets of the real coordinates retain the non-holomorphic polarization inner products.
fn oriented_chirality_forms<const N: usize>(
    transverse: [Jet<N>; 2],
    normal: [Jet<N>; 2],
    pol: u8,
    axis: usize,
    z: [Jet<N>; 2],
) -> Result<[Jet<N>; 3]> {
    let [nr, ni] = normal;
    let sign = helicity_sign(pol);
    // The observable has a smooth limit even where the polarization gauge does not.
    if transverse.iter().all(|q| q.value == Complex::default()) {
        if normal.iter().all(|v| v.value == Complex::default()) {
            return Err(Error::InvalidInput("wavevector must be nonzero".into()));
        }
        return Ok([
            2.0 * sign * mean_exp(-2.0 * ni, z),
            2.0 * sign * mean_exp(2.0 * ni, z),
            Jet::default(),
        ]);
    }
    let mut vector = [Jet::default(); 3];
    vector[axis] = nr + Complex::i() * ni;
    vector[(axis + 1) % 3] = transverse[0];
    vector[(axis + 2) % 3] = transverse[1];
    let up = crate::pw::polarization_from_inputs(vector, pol)?;
    vector[axis] = -vector[axis];
    let down = crate::pw::polarization_from_inputs(vector, pol)?;
    // All six local parameters are real, so conjugation acts on their derivatives.
    let inner = |a: [Jet<N>; 3], b: [Jet<N>; 3]| -> Jet<N> {
        a.into_iter()
            .zip(b)
            .map(|(a, b)| {
                Jet {
                    value: a.value.conj(),
                    derivative: a.derivative.map(|d| d.conj()),
                } * b
            })
            .sum()
    };
    Ok([
        2.0 * sign * inner(up, up) * mean_exp(-2.0 * ni, z),
        2.0 * sign * inner(down, down) * mean_exp(2.0 * ni, z),
        4.0 * sign * inner(down, up) * mean_exp(2.0 * Complex::i() * nr, z),
    ])
}

/// Signed helicity up/down/cross forms, shape (3, modes), for any Cartesian normal.
///
/// The transverse components are real and follow the cyclic order after `axis`.
/// The cross form X enters the density as Re(down* X up). The residual keeps only the
/// geometry.
///
/// Upstream: `treams.chirality_density` in the helicity convention, along any axis,
/// with the differences of the module docs.
pub fn oriented_chirality(
    transverse: Vec<[f64; 2]>,
    normal: Vec<Complex>,
    polarizations: Vec<u8>,
    axis: usize,
    interval: [f64; 2],
) -> Result<(DMatrix<Complex>, OrientedChiralityResidual)> {
    if transverse.is_empty()
        || transverse.len() != normal.len()
        || transverse.len() != polarizations.len()
        || axis > 2
        || transverse.iter().flatten().any(|q| !q.is_finite())
        || normal.iter().any(|&k| !finite(k))
        || polarizations.iter().any(|&p| p > 1)
        || interval.iter().any(|z| !z.is_finite())
    {
        return Err(Error::InvalidInput(
            "chirality requires matching finite geometry, polarization 0/1 and axis 0/1/2".into(),
        ));
    }
    let value = mode_forms(
        normal.len(),
        |j| {
            oriented_chirality_mode::<0>(transverse[j], normal[j], polarizations[j], axis, interval)
        },
        |v| v.value,
    )?;
    Ok((
        value,
        OrientedChiralityResidual {
            transverse,
            normal,
            polarizations,
            axis,
            interval,
        },
    ))
}

impl OrientedChiralityResidual {
    /// Compact output dimensions.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (3, self.normal.len())
    }

    /// Propagate transverse, normal-wavenumber and interval directions together.
    pub fn pushforward(
        &self,
        transverse: &[[f64; 2]],
        normal: &[Complex],
        interval: [f64; 2],
    ) -> Result<DMatrix<Complex>> {
        if transverse.len() != self.transverse.len()
            || normal.len() != self.normal.len()
            || transverse.iter().flatten().any(|v| !v.is_finite())
            || normal.iter().any(|&v| !finite(v))
            || interval.iter().any(|v| !v.is_finite())
        {
            return Err(Error::InvalidInput(
                "chirality tangents must be finite and match inputs".into(),
            ));
        }
        let interval = std::array::from_fn(|j| Jet {
            value: self.interval[j].into(),
            derivative: [interval[j].into()],
        });
        mode_forms(
            self.normal.len(),
            |j| {
                oriented_chirality_forms(
                    std::array::from_fn(|a| Jet {
                        value: self.transverse[j][a].into(),
                        derivative: [transverse[j][a].into()],
                    }),
                    real_direction(self.normal[j], normal[j]),
                    self.polarizations[j],
                    self.axis,
                    interval,
                )
            },
            |v| v.derivative[0],
        )
    }

    /// Recompute six local real derivatives per mode and pair them with the cotangent.
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<OrientedChiralityGradient> {
        let local = mode_cotangents(cotangent, self.normal.len(), |j| {
            oriented_chirality_mode(
                self.transverse[j],
                self.normal[j],
                self.polarizations[j],
                self.axis,
                self.interval,
            )
        })?;
        let mut gradient = OrientedChiralityGradient {
            transverse: Vec::with_capacity(local.len()),
            normal: Vec::with_capacity(local.len()),
            interval: [0.0; 2],
        };
        for g in local {
            gradient.transverse.push([g[Q], g[Q + 1]]);
            gradient.normal.push(Complex::new(g[KZS], g[KZS + 1]));
            gradient.interval[0] += g[INTERVAL];
            gradient.interval[1] += g[INTERVAL + 1];
        }
        Ok(gradient)
    }
}

/// Real and imaginary coordinates of one complex directional input.
fn real_direction(value: Complex, tangent: Complex) -> [Jet<1>; 2] {
    [
        Jet {
            value: value.re.into(),
            derivative: [tangent.re.into()],
        },
        Jet {
            value: value.im.into(),
            derivative: [tangent.im.into()],
        },
    ]
}
