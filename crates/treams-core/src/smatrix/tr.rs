//! Transmittance and reflectance of plane-wave S-matrix columns between two ports.
//!
//! Upstream: `treams.SMatrices.tr`.

use faer::MatRef;
use nalgebra::DMatrix;

use super::{Blocks, StoredBlock, any_nonfinite, interface::port_boundary};
use crate::{
    Complex, Error, Result,
    linalg::{product_adjoint_right, product_views, view},
    numerics::{Jet, finite, label_bits},
};

/// Jet seed offsets of the parameters: the wavenumbers `ks[port][pol]` at
/// `KS + 2 port + pol`, the impedances `zs[port]` at `ZS + port` and the transverse
/// components `q[j]` at `Q + j`.
const KS: usize = 0;
/// See [`KS`].
const ZS: usize = 4;
/// See [`KS`].
const Q: usize = 6;

/// Tangential fields `[E_a, E_b, H_a, H_b]` of the plane waves of one diffraction
/// group, indexed by port, propagation direction and polarization.
type PortWaves<const N: usize> = [[[[Jet<N>; 4]; 2]; 2]; 2];

/// The port fields of one diffraction group, as jets in the eight parameters when
/// `N = 8`; parity bases combine the helicity fields.
fn port_waves<const N: usize>(
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    q: [f64; 2],
    axis: usize,
    helicity: bool,
    fixed_q: bool,
) -> Result<PortWaves<N>> {
    let ks: [[Jet<N>; 2]; 2] = std::array::from_fn(|port| {
        std::array::from_fn(|pol| Jet::variable(ks[port][pol], KS + 2 * port + pol))
    });
    let zs: [Jet<N>; 2] = std::array::from_fn(|port| Jet::variable(zs[port], ZS + port));
    let q = std::array::from_fn(|j| {
        if fixed_q {
            Jet::constant(q[j])
        } else {
            Jet::variable(q[j], Q + j)
        }
    });
    let mut waves = [
        port_boundary(ks[0], zs[0], q, axis)?,
        port_boundary(ks[1], zs[1], q, axis)?,
    ];
    if !helicity {
        for port in &mut waves {
            for polarizations in port {
                let [minus, plus] = *polarizations;
                *polarizations = [
                    std::array::from_fn(|j| (plus[j] - minus[j]) * std::f64::consts::FRAC_1_SQRT_2),
                    std::array::from_fn(|j| (plus[j] + minus[j]) * std::f64::consts::FRAC_1_SQRT_2),
                ];
            }
        }
    }
    Ok(waves)
}

/// The time-averaged power flux `Re(E x H*) / 2` along the normal of tangential
/// fields `e` and `h`.
fn cross(e: [Complex; 4], h: [Complex; 4]) -> f64 {
    0.5 * (e[0] * h[3].conj() - e[1] * h[2].conj()).re
}

/// The cotangents of `e` and `h` in [`cross`] for the flux cotangent `weight`.
fn cross_pullback(e: [Complex; 4], h: [Complex; 4], weight: f64) -> ([Complex; 4], [Complex; 4]) {
    (
        [
            0.5 * weight * h[3],
            -0.5 * weight * h[2],
            Complex::default(),
            Complex::default(),
        ],
        [
            Complex::default(),
            Complex::default(),
            -0.5 * weight * e[1],
            0.5 * weight * e[0],
        ],
    )
}

/// The incident, transmitted and reflected tangential fields of one illumination
/// column, summed over the modes of each diffraction group.
///
/// `direction` is the propagation direction of the illumination; it also indexes the
/// port that receives the transmitted wave.
fn aggregate(
    waves: &[PortWaves<0>],
    modes: &[(usize, u8)],
    incident: &DMatrix<Complex>,
    outgoing: &[DMatrix<Complex>; 2],
    illumination: usize,
    direction: usize,
) -> Vec<[[Complex; 4]; 3]> {
    let transmission = direction;
    let reflection = 1 - transmission;
    let mut fields = vec![[[Complex::default(); 4]; 3]; waves.len()];
    for (mode, &(group, pol)) in modes.iter().enumerate() {
        for (which, (port, wave_direction, amplitude)) in [
            (reflection, transmission, incident[(mode, illumination)]),
            (
                transmission,
                transmission,
                outgoing[0][(mode, illumination)],
            ),
            (reflection, reflection, outgoing[1][(mode, illumination)]),
        ]
        .into_iter()
        .enumerate()
        {
            for (j, field) in fields[group][which].iter_mut().enumerate() {
                *field += waves[group][port][wave_direction][usize::from(pol)][j].value * amplitude;
            }
        }
    }
    fields
}

/// The two plane-wave ports of [`tr`] and [`tr_value`]: port 0 on the positive side
/// (above), port 1 on the negative side (below).
#[derive(Clone, Debug)]
pub struct TrPorts {
    /// Wavenumbers of polarizations 0 and 1 in the medium of each port.
    pub ks: [[Complex; 2]; 2],
    /// Impedance of the medium of each port.
    pub zs: [Complex; 2],
    /// One real transverse wavevector per distinct diffraction group.
    pub q: Vec<[f64; 2]>,
    /// The distinct (diffraction group, polarization) pair of every mode.
    pub modes: Vec<(usize, u8)>,
    /// Cartesian normal axis.
    pub axis: usize,
    /// Helicity polarizations; parity (TE/TM) otherwise.
    pub helicity: bool,
    /// Propagation direction of the illumination: 0 = up (towards the positive side,
    /// incident from the negative side), 1 = down.
    pub direction: usize,
}

impl TrPorts {
    #[allow(clippy::float_cmp)] // Exact diffraction groups.
    fn validate(&self) -> Result<()> {
        if self.axis > 2
            || self.direction > 1
            || self
                .modes
                .iter()
                .any(|&(group, pol)| group >= self.q.len() || pol > 1)
            || self.q.iter().flatten().any(|v| !v.is_finite())
            || self
                .ks
                .iter()
                .flatten()
                .chain(&self.zs)
                .any(|&v| !finite(v) || v == Complex::default())
        {
            return Err(Error::InvalidInput(INVALID_INPUTS.into()));
        }
        if !self.helicity && self.ks.iter().any(|k| k[0] != k[1]) {
            return Err(Error::InvalidInput(
                "parity power requires achiral port media".into(),
            ));
        }
        let mut unique = std::collections::HashSet::new();
        if self.q.iter().any(|q| !unique.insert(q.map(label_bits))) {
            return Err(Error::InvalidInput(
                "transverse groups must be distinct; place coherent modes in the same group".into(),
            ));
        }
        // A repeated mode would add its field twice to the flux and the powers.
        let mut unique = std::collections::HashSet::new();
        if self.modes.iter().any(|&mode| !unique.insert(mode)) {
            return Err(Error::InvalidInput("plane modes must be distinct".into()));
        }
        Ok(())
    }

    /// Tangential port fields of one diffraction group.
    fn waves<const N: usize>(&self, q: [f64; 2], fixed_q: bool) -> Result<PortWaves<N>> {
        port_waves(self.ks, self.zs, q, self.axis, self.helicity, fixed_q)
    }
}

const INVALID_INPUTS: &str = "require matching finite square S blocks, illumination columns, port media and valid plane modes";

/// What [`tr`] saves for its pullback: copies of the transmission and reflection blocks
/// of the illumination direction, the incident fields, the ports and the forward
/// results.
#[derive(Debug)]
pub struct TrResidual {
    matrices: [StoredBlock; 2],
    incident: DMatrix<Complex>,
    ports: TrPorts,
    forward: TrForward,
    fixed_q: bool,
}

/// S-matrix, illumination, port wavenumber/impedance and transverse-wavevector cotangents.
#[derive(Debug)]
pub struct TrGradient {
    /// Cotangents of the four scattering blocks.
    pub matrices: Blocks,
    /// Cotangent of the illumination, one column per independent illumination.
    pub incident: DMatrix<Complex>,
    /// Cotangents of the port 0/1, helicity 0/1 wavenumbers.
    pub ks: [[Complex; 2]; 2],
    /// Cotangents of the port impedances.
    pub zs: [Complex; 2],
    /// Cotangents of the transverse vectors, one per distinct diffraction direction;
    /// zero when the forward holds `q` fixed.
    pub q: Vec<[f64; 2]>,
}

/// Forward values with the intermediates the pullback reuses.
#[derive(Debug)]
struct TrForward {
    /// Rows are transmittance/reflectance; columns are independent illuminations.
    value: DMatrix<f64>,
    flux: Vec<f64>,
    outgoing: [DMatrix<Complex>; 2],
    waves: Vec<PortWaves<0>>,
}

/// Validate the inputs and compute the transmittance and reflectance.
fn evaluate(
    matrices: [MatRef<'_, Complex>; 2],
    incident: &DMatrix<Complex>,
    ports: &TrPorts,
) -> Result<TrForward> {
    let n = ports.modes.len();
    if n == 0
        || incident.nrows() != n
        || incident.ncols() == 0
        || matrices.iter().any(|m| m.nrows() != n || m.ncols() != n)
        || incident.iter().any(|&z| !finite(z))
        || matrices.iter().any(|&m| any_nonfinite(m))
    {
        return Err(Error::InvalidInput(INVALID_INPUTS.into()));
    }
    ports.validate()?;
    let waves = ports
        .q
        .iter()
        .map(|&q| ports.waves::<0>(q, true))
        .collect::<Result<Vec<_>>>()?;
    let outgoing = matrices.map(|m| product_views(m, view(incident)));
    let sign = if ports.direction == 0 { 1.0 } else { -1.0 };
    let mut flux = vec![0.0; incident.ncols()];
    let mut value = DMatrix::zeros(2, incident.ncols());
    for illumination in 0..incident.ncols() {
        let mut transmitted = 0.0;
        let mut reflected = 0.0;
        for [input, trans, reflect] in aggregate(
            &waves,
            &ports.modes,
            incident,
            &outgoing,
            illumination,
            ports.direction,
        ) {
            flux[illumination] +=
                sign * (cross(input, input) + cross(input, reflect) + cross(reflect, input));
            transmitted += sign * cross(trans, trans);
            reflected -= sign * cross(reflect, reflect);
        }
        if !flux[illumination].is_finite() || flux[illumination] <= 0.0 {
            return Err(Error::InvalidInput(
                "transmittance requires positive finite incident power flux".into(),
            ));
        }
        value[(0, illumination)] = transmitted / flux[illumination];
        value[(1, illumination)] = reflected / flux[illumination];
    }
    if value.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFinite(
            "non-finite transmission or reflection".into(),
        ));
    }
    Ok(TrForward {
        value,
        flux,
        outgoing,
        waves,
    })
}

/// The transmittance and reflectance of [`tr`] without the data its pullback needs.
/// Matrix views are the transmission and reflection blocks for the illumination
/// direction.
///
/// Upstream: `treams.SMatrices.tr`.
#[doc(alias = "transmittance")]
#[doc(alias = "power")]
pub fn tr_value(
    matrices: [MatRef<'_, Complex>; 2],
    incident: &DMatrix<Complex>,
    ports: &TrPorts,
) -> Result<DMatrix<f64>> {
    Ok(evaluate(matrices, incident, ports)?.value)
}

/// Transmittance and reflectance of S-matrix columns between two plane-wave ports.
///
/// [`TrResidual::value`] has the rows (transmittance, reflectance) and one column per
/// illumination; the residual keeps it because its pullback reads it. The pullback
/// includes the interference of incident and reflected waves in a lossy medium. With
/// `fixed_q`, it holds the transverse wavevectors fixed and supports exactly normal
/// incidence; otherwise derivatives preserve the distinct diffraction-group topology.
///
/// Upstream: `treams.SMatrices.tr`.
#[doc(alias = "transmittance")]
#[doc(alias = "power")]
pub fn tr(
    matrices: [MatRef<'_, Complex>; 2],
    incident: DMatrix<Complex>,
    ports: TrPorts,
    fixed_q: bool,
) -> Result<TrResidual> {
    let forward = evaluate(matrices, &incident, &ports)?;
    Ok(TrResidual {
        matrices: matrices.map(StoredBlock::copy_view),
        incident,
        ports,
        forward,
        fixed_q,
    })
}

impl TrResidual {
    /// Transmittance and reflectance: the rows, with one column per illumination. The
    /// pullback reads them.
    #[must_use]
    pub const fn value(&self) -> &DMatrix<f64> {
        &self.forward.value
    }

    /// Rows (transmittance, reflectance) and independent illuminations.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        self.forward.value.shape()
    }

    /// Pull back the cotangents of the transmittance and reflectance.
    pub fn pullback(self, cotangent: &DMatrix<f64>) -> Result<TrGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidInput(
                "power cotangent must be finite and match output".into(),
            ));
        }
        let Self {
            matrices: stored,
            incident: input,
            ports,
            forward:
                TrForward {
                    value,
                    flux,
                    outgoing: output,
                    waves,
                },
            fixed_q,
        } = self;
        let n = ports.modes.len();
        let t = ports.direction;
        let r = 1 - t;
        let sign = if t == 0 { 1.0 } else { -1.0 };
        let mut gwaves = vec![[[[[Complex::default(); 4]; 2]; 2]; 2]; ports.q.len()];
        let mut incident = DMatrix::zeros(n, input.ncols());
        let mut outgoing = [incident.clone(), incident.clone()];
        for illumination in 0..input.ncols() {
            let wt = cotangent[(0, illumination)] / flux[illumination];
            let wr = cotangent[(1, illumination)] / flux[illumination];
            let wi = -(cotangent[(0, illumination)] * value[(0, illumination)]
                + cotangent[(1, illumination)] * value[(1, illumination)])
                / flux[illumination];
            let fields = aggregate(&waves, &ports.modes, &input, &output, illumination, t);
            let mut gradient = vec![[[Complex::default(); 4]; 3]; ports.q.len()];
            for (group, f) in fields.iter().enumerate() {
                for (e, h, weight) in [
                    (0, 0, sign * wi),
                    (0, 2, sign * wi),
                    (2, 0, sign * wi),
                    (1, 1, sign * wt),
                    (2, 2, -sign * wr),
                ] {
                    let (ge, gh) = cross_pullback(f[e], f[h], weight);
                    for j in 0..4 {
                        gradient[group][e][j] += ge[j];
                        gradient[group][h][j] += gh[j];
                    }
                }
            }
            for (mode, &(group, pol)) in ports.modes.iter().enumerate() {
                let amplitudes = [
                    input[(mode, illumination)],
                    output[0][(mode, illumination)],
                    output[1][(mode, illumination)],
                ];
                for (which, (port, wave_direction)) in
                    [(r, t), (t, t), (r, r)].into_iter().enumerate()
                {
                    let mut amplitude_gradient = Complex::default();
                    for j in 0..4 {
                        let g = gradient[group][which][j];
                        amplitude_gradient += waves[group][port][wave_direction][usize::from(pol)]
                            [j]
                            .value
                            .conj()
                            * g;
                        gwaves[group][port][wave_direction][usize::from(pol)][j] +=
                            amplitudes[which].conj() * g;
                    }
                    if which == 0 {
                        incident[(mode, illumination)] += amplitude_gradient;
                    } else {
                        outgoing[which - 1][(mode, illumination)] += amplitude_gradient;
                    }
                }
            }
        }
        let mut matrices = std::array::from_fn(|_| DMatrix::zeros(n, n));
        for (which, direction) in [t, r].into_iter().enumerate() {
            matrices[2 * direction + t] = product_adjoint_right(&outgoing[which], &input);
            incident += product_views(stored[which].view().adjoint(), view(&outgoing[which]));
        }
        let mut result = TrGradient {
            matrices,
            incident,
            ks: [[Complex::default(); 2]; 2],
            zs: [Complex::default(); 2],
            q: vec![[0.0; 2]; ports.q.len()],
        };
        for (group, &q) in ports.q.iter().enumerate() {
            if gwaves[group]
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .all(|&v| v == Complex::default())
            {
                continue;
            }
            let jets = ports.waves::<8>(q, fixed_q)?;
            let mut parameters = [Complex::default(); 8];
            for (v, g) in jets
                .iter()
                .flatten()
                .flatten()
                .flatten()
                .zip(gwaves[group].iter().flatten().flatten().flatten())
            {
                for (target, d) in parameters.iter_mut().zip(v.derivative) {
                    *target += g * d.conj();
                }
            }
            for port in 0..2 {
                for pol in 0..2 {
                    result.ks[port][pol] += parameters[KS + 2 * port + pol];
                }
                result.zs[port] += parameters[ZS + port];
            }
            result.q[group] = [parameters[Q].re, parameters[Q + 1].re];
        }
        Ok(result)
    }
}
