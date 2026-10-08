//! Plane-wave channels of periodic arrays: how the multipoles of a 2D array of spheres
//! or a 1D array of cylinders couple to the diffracted plane waves on either side.
//!
//! Upstream: `treams.sw.periodic_to_pw`, `treams.cw.periodic_to_pw`, and the plane-wave
//! expansions inside `treams.SMatrices.from_array`.
//!
//! | Item | Contents | Upstream |
//! |---|---|---|
//! | [`sw_periodic_to_pw`] | one plane-wave amplitude radiated by a spherical multipole of a 2D array | `sw.periodic_to_pw` |
//! | [`cw_periodic_to_pw`] | one plane-wave amplitude radiated by a cylindrical mode of a 1D array | `cw.periodic_to_pw` |
//! | [`spherical_channels`] | the incident and radiated coefficients of a whole spherical basis, with a pullback | `SMatrices.from_array` on a `TMatrix` |
//! | [`cylindrical_channels`] | the same for a cylindrical basis | `SMatrices.from_array` on a `TMatrixC` |
//!
//! Each channel matrix has one column per plane mode and `4 d` rows for `d` multipoles:
//! row `side * d + i` holds the regular expansion of the incident plane wave (the
//! `pw.to_sw` or `pw.to_cw` coefficient, with the phase of the multipole position), and
//! row `(2 + side) * d + i` the amplitude of the plane wave that multipole `i` radiates.
//! On side 0 the plane wave travels up, toward `+z` for spheres and `+y` for cylinders;
//! on side 1 it travels down.
//!
//! Both families share [`ChannelGradient`], input checks and matrix assembly. Each
//! evaluates its own channel type and jet size; pullbacks share an ordered traversal
//! while keeping the family-specific transverse-wavevector rules explicit.
//!
//! The inline `tests` module checks the error paths; the reciprocity, scaling and
//! adjoint identities of the channels are in `properties/plane.rs`.
#![allow(clippy::indexing_slicing)] // Validated modes and fixed-size local jets.

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::{
    Complex, Error, Result,
    basis::ModeLabel,
    numerics::{Jet, finite, parallel::try_fold_ordered},
    saved::{self, Reader, SavedState, Writer},
    special::polarized_angular,
    sw::{Basis, Mode},
};

// Jet slots of the channel derivatives. Both families order their inputs as the three
// position components, the medium wavenumber, the transverse wavevector, and last the
// unit-cell measure.
/// First of the three slots of the multipole position.
const POSITION: usize = 0;
/// Slot of the medium wavenumber `k`.
const K: usize = 3;
/// First transverse-wavevector slot: `(qx, qy)` for spheres, `kx` alone for cylinders.
const Q: usize = 4;
/// Slot of the unit-cell area of a 2D array.
const SPHERICAL_MEASURE: usize = 6;
/// Jet size of a spherical channel.
const SPHERICAL_SLOTS: usize = SPHERICAL_MEASURE + 1;
/// Slot of the period of a 1D array.
const CYLINDRICAL_MEASURE: usize = 5;
/// Jet size of a cylindrical channel.
const CYLINDRICAL_SLOTS: usize = CYLINDRICAL_MEASURE + 1;

/// Assemble incident/radiated pairs, constructing each channel once per side.
/// Each worker owns complete plane-mode columns without buffering intermediate values.
fn channel_matrix<I: Iterator<Item = [Complex; 2]>>(
    multipoles: usize,
    planes: usize,
    entries: impl Fn(usize, usize) -> Result<I> + Sync,
) -> Result<DMatrix<Complex>> {
    let mut value = DMatrix::zeros(4 * multipoles, planes);
    crate::threads::install(|| {
        value
            .as_mut_slice()
            .par_chunks_mut(4 * multipoles)
            .enumerate()
            .try_for_each(|(j, column)| -> Result<()> {
                for side in 0..2 {
                    for (i, [incident, radiated]) in entries(j, side)?.enumerate() {
                        column[side * multipoles + i] = incident;
                        column[(2 + side) * multipoles + i] = radiated;
                    }
                }
                Ok(())
            })
    })?;
    Ok(value)
}

/// One diffraction channel of a 2D array of spheres in one direction: the plane wave
/// `(qx, qy, ±kz)` and the angles at which the multipole coefficients are evaluated,
/// as jets in the slots above.
struct SphericalChannel<const N: usize> {
    /// The plane wavevector `(qx, qy, ±kz)`: `+kz` on side 0, `-kz` on side 1.
    vector: [Jet<N>; 3],
    /// Wavenumber of the medium on the chosen root, see [`SphericalChannel::new`].
    k: Jet<N>,
    /// Normal wavenumber `sqrt(k^2 - q^2)` with a positive imaginary part or a
    /// nonnegative real value.
    kz: Jet<N>,
    /// `sin(theta)` with a nonnegative real part.
    sine: Jet<N>,
    /// `cos(theta) = ±kz / k`.
    cosine: Jet<N>,
    /// `exp(i phi) = (qx + i qy) / |q|`, or 1 at normal incidence.
    azimuth: Jet<N>,
    /// Unit-cell area.
    area: Jet<N>,
}
impl<const N: usize> SphericalChannel<N> {
    /// The channel of transverse wavevector `q` on `side` (0 up, 1 down).
    ///
    /// `k` is the principal root of `k^2`, as upstream `sw.periodic_to_pw` computes
    /// `k = sqrt(kx^2 + ky^2 + kz^2)`, so a wavenumber with a negative real part gives the
    /// same channel as its negative. `kz = sqrt(k^2 - q^2)` takes the root with a positive
    /// imaginary part, or the nonnegative real root, as upstream `kz_s`: the wave decays
    /// or propagates in its direction of travel. Upstream replaces `kz = 0` by
    /// `1e-20 (1 + i)`; this returns an error when the computed `kz` is exactly zero.
    fn new(k: Complex, q: [f64; 2], side: usize, area: f64, fixed_q: bool) -> Result<Self> {
        let input = Jet::variable(k, K);
        let xy: [Jet<N>; 2] = std::array::from_fn(|i| {
            if fixed_q {
                Jet::constant(q[i])
            } else {
                Jet::variable(q[i], Q + i)
            }
        });
        let k = (input * input).sqrt();
        let mut kz = (input * input - xy[0] * xy[0] - xy[1] * xy[1]).sqrt();
        if kz.value == Complex::default() {
            return Err(Error::InvalidInput(
                "plane-wave channel is at a diffraction threshold".into(),
            ));
        }
        if kz.value.im < 0.0 || (kz.value.im == 0.0 && kz.value.re < 0.0) {
            kz = -kz;
        }
        let scale = q[0].abs().max(q[1].abs());
        let (mut sine, azimuth) = if scale == 0.0 {
            if N > 0 && !fixed_q {
                return Err(Error::InvalidInput("normal-incidence azimuth is undefined; hold the transverse wavevector fixed (fixed_q) to differentiate at normal incidence".into()));
            }
            (Jet::default(), Jet::constant(1.0))
        } else {
            let x = xy[0] / scale;
            let y = xy[1] / scale;
            let norm = (x * x + y * y).sqrt();
            (norm * scale / k, (x + Complex::i() * y) / norm)
        };
        if sine.value.re < 0.0 {
            sine = -sine;
        }
        let axial = if side == 0 { kz } else { -kz };
        Ok(Self {
            vector: [xy[0], xy[1], axial],
            k,
            kz,
            sine,
            cosine: axial / k,
            azimuth,
            area: Jet::variable(area, SPHERICAL_MEASURE),
        })
    }

    /// The incident and radiated coefficients of `mode` at `position` for the plane
    /// polarization `pol`: `[pw.to_sw, sw.periodic_to_pw]` with the position phase.
    fn entry(&self, mode: Mode, pol: u8, position: [f64; 3], helicity: bool) -> [Jet<N>; 2] {
        if helicity && mode.pol != pol {
            return [Jet::default(); 2];
        }
        let (l, m) = (mode.l, mode.m);
        let angular = polarized_angular(l, m, [self.cosine, self.sine], [mode.pol, pol], helicity);
        let normalization = (std::f64::consts::PI * f64::from(2 * l + 1) / f64::from(l * (l + 1)))
            .sqrt()
            * crate::special::factorial_ratio_sqrt(l, m);
        let phase = (Complex::i()
            * self
                .vector
                .into_iter()
                .enumerate()
                .map(|(i, k)| k * Jet::variable(position[i], POSITION + i))
                .sum::<Jet<N>>())
        .exp();
        let incident =
            2.0 * Complex::i().powi(l) * normalization * self.azimuth.powi(-m) * angular * phase;
        let outgoing = (-Complex::i()).powi(l) * normalization * self.azimuth.powi(m) * angular
            / (self.area * self.k * self.kz * phase);
        [incident, outgoing]
    }
}

/// Amplitude of the plane wave `vector` that the spherical multipole `mode` of a 2D
/// array with unit-cell `area` radiates. The sign of `kz` (of its imaginary part for a
/// complex `kz`) picks the side.
///
/// Upstream: `treams.sw.periodic_to_pw`.
/// Differences: treams replaces `kz = 0` by `1e-20 (1 + i)`. Here `kz` is recomputed as
/// `sqrt(k^2 - kx^2 - ky^2)`. When that is exactly zero, the call gives an error. When
/// rounding leaves a tiny `kz`, the amplitude is large, about `1 / kz`.
pub fn sw_periodic_to_pw(
    mode: Mode,
    vector: [Complex; 3],
    pol: u8,
    area: f64,
    helicity: bool,
) -> Result<Complex> {
    mode.validate()?;
    if pol > 1
        || vector.iter().any(|&v| !finite(v))
        || vector[0].im != 0.0
        || vector[1].im != 0.0
        || !area.is_finite()
        || area == 0.0
    {
        return Err(Error::InvalidInput("real transverse wavevector, finite axial component, polarization 0/1 and nonzero area required".into()));
    }
    let side = usize::from(vector[2].im < 0.0 || (vector[2].im == 0.0 && vector[2].re < 0.0));
    let k = crate::numerics::complex_sqrt(vector.iter().map(|v| v * v).sum());
    let channel =
        SphericalChannel::<0>::new(k, [vector[0].re, vector[1].re], side, area.abs(), true)?;
    let value = channel.entry(mode, pol, [0.0; 3], helicity)[1].value;
    if !finite(value) {
        return Err(Error::NonFinite(
            "non-finite plane radiation coefficient".into(),
        ));
    }
    Ok(value)
}

/// What [`spherical_channels`] saves for its pullback: its inputs alone.
///
/// The pullback recomputes each channel with its derivatives with respect to the
/// position, the wavenumber, the transverse wavevector and the area.
///
/// The pullback adds the per-column position, wavenumber and area gradients in chunks of
/// consecutive columns fixed by the column count, and the chunk sums in chunk order
/// (`numerics::parallel::try_fold_ordered`), so the thread count does not change them.
/// Each transverse-wavevector gradient belongs to one column, which writes it in place.
#[derive(Clone, Debug)]
pub struct SphericalChannelsResidual {
    basis: Basis,
    ks: [Complex; 2],
    q: Vec<[f64; 2]>,
    polarizations: Vec<u8>,
    area: f64,
    helicity: bool,
    fixed_q: bool,
}

/// Check the inputs both channel families share; `measure` is the area or the period.
fn validate_channels(
    ks: [Complex; 2],
    q: &[[f64; 2]],
    polarizations: &[u8],
    measure: f64,
    helicity: bool,
) -> Result<()> {
    if q.is_empty()
        || q.len() != polarizations.len()
        || q.iter().flatten().any(|v| !v.is_finite())
        || polarizations.iter().any(|&p| p > 1)
        || ks.iter().any(|&k| !finite(k) || k == Complex::default())
        || !measure.is_finite()
        || measure <= 0.0
    {
        return Err(Error::InvalidInput("channels require finite transverse vectors, polarizations 0/1, nonzero wavenumbers and positive unit-cell measure".into()));
    }
    if !helicity && ks[0] != ks[1] {
        return Err(Error::InvalidInput(
            "parity channels require an achiral medium".into(),
        ));
    }
    Ok(())
}

/// Incident and radiated channels of a spherical basis in a 2D array with unit-cell
/// `area`.
///
/// The plane modes have transverse wavevectors `q = (qx, qy)` and polarizations
/// `polarizations`; polarization `pol` sees the medium wavenumber `ks[pol]`.
///
/// The rows of the matrix pack (incident/radiated, up/down, multipole), as the module
/// documentation shows; its columns are plane modes. With `fixed_q`, the pullback holds
/// the incident and diffraction directions fixed.
///
/// Upstream: the plane-wave expansions inside `treams.SMatrices.from_array`.
pub fn spherical_channels(
    basis: Basis,
    ks: [Complex; 2],
    q: Vec<[f64; 2]>,
    polarizations: Vec<u8>,
    area: f64,
    helicity: bool,
    fixed_q: bool,
) -> Result<(DMatrix<Complex>, SphericalChannelsResidual)> {
    basis.validate()?;
    validate_channels(ks, &q, &polarizations, area, helicity)?;
    // The value pass takes no derivatives (N = 0), so `fixed_q` has no effect.
    let positions = &basis.positions;
    let value = channel_matrix(basis.modes.len(), q.len(), |j, side| {
        let pol = polarizations[j];
        let channel = SphericalChannel::<0>::new(ks[usize::from(pol)], q[j], side, area, true)?;
        Ok(basis.modes.iter().map(move |&(p, mode)| {
            channel
                .entry(mode, pol, positions[p], helicity)
                .map(|v| v.value)
        }))
    })?;
    if value.iter().any(|&v| !finite(v)) {
        return Err(Error::NonFinite("non-finite plane-wave channel".into()));
    }
    Ok((
        value,
        SphericalChannelsResidual {
            basis,
            ks,
            q,
            polarizations,
            area,
            helicity,
            fixed_q,
        },
    ))
}

/// Cotangents of positions, medium wavenumbers, transverse plane vectors and the unit-cell
/// measure.
#[derive(Clone, Debug)]
pub struct ChannelGradient {
    /// Multipole position cotangents.
    pub positions: Vec<[f64; 3]>,
    /// Complex medium wavenumber cotangents.
    pub ks: [Complex; 2],
    /// Transverse plane-wave cotangents (zero when `fixed_q` is requested).
    pub q: Vec<[f64; 2]>,
    /// Cotangent of the unit-cell measure: the area of a 2D array or the period of a 1D
    /// array.
    pub measure: f64,
}
impl ChannelGradient {
    fn zeros(positions: usize) -> Self {
        Self {
            positions: vec![[0.0; 3]; positions],
            ks: [Complex::default(); 2],
            q: Vec::new(),
            measure: 0.0,
        }
    }
    // Transverse gradients belong to individual columns, not these partial sums.
    fn add(&mut self, other: Self) {
        for (a, b) in self.positions.iter_mut().zip(other.positions) {
            for (a, b) in a.iter_mut().zip(b) {
                *a += b;
            }
        }
        for (a, b) in self.ks.iter_mut().zip(other.ks) {
            *a += b;
        }
        self.measure += other.measure;
    }
}
/// Reverse the shared matrix assembly in fixed column, side and multipole order.
/// Each column writes its own transverse gradient; only the shared gradients enter
/// the ordered reduction. Construct channels before skipping zero-weight entries,
/// preserving validation and avoiding evaluation of unused, possibly overflowing jets.
fn channel_pullback<M: Copy + Sync, const N: usize, E: Fn(M, [f64; 3]) -> [Jet<N>; 2]>(
    basis: &crate::basis::Basis<M>,
    polarizations: &[u8],
    cotangent: &DMatrix<Complex>,
    entries: impl Fn(usize, usize) -> Result<E> + Sync,
    add_q: impl Fn(&mut [f64; 2], &[Complex; N]) + Sync,
) -> Result<ChannelGradient> {
    let d = basis.modes.len();
    let mut q_gradients = vec![[0.0; 2]; polarizations.len()];
    let columns: Vec<_> = q_gradients.iter_mut().collect();
    let mut result = try_fold_ordered(
        columns,
        true,
        || ChannelGradient::zeros(basis.positions.len()),
        |mut result, j, q_gradient| -> Result<_> {
            let pol = usize::from(polarizations[j]);
            for side in 0..2 {
                let entry = entries(j, side)?;
                for (i, &(p, mode)) in basis.modes.iter().enumerate() {
                    let weights = [
                        cotangent[(side * d + i, j)],
                        cotangent[((2 + side) * d + i, j)],
                    ];
                    if weights.iter().all(|&z| z == Complex::default()) {
                        continue;
                    }
                    let pair = entry(mode, basis.positions[p]);
                    let gradient: [Complex; N] = std::array::from_fn(|a| {
                        weights[0] * pair[0].derivative[a].conj()
                            + weights[1] * pair[1].derivative[a].conj()
                    });
                    for (a, g) in result.positions[p]
                        .iter_mut()
                        .zip(&gradient[POSITION..POSITION + 3])
                    {
                        *a += g.re;
                    }
                    result.ks[pol] += gradient[K];
                    add_q(q_gradient, &gradient);
                    result.measure += gradient[N - 1].re;
                }
            }
            Ok(result)
        },
        |mut total, partial| {
            total.add(partial);
            total
        },
    )?;
    result.q = q_gradients;
    Ok(result)
}

impl SphericalChannelsResidual {
    /// Bytes of saved derivative state for the fixed basis and channel counts.
    pub fn state_size(multipoles: usize, positions: usize, channels: usize) -> Result<usize> {
        channel_state_size(saved::sw_basis_size(multipoles, positions)?, channels)
    }

    /// The shape of the channel matrix: four rows per multipole, one column per plane mode.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (4 * self.basis.modes.len(), self.q.len())
    }

    /// Check finite input tangents and their position and plane counts.
    pub fn validate_tangent(
        &self,
        positions: &[[f64; 3]],
        ks: [Complex; 2],
        q: &[[f64; 2]],
        measure: f64,
    ) -> Result<()> {
        validate_channel_tangent(
            positions,
            ks,
            q,
            measure,
            self.basis.positions.len(),
            self.q.len(),
        )
    }

    /// Directional derivative of the channel matrix. Tangents have the same shapes
    /// and order as the pullback outputs; `q` is held fixed with `fixed_q`. A zero
    /// transverse tangent also permits other input directions at normal incidence.
    pub fn pushforward(
        &self,
        positions: &[[f64; 3]],
        ks: [Complex; 2],
        q: &[[f64; 2]],
        measure: f64,
    ) -> Result<DMatrix<Complex>> {
        self.validate_tangent(positions, ks, q, measure)?;
        channel_matrix(self.basis.modes.len(), self.q.len(), |j, side| {
            let pol = self.polarizations[j];
            // The azimuth gauge at normal incidence is irrelevant when this
            // direction leaves the transverse wavevector fixed.
            let fixed_q = self.fixed_q || q[j].iter().all(|&v| v == 0.0);
            let channel = SphericalChannel::<SPHERICAL_SLOTS>::new(
                self.ks[usize::from(pol)],
                self.q[j],
                side,
                self.area,
                fixed_q,
            )?;
            Ok(self.basis.modes.iter().map(move |&(p, mode)| {
                channel
                    .entry(mode, pol, self.basis.positions[p], self.helicity)
                    .map(|entry| {
                        channel_direction(entry, positions[p], ks[usize::from(pol)], &q[j], measure)
                    })
            }))
        })
    }

    /// Gradients of the positions, the two medium wavenumbers, the transverse wavevectors
    /// (zero with `fixed_q`) and the area, for a `cotangent` of the channel matrix's
    /// shape. Both families traverse columns, sides and multipoles in the same order.
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<ChannelGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid channel cotangent".into()));
        }
        channel_pullback(
            &self.basis,
            &self.polarizations,
            cotangent,
            |j, side| {
                let pol = self.polarizations[j];
                let channel = SphericalChannel::<SPHERICAL_SLOTS>::new(
                    self.ks[usize::from(pol)],
                    self.q[j],
                    side,
                    self.area,
                    self.fixed_q,
                )?;
                Ok(move |mode, position| channel.entry(mode, pol, position, self.helicity))
            },
            |q_gradient, gradient| {
                if !self.fixed_q {
                    for (a, g) in q_gradient.iter_mut().zip(&gradient[Q..Q + 2]) {
                        *a += g.re;
                    }
                }
            },
        )
    }
}

impl SavedState for SphericalChannelsResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(
            self.basis.modes.len(),
            self.basis.positions.len(),
            self.q.len(),
        )?);
        saved::write_sw_basis(&mut writer, &self.basis);
        write_channel_state(
            &mut writer,
            self.ks,
            &self.q,
            &self.polarizations,
            self.area,
        );
        writer.byte(u8::from(self.helicity) | (u8::from(self.fixed_q) << 1));
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let basis = saved::read_sw_basis(&mut reader)?;
        let (ks, q, polarizations, area) = read_channel_state(&mut reader)?;
        let flags = reader.byte()?;
        if flags > 3 {
            return Err(saved::invalid());
        }
        reader.finish()?;
        let helicity = flags & 1 != 0;
        validate_channels(ks, &q, &polarizations, area, helicity)?;
        Ok(Self {
            basis,
            ks,
            q,
            polarizations,
            area,
            helicity,
            fixed_q: flags & 2 != 0,
        })
    }
}

/// The shared numerical channel inputs, with one byte for the family's flags.
fn channel_state_size(basis: usize, channels: usize) -> Result<usize> {
    channels
        .checked_mul(17)
        .and_then(|n| n.checked_add(49))
        .and_then(|n| n.checked_add(basis))
        .ok_or_else(saved::invalid)
}

fn write_channel_state(
    writer: &mut Writer,
    ks: [Complex; 2],
    q: &[[f64; 2]],
    polarizations: &[u8],
    measure: f64,
) {
    for k in ks {
        writer.complex(k);
    }
    writer.usize(q.len());
    for (&[x, y], &pol) in q.iter().zip(polarizations) {
        writer.f64(x);
        writer.f64(y);
        writer.byte(pol);
    }
    writer.f64(measure);
}

type ChannelState = ([Complex; 2], Vec<[f64; 2]>, Vec<u8>, f64);

fn read_channel_state(reader: &mut Reader<'_>) -> Result<ChannelState> {
    let ks = [reader.complex()?, reader.complex()?];
    let count = reader.count(17)?;
    // Only the cell measure and the flag byte follow the channels. Check the exact
    // remaining layout before allocating either of the channel vectors.
    if reader.remaining_len() != channel_state_size(0, count)? - 40 {
        return Err(saved::invalid());
    }
    let mut q = Vec::with_capacity(count);
    let mut polarizations = Vec::with_capacity(count);
    for _ in 0..count {
        q.push([reader.f64()?, reader.f64()?]);
        polarizations.push(reader.byte()?);
    }
    Ok((ks, q, polarizations, reader.f64()?))
}

/// Both families use the same input shapes and slot layout, except that cylinders
/// have only the `kx` transverse slot. Contract each small local jet directly; no
/// global Jacobian or output-sized cotangent seeds are needed.
fn channel_direction<const N: usize>(
    entry: Jet<N>,
    position: [f64; 3],
    k: Complex,
    q: &[f64],
    measure: f64,
) -> Complex {
    position
        .iter()
        .zip(&entry.derivative[POSITION..K])
        .map(|(v, d)| *v * *d)
        .sum::<Complex>()
        + k * entry.derivative[K]
        + q.iter()
            .zip(&entry.derivative[Q..N - 1])
            .map(|(v, d)| *v * *d)
            .sum::<Complex>()
        + measure * entry.derivative[N - 1]
}

fn validate_channel_tangent(
    positions: &[[f64; 3]],
    ks: [Complex; 2],
    q: &[[f64; 2]],
    measure: f64,
    position_count: usize,
    plane_count: usize,
) -> Result<()> {
    if positions.len() != position_count
        || q.len() != plane_count
        || positions.iter().flatten().any(|v| !v.is_finite())
        || ks.iter().any(|&v| !finite(v))
        || q.iter().flatten().any(|v| !v.is_finite())
        || !measure.is_finite()
    {
        return Err(Error::InvalidInput(
            "channel tangents must be finite and match the input shapes".into(),
        ));
    }
    Ok(())
}

/// One diffraction channel of a 1D array of cylinders along x in one direction: the plane
/// wave `(kx, ±ky, kz)`, as jets in the slots above. The axial `kz` is a fixed label.
///
/// Its assembly and pullback traversal are shared with [`SphericalChannel`].
struct CylindricalChannel<const N: usize> {
    /// The plane wavevector `(kx, ±ky, kz)`: `+ky` on side 0, `-ky` on side 1.
    vector: [Jet<N>; 3],
    /// `sqrt(k^2 - kz^2)` with a nonnegative imaginary part: the radial wavenumber of the
    /// cylindrical modes.
    transverse: Jet<N>,
    /// Normal wavenumber `sqrt(k^2 - kz^2 - kx^2)` with a positive imaginary part or a
    /// nonnegative real value.
    normal: Jet<N>,
    /// Lattice period.
    period: Jet<N>,
}
impl<const N: usize> CylindricalChannel<N> {
    /// The channel of `q = (kz, kx)` on `side` (0 toward `+y`, 1 toward `-y`). The normal
    /// wavenumber takes the root with a positive imaginary part, or the nonnegative real
    /// root, as upstream `cw.periodic_to_pw` does for `ky`; a channel at a cutoff or
    /// diffraction threshold is an error.
    fn new(k: Complex, q: [f64; 2], side: usize, period: f64, fixed_q: bool) -> Result<Self> {
        let k = Jet::variable(k, K);
        let kz = Jet::constant(q[0]);
        let kx = if fixed_q {
            Jet::constant(q[1])
        } else {
            Jet::variable(q[1], Q)
        };
        let mut transverse = (k * k - kz * kz).sqrt();
        let normal = (k * k - kz * kz - kx * kx).sqrt();
        if transverse.value == Complex::default() {
            return Err(Error::InvalidInput(
                "cylindrical plane channel is at a cutoff or diffraction threshold".into(),
            ));
        }
        if transverse.value.im < 0.0 {
            transverse = -transverse;
        }
        let mut channel = Self::from_vector([kx, normal, kz], transverse, period)?;
        channel.vector[1] = if side == 0 {
            channel.normal
        } else {
            -channel.normal
        };
        Ok(channel)
    }

    /// Preserve a supplied vector, including its propagation direction and the
    /// scalar API's zero-transverse-wavenumber convention. Matrix channels check
    /// that additional cutoff above because their derivatives are undefined there.
    fn from_vector(vector: [Jet<N>; 3], transverse: Jet<N>, period: f64) -> Result<Self> {
        let mut normal = vector[1];
        if normal.value == Complex::default() {
            return Err(Error::InvalidInput(
                "plane channel is at a diffraction threshold".into(),
            ));
        }
        if normal.value.im < 0.0 || (normal.value.im == 0.0 && normal.value.re < 0.0) {
            normal = -normal;
        }
        Ok(Self {
            vector,
            transverse,
            normal,
            period: Jet::variable(period, CYLINDRICAL_MEASURE),
        })
    }
    /// The incident and radiated coefficients of `mode` at `position` for the plane
    /// polarization `pol`: `[pw.to_cw, cw.periodic_to_pw]` with the position phase.
    fn entry(&self, mode: crate::cw::Mode, pol: u8, position: [f64; 3]) -> [Jet<N>; 2] {
        if !crate::pw::cylindrical_mode_matches(mode, self.vector[2].value, pol) {
            return [Jet::default(); 2];
        }
        let phase = (Complex::i()
            * self
                .vector
                .iter()
                .enumerate()
                .map(|(a, &k)| k * Jet::variable(position[a], POSITION + a))
                .sum::<Jet<N>>())
        .exp();
        let (incoming_angle, outgoing_angle) = if self.transverse.value == Complex::default() {
            (
                Jet::constant(Complex::i().powi(mode.m)),
                Jet::constant((-Complex::i()).powi(mode.m)),
            )
        } else {
            (
                ((Complex::i() * self.vector[0] + self.vector[1]) / self.transverse).powi(mode.m),
                ((-Complex::i() * self.vector[0] + self.vector[1]) / self.transverse).powi(mode.m),
            )
        };
        let incident = incoming_angle * phase;
        let outgoing = 2.0 * outgoing_angle / (self.period * self.normal * phase);
        [incident, outgoing]
    }
}

/// Amplitude of the plane wave `vector = (kx, ky, kz)` that the cylindrical mode `mode`
/// of a 1D array along x with lattice `period` radiates.
///
/// The normal wavenumber is `ky` or `-ky`, whichever has a positive imaginary part or,
/// when real, is nonnegative. A `kz` that differs from the mode's `kz` in any bit gives
/// zero, as in treams.
///
/// Upstream: `treams.cw.periodic_to_pw`.
/// Differences: a channel at a diffraction threshold (`ky = 0`) gives an error, where
/// treams replaces `ky` by `1e-20 (1 + i)`.
#[allow(clippy::float_cmp)] // Direct coefficients preserve exact axial labels.
pub fn cw_periodic_to_pw(
    mode: crate::cw::Mode,
    vector: [Complex; 3],
    pol: u8,
    period: f64,
) -> Result<Complex> {
    mode.validate()?;
    if pol > 1
        || vector.iter().any(|&v| !finite(v))
        || vector[0].im != 0.0
        || vector[2].im != 0.0
        || !period.is_finite()
        || period == 0.0
    {
        return Err(Error::InvalidInput(
            "finite vector with real kx/kz, polarization 0/1 and nonzero period required".into(),
        ));
    }
    if mode.kz != vector[2].re || mode.pol != pol {
        return Ok(Complex::default());
    }
    let channel = CylindricalChannel::<0>::from_vector(
        vector.map(Jet::constant),
        Jet::constant(crate::numerics::complex_sqrt(
            vector[0] * vector[0] + vector[1] * vector[1],
        )),
        period.abs(),
    )?;
    let value = channel.entry(mode, pol, [0.0; 3])[1].value;
    if !finite(value) {
        return Err(Error::NonFinite(
            "non-finite plane radiation coefficient".into(),
        ));
    }
    Ok(value)
}

/// What [`cylindrical_channels`] saves for its pullback: its inputs alone.
///
/// The pullback recomputes each channel with its derivatives with respect to the
/// position, the wavenumber, `kx` and the period.
///
/// The pullback adds the per-column position, wavenumber and period gradients as
/// [`SphericalChannelsResidual`] adds its shared gradients, so the thread count does not
/// change them. Each `kx` gradient belongs to one column, which writes it in place.
#[derive(Clone, Debug)]
pub struct CylindricalChannelsResidual {
    basis: crate::cw::Basis,
    ks: [Complex; 2],
    q: Vec<[f64; 2]>,
    polarizations: Vec<u8>,
    period: f64,
    fixed_q: bool,
}
/// Incident and radiated channels of a cylindrical basis in a 1D array along x with
/// lattice `period`.
///
/// The plane modes are `q = (kz, kx)` with polarizations `polarizations`. The order
/// `(kz, kx)` is that of a plane basis permuted to the zx plane, as treams permutes it
/// in `SMatrices.from_array`. The axial `kz` values are fixed labels, and `pol` has the
/// meaning of [`spherical_channels`].
///
/// The rows of the matrix pack (incident/radiated, up/down, cylindrical mode), as the
/// module documentation shows; its columns are plane modes. With `fixed_q`, the pullback
/// holds the `kx` components fixed.
///
/// Upstream: the plane-wave expansions inside `treams.SMatrices.from_array`.
pub fn cylindrical_channels(
    basis: crate::cw::Basis,
    ks: [Complex; 2],
    q: Vec<[f64; 2]>,
    polarizations: Vec<u8>,
    period: f64,
    helicity: bool,
    fixed_q: bool,
) -> Result<(DMatrix<Complex>, CylindricalChannelsResidual)> {
    basis.validate()?;
    validate_channels(ks, &q, &polarizations, period, helicity)?;
    // The value pass takes no derivatives (N = 0), so `fixed_q` has no effect.
    let positions = &basis.positions;
    let value = channel_matrix(basis.modes.len(), q.len(), |j, side| {
        let pol = polarizations[j];
        let channel = CylindricalChannel::<0>::new(ks[usize::from(pol)], q[j], side, period, true)?;
        Ok(basis
            .modes
            .iter()
            .map(move |&(p, mode)| channel.entry(mode, pol, positions[p]).map(|v| v.value)))
    })?;
    if value.iter().any(|&v| !finite(v)) {
        return Err(Error::NonFinite(
            "non-finite cylindrical plane-wave channel".into(),
        ));
    }
    Ok((
        value,
        CylindricalChannelsResidual {
            basis,
            ks,
            q,
            polarizations,
            period,
            fixed_q,
        },
    ))
}
impl CylindricalChannelsResidual {
    /// Bytes of saved derivative state for the fixed basis and channel counts.
    pub fn state_size(multipoles: usize, positions: usize, channels: usize) -> Result<usize> {
        channel_state_size(saved::cw_basis_size(multipoles, positions)?, channels)
    }

    /// The shape of the channel matrix: four rows per cylindrical mode, one column per
    /// plane mode.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (4 * self.basis.modes.len(), self.q.len())
    }

    /// Check finite input tangents and their position and plane counts.
    pub fn validate_tangent(
        &self,
        positions: &[[f64; 3]],
        ks: [Complex; 2],
        q: &[[f64; 2]],
        measure: f64,
    ) -> Result<()> {
        validate_channel_tangent(
            positions,
            ks,
            q,
            measure,
            self.basis.positions.len(),
            self.q.len(),
        )
    }

    /// Directional derivative of the channel matrix. The axial `q[j][0]` is a fixed
    /// label, and `q[j][1]` is held fixed as well when `fixed_q` was requested.
    pub fn pushforward(
        &self,
        positions: &[[f64; 3]],
        ks: [Complex; 2],
        q: &[[f64; 2]],
        measure: f64,
    ) -> Result<DMatrix<Complex>> {
        self.validate_tangent(positions, ks, q, measure)?;
        channel_matrix(self.basis.modes.len(), self.q.len(), |j, side| {
            let pol = self.polarizations[j];
            let channel = CylindricalChannel::<CYLINDRICAL_SLOTS>::new(
                self.ks[usize::from(pol)],
                self.q[j],
                side,
                self.period,
                self.fixed_q,
            )?;
            Ok(self.basis.modes.iter().map(move |&(p, mode)| {
                channel
                    .entry(mode, pol, self.basis.positions[p])
                    .map(|entry| {
                        channel_direction(
                            entry,
                            positions[p],
                            ks[usize::from(pol)],
                            &q[j][1..],
                            measure,
                        )
                    })
            }))
        })
    }

    /// Gradients of the positions, the two medium wavenumbers, the `kx` components and the
    /// period, for a `cotangent` of the channel matrix's shape. The axial `kz = q[j][0]`
    /// is a fixed label with a zero gradient, and `measure` holds the period gradient. The
    /// traversal is shared with [`SphericalChannelsResidual::pullback`].
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<ChannelGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&v| !finite(v)) {
            return Err(Error::InvalidInput(
                "invalid cylindrical channel cotangent".into(),
            ));
        }
        channel_pullback(
            &self.basis,
            &self.polarizations,
            cotangent,
            |j, side| {
                let pol = self.polarizations[j];
                let channel = CylindricalChannel::<CYLINDRICAL_SLOTS>::new(
                    self.ks[usize::from(pol)],
                    self.q[j],
                    side,
                    self.period,
                    self.fixed_q,
                )?;
                Ok(move |mode, position| channel.entry(mode, pol, position))
            },
            |q_gradient, gradient| q_gradient[1] += gradient[Q].re,
        )
    }
}

impl SavedState for CylindricalChannelsResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(
            self.basis.modes.len(),
            self.basis.positions.len(),
            self.q.len(),
        )?);
        saved::write_cw_basis(&mut writer, &self.basis);
        write_channel_state(
            &mut writer,
            self.ks,
            &self.q,
            &self.polarizations,
            self.period,
        );
        writer.byte(u8::from(self.fixed_q));
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let basis = saved::read_cw_basis(&mut reader)?;
        let (ks, q, polarizations, period) = read_channel_state(&mut reader)?;
        let fixed_q = match reader.byte()? {
            0 => false,
            1 => true,
            _ => return Err(saved::invalid()),
        };
        reader.finish()?;
        // Cylindrical channels do not mix polarizations, so the original helicity
        // convention has no further role once the forward inputs were checked.
        validate_channels(ks, &q, &polarizations, period, true)?;
        Ok(Self {
            basis,
            ks,
            q,
            polarizations,
            period,
            fixed_q,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_same_bits_on_pools, bits, cylindrical_basis, patterned};

    /// The direct-vector API preserves the supplied normal component and the
    /// upstream zero-transverse convention, while rejecting a zero normal component.
    #[test]
    fn cylindrical_scalar_channels_preserve_the_supplied_vector() {
        let mode = crate::cw::Mode {
            kz: 0.2,
            m: 2,
            pol: 1,
        };
        for normal in [Complex::i(), -Complex::i()] {
            let vector = [Complex::from(1.0), normal, Complex::from(mode.kz)];
            let channel = CylindricalChannel::<0>::from_vector(
                vector.map(Jet::constant),
                Jet::default(),
                2.0,
            )
            .unwrap();
            assert_eq!(channel.vector.map(|v| v.value), vector);
            assert_eq!(channel.normal.value, Complex::i());
            assert_eq!(
                cw_periodic_to_pw(mode, vector, 1, -2.0).unwrap(),
                Complex::i()
            );
        }
        let vector = [
            Complex::from(1.0),
            Complex::default(),
            Complex::from(mode.kz),
        ];
        assert!(matches!(
            cw_periodic_to_pw(mode, vector, 1, 2.0),
            Err(Error::InvalidInput(_))
        ));
        // A mismatched polarization has no radiation even at the threshold.
        assert_eq!(
            cw_periodic_to_pw(mode, vector, 0, 2.0).unwrap(),
            Complex::default()
        );
    }

    /// Zero weights must skip entry evaluation, including unused jets that would
    /// overflow. Channel validation still runs once on each side.
    #[test]
    fn channel_pullbacks_skip_zero_weight_entries_after_channel_validation() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let basis = cylindrical_basis(1, 0.2, [0.0; 3]);
        let weights = DMatrix::zeros(4 * basis.modes.len(), 1);
        let visits = AtomicUsize::new(0);
        let gradient = channel_pullback(
            &basis,
            &[1],
            &weights,
            |_, _| {
                visits.fetch_add(1, Ordering::Relaxed);
                Ok(|_, _| -> [Jet<CYLINDRICAL_SLOTS>; 2] { panic!("unused entry evaluated") })
            },
            |_, _| panic!("unused gradient accumulated"),
        )
        .unwrap();
        assert_eq!(visits.load(Ordering::Relaxed), 2);
        assert!(
            bits(&[
                &gradient.positions,
                &gradient.ks,
                &gradient.q,
                &gradient.measure,
            ])
            .iter()
            .all(|&bit| bit == 0)
        );
    }

    fn roundtrip<R: SavedState>(residual: &R, expected_size: usize) -> R {
        let mut bytes = residual.save_state().unwrap();
        assert_eq!(bytes.len(), expected_size);
        let restored = R::from_state(&bytes).unwrap();
        assert_eq!(bytes, restored.save_state().unwrap());
        assert!(R::from_state(&bytes[..bytes.len() - 1]).is_err());
        bytes.push(0);
        assert!(R::from_state(&bytes).is_err());
        bytes.pop();
        bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(R::from_state(&bytes).is_err());
        restored
    }

    fn assert_channel_gradient_same(left: &ChannelGradient, right: &ChannelGradient) {
        assert_eq!(
            bits(&[&left.positions, &left.ks, &left.q, &left.measure]),
            bits(&[&right.positions, &right.ks, &right.q, &right.measure]),
        );
    }

    /// Restoring numerical state preserves both derivative directions, including
    /// the fixed-direction flags, and never consumes the original or restored state.
    #[test]
    fn saved_channels_preserve_both_derivatives() {
        let ks = [Complex::new(1.3, 0.05); 2];
        let dp = [[0.1, -0.2, 0.3]];
        let dk = [Complex::new(0.04, -0.02); 2];
        let dq = [[0.03, -0.02], [-0.01, 0.04]];
        for helicity in [false, true] {
            for fixed_q in [false, true] {
                let basis = crate::test_support::spherical_basis(1, [0.1, -0.2, 0.05]);
                let size = SphericalChannelsResidual::state_size(basis.modes.len(), 1, 2).unwrap();
                let (value, residual) = spherical_channels(
                    basis,
                    ks,
                    vec![[0.3, 0.2], [-0.4, 0.1]],
                    vec![1, 0],
                    2.0,
                    helicity,
                    fixed_q,
                )
                .unwrap();
                let restored = roundtrip(&residual, size);
                let g = patterned(value.nrows(), value.ncols(), 0.3);
                assert_channel_gradient_same(
                    &residual.pullback(&g).unwrap(),
                    &restored.pullback(&g).unwrap(),
                );
                assert_eq!(
                    residual.pushforward(&dp, dk, &dq, 0.07).unwrap(),
                    restored.pushforward(&dp, dk, &dq, 0.07).unwrap(),
                );
                assert_channel_gradient_same(
                    &residual.pullback(&g).unwrap(),
                    &restored.pullback(&g).unwrap(),
                );

                let basis = cylindrical_basis(1, 0.2, [0.1, -0.2, 0.05]);
                let size =
                    CylindricalChannelsResidual::state_size(basis.modes.len(), 1, 2).unwrap();
                let (value, residual) = cylindrical_channels(
                    basis,
                    ks,
                    vec![[0.2, 0.3], [0.2, -0.4]],
                    vec![1, 0],
                    2.0,
                    helicity,
                    fixed_q,
                )
                .unwrap();
                let restored = roundtrip(&residual, size);
                let g = patterned(value.nrows(), value.ncols(), 0.3);
                assert_channel_gradient_same(
                    &residual.pullback(&g).unwrap(),
                    &restored.pullback(&g).unwrap(),
                );
                assert_eq!(
                    residual.pushforward(&dp, dk, &dq, 0.07).unwrap(),
                    restored.pushforward(&dp, dk, &dq, 0.07).unwrap(),
                );
                assert_channel_gradient_same(
                    &residual.pullback(&g).unwrap(),
                    &restored.pullback(&g).unwrap(),
                );
            }
        }
    }

    /// A strongly evanescent order overflows the phase of a position far from the
    /// array axis; both families reject the non-finite channel instead of returning it.
    #[test]
    fn nonfinite_channels_are_rejected() {
        let k = [Complex::new(1.3, 0.0); 2];
        let cylinders = cylindrical_basis(1, 0.2, [0.0, 20.0, 0.0]);
        let result =
            cylindrical_channels(cylinders, k, vec![[0.2, 50.0]], vec![1], 2.0, true, false);
        assert!(matches!(result, Err(Error::NonFinite(_))), "{result:?}");
        let spheres = crate::test_support::spherical_basis(1, [0.0, 0.0, 20.0]);
        let result = spherical_channels(spheres, k, vec![[50.0, 0.0]], vec![1], 2.0, true, false);
        assert!(matches!(result, Err(Error::NonFinite(_))), "{result:?}");
    }

    /// Both families add their shared gradients in chunks fixed by the column count:
    /// with more columns than chunks, every gradient repeats bit for bit on every pool
    /// size.
    #[test]
    fn pullbacks_do_not_depend_on_the_thread_count() {
        let ks = [Complex::new(1.2, 0.05), Complex::new(1.4, 0.03)];
        let columns = 150_u32;
        let q: Vec<[f64; 2]> = (0..columns)
            .map(|j| {
                let t = f64::from(j);
                [0.9 * (0.7 * t).sin(), 0.8 * (1.3 * t).cos()]
            })
            .collect();
        let polarizations: Vec<u8> = (0..columns).map(|j| u8::from(j % 3 != 0)).collect();
        let positions = vec![[0.1, -0.2, 0.05], [-0.15, 0.1, -0.1]];
        let modes = crate::sw::modes(2).unwrap();
        let spheres = Basis {
            modes: (0..2)
                .flat_map(|p| modes.iter().map(move |&mode| (p, mode)))
                .collect(),
            positions: positions.clone(),
        };
        let g = patterned(4 * spheres.modes.len(), q.len(), 0.3);
        assert_same_bits_on_pools(|| {
            let (_, residual) = spherical_channels(
                spheres.clone(),
                ks,
                q.clone(),
                polarizations.clone(),
                2.0,
                true,
                false,
            )
            .unwrap();
            let g = residual.pullback(&g).unwrap();
            bits(&[&g.positions, &g.ks, &g.q, &g.measure])
        });
        let cylinders = crate::cw::Basis {
            modes: (0..2)
                .flat_map(|p| {
                    (-3..=3).flat_map(move |m| {
                        [1, 0].map(|pol| (p, crate::cw::Mode { kz: 0.2, m, pol }))
                    })
                })
                .collect(),
            positions,
        };
        let q: Vec<_> = q.iter().map(|&[kx, _]| [0.2, kx]).collect();
        let g = patterned(4 * cylinders.modes.len(), q.len(), 0.6);
        assert_same_bits_on_pools(|| {
            let (_, residual) = cylindrical_channels(
                cylinders.clone(),
                ks,
                q.clone(),
                polarizations.clone(),
                2.0,
                true,
                false,
            )
            .unwrap();
            let g = residual.pullback(&g).unwrap();
            bits(&[&g.positions, &g.ks, &g.q, &g.measure])
        });
    }
}
