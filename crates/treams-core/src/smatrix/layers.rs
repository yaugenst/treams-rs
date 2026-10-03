//! S-matrices of planar multilayer stacks: one S-matrix of 2-by-2 blocks per
//! transverse wavevector, built from [`interface()`](super::interface()) and
//! [`add`](super::add), with analytic pullbacks.
//!
//! Upstream: `treams.SMatrices.stack` of interfaces and propagations.
#![allow(clippy::indexing_slicing)] // Validated layer counts and two-polarization channels.

use nalgebra::DMatrix;
use rayon::prelude::*;

use super::{AddGradient, AddResidual, Blocks, InterfaceGradient, InterfaceResidual};
use crate::{Complex, Error, Result, linalg::product, numerics::finite, pw::wave_vector_z};

/// One interior layer: propagation across it, then the interface above it.
#[derive(Debug)]
struct Step {
    /// Normal wavenumber of each polarization in the layer.
    normal: [Complex; 2],
    /// Phase `exp(i kz d)` of each polarization across the layer.
    phase: [Complex; 2],
    /// Blocks 0, 1 and 3 of the stack below the layer.
    below: [DMatrix<Complex>; 3],
    boundary: AddResidual,
    interface: InterfaceResidual,
}

/// The stack of one transverse wavevector: the lowest interface, then one step per
/// interior layer.
#[derive(Debug)]
struct Channel {
    initial: InterfaceResidual,
    steps: Vec<Step>,
}

/// What [`layer_stack`] saves for its pullback: the solves of every transverse
/// wavevector, each on its own, without a dense matrix over all of them.
///
/// The pullback adds the medium and thickness gradients of all wavevectors with Rayon's
/// `try_fold` and `try_reduce`, so their last bits can change with the thread count and
/// from run to run. Each `q` gradient belongs to one wavevector and does not.
#[derive(Debug)]
pub struct LayerStackResidual {
    ks: Vec<[Complex; 2]>,
    q: Vec<[f64; 2]>,
    thickness: Vec<f64>,
    channels: Vec<Channel>,
    fixed_q: bool,
}

/// Multilayer parameter cotangents under the real Hermitian pairing.
#[derive(Debug)]
pub struct LayerStackGradient {
    /// Two wavenumber cotangents per medium, ordered from the negative side (below) to
    /// the positive side (above).
    pub ks: Vec<[Complex; 2]>,
    /// Impedance cotangents per medium.
    pub zs: Vec<Complex>,
    /// Real transverse-component cotangents per channel; zero when the forward holds
    /// `q` fixed.
    pub q: Vec<[f64; 2]>,
    /// Real thickness cotangents for interior media.
    pub thickness: Vec<f64>,
}

/// Place the propagation `P = diag(phase)` across a layer above the stack `below`.
///
/// Propagation reflects nothing, so the Redheffer product reduces to the closed
/// form `[P V0, P V1 P, V2, V3 P]`. Returns it with the blocks its pullback reads.
fn propagate(below: Blocks, phase: [Complex; 2]) -> (Blocks, [DMatrix<Complex>; 3]) {
    let [v0, v1, v2, v3] = below;
    // The dense products are those of `smatrix::add` with the exact identity
    // operator of a reflectionless upper stack, so the blocks are bitwise equal to
    // the generic composition; eigen-orderings of downstream bands depend on it.
    let diagonal = DMatrix::from_fn(
        2,
        2,
        |i, j| {
            if i == j { phase[i] } else { Complex::default() }
        },
    );
    let spaced = [
        product(&diagonal, &v0),
        product(&diagonal, &product(&v1, &diagonal)),
        v2,
        product(&v3, &diagonal),
    ];
    (spaced, [v0, v1, v3])
}

/// Pull the cotangent `g` of [`propagate`] back to the stack below and the phases.
fn propagate_pullback(
    g: Blocks,
    below: &[DMatrix<Complex>; 3],
    phase: [Complex; 2],
) -> (Blocks, [Complex; 2]) {
    let [v0, v1, v3] = below;
    let mut gradient = [Complex::default(); 2];
    for i in 0..2 {
        for j in 0..2 {
            gradient[i] +=
                g[0][(i, j)] * v0[(i, j)].conj() + g[1][(i, j)] * (v1[(i, j)] * phase[j]).conj();
            gradient[j] +=
                g[1][(i, j)] * (phase[i] * v1[(i, j)]).conj() + g[3][(i, j)] * v3[(i, j)].conj();
        }
    }
    let [g0, g1, g2, g3] = g;
    let conjugate = phase.map(|p| p.conj());
    let previous = [
        DMatrix::from_fn(2, 2, |i, j| conjugate[i] * g0[(i, j)]),
        DMatrix::from_fn(2, 2, |i, j| conjugate[i] * g1[(i, j)] * conjugate[j]),
        g2,
        DMatrix::from_fn(2, 2, |i, j| g3[(i, j)] * conjugate[j]),
    ];
    (previous, gradient)
}

/// Stack layers independently for each transverse wavevector; the output is one S-matrix
/// of four 2-by-2 blocks per wavevector.
///
/// `ks` holds the wavenumbers of polarizations 0 and 1 and `zs` the impedance of every
/// medium, from the negative side (below) to the positive side (above); `thickness`
/// holds the interior layers in the same order. `axis` is the Cartesian normal of the
/// layers, and each `q` holds the two transverse components after it in cyclic order.
/// With `fixed_q`, the pullback holds the transverse components fixed and returns
/// zero `q` gradients.
///
/// Upstream: `treams.SMatrices.stack` of alternating `SMatrices.interface` and
/// `SMatrices.propagation`, here for each transverse wavevector separately.
pub fn layer_stack(
    ks: Vec<[Complex; 2]>,
    zs: &[Complex],
    q: Vec<[f64; 2]>,
    thickness: &[f64],
    axis: usize,
    fixed_q: bool,
) -> Result<(Vec<Blocks>, LayerStackResidual)> {
    if ks.len() < 2
        || ks.len() != zs.len()
        || ks.len() != thickness.len() + 2
        || q.is_empty()
        || thickness.iter().any(|&d| !d.is_finite() || d < 0.0)
    {
        return Err(Error::InvalidInput("layers require at least two media, matching impedances, nonempty transverse channels and nonnegative interior thicknesses".into()));
    }
    let results: Vec<_> = q
        .par_iter()
        .map(|&q| -> Result<_> {
            let mut steps = Vec::with_capacity(thickness.len());
            let (mut value, initial) =
                super::interface([ks[0], ks[1]], [zs[0], zs[1]], q, axis, fixed_q)?;
            for (layer, &d) in thickness.iter().enumerate() {
                let medium = layer + 1;
                // Outgoing normal wavenumbers keep |phase| <= 1 for d >= 0.
                let normal = ks[medium].map(|k| wave_vector_z(q[0].into(), q[1].into(), k));
                let phase = normal.map(|kz| (Complex::i() * (kz * d)).exp());
                let (spaced, below) = propagate(value, phase);
                let (matching, interface) = super::interface(
                    [ks[medium], ks[medium + 1]],
                    [zs[medium], zs[medium + 1]],
                    q,
                    axis,
                    fixed_q,
                )?;
                let boundary;
                (value, boundary) = super::add(spaced, matching)?;
                steps.push(Step {
                    normal,
                    phase,
                    below,
                    boundary,
                    interface,
                });
            }
            Ok((value, Channel { initial, steps }))
        })
        .collect::<Result<_>>()?;
    let (values, channels) = results.into_iter().unzip();
    Ok((
        values,
        LayerStackResidual {
            ks,
            q,
            thickness: thickness.to_vec(),
            channels,
            fixed_q,
        },
    ))
}

impl LayerStackGradient {
    /// Add the cotangents of another partial sum, entry by entry.
    fn add(&mut self, other: &Self) {
        for (a, b) in self
            .ks
            .iter_mut()
            .flatten()
            .chain(&mut self.zs)
            .zip(other.ks.iter().flatten().chain(&other.zs))
        {
            *a += b;
        }
        for (a, b) in self
            .q
            .iter_mut()
            .flatten()
            .chain(&mut self.thickness)
            .zip(other.q.iter().flatten().chain(&other.thickness))
        {
            *a += b;
        }
    }

    /// Add the gradient of the interface between `medium` and the medium above it.
    fn add_interface(&mut self, medium: usize, channel: usize, gradient: InterfaceGradient) {
        let InterfaceGradient {
            ks: gk,
            z: gz,
            q: gq,
        } = gradient;
        for side in 0..2 {
            for (target, g) in self.ks[medium + side].iter_mut().zip(gk[side]) {
                *target += g;
            }
            self.zs[medium + side] += gz[side];
        }
        for (target, g) in self.q[channel].iter_mut().zip(gq) {
            *target += g;
        }
    }
}

impl LayerStackResidual {
    /// Number of independent transverse channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.q.len()
    }

    /// Reuse every two-polarization solve; accumulate shared medium and thickness derivatives.
    pub fn pullback(self, cotangent: Vec<Blocks>) -> Result<LayerStackGradient> {
        if cotangent.len() != self.q.len()
            || cotangent
                .iter()
                .flatten()
                .any(|b| b.shape() != (2, 2) || b.iter().any(|&v| !finite(v)))
        {
            return Err(Error::InvalidInput(
                "invalid compact layer-stack cotangent".into(),
            ));
        }
        let fixed_q = self.fixed_q;
        let zero = || LayerStackGradient {
            ks: vec![[Complex::default(); 2]; self.ks.len()],
            zs: vec![Complex::default(); self.ks.len()],
            q: vec![[0.0; 2]; self.q.len()],
            thickness: vec![0.0; self.ks.len() - 2],
        };
        self.channels
            .into_par_iter()
            .zip(cotangent)
            .enumerate()
            .try_fold(
                zero,
                |mut result, (i, (channel, mut cotangent))| -> Result<_> {
                    let q = self.q[i];
                    for (layer, step) in channel.steps.into_iter().enumerate().rev() {
                        let medium = layer + 1;
                        let AddGradient {
                            lower: spaced_g,
                            upper: interface_g,
                        } = step.boundary.pullback(&cotangent)?;
                        let interface_g = step.interface.pullback(&interface_g)?;
                        result.add_interface(medium, i, interface_g);
                        let (previous, phase_g) =
                            propagate_pullback(spaced_g, &step.below, step.phase);
                        let d = self.thickness[layer];
                        for (pol, (&phase, &normal)) in
                            step.phase.iter().zip(&step.normal).enumerate()
                        {
                            let phase_g = phase_g[pol];
                            // phase = exp(i kz d): dphase = i phase (d dkz + kz dd).
                            let slope = Complex::i() * phase;
                            result.thickness[layer] += (phase_g.conj() * slope * normal).re;
                            let normal_g = phase_g * (slope * d).conj();
                            // kz = sqrt(k^2 - |q|^2): dkz = (k dk - q . dq) / kz.
                            result.ks[medium][pol] +=
                                normal_g * (self.ks[medium][pol] / normal).conj();
                            if !fixed_q {
                                for (target, &q) in result.q[i].iter_mut().zip(&q) {
                                    *target -= (normal_g.conj() * q / normal).re;
                                }
                            }
                        }
                        cotangent = previous;
                    }
                    let initial_g = channel.initial.pullback(&cotangent)?;
                    result.add_interface(0, i, initial_g);
                    Ok(result)
                },
            )
            .try_reduce(zero, |mut a, b| {
                a.add(&b);
                Ok(a)
            })
    }
}
