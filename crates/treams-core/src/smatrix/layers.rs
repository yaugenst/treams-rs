//! S-matrices of planar multilayer stacks: one S-matrix of 2-by-2 blocks per
//! transverse wavevector, built from [`interface()`](super::interface()) and
//! [`add`](super::add), with analytic pullbacks.
//!
//! Upstream: `treams.SMatrices.stack` of interfaces and propagations.
#![allow(clippy::indexing_slicing)] // Validated layer counts and two-polarization channels.

mod saved;

use nalgebra::DMatrix;

use super::{AddGradient, AddResidual, Blocks, InterfaceGradient, InterfaceResidual};
use crate::{
    Complex, Error, Result,
    linalg::product,
    numerics::{
        finite,
        parallel::{try_fold_ordered, try_map},
    },
    pw::wave_vector_z,
};

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
/// The pullback adds the medium and thickness gradients of all wavevectors in chunks of
/// consecutive wavevectors fixed by their count, and the chunk sums in chunk order
/// (`numerics::parallel::try_fold_ordered`), so the thread count does not change them.
/// Each `q` gradient belongs to one wavevector, which writes it in place.
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
    let results = try_map(q.len(), q.len() > 1, |channel| -> Result<_> {
        let q = q[channel];
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
    })?;
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

    /// Add the gradient of the interface between `medium` and the medium above it; its
    /// transverse-wavevector part goes to `q`, the gradient of the channel's `q`.
    fn add_interface(&mut self, medium: usize, q: &mut [f64; 2], gradient: InterfaceGradient) {
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
        for (target, g) in q.iter_mut().zip(gq) {
            *target += g;
        }
    }
}

impl LayerStackResidual {
    /// Number of media, including the exterior half-spaces.
    #[must_use]
    pub fn medium_count(&self) -> usize {
        self.ks.len()
    }

    /// Number of independent transverse channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.q.len()
    }

    /// Propagate one parameter direction through the independent two-polarization
    /// channels, reusing every saved interface and internal-field factorization.
    pub fn pushforward(
        &self,
        ks: &[[Complex; 2]],
        zs: &[Complex],
        q: &[[f64; 2]],
        thickness: &[f64],
    ) -> Result<Vec<Blocks>> {
        if ks.len() != self.ks.len()
            || zs.len() != self.ks.len()
            || q.len() != self.q.len()
            || thickness.len() != self.thickness.len()
            || ks.iter().flatten().chain(zs).any(|&v| !finite(v))
            || q.iter().flatten().chain(thickness).any(|v| !v.is_finite())
        {
            return Err(Error::InvalidInput("invalid layer-stack tangents".into()));
        }
        try_map(self.channels.len(), self.channels.len() > 1, |i| {
            let channel = &self.channels[i];
            let dq = if self.fixed_q { [0.0; 2] } else { q[i] };
            let mut tangent = channel
                .initial
                .pushforward([ks[0], ks[1]], [zs[0], zs[1]], dq)?;
            for (layer, step) in channel.steps.iter().enumerate() {
                let medium = layer + 1;
                let phase_tangent: [Complex; 2] = std::array::from_fn(|pol| {
                    let normal_tangent = (self.ks[medium][pol] * ks[medium][pol]
                        - self.q[i][0] * dq[0]
                        - self.q[i][1] * dq[1])
                        / step.normal[pol];
                    Complex::i()
                        * step.phase[pol]
                        * (self.thickness[layer] * normal_tangent
                            + step.normal[pol] * thickness[layer])
                });
                let [dv0, dv1, dv2, dv3] = tangent;
                let [v0, v1, v3] = &step.below;
                let phase = step.phase;
                let spaced = [
                    DMatrix::from_fn(2, 2, |i, j| {
                        phase_tangent[i] * v0[(i, j)] + phase[i] * dv0[(i, j)]
                    }),
                    DMatrix::from_fn(2, 2, |i, j| {
                        phase_tangent[i] * v1[(i, j)] * phase[j]
                            + phase[i] * dv1[(i, j)] * phase[j]
                            + phase[i] * v1[(i, j)] * phase_tangent[j]
                    }),
                    dv2,
                    DMatrix::from_fn(2, 2, |i, j| {
                        dv3[(i, j)] * phase[j] + v3[(i, j)] * phase_tangent[j]
                    }),
                ];
                let boundary = step.interface.pushforward(
                    [ks[medium], ks[medium + 1]],
                    [zs[medium], zs[medium + 1]],
                    dq,
                )?;
                tangent = step.boundary.pushforward(&spaced, &boundary)?;
            }
            Ok(tangent)
        })
    }

    /// Reuse every two-polarization solve; accumulate shared medium and thickness derivatives.
    pub fn pullback(&self, cotangent: Vec<Blocks>) -> Result<LayerStackGradient> {
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
        let Self {
            ks,
            q,
            thickness,
            channels,
            fixed_q,
        } = self;
        // Each channel writes its own `q` gradient in place; the partial sums carry only
        // the medium and thickness gradients that all channels share.
        let mut q_gradients = vec![[0.0; 2]; q.len()];
        let items: Vec<_> = channels
            .iter()
            .zip(cotangent)
            .zip(&mut q_gradients)
            .collect();
        let mut result = try_fold_ordered(
            items,
            true,
            || LayerStackGradient {
                ks: vec![[Complex::default(); 2]; ks.len()],
                zs: vec![Complex::default(); ks.len()],
                q: Vec::new(),
                thickness: vec![0.0; ks.len() - 2],
            },
            |mut result, i, ((channel, mut cotangent), q_gradient)| -> Result<_> {
                let wavevector = q[i];
                for (layer, step) in channel.steps.iter().enumerate().rev() {
                    let medium = layer + 1;
                    let AddGradient {
                        lower: spaced_g,
                        upper: interface_g,
                    } = step.boundary.pullback(&cotangent)?;
                    let interface_g = step.interface.pullback(&interface_g)?;
                    result.add_interface(medium, q_gradient, interface_g);
                    let (previous, phase_g) = propagate_pullback(spaced_g, &step.below, step.phase);
                    let d = thickness[layer];
                    for (pol, (&phase, &normal)) in step.phase.iter().zip(&step.normal).enumerate()
                    {
                        let phase_g = phase_g[pol];
                        // phase = exp(i kz d): dphase = i phase (d dkz + kz dd).
                        let slope = Complex::i() * phase;
                        result.thickness[layer] += (phase_g.conj() * slope * normal).re;
                        let normal_g = phase_g * (slope * d).conj();
                        // kz = sqrt(k^2 - |q|^2): dkz = (k dk - q . dq) / kz.
                        result.ks[medium][pol] += normal_g * (ks[medium][pol] / normal).conj();
                        if !fixed_q {
                            for (target, &component) in q_gradient.iter_mut().zip(&wavevector) {
                                *target -= (normal_g.conj() * component / normal).re;
                            }
                        }
                    }
                    cotangent = previous;
                }
                let initial_g = channel.initial.pullback(&cotangent)?;
                result.add_interface(0, q_gradient, initial_g);
                Ok(result)
            },
            |mut total, partial| {
                total.add(&partial);
                total
            },
        )?;
        result.q = q_gradients;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::{Blocks, layer_stack};
    use crate::{
        Complex,
        test_support::{assert_same_bits_on_pools, bits, patterned},
    };

    /// The medium and thickness gradients add in chunks fixed by the channel count:
    /// with more channels than chunks, every gradient repeats bit for bit on every pool
    /// size.
    #[test]
    fn pullback_does_not_depend_on_the_thread_count() {
        let c = Complex::new;
        let ks = vec![
            [c(1.0, 0.0); 2],
            [c(1.7, 0.02), c(1.8, 0.01)],
            [c(1.3, 0.0), c(1.35, 0.0)],
            [c(2.1, 0.05); 2],
            [c(1.0, 0.0); 2],
        ];
        let zs = [
            c(1.0, 0.0),
            c(0.7, 0.01),
            c(0.8, 0.0),
            c(0.6, 0.02),
            c(1.0, 0.0),
        ];
        let thickness = [0.3, 0.45, 0.2];
        let channels = 100_u32;
        let q: Vec<[f64; 2]> = (0..channels)
            .map(|j| {
                let t = f64::from(j);
                [0.9 * (0.7 * t).sin(), 0.8 * (1.3 * t).cos()]
            })
            .collect();
        let g: Vec<Blocks> = (0..channels)
            .map(|j| [0.0, 0.25, 0.5, 0.75].map(|seed| patterned(2, 2, f64::from(j) + seed)))
            .collect();
        assert_same_bits_on_pools(|| {
            let (_, residual) =
                layer_stack(ks.clone(), &zs, q.clone(), &thickness, 2, false).unwrap();
            let g = residual.pullback(g.clone()).unwrap();
            bits(&[&g.ks, &g.zs, &g.q, &g.thickness])
        });
    }
}
