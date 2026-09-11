//! Independent transverse channels for planar multilayers, with linear-size residuals.
#![allow(clippy::indexing_slicing)] // Validated layer counts and two-polarization channels.

use rayon::prelude::*;

use crate::{
    Complex, Error, Result, finite,
    jet::Jet,
    smatrix::{self, Blocks, InterfaceResidual, PropagationResidual, StackResidual},
};

#[derive(Debug)]
struct Step {
    propagation: PropagationResidual,
    spacer: StackResidual,
    boundary: StackResidual,
    interface: InterfaceResidual,
}
#[derive(Debug)]
struct Channel {
    initial: InterfaceResidual,
    steps: Vec<Step>,
}
/// Independent per-channel solves; no dense all-channel matrix is retained.
#[derive(Debug)]
pub struct LayersResidual {
    ks: Vec<[Complex; 2]>,
    q: Vec<[f64; 2]>,
    channels: Vec<Channel>,
}
/// Multilayer parameter cotangents under the real Hermitian pairing.
#[derive(Debug)]
pub struct LayersGradient {
    /// Two wavenumber cotangents per medium, ordered below to above.
    pub ks: Vec<[Complex; 2]>,
    /// Impedance cotangents per medium.
    pub zs: Vec<Complex>,
    /// Real transverse-component cotangents per channel.
    pub q: Vec<[f64; 2]>,
    /// Real thickness cotangents for interior media.
    pub thickness: Vec<f64>,
}

fn normals(ks: [Complex; 2], q: [f64; 2]) -> Result<[Complex; 2]> {
    Ok([
        smatrix::normal_component::<0>(Jet::constant(ks[0]), q.map(Jet::constant))?.value,
        smatrix::normal_component::<0>(Jet::constant(ks[1]), q.map(Jet::constant))?.value,
    ])
}

/// Stack layers independently for each transverse pair; output is one four-block 2-by-2 S matrix per pair.
#[allow(clippy::assigning_clones)] // Previous blocks were moved into the retained spacer solve.
pub fn stack(
    ks: Vec<[Complex; 2]>,
    zs: &[Complex],
    q: Vec<[f64; 2]>,
    thickness: &[f64],
    axis: usize,
) -> Result<(Vec<Blocks>, LayersResidual)> {
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
            let initial = smatrix::interface([ks[0], ks[1]], [zs[0], zs[1]], q, axis)?;
            let mut value = initial.value.clone();
            for (layer, &d) in thickness.iter().enumerate() {
                let medium = layer + 1;
                let kn = normals(ks[medium], q)?;
                let vectors = kn
                    .map(|k| [Complex::new(q[0], 0.0), Complex::new(q[1], 0.0), k])
                    .to_vec();
                let propagation = smatrix::propagation(vectors, [0.0, 0.0, d])?;
                let spacer = smatrix::add(value, propagation.value.clone())?;
                let interface = smatrix::interface(
                    [ks[medium], ks[medium + 1]],
                    [zs[medium], zs[medium + 1]],
                    q,
                    axis,
                )?;
                let boundary = smatrix::add(spacer.value.clone(), interface.value.clone())?;
                value = boundary.value.clone();
                steps.push(Step {
                    propagation,
                    spacer,
                    boundary,
                    interface,
                });
            }
            Ok((value, Channel { initial, steps }))
        })
        .collect::<Result<_>>()?;
    let (values, channels) = results.into_iter().unzip();
    Ok((values, LayersResidual { ks, q, channels }))
}
impl LayersResidual {
    /// Number of independent transverse channels.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.q.len()
    }
    /// Reuse every two-polarization solve; accumulate shared medium and thickness derivatives.
    pub fn pullback(self, g: Vec<Blocks>, fixed_q: bool) -> Result<LayersGradient> {
        if g.len() != self.q.len()
            || g.iter()
                .flatten()
                .any(|b| b.shape() != (2, 2) || b.iter().any(|&v| !finite(v)))
        {
            return Err(Error::InvalidInput(
                "invalid compact layer-stack cotangent".into(),
            ));
        }
        let zero = || LayersGradient {
            ks: vec![[Complex::default(); 2]; self.ks.len()],
            zs: vec![Complex::default(); self.ks.len()],
            q: vec![[0.0; 2]; self.q.len()],
            thickness: vec![0.0; self.ks.len() - 2],
        };
        self.channels
            .into_par_iter()
            .zip(g)
            .enumerate()
            .try_fold(
                zero,
                |mut result, (i, (channel, mut cotangent))| -> Result<_> {
                    let q = self.q[i];
                    for (layer, step) in channel.steps.into_iter().enumerate().rev() {
                        let medium = layer + 1;
                        let (spacer_g, interface_g) = step.boundary.pullback(&cotangent)?;
                        let (gk, gz, gq) = step.interface.pullback(&interface_g, fixed_q)?;
                        for side in 0..2 {
                            for (target, g) in result.ks[medium + side].iter_mut().zip(gk[side]) {
                                *target += g;
                            }
                            result.zs[medium + side] += gz[side];
                            result.q[i][side] += gq[side];
                        }
                        let (previous, propagation_g) = step.spacer.pullback(&spacer_g)?;
                        let (gv, gd) = step.propagation.pullback(&propagation_g)?;
                        result.thickness[layer] += gd[2];
                        let kn = normals(self.ks[medium], q)?;
                        for (pol, gradient) in gv.iter().enumerate() {
                            result.ks[medium][pol] +=
                                gradient[2] * (self.ks[medium][pol] / kn[pol]).conj();
                            if !fixed_q {
                                for (a, &q) in q.iter().enumerate() {
                                    result.q[i][a] -= (gradient[2].conj() * q / kn[pol]).re;
                                }
                            }
                        }
                        cotangent = previous;
                    }
                    let (gk, gz, gq) = channel.initial.pullback(&cotangent, fixed_q)?;
                    for side in 0..2 {
                        for (target, g) in result.ks[side].iter_mut().zip(gk[side]) {
                            *target += g;
                        }
                        result.zs[side] += gz[side];
                        result.q[i][side] += gq[side];
                    }
                    Ok(result)
                },
            )
            .try_reduce(zero, |mut a, b| {
                for (a, b) in
                    a.ks.iter_mut()
                        .flatten()
                        .chain(&mut a.zs)
                        .zip(b.ks.iter().flatten().chain(&b.zs))
                {
                    *a += b;
                }
                for (a, b) in
                    a.q.iter_mut()
                        .flatten()
                        .chain(&mut a.thickness)
                        .zip(b.q.iter().flatten().chain(&b.thickness))
                {
                    *a += b;
                }
                Ok(a)
            })
    }
}
