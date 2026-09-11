//! Axisymmetric extended-boundary-condition surface integrals and native adjoints.
#![allow(clippy::indexing_slicing)] // Validated mode/sample dimensions and Cartesian axes.

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::{
    Complex, Error, Result,
    fields::{VectorWave, spherical_wave_impl},
    finite,
    special::Radial,
    waves::Mode,
};

/// A radial surface sampled at fixed polar quadrature nodes in [0, pi].
#[derive(Debug)]
pub struct Surface {
    /// Polar angles; these and the integration weights are held fixed in reverse.
    pub theta: Vec<f64>,
    /// Integration weights for d theta, without the sin(theta) Jacobian.
    pub weights: Vec<f64>,
    /// Positive surface radius at each node.
    pub radii: Vec<f64>,
    /// d radius / d theta at each node, independently differentiable here.
    pub slopes: Vec<f64>,
}

/// Geometry-only Q-matrix residual, recomputing local waves during reverse.
#[derive(Debug)]
pub struct QResidual {
    to: Vec<Mode>,
    source: Vec<Mode>,
    surface: Surface,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    radial: Radial,
    legacy: bool,
}

/// Cotangents of all sampled shape and continuous medium parameters.
#[derive(Debug)]
pub struct QGradient {
    /// Surface-radius cotangents.
    pub radii: Vec<f64>,
    /// Surface-slope cotangents.
    pub slopes: Vec<f64>,
    /// Inner/outer complex wavenumbers, each ordered negative/positive helicity.
    pub ks: [[Complex; 2]; 2],
    /// Inner/outer complex impedances.
    pub zs: [Complex; 2],
}

fn geometry(surface: &Surface, i: usize) -> [[f64; 3]; 4] {
    let (s, c) = surface.theta[i].sin_cos();
    let rhat = [s, 0.0, c];
    let slope = [-c, 0.0, s];
    [
        rhat.map(|x| x * surface.radii[i]),
        std::array::from_fn(|j| surface.radii[i] * rhat[j] + surface.slopes[i] * slope[j]),
        rhat,
        slope,
    ]
}

fn triple(n: [f64; 3], a: [Complex; 3], b: [Complex; 3]) -> Complex {
    n[0] * (a[1] * b[2] - a[2] * b[1])
        + n[1] * (a[2] * b[0] - a[0] * b[2])
        + n[2] * (a[0] * b[1] - a[1] * b[0])
}

fn waves<const D: bool>(
    modes: &[Mode],
    ks: [Complex; 2],
    position: [f64; 3],
    radial: Radial,
    reverse_m: bool,
) -> Result<Vec<VectorWave>> {
    modes
        .iter()
        .map(|&mode| {
            spherical_wave_impl::<D>(
                Mode {
                    m: if reverse_m { -mode.m } else { mode.m },
                    ..mode
                },
                ks[usize::from(mode.pol)],
                position,
                true,
                radial,
            )
        })
        .collect()
}

/// Integrate Q between regular inner test waves and regular/outgoing outer waves.
/// Set legacy to reproduce treams' integral with its radial area factor omitted.
pub fn qmat(
    to: Vec<Mode>,
    source: Vec<Mode>,
    surface: Surface,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    singular: bool,
    legacy: bool,
) -> Result<(DMatrix<Complex>, QResidual)> {
    let n = surface.theta.len();
    if n == 0
        || to.is_empty()
        || source.is_empty()
        || surface.weights.len() != n
        || surface.radii.len() != n
        || surface.slopes.len() != n
        || surface
            .theta
            .iter()
            .any(|t| !t.is_finite() || !(0.0..=std::f64::consts::PI).contains(t))
        || surface.radii.iter().any(|r| !r.is_finite() || *r <= 0.0)
        || surface
            .weights
            .iter()
            .chain(&surface.slopes)
            .any(|x| !x.is_finite())
        || ks.iter().flatten().chain(&zs).any(|&z| !finite(z))
    {
        return Err(Error::InvalidInput("require nonempty modes, finite matching surface samples, positive radii and polar angles in [0, pi]".into()));
    }
    for mode in to.iter().chain(&source) {
        mode.validate()?;
    }
    let residual = QResidual {
        to,
        source,
        surface,
        ks,
        zs,
        legacy,
        radial: if singular {
            Radial::Outgoing
        } else {
            Radial::Regular
        },
    };
    let rows = residual.to.len();
    let mut value = DMatrix::zeros(rows, residual.source.len());
    for node in 0..n {
        let [point, normal, _, _] = geometry(&residual.surface, node);
        let a = waves::<false>(&residual.to, ks[0], point, Radial::Regular, true)?;
        let b = waves::<false>(&residual.source, ks[1], point, residual.radial, false)?;
        let weight = residual.surface.weights[node]
            * residual.surface.theta[node].sin()
            * if legacy {
                1.0
            } else {
                residual.surface.radii[node]
            };
        let fill = |(j, column): (usize, &mut [Complex])| {
            for (i, entry) in column.iter_mut().enumerate() {
                if residual.to[i].m == residual.source[j].m {
                    let factor = (2.0 * f64::from(residual.to[i].pol) - 1.0) * zs[1]
                        + (2.0 * f64::from(residual.source[j].pol) - 1.0) * zs[0];
                    *entry += weight * factor * triple(normal, a[i].value, b[j].value);
                }
            }
        };
        if value.len() >= 4096 {
            value
                .as_mut_slice()
                .par_chunks_mut(rows)
                .enumerate()
                .for_each(fill);
        } else {
            value
                .as_mut_slice()
                .chunks_mut(rows)
                .enumerate()
                .for_each(fill);
        }
    }
    if value.iter().any(|&z| !finite(z)) {
        return Err(Error::SpecialFunction("non-finite EBCM integral".into()));
    }
    Ok((value, residual))
}

impl QResidual {
    /// Matrix shape, for binding-level cotangent validation.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.to.len(), self.source.len())
    }

    /// Analytic surface and medium VJP at fixed quadrature nodes and weights.
    pub fn pullback(self, g: &DMatrix<Complex>) -> Result<QGradient> {
        if g.shape() != self.shape() || g.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid EBCM cotangent".into()));
        }
        let nodes = (0..self.surface.theta.len())
            .into_par_iter()
            .map(|node| {
                let [point, normal, rhat, slope] = geometry(&self.surface, node);
                let a = waves::<true>(&self.to, self.ks[0], point, Radial::Regular, true)?;
                let b = waves::<true>(&self.source, self.ks[1], point, self.radial, false)?;
                let measure = self.surface.weights[node] * self.surface.theta[node].sin();
                let weight = measure
                    * if self.legacy {
                        1.0
                    } else {
                        self.surface.radii[node]
                    };
                let mut result = NodeGradient::default();
                for (j, vb) in b.iter().enumerate() {
                    let pb = usize::from(self.source[j].pol);
                    for (i, va) in a.iter().enumerate() {
                        if self.to[i].m != self.source[j].m || g[(i, j)] == Complex::default() {
                            continue;
                        }
                        let pa = usize::from(self.to[i].pol);
                        let sa = 2.0 * f64::from(self.to[i].pol) - 1.0;
                        let sb = 2.0 * f64::from(self.source[j].pol) - 1.0;
                        let factor = sa * self.zs[1] + sb * self.zs[0];
                        let ar = std::array::from_fn(|c| {
                            (0..3).map(|d| va.position[c][d] * rhat[d]).sum()
                        });
                        let br = std::array::from_fn(|c| {
                            (0..3).map(|d| vb.position[c][d] * rhat[d]).sum()
                        });
                        let dr = triple(rhat, va.value, vb.value)
                            + triple(normal, ar, vb.value)
                            + triple(normal, va.value, br);
                        let cotangent = g[(i, j)];
                        result.radius += (cotangent.conj() * weight * factor * dr).re;
                        if !self.legacy {
                            result.radius += (cotangent.conj()
                                * measure
                                * factor
                                * triple(normal, va.value, vb.value))
                            .re;
                        }
                        result.slope += (cotangent.conj()
                            * weight
                            * factor
                            * triple(slope, va.value, vb.value))
                        .re;
                        result.ks[0][pa] +=
                            cotangent * (weight * factor * triple(normal, va.k, vb.value)).conj();
                        result.ks[1][pb] +=
                            cotangent * (weight * factor * triple(normal, va.value, vb.k)).conj();
                        let base = weight * triple(normal, va.value, vb.value);
                        result.zs[0] += cotangent * (sb * base).conj();
                        result.zs[1] += cotangent * (sa * base).conj();
                    }
                }
                Ok(result)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut result = QGradient {
            radii: Vec::with_capacity(nodes.len()),
            slopes: Vec::with_capacity(nodes.len()),
            ks: [[Complex::default(); 2]; 2],
            zs: [Complex::default(); 2],
        };
        for node in nodes {
            result.radii.push(node.radius);
            result.slopes.push(node.slope);
            for side in 0..2 {
                result.zs[side] += node.zs[side];
                for pol in 0..2 {
                    result.ks[side][pol] += node.ks[side][pol];
                }
            }
        }
        Ok(result)
    }
}

#[derive(Default)]
struct NodeGradient {
    radius: f64,
    slope: f64,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
}
