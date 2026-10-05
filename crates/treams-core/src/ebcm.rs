//! Q-matrices of the extended boundary condition method (EBCM) for particles that are
//! symmetric about the z axis, with analytic gradients for the surface and the media.
//!
//! Upstream: `treams.ebcm`. [`qmat`] computes `treams.ebcm.qmat` with two differences:
//!
//! - `radial_area_factor = true` includes the factor `r` of the surface element
//!   `(r rhat - r' theta-hat) r sin(theta) dtheta dphi`, which treams omits; `false`
//!   reproduces treams. See
//!   <https://yaugenst.github.io/treams-rs/latest/coming-from-treams/differences/>.
//! - It sums over the fixed polar nodes and weights of a [`Surface`]; the Python
//!   `ebcm.qmat` passes Gauss-Legendre nodes. treams integrates each matrix entry with
//!   the adaptive `scipy.integrate.quad`.
#![allow(clippy::indexing_slicing)] // Validated mode/sample dimensions and Cartesian axes.

use std::collections::HashMap;

use nalgebra::DMatrix;
use rayon::prelude::*;

use crate::{
    Complex, Error, Result,
    basis::{ModeLabel, MultipoleBasis},
    fields::{SPATIAL_AND_K, VALUES, VectorWave, WaveSet},
    numerics::{self, finite},
    saved::{Reader, SavedState, Writer, invalid},
    special::Radial,
    sw::Mode,
};

/// A surface of revolution `r(theta)` about the z axis, sampled at fixed polar
/// quadrature nodes in [0, pi].
#[derive(Debug)]
pub struct Surface {
    /// Polar angles; the pullback holds these and the integration weights fixed.
    pub theta: Vec<f64>,
    /// Integration weights for d theta, without the sin(theta) Jacobian.
    pub weights: Vec<f64>,
    /// Positive surface radius at each node.
    pub radii: Vec<f64>,
    /// The slope `dr/dtheta` at each node. The pullback treats it as an input
    /// independent of the radii.
    pub slopes: Vec<f64>,
}

/// What [`qmat`] saves for its pullback: the modes, the surface and the media, without
/// any wave values; the pullback evaluates the waves again.
#[derive(Debug)]
pub struct QmatResidual {
    destination: Vec<Mode>,
    source: Vec<Mode>,
    surface: Surface,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    radial: Radial,
    radial_area_factor: bool,
}

/// Input gradients of [`QmatResidual::pullback`].
#[derive(Debug)]
pub struct QmatGradient {
    /// Gradients of the surface radii, one per node.
    pub radii: Vec<f64>,
    /// Gradients of the surface slopes, one per node.
    pub slopes: Vec<f64>,
    /// Gradients of the inner and outer wavenumbers, each ordered negative, positive
    /// helicity.
    pub ks: [[Complex; 2]; 2],
    /// Gradients of the inner and outer impedances.
    pub zs: [Complex; 2],
}

/// The point of node `i` in the x-z plane, its unnormalized surface normal
/// `r rhat - r' theta-hat` (`r'` is the slope `dr/dtheta`) and the unit vectors `rhat`
/// and `-theta-hat`, along which the normal changes with the radius and the slope.
fn node_geometry(surface: &Surface, i: usize) -> [[f64; 3]; 4] {
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

/// `a x n` of a complex and a real vector.
fn cross(a: [Complex; 3], n: [f64; 3]) -> [Complex; 3] {
    [
        a[1] * n[2] - a[2] * n[1],
        a[2] * n[0] - a[0] * n[2],
        a[0] * n[1] - a[1] * n[0],
    ]
}

/// The bilinear (unconjugated) product `a . b`.
fn dot(a: [Complex; 3], b: [Complex; 3]) -> Complex {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The triple product `n . (a x b)`.
fn triple(n: [f64; 3], a: [Complex; 3], b: [Complex; 3]) -> Complex {
    n[0] * (a[1] * b[2] - a[2] * b[1])
        + n[1] * (a[2] * b[0] - a[0] * b[2])
        + n[2] * (a[0] * b[1] - a[1] * b[0])
}

/// Spherical helicity waves of `modes` about the origin with wavenumbers `ks`. Only
/// the wavenumbers of helicities with modes are evaluated. The other may be zero,
/// which [`WaveSet::new`] rejects, so it takes the evaluated one's value.
fn origin_waves(
    modes: impl Iterator<Item = Mode>,
    ks: [Complex; 2],
    radial: Radial,
) -> Result<WaveSet> {
    let modes: Vec<_> = modes.map(|mode| (0, mode)).collect();
    let used = |pol: usize| modes.iter().any(|(_, mode)| usize::from(mode.pol) == pol);
    let ks = [0, 1].map(|pol| if used(pol) { ks[pol] } else { ks[1 - pol] });
    let basis = crate::sw::Basis {
        modes,
        positions: vec![[0.0; 3]],
    };
    WaveSet::new(MultipoleBasis::Spherical(basis), ks, true, radial)
}

/// The first `count` waves of `waves` at `point`, which share their radial functions
/// per degree and wavenumber and their solid harmonics per degree and order. `N` is
/// the derivative level, [`VALUES`] or [`SPATIAL_AND_K`].
fn evaluate_waves<const N: usize>(
    waves: &WaveSet,
    count: usize,
    point: [f64; 3],
) -> Result<Vec<VectorWave>> {
    let mut cache = waves.cache::<N>();
    // The loop writes each 240-byte wave once into a presized vector; collecting the
    // `Result`s copies it through an iterator adapter and grows the vector stepwise.
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        values.push(waves.wave::<N>(i, point, &mut cache)?.0);
    }
    Ok(values)
}

/// Integrate Q between regular inner test waves and regular or singular outer waves.
///
/// The surface element of node `i` is `(r rhat - r' theta-hat) r sin(theta) w_i`, where
/// `w_i` is the quadrature weight. With `radial_area_factor` false the factor `r` is
/// omitted. Each entry sums its nodes in order, so the result does not depend on the
/// thread count.
///
/// Upstream: `treams.ebcm.qmat`. Differences: treams omits the radial area factor `r`,
/// and `radial_area_factor = false` reproduces treams; treams integrates each entry
/// with adaptive quadrature, while this sums over the fixed nodes of `surface`.
pub fn qmat(
    destination: Vec<Mode>,
    source: Vec<Mode>,
    surface: Surface,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
    radial: Radial,
    radial_area_factor: bool,
) -> Result<(DMatrix<Complex>, QmatResidual)> {
    let n = surface.theta.len();
    let residual = QmatResidual {
        destination,
        source,
        surface,
        ks,
        zs,
        radial,
        radial_area_factor,
    };
    residual.validate()?;
    // The wave evaluations dominate; each node is independent.
    let [destination, source] = residual.wave_sets()?;
    let samples = crate::threads::install(|| {
        (0..n)
            .into_par_iter()
            .map(|node| -> Result<Sample> {
                let [point, normal, _, _] = node_geometry(&residual.surface, node);
                let values = |waves: Vec<VectorWave>| waves.into_iter().map(|w| w.value).collect();
                Ok(Sample {
                    weight: residual.weight(node),
                    normal,
                    destination: values(evaluate_waves::<VALUES>(
                        &destination,
                        residual.destination.len(),
                        point,
                    )?),
                    source: values(evaluate_waves::<VALUES>(
                        &source,
                        residual.source.len(),
                        point,
                    )?),
                })
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let rows = residual.rows_by_order();
    let factors = residual.factors();
    let mut value = numerics::zeros(residual.destination.len(), residual.source.len())?;
    // Each entry sums its nodes in order, so the result is independent of threading.
    crate::threads::install(|| {
        value
            .as_mut_slice()
            .par_chunks_mut(residual.destination.len())
            .zip(&residual.source)
            .enumerate()
            .for_each(|(j, (column, from))| {
                let rows = rows.get(&from.m).map_or(&[][..], Vec::as_slice);
                for sample in &samples {
                    for &i in rows {
                        let factor = factors[usize::from(residual.destination[i].pol)]
                            [usize::from(from.pol)];
                        column[i] += sample.weight
                            * factor
                            * triple(sample.normal, sample.destination[i], sample.source[j]);
                    }
                }
            });
    });
    if value.iter().any(|&z| !finite(z)) {
        return Err(Error::NonFinite("non-finite EBCM integral".into()));
    }
    Ok((value, residual))
}

/// The sign `2 pol - 1` of polarizations 0 and 1 (negative and positive helicity), as
/// [`helicity_sign`](crate::special::helicity_sign) gives it. A pair of helicities
/// couples through the impedance factor `HELICITY[a] z_outer + HELICITY[b] z_inner`,
/// so the impedance gradients use the same signs.
const HELICITY: [f64; 2] = [-1.0, 1.0];

/// Surface weight and normal of one node with the waves evaluated there.
struct Sample {
    weight: f64,
    normal: [f64; 3],
    destination: Vec<[Complex; 3]>,
    source: Vec<[Complex; 3]>,
}

/// The value and directional derivative of a node's integrand factors. Wave
/// derivatives are contracted here, before assembling any matrix entries.
struct SampleTangent {
    value: Sample,
    weight: f64,
    normal: [f64; 3],
    destination: Vec<[Complex; 3]>,
    source: Vec<[Complex; 3]>,
}

impl QmatResidual {
    fn validate(&self) -> Result<()> {
        let surface = &self.surface;
        let n = surface.theta.len();
        if n == 0
            || self.destination.is_empty()
            || self.source.is_empty()
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
            || self
                .ks
                .iter()
                .flatten()
                .chain(&self.zs)
                .any(|&z| !finite(z))
        {
            return Err(Error::InvalidInput("require nonempty modes, finite matching surface samples, positive radii and polar angles in [0, pi]".into()));
        }
        for mode in self.destination.iter().chain(&self.source) {
            mode.validate()?;
        }
        Ok(())
    }

    /// Bytes in the private callback state for the quadrature and mode counts.
    pub fn state_size(samples: usize, destinations: usize, sources: usize) -> Result<usize> {
        if samples == 0 || destinations == 0 || sources == 0 {
            return Err(invalid());
        }
        // Three counts, two flags, six complex media values, then the mode labels
        // and four real surface arrays. No wave values or Q matrix are cached.
        destinations
            .checked_add(sources)
            .and_then(|modes| modes.checked_mul(9))
            .and_then(|modes| samples.checked_mul(32).and_then(|n| modes.checked_add(n)))
            .and_then(|bytes| bytes.checked_add(122))
            .ok_or_else(invalid)
    }

    /// The inner test waves, whose orders are reversed, and the outer waves.
    fn wave_sets(&self) -> Result<[WaveSet; 2]> {
        let reversed = self
            .destination
            .iter()
            .map(|&mode| Mode { m: -mode.m, ..mode });
        Ok([
            origin_waves(reversed, self.ks[0], Radial::Regular)?,
            origin_waves(self.source.iter().copied(), self.ks[1], self.radial)?,
        ])
    }

    /// Integration weight of a node: the quadrature weight, `sin(theta)` and, with
    /// `radial_area_factor`, the radius.
    fn weight(&self, node: usize) -> f64 {
        self.measure(node)
            * if self.radial_area_factor {
                self.surface.radii[node]
            } else {
                1.0
            }
    }

    /// The quadrature weight times `sin(theta)`, which does not depend on the radius.
    fn measure(&self, node: usize) -> f64 {
        self.surface.weights[node] * self.surface.theta[node].sin()
    }

    /// The rows of each azimuthal order; only equal orders couple.
    fn rows_by_order(&self) -> HashMap<i32, Vec<usize>> {
        let mut rows = HashMap::<i32, Vec<usize>>::new();
        for (i, mode) in self.destination.iter().enumerate() {
            rows.entry(mode.m).or_default().push(i);
        }
        rows
    }

    /// The impedance factor `HELICITY[a] z_outer + HELICITY[b] z_inner` of each (row
    /// helicity `a`, column helicity `b`) pair.
    fn factors(&self) -> [[Complex; 2]; 2] {
        HELICITY.map(|a| HELICITY.map(|b| a * self.zs[1] + b * self.zs[0]))
    }

    /// Matrix shape, for binding-level cotangent validation.
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.destination.len(), self.source.len())
    }

    /// Number of fixed quadrature nodes, for binding-level tangent validation.
    #[must_use]
    pub fn samples(&self) -> usize {
        self.surface.radii.len()
    }

    /// Directional derivative with respect to the surface radii and slopes,
    /// wavenumbers and impedances. Quadrature nodes and weights stay fixed.
    /// Each wave's local derivatives are contracted once per node, before the
    /// matrix entries are assembled, sharing the same cached waves as the pullback.
    pub fn pushforward(
        &self,
        radii: &[f64],
        slopes: &[f64],
        ks: [[Complex; 2]; 2],
        zs: [Complex; 2],
    ) -> Result<DMatrix<Complex>> {
        if radii.len() != self.samples()
            || slopes.len() != self.samples()
            || radii.iter().chain(slopes).any(|x| !x.is_finite())
            || ks.iter().flatten().chain(&zs).any(|&z| !finite(z))
        {
            return Err(Error::InvalidInput("invalid EBCM tangents".into()));
        }
        let [destination, source] = self.wave_sets()?;
        let samples = crate::threads::install(|| {
            (0..self.samples())
                .into_par_iter()
                .map(|node| -> Result<SampleTangent> {
                    let [point, normal, rhat, slope] = node_geometry(&self.surface, node);
                    let point_tangent = rhat.map(|x| x * radii[node]);
                    let contract = |waves: Vec<VectorWave>, modes: &[Mode], dk: [Complex; 2]| {
                        waves
                            .into_iter()
                            .zip(modes)
                            .map(|(wave, mode)| {
                                let tangent = std::array::from_fn(|component| {
                                    wave.k[component] * dk[usize::from(mode.pol)]
                                        + (0..3)
                                            .map(|axis| {
                                                wave.position[component][axis] * point_tangent[axis]
                                            })
                                            .sum::<Complex>()
                                });
                                (wave.value, tangent)
                            })
                            .unzip()
                    };
                    let (a, da) = contract(
                        evaluate_waves::<SPATIAL_AND_K>(
                            &destination,
                            self.destination.len(),
                            point,
                        )?,
                        &self.destination,
                        ks[0],
                    );
                    let (b, db) = contract(
                        evaluate_waves::<SPATIAL_AND_K>(&source, self.source.len(), point)?,
                        &self.source,
                        ks[1],
                    );
                    Ok(SampleTangent {
                        value: Sample {
                            weight: self.weight(node),
                            normal,
                            destination: a,
                            source: b,
                        },
                        weight: if self.radial_area_factor {
                            self.measure(node) * radii[node]
                        } else {
                            0.0
                        },
                        normal: std::array::from_fn(|axis| {
                            radii[node] * rhat[axis] + slopes[node] * slope[axis]
                        }),
                        destination: da,
                        source: db,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let rows = self.rows_by_order();
        let factors = self.factors();
        let factor_tangents = HELICITY.map(|a| HELICITY.map(|b| a * zs[1] + b * zs[0]));
        let mut tangent = numerics::zeros(self.destination.len(), self.source.len())?;
        crate::threads::install(|| {
            tangent
                .as_mut_slice()
                .par_chunks_mut(self.destination.len())
                .zip(&self.source)
                .enumerate()
                .for_each(|(j, (column, from))| {
                    let rows = rows.get(&from.m).map_or(&[][..], Vec::as_slice);
                    let pb = usize::from(from.pol);
                    for sample in &samples {
                        let value = &sample.value;
                        for &i in rows {
                            let pa = usize::from(self.destination[i].pol);
                            let factor = factors[pa][pb];
                            let a = value.destination[i];
                            let b = value.source[j];
                            let integral = triple(value.normal, a, b);
                            let derivative = triple(sample.normal, a, b)
                                + triple(value.normal, sample.destination[i], b)
                                + triple(value.normal, a, sample.source[j]);
                            column[i] += (sample.weight * factor
                                + value.weight * factor_tangents[pa][pb])
                                * integral
                                + value.weight * factor * derivative;
                        }
                    }
                });
        });
        if tangent.iter().any(|&z| !finite(z)) {
            return Err(Error::NonFinite("non-finite EBCM tangent".into()));
        }
        Ok(tangent)
    }

    /// Gradients of the surface radii and slopes, the wavenumbers and the impedances
    /// from `cotangent`, the gradient of a real loss with respect to the matrix. The
    /// quadrature nodes and weights stay fixed. The sums run over the nodes in order,
    /// so the gradients do not depend on the thread count.
    pub fn pullback(&self, cotangent: &DMatrix<Complex>) -> Result<QmatGradient> {
        if cotangent.shape() != self.shape() || cotangent.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("invalid EBCM cotangent".into()));
        }
        let rows = self.rows_by_order();
        let factors = self.factors();
        let [destination, source] = self.wave_sets()?;
        let nodes = crate::threads::install(|| {
            (0..self.surface.theta.len())
                .into_par_iter()
                .map(|node| {
                    let [point, normal, rhat, slope] = node_geometry(&self.surface, node);
                    let a = evaluate_waves::<SPATIAL_AND_K>(
                        &destination,
                        self.destination.len(),
                        point,
                    )?;
                    let b = evaluate_waves::<SPATIAL_AND_K>(&source, self.source.len(), point)?;
                    let (measure, weight) = (self.measure(node), self.weight(node));
                    let along_radius = |wave: &VectorWave| -> [Complex; 3] {
                        std::array::from_fn(|c| (0..3).map(|d| wave.position[c][d] * rhat[d]).sum())
                    };
                    let ar: Vec<_> = a.iter().map(along_radius).collect();
                    let mut result = NodeGradient::default();
                    for (j, (vb, from)) in b.iter().zip(&self.source).enumerate() {
                        let pb = usize::from(from.pol);
                        // With triple(n, a, b) = a . (b x n), each outer wave's cross
                        // products serve all of its inner partners.
                        let normal_cross = cross(vb.value, normal);
                        let rhat_cross = cross(vb.value, rhat);
                        let slope_cross = cross(vb.value, slope);
                        let radius_cross = cross(along_radius(vb), normal);
                        let k_cross = cross(vb.k, normal);
                        for &i in rows.get(&from.m).into_iter().flatten() {
                            let cot = cotangent[(i, j)];
                            if cot == Complex::default() {
                                continue;
                            }
                            let va = &a[i];
                            let pa = usize::from(self.destination[i].pol);
                            let factor = factors[pa][pb];
                            let t = dot(va.value, normal_cross);
                            let dr = dot(va.value, rhat_cross)
                                + dot(ar[i], normal_cross)
                                + dot(va.value, radius_cross);
                            let scaled = cot.conj() * weight;
                            let weighted = scaled * factor;
                            result.radius += (weighted * dr).re;
                            if self.radial_area_factor {
                                result.radius += (cot.conj() * measure * factor * t).re;
                            }
                            result.slope += (weighted * dot(va.value, slope_cross)).re;
                            result.ks[0][pa] += (weighted * dot(va.k, normal_cross)).conj();
                            result.ks[1][pb] += (weighted * dot(va.value, k_cross)).conj();
                            result.zs[0] += (scaled * HELICITY[pb] * t).conj();
                            result.zs[1] += (scaled * HELICITY[pa] * t).conj();
                        }
                    }
                    Ok(result)
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let mut result = QmatGradient {
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

impl SavedState for QmatResidual {
    fn save_state(&self) -> Result<Vec<u8>> {
        let mut writer = Writer::new(Self::state_size(
            self.samples(),
            self.destination.len(),
            self.source.len(),
        )?);
        writer.usize(self.samples());
        writer.usize(self.destination.len());
        writer.usize(self.source.len());
        writer.byte(u8::from(self.radial == Radial::Singular));
        writer.byte(u8::from(self.radial_area_factor));
        for mode in self.destination.iter().chain(&self.source) {
            writer.i32(mode.l);
            writer.i32(mode.m);
            writer.byte(mode.pol);
        }
        for values in [
            &self.surface.theta,
            &self.surface.weights,
            &self.surface.radii,
            &self.surface.slopes,
        ] {
            for &value in values {
                writer.f64(value);
            }
        }
        for &value in self.ks.iter().flatten().chain(&self.zs) {
            writer.complex(value);
        }
        Ok(writer.finish())
    }

    fn from_state(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader::new(bytes);
        let samples = reader.usize()?;
        let destinations = reader.usize()?;
        let sources = reader.usize()?;
        if bytes.len() != Self::state_size(samples, destinations, sources)? {
            return Err(invalid());
        }
        let radial = match reader.byte()? {
            0 => Radial::Regular,
            1 => Radial::Singular,
            _ => return Err(invalid()),
        };
        let radial_area_factor = match reader.byte()? {
            0 => false,
            1 => true,
            _ => return Err(invalid()),
        };
        let mut modes = |count| {
            (0..count)
                .map(|_| {
                    Ok(Mode {
                        l: reader.i32()?,
                        m: reader.i32()?,
                        pol: reader.byte()?,
                    })
                })
                .collect::<Result<Vec<_>>>()
        };
        let destination = modes(destinations)?;
        let source = modes(sources)?;
        let mut surface_values = || {
            (0..samples)
                .map(|_| reader.f64())
                .collect::<Result<Vec<_>>>()
        };
        let surface = Surface {
            theta: surface_values()?,
            weights: surface_values()?,
            radii: surface_values()?,
            slopes: surface_values()?,
        };
        let ks = [
            [reader.complex()?, reader.complex()?],
            [reader.complex()?, reader.complex()?],
        ];
        let zs = [reader.complex()?, reader.complex()?];
        reader.finish()?;
        let residual = Self {
            destination,
            source,
            surface,
            ks,
            zs,
            radial,
            radial_area_factor,
        };
        residual.validate()?;
        Ok(residual)
    }
}

/// The gradient contributions of one node: its radius and slope, and its share of
/// the wavenumber and impedance gradients.
#[derive(Default)]
struct NodeGradient {
    radius: f64,
    slope: f64,
    ks: [[Complex; 2]; 2],
    zs: [Complex; 2],
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // Saved state must preserve every derivative exactly.

    use super::{QmatResidual, Surface, qmat};
    use crate::{Complex, Result, saved::SavedState, special::Radial, sw::Mode};
    use nalgebra::DMatrix;

    /// Only the wavenumbers of helicities with modes are evaluated, so the others may
    /// be zero, in the forward integral and in its pullback.
    #[test]
    fn unused_helicities_accept_zero_wavenumbers() {
        let surface = || Surface {
            theta: vec![0.5, 1.5, 2.5],
            weights: vec![1.0; 3],
            radii: vec![0.3, 0.35, 0.3],
            slopes: vec![0.1, 0.0, -0.1],
        };
        let modes = vec![Mode { l: 1, m: 0, pol: 1 }, Mode { l: 2, m: 1, pol: 1 }];
        let (k, zero) = (Complex::new(1.3, 0.1), Complex::default());
        let zs = [Complex::new(1.0, 0.0), Complex::new(0.8, 0.1)];
        let q = |ks| {
            qmat(
                modes.clone(),
                modes.clone(),
                surface(),
                ks,
                zs,
                Radial::Singular,
                true,
            )
        };
        let (value, residual) = q([[zero, k], [zero, k * 2.0]]).unwrap();
        let (expected, _) = q([[k, k], [k, k * 2.0]]).unwrap();
        assert_eq!(value, expected);
        let gradient = residual.pullback(&DMatrix::from_element(2, 2, k)).unwrap();
        assert_eq!(gradient.ks[0][0], zero);
        assert!(q([[k, zero], [k, k]]).is_err());
    }

    fn state_fixture(radial: Radial, radial_area_factor: bool) -> Result<QmatResidual> {
        let destination = vec![
            Mode { l: 1, m: 0, pol: 0 },
            Mode { l: 2, m: 1, pol: 1 },
            Mode {
                l: 1,
                m: -1,
                pol: 0,
            },
        ];
        let source = vec![Mode { l: 2, m: 1, pol: 0 }, Mode { l: 1, m: 0, pol: 1 }];
        let surface = Surface {
            theta: vec![0.5, 1.5, 2.5],
            weights: vec![0.7, 0.9, 0.8],
            radii: vec![0.3, 0.35, 0.3],
            slopes: vec![0.1, 0.0, -0.1],
        };
        let ks = [
            [Complex::new(1.3, 0.1), Complex::new(1.4, 0.12)],
            [Complex::new(1.1, 0.0), Complex::new(1.2, 0.02)],
        ];
        let zs = [Complex::new(0.9, 0.03), Complex::new(0.8, 0.1)];
        qmat(
            destination,
            source,
            surface,
            ks,
            zs,
            radial,
            radial_area_factor,
        )
        .map(|(_, residual)| residual)
    }

    #[test]
    fn saved_state_preserves_repeated_derivatives() -> Result<()> {
        let radii = [0.03, -0.02, 0.01];
        let slopes = [0.02, 0.01, -0.04];
        let ks = [
            [Complex::new(0.02, 0.01), Complex::new(-0.03, 0.02)],
            [Complex::new(0.01, -0.02), Complex::new(-0.02, -0.01)],
        ];
        let zs = [Complex::new(0.02, -0.01), Complex::new(-0.01, 0.03)];
        let cotangent = DMatrix::from_row_slice(
            3,
            2,
            &[
                Complex::new(0.1, 0.2),
                Complex::new(-0.2, 0.3),
                Complex::new(0.3, -0.1),
                Complex::new(-0.4, 0.1),
                Complex::new(0.2, 0.4),
                Complex::new(0.1, -0.3),
            ],
        );
        for radial in [Radial::Regular, Radial::Singular] {
            for radial_area_factor in [false, true] {
                let residual = state_fixture(radial, radial_area_factor)?;
                let state = residual.save_state()?;
                assert_eq!(state.len(), QmatResidual::state_size(3, 3, 2)?);
                let restored = QmatResidual::from_state(&state)?;
                assert_eq!(restored.shape(), (3, 2));
                assert_eq!(restored.samples(), 3);
                let expected = residual.pushforward(&radii, &slopes, ks, zs)?;
                let expected_gradient = residual.pullback(&cotangent)?;
                for _ in 0..2 {
                    assert_eq!(restored.pushforward(&radii, &slopes, ks, zs)?, expected);
                    let gradient = restored.pullback(&cotangent)?;
                    assert_eq!(gradient.radii, expected_gradient.radii);
                    assert_eq!(gradient.slopes, expected_gradient.slopes);
                    assert_eq!(gradient.ks, expected_gradient.ks);
                    assert_eq!(gradient.zs, expected_gradient.zs);
                }
                assert_eq!(restored.save_state()?, state);
            }
        }
        Ok(())
    }

    #[test]
    fn saved_state_rejects_malformed_layout_and_invalid_domain() -> Result<()> {
        let state = state_fixture(Radial::Singular, true)?.save_state()?;
        for length in [0, 1, 23, 24, 25, state.len() - 1] {
            assert!(QmatResidual::from_state(&state[..length]).is_err());
        }
        let mut trailing = state.clone();
        trailing.push(0);
        assert!(QmatResidual::from_state(&trailing).is_err());
        for offset in [0, 8, 16] {
            for count in [0_u64, u64::MAX] {
                let mut malformed = state.clone();
                malformed[offset..offset + 8].copy_from_slice(&count.to_le_bytes());
                assert!(QmatResidual::from_state(&malformed).is_err());
            }
        }
        // Radial kind, area-factor flag and the first mode's polarization.
        for offset in [24, 25, 34] {
            let mut malformed = state.clone();
            malformed[offset] = 2;
            assert!(QmatResidual::from_state(&malformed).is_err());
        }
        let mut malformed = state;
        let first_radius = 26 + 9 * 5 + 16 * 3;
        malformed[first_radius..first_radius + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        assert!(QmatResidual::from_state(&malformed).is_err());
        assert!(QmatResidual::state_size(usize::MAX, 1, 1).is_err());
        assert!(QmatResidual::state_size(1, usize::MAX, 1).is_err());
        assert!(QmatResidual::state_size(1, 1, usize::MAX).is_err());
        Ok(())
    }
}
