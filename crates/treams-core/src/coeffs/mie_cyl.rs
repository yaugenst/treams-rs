//! Mie coefficients of multilayer chiral cylinders and their analytic pullback.
//!
//! Upstream: `treams.coeffs.mie_cyl`. The pullback is a treams-rs extension.
#![allow(clippy::indexing_slicing)] // Fixed four-channel interface matrices.

mod saved;

use super::{LayerGradient, Material, Matrix2, Matrix4, Matrix42};
use crate::{
    Complex, Error, MAX_DEGREE, Result,
    numerics::finite,
    special::{Radial, RadialJet, cylindrical_radial},
};

/// One boundary between two media, as the pullback reads it.
#[derive(Clone, Debug)]
struct Interface {
    /// The inner medium at the boundary.
    inside: Side,
    /// The outer medium at the boundary.
    outside: Side,
    /// The inverse of the outer boundary matrix.
    inverse_outer: Matrix4,
    /// The map from inner to outer coefficients: `inverse_outer` times the inner
    /// boundary matrix.
    transfer: Matrix4,
    /// The product of the transfers inside this boundary, applied to the two regular
    /// core channels.
    before: Matrix42,
}

/// What [`mie_cyl`] saves for its pullback: the four-channel interface states and
/// radial functions, so the pullback evaluates no Bessel functions.
#[derive(Clone, Debug)]
pub struct MieCylResidual {
    order: i32,
    kz: f64,
    k0: f64,
    radii: Vec<f64>,
    materials: Vec<Material>,
    interfaces: Vec<Interface>,
    inverse: Matrix2,
    value: Matrix2,
}

/// Gradients of the inputs of [`mie_cyl`], in the order of its arguments.
#[derive(Clone, Debug)]
pub struct MieCylGradient {
    /// Gradient of the axial wavenumber.
    pub kz: f64,
    /// Gradient of the vacuum wavenumber.
    pub k0: f64,
    /// Gradients of the radii and the materials.
    pub layers: LayerGradient,
}

/// A direction of change of one medium at one boundary: of its helicity wavenumbers,
/// the axial wavenumber, the radius and its impedance.
#[derive(Clone, Copy, Debug, Default)]
struct Direction {
    ks: [Complex; 2],
    kz: Complex,
    radius: Complex,
    impedance: Complex,
}

/// One medium at a boundary: its helicity wavenumbers, impedance, radial wavenumbers
/// `kr` and arguments `x = kr radius`, and the regular and (except in the core)
/// singular radial functions of each helicity.
#[derive(Clone, Copy, Debug)]
struct Side {
    ks: [Complex; 2],
    impedance: Complex,
    kr: [Complex; 2],
    x: [Complex; 2],
    radial: [[Option<RadialJet>; 2]; 2],
}

impl Side {
    fn new(
        order: i32,
        kz: f64,
        ks: [Complex; 2],
        radius: f64,
        impedance: Complex,
        regular_only: bool,
    ) -> Result<Self> {
        let mut side = Self {
            ks,
            impedance,
            kr: [Complex::default(); 2],
            x: [Complex::default(); 2],
            radial: [[None; 2]; 2],
        };
        for (pol, &k) in ks.iter().enumerate() {
            let kr = (k * k - kz * kz).sqrt();
            if k.norm_sqr() == 0.0 || kr.norm_sqr() == 0.0 {
                return Err(Error::InvalidInput(
                    "cylindrical cutoff k_rho=0 is not supported".into(),
                ));
            }
            let x = kr * radius;
            (side.kr[pol], side.x[pol]) = (kr, x);
            let kinds: &[Radial] = if regular_only {
                &[Radial::Regular]
            } else {
                &[Radial::Regular, Radial::Singular]
            };
            for (kind, &radial) in kinds.iter().enumerate() {
                side.radial[pol][kind] = Some(cylindrical_radial(order, x, radial)?);
            }
        }
        Ok(side)
    }

    /// The columns `(pol, kind)` of the boundary matrix with their radial function.
    fn columns(&self) -> impl Iterator<Item = (usize, usize, RadialJet)> + '_ {
        (0..2).flat_map(move |kind| {
            (0..2).filter_map(move |pol| self.radial[pol][kind].map(|f| (pol, kind, f)))
        })
    }

    /// The boundary matrix: tangential field components of each wave.
    fn matrix(&self, order: i32, kz: f64) -> Matrix4 {
        let mut value = Matrix4::zeros();
        for (pol, kind, f) in self.columns() {
            let (k, kr, x) = (self.ks[pol], self.kr[pol], self.x[pol]);
            let phi = -kr * f.value / k;
            let sign = if pol == 0 { 1.0 } else { -1.0 };
            let z = -kz * f64::from(order) * f.value / (k * x) + sign * f.first;
            let col = pol + 2 * kind;
            value[(0, col)] = phi;
            value[(1, col)] = z;
            value[(2, col)] = -sign * phi / self.impedance;
            value[(3, col)] = -sign * z / self.impedance;
        }
        value
    }

    /// The directional derivative of [`Self::matrix`], linear in the direction.
    fn derivative(&self, order: i32, kz: f64, radius: f64, direction: Direction) -> Matrix4 {
        let mut derivative = Matrix4::zeros();
        for (pol, kind, f) in self.columns() {
            let (k, kr, x) = (self.ks[pol], self.kr[pol], self.x[pol]);
            let dk = direction.ks[pol];
            let dkr = (k * dk - kz * direction.kz) / kr;
            let dx = dkr * radius + kr * direction.radius;
            let phi = -kr * f.value / k;
            let dphi = -(dkr * f.value + kr * f.first * dx) / k - phi * dk / k;
            let sign = if pol == 0 { 1.0 } else { -1.0 };
            let z = -kz * f64::from(order) * f.value / (k * x) + sign * f.first;
            let dz = -f64::from(order) / (k * x)
                * (direction.kz * f.value + kz * f.first * dx - kz * f.value * (dk / k + dx / x))
                + sign * f.second * dx;
            let (impedance, dimpedance) = (self.impedance, direction.impedance);
            let col = pol + 2 * kind;
            derivative[(0, col)] = dphi;
            derivative[(1, col)] = dz;
            derivative[(2, col)] = -sign * (dphi - phi * dimpedance / impedance) / impedance;
            derivative[(3, col)] = -sign * (dz - z * dimpedance / impedance) / impedance;
        }
        derivative
    }
}

/// Mie coefficients of concentric chiral cylinders at the axial wavenumber `kz` and the
/// azimuthal order `m` (`order`).
///
/// `radii` are the boundary radii, from inner to outer, and `materials` has one more
/// entry, the embedding medium last.
///
/// Upstream: `treams.coeffs.mie_cyl`.
pub fn mie_cyl(
    kz: f64,
    order: i32,
    k0: f64,
    radii: &[f64],
    materials: &[Material],
) -> Result<MieCylResidual> {
    crate::coeffs::validate_layers(radii, materials)?;
    if !kz.is_finite()
        || !k0.is_finite()
        || k0 <= 0.0
        || order.unsigned_abs() > MAX_DEGREE.unsigned_abs()
    {
        return Err(Error::InvalidInput(
            // 128 is MAX_DEGREE.
            "require finite kz, positive k0 and |m| <= 128".into(),
        ));
    }
    let mut q = Matrix42::zeros();
    q[(0, 0)] = Complex::new(1.0, 0.0);
    q[(1, 1)] = Complex::new(1.0, 0.0);
    let mut interfaces = Vec::with_capacity(radii.len());
    for (i, &radius) in radii.iter().enumerate() {
        let side = |m: Material, regular_only| {
            let ks = [k0 * (m.index() - m.kappa), k0 * (m.index() + m.kappa)];
            Side::new(order, kz, ks, radius, m.impedance(), regular_only)
        };
        let inside = side(materials[i], i == 0)?;
        let outside = side(materials[i + 1], false)?;
        let inverse_outer = outside
            .matrix(order, kz)
            .try_inverse()
            .ok_or(Error::Singular)?;
        let transfer = inverse_outer * inside.matrix(order, kz);
        interfaces.push(Interface {
            inside,
            outside,
            inverse_outer,
            transfer,
            before: q,
        });
        q = transfer * q;
    }
    let inverse = q
        .fixed_rows::<2>(0)
        .into_owned()
        .try_inverse()
        .ok_or(Error::Singular)?;
    let value = q.fixed_rows::<2>(2) * inverse;
    if value.iter().any(|&z| !finite(z)) {
        return Err(Error::Singular);
    }
    Ok(MieCylResidual {
        order,
        kz,
        k0,
        radii: radii.to_vec(),
        materials: materials.to_vec(),
        interfaces,
        inverse,
        value,
    })
}
impl MieCylResidual {
    /// The number of recorded layer boundaries.
    #[must_use]
    pub fn boundaries(&self) -> usize {
        self.radii.len()
    }

    /// The coefficients in negative, positive helicity order, which the pullback reads.
    #[must_use]
    pub const fn value(&self) -> &Matrix2 {
        &self.value
    }

    /// The coefficient tangent along changes of `kz`, `k0`, the radii and the
    /// materials. Material entries store changes of epsilon, mu and kappa.
    /// Cached interface inverses and radial derivatives are reused for one forward
    /// pass, with no additional factorizations or Bessel evaluations.
    pub fn pushforward(
        &self,
        kz: f64,
        k0: f64,
        radii: &[f64],
        materials: &[Material],
    ) -> Result<Matrix2> {
        super::validate_layer_tangents(self.radii.len(), radii, materials)?;
        if !kz.is_finite() || !k0.is_finite() {
            return Err(Error::InvalidInput(
                "wavenumber tangents must be finite".into(),
            ));
        }
        let optical: Vec<_> = self
            .materials
            .iter()
            .zip(materials)
            .map(|(&material, &direction)| material.pushforward(direction))
            .collect();
        let mut dq = Matrix42::zeros();
        for (i, interface) in self.interfaces.iter().enumerate() {
            let derivative = |side: usize, boundary: &Side| {
                let tangent = optical[i + side];
                let dn = [tangent.index - tangent.kappa, tangent.index + tangent.kappa];
                boundary.derivative(
                    self.order,
                    self.kz,
                    self.radii[i],
                    Direction {
                        ks: std::array::from_fn(|pol| {
                            boundary.ks[pol] * (k0 / self.k0) + self.k0 * dn[pol]
                        }),
                        kz: kz.into(),
                        radius: radii[i].into(),
                        impedance: tangent.impedance,
                    },
                )
            };
            let inner = derivative(0, &interface.inside);
            let outer = derivative(1, &interface.outside);
            let transfer = interface.inverse_outer * (inner - outer * interface.transfer);
            dq = transfer * interface.before + interface.transfer * dq;
        }
        Ok((dq.fixed_rows::<2>(2) - self.value * dq.fixed_rows::<2>(0)) * self.inverse)
    }

    /// Gradients of `kz`, `k0`, the radii and the materials from `cotangent`, the
    /// gradient of a real loss with respect to the coefficient matrix.
    ///
    /// The pullback runs on the calling thread and adds the boundaries in order, from
    /// the outermost in.
    pub fn pullback(&self, cotangent: &Matrix2) -> Result<MieCylGradient> {
        if cotangent.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("cotangent must be finite".into()));
        }
        let mut result = MieCylGradient {
            kz: 0.0,
            k0: 0.0,
            layers: LayerGradient::zeros(self.radii.len()),
        };
        let lower = cotangent * self.inverse.adjoint();
        let mut gq = Matrix42::zeros();
        gq.fixed_rows_mut::<2>(0)
            .copy_from(&(-self.value.adjoint() * lower));
        gq.fixed_rows_mut::<2>(2).copy_from(&lower);
        for (i, interface) in self.interfaces.iter().enumerate().rev() {
            let gf = gq * interface.before.adjoint();
            let inner = interface.inverse_outer.adjoint() * gf;
            let outer = -inner * interface.transfer.adjoint();
            let sides = [(&interface.inside, inner), (&interface.outside, outer)];
            for (side, (boundary, g)) in sides.into_iter().enumerate() {
                let material = self.materials[i + side];
                let n = material.index();
                let contract = |direction: Direction| -> Complex {
                    let df = boundary.derivative(self.order, self.kz, self.radii[i], direction);
                    g.dotc(&df).conj()
                };
                let one = Complex::new(1.0, 0.0);
                result.layers.radii[i] += contract(Direction {
                    radius: one,
                    ..Direction::default()
                })
                .re;
                result.kz += contract(Direction {
                    kz: one,
                    ..Direction::default()
                })
                .re;
                result.k0 += contract(Direction {
                    ks: [n - material.kappa, n + material.kappa],
                    ..Direction::default()
                })
                .re;
                let layers = &mut result.layers;
                let targets = [&mut layers.epsilon, &mut layers.mu, &mut layers.kappa];
                for (tangent, target) in material.tangents().into_iter().zip(targets) {
                    target[i + side] += contract(Direction {
                        ks: [
                            self.k0 * (tangent.index - tangent.kappa),
                            self.k0 * (tangent.index + tangent.kappa),
                        ],
                        impedance: tangent.impedance,
                        ..Direction::default()
                    });
                }
            }
            gq = interface.transfer.adjoint() * gq;
        }
        Ok(result)
    }
}
