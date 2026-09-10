//! Multilayer chiral cylinder scattering and analytic boundary pullbacks.
#![allow(clippy::indexing_slicing)] // Fixed four-channel interface matrices.

use crate::{
    Complex, Error, Result,
    coeffs::{Material, Matrix2, MieGradient},
    finite,
    special::{Radial, cylindrical},
};
use nalgebra::SMatrix;
type Matrix4 = SMatrix<Complex, 4, 4>;
type Matrix42 = SMatrix<Complex, 4, 2>;

#[derive(Clone, Debug)]
struct Interface {
    inverse_outer: Matrix4,
    transfer: Matrix4,
    before: Matrix42,
}

/// Native cylinder execution, retaining only four-channel interface states.
#[derive(Clone, Debug)]
pub struct CylinderResidual {
    order: i32,
    kz: f64,
    k0: f64,
    radii: Vec<f64>,
    materials: Vec<Material>,
    interfaces: Vec<Interface>,
    inverse: Matrix2,
    /// Coefficients in negative, positive helicity order.
    pub value: Matrix2,
}

/// Cylinder parameter cotangents under the real Hermitian pairing.
#[derive(Clone, Debug)]
pub struct CylinderGradient {
    /// Axial wave number derivative.
    pub kz: f64,
    /// Vacuum wave number derivative.
    pub k0: f64,
    /// Radius derivatives in `sizes`, plus all material cotangents.
    pub layers: MieGradient,
}

#[derive(Clone, Copy, Debug, Default)]
struct Direction {
    ks: [Complex; 2],
    kz: Complex,
    radius: Complex,
    impedance: Complex,
}

fn boundary(
    order: i32,
    kz: f64,
    ks: [Complex; 2],
    radius: f64,
    impedance: Complex,
    direction: Direction,
    regular_only: bool,
) -> Result<(Matrix4, Matrix4)> {
    let mut value = Matrix4::zeros();
    let mut derivative = Matrix4::zeros();
    for (pol, &k) in ks.iter().enumerate() {
        let dk = direction.ks[pol];
        let kr = (k * k - kz * kz).sqrt();
        let dkr = (k * dk - kz * direction.kz) / kr;
        let x = kr * radius;
        let dx = dkr * radius + kr * direction.radius;
        if k.norm_sqr() == 0.0 || kr.norm_sqr() == 0.0 {
            return Err(Error::InvalidInput(
                "cylindrical cutoff k_rho=0 is not supported".into(),
            ));
        }
        for (kind, radial) in [Radial::Regular, Radial::Outgoing].into_iter().enumerate() {
            if kind == 1 && regular_only {
                continue;
            }
            let f = cylindrical(order, x, radial)?;
            let phi = -kr * f.value / k;
            let dphi = -(dkr * f.value + kr * f.first * dx) / k - phi * dk / k;
            let sign = 1.0
                - 2.0
                    * f64::from(
                        u32::try_from(pol)
                            .map_err(|_| Error::InvalidInput("invalid polarization".into()))?,
                    );
            let z = -kz * f64::from(order) * f.value / (k * x) + sign * f.first;
            let dz = -f64::from(order) / (k * x)
                * (direction.kz * f.value + kz * f.first * dx - kz * f.value * (dk / k + dx / x))
                + sign * f.second * dx;
            let col = pol + 2 * kind;
            value[(0, col)] = phi;
            value[(1, col)] = z;
            value[(2, col)] = -sign * phi / impedance;
            value[(3, col)] = -sign * z / impedance;
            derivative[(0, col)] = dphi;
            derivative[(1, col)] = dz;
            derivative[(2, col)] =
                -sign * (dphi - phi * direction.impedance / impedance) / impedance;
            derivative[(3, col)] = -sign * (dz - z * direction.impedance / impedance) / impedance;
        }
    }
    Ok((value, derivative))
}

/// Scattering coefficients of concentric cylinders, with full material chirality.
pub fn mie_cyl(
    kz: f64,
    order: i32,
    k0: f64,
    radii: &[f64],
    materials: &[Material],
) -> Result<CylinderResidual> {
    crate::coeffs::validate_layers(radii, materials)?;
    if !kz.is_finite() || !k0.is_finite() || k0 <= 0.0 || order.unsigned_abs() > 128 {
        return Err(Error::InvalidInput(
            "require finite kz, positive k0 and |m| <= 128".into(),
        ));
    }
    let mut q = Matrix42::zeros();
    q[(0, 0)] = Complex::new(1.0, 0.0);
    q[(1, 1)] = Complex::new(1.0, 0.0);
    let mut interfaces = Vec::with_capacity(radii.len());
    for (i, &radius) in radii.iter().enumerate() {
        let inner = materials[i];
        let outer = materials[i + 1];
        let ks = |m: Material| [k0 * (m.index() - m.kappa), k0 * (m.index() + m.kappa)];
        let (inside, _) = boundary(
            order,
            kz,
            ks(inner),
            radius,
            inner.impedance(),
            Direction::default(),
            i == 0,
        )?;
        let (outside, _) = boundary(
            order,
            kz,
            ks(outer),
            radius,
            outer.impedance(),
            Direction::default(),
            false,
        )?;
        let inverse_outer = outside.try_inverse().ok_or(Error::Singular)?;
        let transfer = inverse_outer * inside;
        interfaces.push(Interface {
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
    Ok(CylinderResidual {
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
impl CylinderResidual {
    /// Consume the residual and contract an arbitrary complex output cotangent.
    pub fn pullback(self, cotangent: &Matrix2) -> Result<CylinderGradient> {
        if cotangent.iter().any(|&z| !finite(z)) {
            return Err(Error::InvalidInput("cotangent must be finite".into()));
        }
        let mut result = CylinderGradient {
            kz: 0.0,
            k0: 0.0,
            layers: MieGradient {
                sizes: vec![0.0; self.radii.len()],
                epsilon: vec![Complex::default(); self.materials.len()],
                mu: vec![Complex::default(); self.materials.len()],
                kappa: vec![Complex::default(); self.materials.len()],
            },
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
            for (side, g) in [inner, outer].into_iter().enumerate() {
                let material = self.materials[i + side];
                let n = material.index();
                let z = material.impedance();
                let indices = [n - material.kappa, n + material.kappa];
                let ks = indices.map(|n| self.k0 * n);
                let contract = |direction: Direction| -> Result<Complex> {
                    let (_, df) = boundary(
                        self.order,
                        self.kz,
                        ks,
                        self.radii[i],
                        z,
                        direction,
                        i == 0 && side == 0,
                    )?;
                    Ok(g.dotc(&df).conj())
                };
                let one = Complex::new(1.0, 0.0);
                result.layers.sizes[i] += contract(Direction {
                    radius: one,
                    ..Direction::default()
                })?
                .re;
                result.kz += contract(Direction {
                    kz: one,
                    ..Direction::default()
                })?
                .re;
                result.k0 += contract(Direction {
                    ks: indices,
                    ..Direction::default()
                })?
                .re;
                result.layers.epsilon[i + side] += contract(Direction {
                    ks: [self.k0 * material.mu / (2.0 * n); 2],
                    impedance: -z / (2.0 * material.epsilon),
                    ..Direction::default()
                })?;
                result.layers.mu[i + side] += contract(Direction {
                    ks: [self.k0 * material.epsilon / (2.0 * n); 2],
                    impedance: z / (2.0 * material.mu),
                    ..Direction::default()
                })?;
                result.layers.kappa[i + side] += contract(Direction {
                    ks: [-self.k0 * one, self.k0 * one],
                    ..Direction::default()
                })?;
            }
            gq = interface.transfer.adjoint() * gq;
        }
        Ok(result)
    }
}

/// Multilayer cylinder T-matrix with one retained boundary solve per axial/order pair.
#[derive(Debug)]
pub struct CylinderMatrixResidual {
    blocks: Vec<CylinderResidual>,
    orders: usize,
    /// T-matrix ordered by kz, ascending m and positive/negative helicity.
    pub value: nalgebra::DMatrix<Complex>,
}

/// Cylinder T-matrix parameter cotangents.
#[derive(Debug)]
pub struct CylinderMatrixGradient {
    /// Axial wavenumber cotangent for each input kz.
    pub kzs: Vec<f64>,
    /// Vacuum wavenumber cotangent.
    pub k0: f64,
    /// Radius and material cotangents.
    pub layers: MieGradient,
}

/// Assemble a cylinder T-matrix, parallelizing the independent boundary solves.
pub fn cylinder(
    kzs: &[f64],
    mmax: u32,
    k0: f64,
    radii: &[f64],
    materials: &[Material],
) -> Result<CylinderMatrixResidual> {
    use rayon::prelude::*;
    if kzs.is_empty() || kzs.iter().any(|k| !k.is_finite()) || mmax > 128 {
        return Err(Error::InvalidInput(
            "require finite nonempty kzs and mmax <= 128".into(),
        ));
    }
    let mut seen = std::collections::HashSet::new();
    if kzs
        .iter()
        .any(|&k| !seen.insert(if k == 0.0 { 0 } else { k.to_bits() }))
    {
        return Err(Error::InvalidInput(
            "axial wavenumbers must be unique".into(),
        ));
    }
    let orders = (2 * mmax + 1) as usize;
    let bound = i32::try_from(mmax).map_err(|_| Error::InvalidInput("invalid mmax".into()))?;
    let blocks = kzs
        .par_iter()
        .flat_map(|&kz| {
            (-bound..=bound)
                .into_par_iter()
                .map(move |order| mie_cyl(kz, order, k0, radii, materials))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut value = nalgebra::DMatrix::zeros(2 * blocks.len(), 2 * blocks.len());
    for (index, block) in blocks.iter().enumerate() {
        for row in 0..2 {
            for col in 0..2 {
                value[(2 * index + row, 2 * index + col)] = block.value[(1 - row, 1 - col)];
            }
        }
    }
    Ok(CylinderMatrixResidual {
        blocks,
        orders,
        value,
    })
}

impl CylinderMatrixResidual {
    /// Reverse the complete matrix through the exact forward boundary solves.
    pub fn pullback(
        self,
        cotangent: &nalgebra::DMatrix<Complex>,
    ) -> Result<CylinderMatrixGradient> {
        use rayon::prelude::*;
        if cotangent.shape() != self.value.shape() || cotangent.iter().any(|&g| !finite(g)) {
            return Err(Error::InvalidInput(
                "invalid cylinder T-matrix cotangent".into(),
            ));
        }
        let first = &self.blocks[0];
        let mut result = CylinderMatrixGradient {
            kzs: vec![0.0; self.blocks.len() / self.orders],
            k0: 0.0,
            layers: MieGradient {
                sizes: vec![0.0; first.radii.len()],
                epsilon: vec![Complex::default(); first.materials.len()],
                mu: vec![Complex::default(); first.materials.len()],
                kappa: vec![Complex::default(); first.materials.len()],
            },
        };
        let gradients = self
            .blocks
            .into_par_iter()
            .enumerate()
            .map(|(index, block)| {
                block.pullback(&Matrix2::from_fn(|i, j| {
                    cotangent[(2 * index + 1 - i, 2 * index + 1 - j)]
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        for (index, g) in gradients.into_iter().enumerate() {
            result.kzs[index / self.orders] += g.kz;
            result.k0 += g.k0;
            for (a, b) in result.layers.sizes.iter_mut().zip(g.layers.sizes) {
                *a += b;
            }
            for (a, b) in result.layers.epsilon.iter_mut().zip(g.layers.epsilon) {
                *a += b;
            }
            for (a, b) in result.layers.mu.iter_mut().zip(g.layers.mu) {
                *a += b;
            }
            for (a, b) in result.layers.kappa.iter_mut().zip(g.layers.kappa) {
                *a += b;
            }
        }
        Ok(result)
    }
}
