//! Multilayer chiral sphere coefficients and a native analytic reverse pass.
// Fixed-size matrices below have statically known dimensions; their indices are
// the regular/outgoing and helicity channels in the interface equations.
#![allow(clippy::indexing_slicing)]

use nalgebra::SMatrix;

use crate::special::{Radial, spherical};
use crate::{Complex, Error, Result, finite};

type Matrix4 = SMatrix<Complex, 4, 4>;
type Matrix42 = SMatrix<Complex, 4, 2>;
/// Two helicity channels, in treams coefficient order (negative, positive).
pub type Matrix2 = SMatrix<Complex, 2, 2>;

/// Isotropic reciprocal material, optionally chiral.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// Relative permittivity.
    pub epsilon: Complex,
    /// Relative permeability.
    pub mu: Complex,
    /// Chirality parameter.
    pub kappa: Complex,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            epsilon: Complex::new(1.0, 0.0),
            mu: Complex::new(1.0, 0.0),
            kappa: Complex::default(),
        }
    }
}

impl Material {
    /// Principal-branch refractive index, following treams.
    #[must_use]
    pub fn index(self) -> Complex {
        crate::complex_sqrt(self.epsilon * self.mu)
    }
    /// Negative/positive-helicity indices on the outgoing branch.
    #[must_use]
    pub fn indices(self) -> [Complex; 2] {
        let n = self.index();
        [n - self.kappa, n + self.kappa].map(|v| if v.im < 0.0 { -v } else { v })
    }
    /// Principal-branch wave impedance, following treams.
    #[must_use]
    pub fn impedance(self) -> Complex {
        crate::complex_sqrt(self.mu / self.epsilon)
    }
}

#[derive(Clone, Debug)]
struct Interface {
    x: [[Complex; 2]; 2],
    z: [Complex; 2],
    matrix: Matrix4,
    before: Matrix42,
}

/// Retained native execution for one multipole order.
#[derive(Clone, Debug)]
pub struct MieResidual {
    l: u32,
    sizes: Vec<f64>,
    materials: Vec<Material>,
    interfaces: Vec<Interface>,
    inverse: Matrix2,
    /// Forward Mie coefficient matrix.
    pub value: Matrix2,
}

/// Input cotangents under `dL = Re(sum(conj(g) * dx))`.
#[derive(Clone, Debug)]
pub struct MieGradient {
    /// Size-parameter gradients, from inner to outer boundary.
    pub sizes: Vec<f64>,
    /// Permittivity gradients, including the embedding medium.
    pub epsilon: Vec<Complex>,
    /// Permeability gradients, including the embedding medium.
    pub mu: Vec<Complex>,
    /// Chirality gradients, including the embedding medium.
    pub kappa: Vec<Complex>,
}

// Returns F and an analytic directional derivative of this four-channel
// interface. The radial differential equation supplies second derivatives.
fn interface(
    l: u32,
    x: [[Complex; 2]; 2],
    z: [Complex; 2],
    dx: [[Complex; 2]; 2],
    dz: [Complex; 2],
) -> Result<(Matrix4, Matrix4)> {
    let mut waves = [[[(Complex::default(), Complex::default()); 2]; 2]; 2];
    let mut derivatives = waves;
    for side in 0..2 {
        for pol in 0..2 {
            for (kind, radial) in [Radial::Regular, Radial::Outgoing].into_iter().enumerate() {
                let arg = x[side][pol];
                let f = spherical(l, arg, radial)?;
                waves[side][pol][kind] = (f.value, f.first + f.value / arg);
                derivatives[side][pol][kind] = (
                    f.first * dx[side][pol],
                    (f.second + f.first / arg - f.value / arg.powu(2)) * dx[side][pol],
                );
            }
        }
    }
    let mut f = Matrix4::zeros();
    let mut df = Matrix4::zeros();
    for p in 0..2 {
        for q in 0..2 {
            let sign = if p == q { 1.0 } else { -1.0 };
            let ratio = z[1] / z[0];
            let a = (ratio + sign) / (2.0 * Complex::i());
            let da = (dz[1] / z[0] - ratio * dz[0] / z[0]) / (2.0 * Complex::i());
            let factor = a * x[1][p].powu(2);
            let dfactor = da * x[1][p].powu(2) + 2.0 * a * x[1][p] * dx[1][p];
            for kind in 0..2 {
                let (inner, dinner) = (waves[0][q][kind], derivatives[0][q][kind]);
                for out_kind in 0..2 {
                    let (outer, douter) =
                        (waves[1][p][1 - out_kind], derivatives[1][p][1 - out_kind]);
                    let out_sign = if out_kind == 0 { 1.0 } else { -1.0 };
                    let term = out_sign * (outer.1 * inner.0 - sign * outer.0 * inner.1);
                    let dterm = out_sign
                        * (douter.1 * inner.0 + outer.1 * dinner.0
                            - sign * (douter.0 * inner.1 + outer.0 * dinner.1));
                    f[(p + out_kind * 2, q + kind * 2)] = term * factor;
                    df[(p + out_kind * 2, q + kind * 2)] = dterm * factor + term * dfactor;
                }
            }
        }
    }
    Ok((f, df))
}

/// Solve a concentric multilayer sphere, retaining the native reverse context.
pub fn mie_forward(l: u32, sizes: &[f64], materials: &[Material]) -> Result<MieResidual> {
    if !(1..=128).contains(&l) || sizes.is_empty() || materials.len() != sizes.len() + 1 {
        return Err(Error::InvalidInput(
            "require 1 <= l <= 128 and one more material than radii".into(),
        ));
    }
    validate_layers(sizes, materials)?;
    let mut q = Matrix42::zeros();
    q[(0, 0)] = Complex::new(1.0, 0.0);
    q[(1, 1)] = Complex::new(1.0, 0.0);
    let mut interfaces = Vec::with_capacity(sizes.len());
    for (i, size) in sizes.iter().enumerate() {
        let pair = [materials[i], materials[i + 1]];
        let x = pair.map(|m| [size * (m.index() - m.kappa), size * (m.index() + m.kappa)]);
        let z = pair.map(Material::impedance);
        let (matrix, _) = interface(
            l,
            x,
            z,
            [[Complex::default(); 2]; 2],
            [Complex::default(); 2],
        )?;
        interfaces.push(Interface {
            x,
            z,
            matrix,
            before: q,
        });
        q = matrix * q;
    }
    let upper = q.fixed_rows::<2>(0).into_owned();
    let inverse = upper.try_inverse().ok_or(Error::Singular)?;
    let value = q.fixed_rows::<2>(2) * inverse;
    if value.iter().any(|z| !finite(*z)) {
        return Err(Error::Singular);
    }
    Ok(MieResidual {
        l,
        sizes: sizes.to_vec(),
        materials: materials.to_vec(),
        interfaces,
        inverse,
        value,
    })
}

impl MieResidual {
    /// Consume a forward residual and apply its analytic reverse pass.
    pub fn pullback(self, cotangent: &Matrix2) -> Result<MieGradient> {
        if cotangent.iter().any(|z| !finite(*z)) {
            return Err(Error::InvalidInput("cotangents must be finite".into()));
        }
        let mut result = MieGradient {
            sizes: vec![0.0; self.sizes.len()],
            epsilon: vec![Complex::default(); self.materials.len()],
            mu: vec![Complex::default(); self.materials.len()],
            kappa: vec![Complex::default(); self.materials.len()],
        };
        let lower = cotangent * self.inverse.adjoint();
        let upper = -self.value.adjoint() * lower;
        let mut gq = Matrix42::zeros();
        gq.fixed_rows_mut::<2>(0).copy_from(&upper);
        gq.fixed_rows_mut::<2>(2).copy_from(&lower);
        for (i, layer) in self.interfaces.iter().enumerate().rev() {
            let gf = gq * layer.before.adjoint();
            let contract = |dx, dz| -> Result<Complex> {
                let (_, df) = interface(self.l, layer.x, layer.z, dx, dz)?;
                Ok(gf
                    .iter()
                    .zip(df.iter())
                    .map(|(g, d)| g.conj() * d)
                    .sum::<Complex>()
                    .conj())
            };
            result.sizes[i] = contract(
                layer.x.map(|row| row.map(|x| x / self.sizes[i])),
                [Complex::default(); 2],
            )?
            .re;
            for side in 0..2 {
                let material = self.materials[i + side];
                let n = material.index();
                let impedance = layer.z[side];
                for parameter in 0..3 {
                    let (dn, dz, dk) = match parameter {
                        0 => (
                            material.mu / (2.0 * n),
                            -impedance / (2.0 * material.epsilon),
                            Complex::default(),
                        ),
                        1 => (
                            material.epsilon / (2.0 * n),
                            impedance / (2.0 * material.mu),
                            Complex::default(),
                        ),
                        _ => (
                            Complex::default(),
                            Complex::default(),
                            Complex::new(1.0, 0.0),
                        ),
                    };
                    let mut dx = [[Complex::default(); 2]; 2];
                    dx[side] = [self.sizes[i] * (dn - dk), self.sizes[i] * (dn + dk)];
                    let mut z_tangent = [Complex::default(); 2];
                    z_tangent[side] = dz;
                    let gradient = contract(dx, z_tangent)?;
                    match parameter {
                        0 => result.epsilon[i + side] += gradient,
                        1 => result.mu[i + side] += gradient,
                        _ => result.kappa[i + side] += gradient,
                    }
                }
            }
            gq = layer.matrix.adjoint() * gq;
        }
        Ok(result)
    }
}

/// Shared material and concentric-boundary validation.
pub(crate) fn validate_layers(sizes: &[f64], materials: &[Material]) -> Result<()> {
    if sizes.is_empty() || materials.len() != sizes.len() + 1 {
        return Err(Error::InvalidInput(
            "require one more material than radii".into(),
        ));
    }
    if sizes.iter().any(|x| !x.is_finite() || *x <= 0.0) || sizes.windows(2).any(|x| x[0] >= x[1]) {
        return Err(Error::InvalidInput(
            "size parameters must be finite, positive and strictly increasing".into(),
        ));
    }
    for material in materials {
        if !finite(material.epsilon)
            || !finite(material.mu)
            || !finite(material.kappa)
            || material.epsilon.norm_sqr() == 0.0
            || material.mu.norm_sqr() == 0.0
        {
            return Err(Error::InvalidInput(
                "require finite materials with nonzero epsilon and mu".into(),
            ));
        }
    }
    Ok(())
}
