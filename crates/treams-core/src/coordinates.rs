//! Coordinate and vector-component transformations in orthonormal local frames.
#![allow(clippy::indexing_slicing)] // Fixed Cartesian triples.

use crate::{Complex, Error, Result, finite};

/// Coordinate conversion; vector positions are expressed in the input system.
#[derive(Clone, Copy, Debug)]
pub enum Transform {
    /// Cartesian (x,y,z) to cylindrical (rho,phi,z).
    CarToCyl,
    /// Cartesian (x,y,z) to spherical (r,theta,phi).
    CarToSph,
    /// Cylindrical to Cartesian.
    CylToCar,
    /// Cylindrical to spherical.
    CylToSph,
    /// Spherical to Cartesian.
    SphToCar,
    /// Spherical to cylindrical.
    SphToCyl,
    /// Cartesian (x,y) to polar (rho,phi).
    CarToPol,
    /// Polar to Cartesian (x,y).
    PolToCar,
}
impl Transform {
    /// Number of coordinate or vector components.
    #[must_use]
    pub fn dimension(self) -> usize {
        match self {
            Self::CarToPol | Self::PolToCar => 2,
            _ => 3,
        }
    }
}

#[inline]
fn radius(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs());
    if (1e-150..=1e150).contains(&scale) {
        a.mul_add(a, b * b).sqrt()
    } else {
        a.hypot(b)
    }
}

fn validate(position: [f64; 3]) -> Result<()> {
    if position.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput("coordinates must be finite".into()));
    }
    Ok(())
}

/// Transform a point. Polar conversions ignore and return zero in the third slot.
#[inline]
pub fn point(position: [f64; 3], transform: Transform) -> Result<[f64; 3]> {
    use Transform::{
        CarToCyl, CarToPol, CarToSph, CylToCar, CylToSph, PolToCar, SphToCar, SphToCyl,
    };
    validate(position)?;
    let [a, b, c] = position;
    Ok(match transform {
        CarToCyl | CarToPol => [
            radius(a, b),
            b.atan2(a),
            if matches!(transform, CarToPol) {
                0.0
            } else {
                c
            },
        ],
        CarToSph => {
            let rho = radius(a, b);
            [radius(rho, c), rho.atan2(c), b.atan2(a)]
        }
        CylToCar | PolToCar => {
            let (sin, cos) = b.sin_cos();
            [
                a * cos,
                a * sin,
                if matches!(transform, PolToCar) {
                    0.0
                } else {
                    c
                },
            ]
        }
        CylToSph => [radius(a, c), a.atan2(c), b],
        SphToCar => {
            let (st, ct) = b.sin_cos();
            let (sp, cp) = c.sin_cos();
            [a * st * cp, a * st * sp, a * ct]
        }
        SphToCyl => {
            let (sin, cos) = b.sin_cos();
            [a * sin, c, a * cos]
        }
    })
}

/// Contract a point-coordinate Jacobian. Undefined angular derivatives at axes
/// are rejected only when their output cotangent is nonzero.
pub fn point_pullback(position: [f64; 3], transform: Transform, g: [f64; 3]) -> Result<[f64; 3]> {
    use Transform::{
        CarToCyl, CarToPol, CarToSph, CylToCar, CylToSph, PolToCar, SphToCar, SphToCyl,
    };
    validate(position)?;
    validate(g)?;
    let [a, b, c] = position;
    let undefined =
        || Error::InvalidInput("coordinate derivative is undefined at this axis or origin".into());
    let mut result = [0.0; 3];
    match transform {
        CarToCyl | CarToPol | CarToSph => {
            let rho = radius(a, b);
            let spherical = matches!(transform, CarToSph);
            let phi_g = if spherical { g[2] } else { g[1] };
            if rho == 0.0 && (phi_g != 0.0 || if spherical { g[1] != 0.0 } else { g[0] != 0.0 }) {
                return Err(undefined());
            }
            let (sp, cp) = if rho == 0.0 {
                (0.0, 1.0)
            } else {
                (b / rho, a / rho)
            };
            let (radial, axial) = if spherical {
                let r = radius(rho, c);
                if r == 0.0 && g[0] != 0.0 {
                    return Err(undefined());
                }
                if r == 0.0 {
                    (0.0, 0.0)
                } else {
                    (
                        g[0] * (rho / r) + g[1] * (c / r) / r,
                        g[0] * (c / r) - g[1] * (rho / r) / r,
                    )
                }
            } else {
                (
                    g[0],
                    if matches!(transform, CarToPol) {
                        0.0
                    } else {
                        g[2]
                    },
                )
            };
            let angular = if phi_g == 0.0 { 0.0 } else { phi_g / rho };
            result = [
                radial * cp - angular * sp,
                radial * sp + angular * cp,
                axial,
            ];
        }
        CylToCar | PolToCar => {
            let (sin, cos) = b.sin_cos();
            result = [
                g[0] * cos + g[1] * sin,
                a * (-g[0] * sin + g[1] * cos),
                if matches!(transform, PolToCar) {
                    0.0
                } else {
                    g[2]
                },
            ];
        }
        CylToSph => {
            let r = radius(a, c);
            if r == 0.0 && (g[0] != 0.0 || g[1] != 0.0) {
                return Err(undefined());
            }
            if r != 0.0 {
                result[0] = g[0] * (a / r) + g[1] * (c / r) / r;
                result[2] = g[0] * (c / r) - g[1] * (a / r) / r;
            }
            result[1] = g[2];
        }
        SphToCar => {
            let (st, ct) = b.sin_cos();
            let (sp, cp) = c.sin_cos();
            let radial = g[0] * cp + g[1] * sp;
            result = [
                radial * st + g[2] * ct,
                a * (radial * ct - g[2] * st),
                a * st * (-g[0] * sp + g[1] * cp),
            ];
        }
        SphToCyl => {
            let (sin, cos) = b.sin_cos();
            result = [g[0] * sin + g[2] * cos, a * (g[0] * cos - g[2] * sin), g[1]];
        }
    }
    Ok(result)
}

/// Rotation of vector components, and angle derivatives of that rotation.
/// Polar vectors use the first two rows/columns; no radial metric factors occur.
fn frame(position: [f64; 3], transform: Transform) -> ([[f64; 3]; 3], [[[f64; 3]; 3]; 2]) {
    use Transform::{
        CarToCyl, CarToPol, CarToSph, CylToCar, CylToSph, PolToCar, SphToCar, SphToCyl,
    };
    let [a, b, c] = position;
    let zero = [[0.0; 3]; 3];
    match transform {
        CarToCyl | CarToPol | CylToCar | PolToCar => {
            let inverse = matches!(transform, CylToCar | PolToCar);
            let phi = if inverse { b } else { b.atan2(a) };
            let (mut sin, cos) = phi.sin_cos();
            let sign = if inverse { -1.0 } else { 1.0 };
            sin *= sign;
            (
                [[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]],
                [
                    [
                        [-sign * sin, sign * cos, 0.0],
                        [-sign * cos, -sign * sin, 0.0],
                        [0.0; 3],
                    ],
                    zero,
                ],
            )
        }
        CylToSph | SphToCyl => {
            let inverse = matches!(transform, SphToCyl);
            let theta = if inverse { b } else { a.atan2(c) };
            let (sin, cos) = theta.sin_cos();
            let r = [[sin, 0.0, cos], [cos, 0.0, -sin], [0.0, 1.0, 0.0]];
            let d = [[cos, 0.0, -sin], [-sin, 0.0, -cos], [0.0; 3]];
            if inverse {
                (transpose(r), [transpose(d), zero])
            } else {
                (r, [d, zero])
            }
        }
        CarToSph | SphToCar => {
            let inverse = matches!(transform, SphToCar);
            let theta = if inverse { b } else { radius(a, b).atan2(c) };
            let phi = if inverse { c } else { b.atan2(a) };
            let (st, ct) = theta.sin_cos();
            let (sp, cp) = phi.sin_cos();
            let r = [
                [st * cp, st * sp, ct],
                [ct * cp, ct * sp, -st],
                [-sp, cp, 0.0],
            ];
            let dt = [[ct * cp, ct * sp, -st], [-st * cp, -st * sp, -ct], [0.0; 3]];
            let dp = [
                [-st * sp, st * cp, 0.0],
                [-ct * sp, ct * cp, 0.0],
                [-cp, -sp, 0.0],
            ];
            if inverse {
                (transpose(r), [transpose(dt), transpose(dp)])
            } else {
                (r, [dt, dp])
            }
        }
    }
}
fn transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| matrix[j][i]))
}
fn apply(matrix: [[f64; 3]; 3], vector: [Complex; 3]) -> [Complex; 3] {
    matrix.map(|row| row.into_iter().zip(vector).map(|(a, b)| a * b).sum())
}

/// Transform vector components at a point in the source coordinate system.
#[inline]
pub fn vector(
    value: [Complex; 3],
    position: [f64; 3],
    transform: Transform,
) -> Result<[Complex; 3]> {
    validate(position)?;
    if value.into_iter().any(|v| !finite(v)) {
        return Err(Error::InvalidInput(
            "vector components must be finite".into(),
        ));
    }
    use Transform::{
        CarToCyl, CarToPol, CarToSph, CylToCar, CylToSph, PolToCar, SphToCar, SphToCyl,
    };
    let [a, b, c] = position;
    let [x, y, z] = value;
    Ok(match transform {
        CarToCyl | CarToPol => {
            let (sin, cos) = b.atan2(a).sin_cos();
            [x * cos + y * sin, -x * sin + y * cos, z]
        }
        CylToCar | PolToCar => {
            let (sin, cos) = b.sin_cos();
            [x * cos - y * sin, x * sin + y * cos, z]
        }
        CylToSph => {
            let (sin, cos) = a.atan2(c).sin_cos();
            [x * sin + z * cos, x * cos - z * sin, y]
        }
        SphToCyl => {
            let (sin, cos) = b.sin_cos();
            [x * sin + y * cos, z, x * cos - y * sin]
        }
        CarToSph => {
            let (st, ct) = radius(a, b).atan2(c).sin_cos();
            let (sp, cp) = b.atan2(a).sin_cos();
            let radial = x * cp + y * sp;
            [radial * st + z * ct, radial * ct - z * st, -x * sp + y * cp]
        }
        SphToCar => {
            let (st, ct) = b.sin_cos();
            let (sp, cp) = c.sin_cos();
            let radial = x * st + y * ct;
            [radial * cp - z * sp, radial * sp + z * cp, x * ct - y * st]
        }
    })
}

/// Contract derivatives with respect to complex vector components and real position.
pub fn vector_pullback(
    value: [Complex; 3],
    position: [f64; 3],
    transform: Transform,
    g: [Complex; 3],
) -> Result<([Complex; 3], [f64; 3])> {
    use Transform::{
        CarToCyl, CarToPol, CarToSph, CylToCar, CylToSph, PolToCar, SphToCar, SphToCyl,
    };
    validate(position)?;
    if value.into_iter().chain(g).any(|v| !finite(v)) {
        return Err(Error::InvalidInput(
            "vector and cotangent must be finite".into(),
        ));
    }
    let (rotation, derivatives) = frame(position, transform);
    let angles = derivatives.map(|d| {
        apply(d, value)
            .into_iter()
            .zip(g)
            .map(|(d, g)| (g.conj() * d).re)
            .sum::<f64>()
    });
    let point = match transform {
        CarToCyl | CarToPol | CylToSph => {
            point_pullback(position, transform, [0.0, angles[0], 0.0])?
        }
        CarToSph => point_pullback(position, transform, [0.0, angles[0], angles[1]])?,
        CylToCar | PolToCar | SphToCyl => [0.0, angles[0], 0.0],
        SphToCar => [0.0, angles[0], angles[1]],
    };
    Ok((apply(transpose(rotation), g), point))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Property failures retain reproducible inputs.
    use super::*;
    #[test]
    fn coordinate_norm_preserves_extreme_scales() {
        for scale in [
            f64::from_bits(1),
            1e-300,
            1e-151,
            1e-149,
            1.0,
            1e149,
            1e151,
            1e300,
            f64::MAX / 2.0,
        ] {
            for (a, b) in [(0.0, 1.0), (1.0, 0.0), (0.3, -0.4), (1.0, -1.0)] {
                let a = a * scale;
                let b = b * scale;
                let expected = a.hypot(b);
                assert!((radius(a, b) - expected).abs() <= 3.0 * f64::EPSILON * expected);
            }
        }
    }

    use proptest::prelude::*;
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]
        #[test]
        fn coordinate_and_vector_adjoints(a in 0.2_f64..2.0,b in -1.0_f64..1.0,c in -1.0_f64..1.0) {
            use Transform::*;
            let h=1e-6;
            for transform in [CarToCyl,CarToSph,CylToCar,CylToSph,SphToCar,SphToCyl,CarToPol,PolToCar] {
                let p=[a,b,if transform.dimension()==2 {0.0}else{c}];
                let g=[0.3,-0.2,if transform.dimension()==2 {0.0}else{0.4}];
                let v=[Complex::new(0.4,0.2),Complex::new(-0.3,0.1),if transform.dimension()==2 {Complex::default()}else{Complex::new(0.7,-0.2)}];
                let cotangent=g.map(|x|Complex::new(x,0.5*x));
                let point_g=point_pullback(p,transform,g).unwrap();
                let (vector_g,position_g)=vector_pullback(v,p,transform,cotangent).unwrap();
                for axis in 0..transform.dimension() {
                    let mut plus=p;plus[axis]+=h;
                    let mut minus=p;minus[axis]-=h;
                    let difference=point(plus,transform).unwrap().into_iter().zip(point(minus,transform).unwrap()).zip(g).map(|((p,m),g)|g*(p-m)/(2.0*h)).sum::<f64>();
                    prop_assert!((difference-point_g[axis]).abs()<2e-8*(1.0+difference.abs()));
                    let difference=vector(v,plus,transform).unwrap().into_iter().zip(vector(v,minus,transform).unwrap()).zip(cotangent).map(|((p,m),g)|(g.conj()*(p-m)).re/(2.0*h)).sum::<f64>();
                    prop_assert!((difference-position_g[axis]).abs()<2e-8*(1.0+difference.abs()));
                    let direction=Complex::new(0.2,-0.4);
                    let mut plus=v;plus[axis]+=h*direction;
                    let mut minus=v;minus[axis]-=h*direction;
                    let difference=vector(plus,p,transform).unwrap().into_iter().zip(vector(minus,p,transform).unwrap()).zip(cotangent).map(|((p,m),g)|(g.conj()*(p-m)).re/(2.0*h)).sum::<f64>();
                    prop_assert!((difference-(vector_g[axis].conj()*direction).re).abs()<2e-8*(1.0+difference.abs()));
                }
            }
        }
    }
    #[test]
    #[allow(clippy::float_cmp)] // Axis normals and zero cotangents have exact results.
    fn coordinate_axis_derivative_contract() {
        assert!(point_pullback([0.0, 0.0, 1.0], Transform::CarToSph, [0.0, 0.0, 1.0]).is_err());
        assert_eq!(
            point_pullback([0.0, 0.0, -1.0], Transform::CarToSph, [1.0, 0.0, 0.0]).unwrap(),
            [0.0, 0.0, -1.0]
        );
        assert_eq!(
            point_pullback([0.0; 3], Transform::CarToPol, [0.0; 3]).unwrap(),
            [0.0; 3]
        );
    }
}
