//! Coordinate and vector-component transformations in orthonormal local frames.
//!
//! Upstream: `treams.special.car2cyl`, `car2sph`, `cyl2car`, `cyl2sph`, `sph2car`,
//! `sph2cyl`, `car2pol`, `pol2car` ([`point`]) and their `v`-prefixed vector twins such
//! as `vcar2sph` ([`vector`]), from `_coord.pyx`. Spherical coordinates are
//! `(r, theta, phi)` with the polar angle `theta`, cylindrical ones `(rho, phi, z)`, as
//! in treams.
#![allow(clippy::indexing_slicing)] // Fixed Cartesian triples.

use crate::{Complex, Error, Result, numerics::finite};

/// Coordinate conversion; vector positions are expressed in the input system.
///
/// Upstream: each variant names a function of `treams.special` for [`point`]
/// (`CarToCyl` is `car2cyl`); [`vector`] takes the `v`-prefixed one (`vcar2cyl`).
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
use Transform::{CarToCyl, CarToPol, CarToSph, CylToCar, CylToSph, PolToCar, SphToCar, SphToCyl};

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

/// `sqrt(a^2 + b^2)`: a fused multiply-add where the squares stay normal, `hypot`
/// elsewhere.
#[inline]
fn radius(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs());
    if (1e-150..=1e150).contains(&scale) {
        a.mul_add(a, b * b).sqrt()
    } else {
        a.hypot(b)
    }
}

/// Checks that every component is finite.
fn validate(position: [f64; 3]) -> Result<()> {
    if position.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput("coordinates must be finite".into()));
    }
    Ok(())
}

/// Cylindrical `(rho, phi, z)` of Cartesian `(x, y, z)`.
#[inline]
fn cylindrical_from_cartesian([a, b, c]: [f64; 3]) -> [f64; 3] {
    [radius(a, b), b.atan2(a), c]
}

/// Cartesian `(x, y, z)` of cylindrical `(rho, phi, z)`.
#[inline]
fn cartesian_from_cylindrical([a, b, c]: [f64; 3]) -> [f64; 3] {
    let (sin, cos) = b.sin_cos();
    [a * cos, a * sin, c]
}

/// Spherical `(r, theta, phi)` of cylindrical `(rho, phi, z)`.
#[inline]
fn spherical_from_cylindrical([a, b, c]: [f64; 3]) -> [f64; 3] {
    [radius(a, c), a.atan2(c), b]
}

/// Cylindrical `(rho, phi, z)` of spherical `(r, theta, phi)`.
#[inline]
fn cylindrical_from_spherical([a, b, c]: [f64; 3]) -> [f64; 3] {
    let (sin, cos) = b.sin_cos();
    [a * sin, c, a * cos]
}

/// Transform a point. Polar conversions ignore and return zero in the third slot.
///
/// Cartesian and spherical coordinates convert through cylindrical coordinates.
///
/// Upstream: `treams.special.car2cyl` and the other point conversions of
/// [`Transform`]. Differences: non-finite coordinates give an error.
// Always inlined: the ufunc loops pass a constant transform, which then selects
// one chart at compile time (a call per point costs a third more).
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn point(position: [f64; 3], transform: Transform) -> Result<[f64; 3]> {
    validate(position)?;
    Ok(match transform {
        CarToCyl => cylindrical_from_cartesian(position),
        CarToPol => {
            let [rho, phi, _] = cylindrical_from_cartesian(position);
            [rho, phi, 0.0]
        }
        CylToCar => cartesian_from_cylindrical(position),
        PolToCar => {
            let [x, y, _] = cartesian_from_cylindrical(position);
            [x, y, 0.0]
        }
        CylToSph => spherical_from_cylindrical(position),
        SphToCyl => cylindrical_from_spherical(position),
        // A cylindrical radius that overflows gives an infinite spherical radius with
        // exact angles, as in treams.
        CarToSph => spherical_from_cylindrical(cylindrical_from_cartesian(position)),
        SphToCar => cartesian_from_cylindrical(cylindrical_from_spherical(position)),
    })
}

/// The gradient with respect to the input coordinates of [`point`], given the gradient
/// `cotangent` with respect to its output.
///
/// Angular derivatives are undefined on the axes; they give an error only where their
/// `cotangent` component is nonzero.
pub fn point_pullback(
    position: [f64; 3],
    transform: Transform,
    cotangent: [f64; 3],
) -> Result<[f64; 3]> {
    validate(position)?;
    validate(cotangent)?;
    let [a, b, c] = position;
    let undefined =
        || Error::InvalidInput("coordinate derivative is undefined at this axis or origin".into());
    Ok(match transform {
        CarToCyl | CarToPol => {
            let rho = radius(a, b);
            if rho == 0.0 && (cotangent[0] != 0.0 || cotangent[1] != 0.0) {
                return Err(undefined());
            }
            let (sp, cp) = if rho == 0.0 {
                (0.0, 1.0)
            } else {
                (b / rho, a / rho)
            };
            let angular = if cotangent[1] == 0.0 {
                0.0
            } else {
                cotangent[1] / rho
            };
            [
                cotangent[0] * cp - angular * sp,
                cotangent[0] * sp + angular * cp,
                if matches!(transform, CarToPol) {
                    0.0
                } else {
                    cotangent[2]
                },
            ]
        }
        CylToCar | PolToCar => {
            let (sin, cos) = b.sin_cos();
            [
                cotangent[0] * cos + cotangent[1] * sin,
                a * (-cotangent[0] * sin + cotangent[1] * cos),
                if matches!(transform, PolToCar) {
                    0.0
                } else {
                    cotangent[2]
                },
            ]
        }
        CylToSph => {
            let r = radius(a, c);
            if r == 0.0 && (cotangent[0] != 0.0 || cotangent[1] != 0.0) {
                return Err(undefined());
            }
            if r == 0.0 {
                [0.0, cotangent[2], 0.0]
            } else {
                [
                    cotangent[0] * (a / r) + cotangent[1] * (c / r) / r,
                    cotangent[2],
                    cotangent[0] * (c / r) - cotangent[1] * (a / r) / r,
                ]
            }
        }
        SphToCyl => {
            let (sin, cos) = b.sin_cos();
            [
                cotangent[0] * sin + cotangent[2] * cos,
                a * (cotangent[0] * cos - cotangent[2] * sin),
                cotangent[1],
            ]
        }
        CarToSph => {
            // Polar and azimuthal derivatives are undefined on the whole z axis. Check
            // here: the intermediate cylindrical radial cotangent may underflow to zero.
            if radius(a, b) == 0.0 && (cotangent[1] != 0.0 || cotangent[2] != 0.0) {
                return Err(undefined());
            }
            let cylindrical = point(position, CarToCyl)?;
            point_pullback(
                position,
                CarToCyl,
                point_pullback(cylindrical, CylToSph, cotangent)?,
            )?
        }
        SphToCar => {
            let cylindrical = point(position, SphToCyl)?;
            point_pullback(
                position,
                SphToCyl,
                point_pullback(cylindrical, CylToCar, cotangent)?,
            )?
        }
    })
}

/// Rotation of vector components into the output frame of `transform`, and its
/// derivatives with respect to the (at most two) angles it depends on. Inverse
/// transforms use the transposed matrices. Polar vectors use the first two
/// rows/columns; no radial metric factors occur.
#[inline]
fn frame(position: [f64; 3], transform: Transform) -> ([[f64; 3]; 3], [[[f64; 3]; 3]; 2]) {
    let [a, b, c] = position;
    let inverse = matches!(transform, CylToCar | PolToCar | SphToCyl | SphToCar);
    let zero = [[0.0; 3]; 3];
    let (rotation, derivatives) = match transform {
        CarToCyl | CarToPol | CylToCar | PolToCar => {
            let phi = if inverse { b } else { b.atan2(a) };
            let (sin, cos) = phi.sin_cos();
            (
                [[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]],
                [[[-sin, cos, 0.0], [-cos, -sin, 0.0], [0.0; 3]], zero],
            )
        }
        CylToSph | SphToCyl => {
            let theta = if inverse { b } else { a.atan2(c) };
            let (sin, cos) = theta.sin_cos();
            (
                [[sin, 0.0, cos], [cos, 0.0, -sin], [0.0, 1.0, 0.0]],
                [[[cos, 0.0, -sin], [-sin, 0.0, -cos], [0.0; 3]], zero],
            )
        }
        CarToSph | SphToCar => {
            let theta = if inverse { b } else { radius(a, b).atan2(c) };
            let phi = if inverse { c } else { b.atan2(a) };
            let (st, ct) = theta.sin_cos();
            let (sp, cp) = phi.sin_cos();
            (
                [
                    [st * cp, st * sp, ct],
                    [ct * cp, ct * sp, -st],
                    [-sp, cp, 0.0],
                ],
                [
                    [[ct * cp, ct * sp, -st], [-st * cp, -st * sp, -ct], [0.0; 3]],
                    [
                        [-st * sp, st * cp, 0.0],
                        [-ct * sp, ct * cp, 0.0],
                        [-cp, -sp, 0.0],
                    ],
                ],
            )
        }
    };
    if inverse {
        (transpose(rotation), derivatives.map(transpose))
    } else {
        (rotation, derivatives)
    }
}

/// The transpose of a 3 x 3 matrix.
fn transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| matrix[j][i]))
}

/// The product of a real 3 x 3 matrix and a complex vector.
fn apply(matrix: [[f64; 3]; 3], vector: [Complex; 3]) -> [Complex; 3] {
    matrix.map(|row| row.into_iter().zip(vector).map(|(a, b)| a * b).sum())
}

/// Transform vector components at a point in the source coordinate system.
///
/// Upstream: `treams.special.vcar2cyl` and the other vector conversions of
/// [`Transform`]. Differences: non-finite components or coordinates give an error.
// Always inlined, as `point`: a constant transform selects one frame.
#[allow(clippy::inline_always)]
#[inline(always)]
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
    // Apply `frame`'s rotation without its structural zeros and ones: the dense
    // product costs a third more per point, since x * 0.0 cannot be folded.
    let r = frame(position, transform).0;
    let [x, y, z] = value;
    let row = |i: usize| x * r[i][0] + y * r[i][1];
    Ok(match transform {
        CarToCyl | CarToPol | CylToCar | PolToCar => [row(0), row(1), z],
        CylToSph => [x * r[0][0] + z * r[0][2], x * r[1][0] + z * r[1][2], y],
        SphToCyl => [row(0), z, row(2)],
        CarToSph | SphToCar => [row(0) + z * r[0][2], row(1) + z * r[1][2], row(2)],
    })
}

/// The gradients with respect to the vector components and the position of [`vector`],
/// given the gradient `cotangent` with respect to its output components.
pub fn vector_pullback(
    value: [Complex; 3],
    position: [f64; 3],
    transform: Transform,
    cotangent: [Complex; 3],
) -> Result<([Complex; 3], [f64; 3])> {
    validate(position)?;
    if value.into_iter().chain(cotangent).any(|v| !finite(v)) {
        return Err(Error::InvalidInput(
            "vector and cotangent must be finite".into(),
        ));
    }
    let (rotation, derivatives) = frame(position, transform);
    let angles = derivatives.map(|d| {
        apply(d, value)
            .into_iter()
            .zip(cotangent)
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
    Ok((apply(transpose(rotation), cotangent), point))
}

#[cfg(test)]
mod tests {
    //! Charts against their inverses and compositions, and coordinate pullbacks against
    //! finite differences.

    use super::{Complex, Transform, point, point_pullback, radius, vector, vector_pullback};
    use crate::test_support::{
        ALGEBRA_CASES, DEFAULT_CASES, complex, log_uniform, prop_assert_close, re_dot,
    };
    use Transform::{
        CarToCyl, CarToPol, CarToSph, CylToCar, CylToSph, PolToCar, SphToCar, SphToCyl,
    };
    use proptest::{prelude::*, test_runner::TestCaseError};

    /// Each chart and its inverse.
    const PAIRS: [(Transform, Transform); 4] = [
        (CarToCyl, CylToCar),
        (CarToSph, SphToCar),
        (CylToSph, SphToCyl),
        (CarToPol, PolToCar),
    ];

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

    /// Cartesian points from 1e-300 to 1e300, a quarter of them on the polar axis on
    /// either side of the origin.
    fn cartesian_point() -> impl Strategy<Value = [f64; 3]> {
        let direction = prop_oneof![
            3 => prop::array::uniform3(-2.0_f64..2.0),
            1 => (-2.0_f64..2.0).prop_map(|z| [0.0, 0.0, z]),
        ];
        let scale = prop_oneof![Just(1e-300), Just(1.0), Just(1e300), log_uniform(-3.0..3.0),];
        (direction, scale).prop_map(|(direction, scale)| direction.map(|v| v * scale))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn charts_invert_compose_and_rotate_vectors_orthogonally(
            position in cartesian_point(),
            value in prop::array::uniform3(complex(1.0)),
        ) {
            check_charts(position, value)?;
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

        #[test]
        fn coordinate_and_vector_adjoints(
            a in 0.2_f64..2.0,
            b in -1.0_f64..1.0,
            c in -1.0_f64..1.0,
            value in prop::array::uniform3(complex(1.0)),
            cotangent in prop::array::uniform3(complex(1.0)),
        ) {
            check_adjoints([a, b, c], value, cotangent)?;
        }
    }

    /// The point and vector components of `transform`'s input system, restricted to
    /// the plane for polar charts.
    fn input(
        position: [f64; 3],
        value: [Complex; 3],
        transform: Transform,
    ) -> ([f64; 3], [Complex; 3]) {
        let source = match transform {
            CylToCar | CylToSph => Some(CarToCyl),
            SphToCar | SphToCyl => Some(CarToSph),
            PolToCar => Some(CarToPol),
            CarToCyl | CarToSph | CarToPol => None,
        };
        let planar = transform.dimension() == 2;
        let (position, value) = if planar {
            (
                [position[0], position[1], 0.0],
                [value[0], value[1], Complex::default()],
            )
        } else {
            (position, value)
        };
        match source {
            Some(source) => (
                point(position, source).unwrap(),
                vector(value, position, source).unwrap(),
            ),
            None => (position, value),
        }
    }

    /// Per-slot tolerances of coordinates in the output system of `transform` for
    /// points of norm `scale`: relative for lengths and absolute for angles.
    fn slot_tolerances(transform: Transform, scale: f64) -> [f64; 3] {
        let length = 1e-14 * scale + f64::MIN_POSITIVE;
        match transform {
            CarToCyl | SphToCyl | CarToPol => [length, 1e-14, length],
            CarToSph | CylToSph => [length, 1e-14, 1e-14],
            CylToCar | SphToCar | PolToCar => [length; 3],
        }
    }

    /// Every chart inverts, `CylToSph ∘ CarToCyl = CarToSph` and
    /// `CylToCar ∘ SphToCyl = SphToCar` exactly, and vector frames are orthogonal:
    /// they invert, preserve the norm and compose.
    #[allow(clippy::float_cmp)] // Chart compositions are exact.
    fn check_charts(cartesian: [f64; 3], value: [Complex; 3]) -> Result<(), TestCaseError> {
        let scale = cartesian.iter().map(|v| v.abs()).fold(0.0, f64::max);
        for (forward, inverse) in PAIRS {
            for (from, to) in [(forward, inverse), (inverse, forward)] {
                let (position, value) = input(cartesian, value, from);
                let image = point(position, from).unwrap();
                let back = point(image, to).unwrap();
                let tolerances = slot_tolerances(to, scale);
                for ((back, position), tolerance) in back.into_iter().zip(position).zip(tolerances)
                {
                    prop_assert_close!(back, position, tolerance, "{from:?} then {to:?}");
                }
                let rotated = vector(value, position, from).unwrap();
                let norm = re_dot(value, value);
                prop_assert_close!(re_dot(rotated, rotated), norm, 1e-14 * norm, "{from:?}");
                let restored = vector(rotated, image, to).unwrap();
                prop_assert_close!(restored, value, 1e-14 * norm.sqrt(), "{from:?} then {to:?}");
            }
        }
        let cylindrical = point(cartesian, CarToCyl).unwrap();
        let spherical = point(cartesian, CarToSph).unwrap();
        prop_assert_eq!(point(cylindrical, CylToSph).unwrap(), spherical);
        prop_assert_eq!(
            point(point(spherical, SphToCyl).unwrap(), CylToCar).unwrap(),
            point(spherical, SphToCar).unwrap()
        );
        let tolerance = 1e-14 * re_dot(value, value).sqrt();
        let via_cylinder = vector(
            vector(value, cartesian, CarToCyl).unwrap(),
            cylindrical,
            CylToSph,
        );
        prop_assert_close!(
            via_cylinder.unwrap(),
            vector(value, cartesian, CarToSph).unwrap(),
            tolerance
        );
        let components = vector(value, spherical, SphToCyl).unwrap();
        let cylinder = point(spherical, SphToCyl).unwrap();
        let via_cylinder = vector(components, cylinder, CylToCar).unwrap();
        prop_assert_close!(
            via_cylinder,
            vector(value, spherical, SphToCar).unwrap(),
            tolerance
        );
        Ok(())
    }

    /// Point pullbacks match central differences and invert with their chart, and
    /// vector pullbacks are the transposed rotation (exactly, since `vector` is linear
    /// in its components) plus a position term matching central differences.
    fn check_adjoints(
        cartesian: [f64; 3],
        value: [Complex; 3],
        cotangent: [Complex; 3],
    ) -> Result<(), TestCaseError> {
        let h = 1e-6;
        for (forward, inverse) in PAIRS {
            for (transform, other) in [(forward, inverse), (inverse, forward)] {
                let (position, value) = input(cartesian, value, transform);
                let planar = transform.dimension() == 2;
                let cotangent = if planar {
                    [cotangent[0], cotangent[1], Complex::default()]
                } else {
                    cotangent
                };
                let g = cotangent.map(|g| g.re);
                let point_g = point_pullback(position, transform, g).unwrap();
                let (vector_g, position_g) =
                    vector_pullback(value, position, transform, cotangent).unwrap();
                let rotated = vector(value, position, transform).unwrap();
                prop_assert_close!(
                    re_dot(cotangent, rotated),
                    re_dot(vector_g, value),
                    1e-14 * (1.0 + re_dot(cotangent, cotangent) + re_dot(value, value)),
                    "{transform:?}"
                );
                let image = point(position, transform).unwrap();
                let chained = point_pullback(
                    position,
                    transform,
                    point_pullback(image, other, g).unwrap(),
                );
                prop_assert_close!(
                    chained.unwrap(),
                    g,
                    1e-13 * (1.0 + g.iter().map(|v| v.abs()).sum::<f64>()),
                    "{transform:?}"
                );
                for axis in 0..transform.dimension() {
                    let step = |t: f64| {
                        let mut shifted = position;
                        shifted[axis] += t * h;
                        shifted
                    };
                    let numeric = (point(step(1.0), transform).unwrap().into_iter())
                        .zip(point(step(-1.0), transform).unwrap())
                        .zip(g)
                        .map(|((p, m), g)| g * (p - m) / (2.0 * h))
                        .sum::<f64>();
                    prop_assert_close!(
                        point_g[axis],
                        numeric,
                        2e-8 * (1.0 + numeric.abs()),
                        "{transform:?}"
                    );
                    let numeric = re_dot(
                        cotangent,
                        (0..3).map(|c| {
                            (vector(value, step(1.0), transform).unwrap()[c]
                                - vector(value, step(-1.0), transform).unwrap()[c])
                                / (2.0 * h)
                        }),
                    );
                    prop_assert_close!(
                        position_g[axis],
                        numeric,
                        2e-8 * (1.0 + numeric.abs()),
                        "{transform:?}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[allow(clippy::float_cmp)] // Axis normals and zero cotangents have exact results.
    fn coordinate_axis_derivative_contract() {
        assert!(point_pullback([0.0, 0.0, 1.0], CarToSph, [0.0, 0.0, 1.0]).is_err());
        // The polar cotangent divided by r underflows in the cylindrical composition,
        // so only the explicit axis check rejects it.
        assert!(point_pullback([0.0, 0.0, 1e300], CarToSph, [0.0, 1e-30, 0.0]).is_err());
        assert_eq!(
            point_pullback([0.0, 0.0, -1.0], CarToSph, [1.0, 0.0, 0.0]).unwrap(),
            [0.0, 0.0, -1.0]
        );
        assert_eq!(
            point_pullback([0.0; 3], CarToPol, [0.0; 3]).unwrap(),
            [0.0; 3]
        );
    }
}
