//! Small lattice geometry and integer mode enumeration, without Python allocation loops.
#![allow(clippy::indexing_slicing)] // Dimensions are validated before fixed-array indexing.
use crate::{Error, Result};
use nalgebra::{Matrix3, Vector3};
use std::{
    f64::consts::TAU,
    ops::{Add, Mul, Sub},
};

/// Signed determinant for a one-, two- or three-dimensional cell.
#[inline]
pub fn volume<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>>(
    a: [[T; 3]; 3],
    dim: usize,
) -> Result<T> {
    match dim {
        1 => Ok(a[0][0]),
        2 => Ok(a[0][0] * a[1][1] - a[0][1] * a[1][0]),
        3 => Ok(a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            + a[0][1] * (a[1][2] * a[2][0] - a[1][0] * a[2][2])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])),
        _ => Err(Error::InvalidInput(
            "cell dimension must be 1, 2 or 3".into(),
        )),
    }
}

/// Reciprocal row vectors, padded by unused identity axes for dimensions below three.
pub fn reciprocal(a: [[f64; 3]; 3], dim: usize) -> Result<[[f64; 3]; 3]> {
    if !(1..=3).contains(&dim)
        || a[..dim]
            .iter()
            .any(|r| r[..dim].iter().any(|v| !v.is_finite()))
    {
        return Err(Error::InvalidInput(
            "require finite lattice vectors in dimension 1, 2 or 3".into(),
        ));
    }
    let matrix = Matrix3::from_fn(|i, j| {
        if i < dim && j < dim {
            a[i][j]
        } else {
            f64::from(i == j)
        }
    });
    let inverse = matrix
        .try_inverse()
        .ok_or_else(|| Error::InvalidInput("lattice vectors must be independent".into()))?;
    let result = inverse.transpose() * TAU;
    if result.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "lattice is numerically singular".into(),
        ));
    }
    Ok(std::array::from_fn(|i| {
        std::array::from_fn(|j| result[(i, j)])
    }))
}

/// All integer cube points, or just its boundary, in lexicographic row order.
pub fn cube(dim: usize, n: i64, edge: bool) -> Result<Vec<i64>> {
    if !(1..=3).contains(&dim) || n < 0 {
        return Err(Error::InvalidInput(
            "cube requires dimension 1, 2 or 3 and nonnegative size".into(),
        ));
    }
    let count = usize::try_from(n)
        .ok()
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(1))
        .and_then(|side| {
            side.checked_pow(u32::try_from(dim).ok()?).map(|outer| {
                if edge && side > 1 {
                    outer - (side - 2).pow(u32::try_from(dim).unwrap_or_default())
                } else {
                    outer
                }
            })
        })
        .and_then(|rows| rows.checked_mul(dim))
        .ok_or_else(|| Error::InvalidInput("cube output size overflows".into()))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|e| Error::InvalidInput(e.to_string()))?;
    visit_cube(dim, n, edge, |point| {
        values.extend_from_slice(&point[..dim]);
        Ok(())
    })?;
    Ok(values)
}

/// Visit cube points without allocating an integer point table.
pub(crate) fn visit_cube(
    dim: usize,
    n: i64,
    edge: bool,
    mut visitor: impl FnMut([i64; 3]) -> Result<()>,
) -> Result<()> {
    if !(1..=3).contains(&dim) || n < 0 {
        return Err(Error::InvalidInput(
            "cube requires dimension 1, 2 or 3 and nonnegative size".into(),
        ));
    }
    fn append(
        point: &mut [i64; 3],
        dim: usize,
        axis: usize,
        n: i64,
        edge: bool,
        boundary: bool,
        visitor: &mut impl FnMut([i64; 3]) -> Result<()>,
    ) -> Result<()> {
        if axis == dim {
            return visitor(*point);
        }
        if edge && !boundary && axis + 1 == dim && n > 0 {
            for value in [-n, n] {
                point[axis] = value;
                append(point, dim, axis + 1, n, edge, true, visitor)?;
            }
        } else {
            for value in -n..=n {
                point[axis] = value;
                append(
                    point,
                    dim,
                    axis + 1,
                    n,
                    edge,
                    boundary || value.abs() == n,
                    visitor,
                )?;
            }
        }
        Ok(())
    }
    append(&mut [0; 3], dim, 0, n, edge, false, &mut visitor)
}

/// Diffraction orders inside a reciprocal-space circle, with adjacent opposite pairs.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Integer bounds are checked before conversion.
pub fn diffraction_orders(b: [[f64; 2]; 2], radius: f64) -> Result<Vec<i64>> {
    if !radius.is_finite() || b.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "diffraction geometry must be finite".into(),
        ));
    }
    let det = b[0][0] * b[1][1] - b[0][1] * b[1][0];
    let length = b[1][0].hypot(b[1][1]);
    if det == 0.0 || !det.is_finite() || !length.is_finite() {
        return Err(Error::InvalidInput(
            "lattice vectors must be independent".into(),
        ));
    }
    if radius < 0.0 {
        return Ok(Vec::new());
    }
    let bound = (radius * length / det.abs()).ceil();
    let nbound = radius * b[0][0].hypot(b[0][1]) / det.abs();
    if !bound.is_finite()
        || !nbound.is_finite()
        || bound >= i64::MAX as f64 - 4.0
        || nbound >= i64::MAX as f64 - 4.0
    {
        return Err(Error::InvalidInput(
            "diffraction order bounds overflow".into(),
        ));
    }
    let mut orders = vec![0, 0];
    for m in 0..=bound as i64 {
        // Rounding can place a row holding a boundary order just beyond the radius;
        // the padded window keeps it and the hypot test below decides.
        let distance = (m as f64 * det / length).abs();
        let center =
            -(m as f64) * (b[0][0] * (b[1][0] / length) + b[0][1] * (b[1][1] / length)) / length;
        let half = ((radius - distance) * (radius + distance)).max(0.0).sqrt() / length;
        let lower = (center - half).floor() as i64 - 1;
        let upper = (center + half).ceil() as i64 + 1;
        let mut emit = |n: i64| {
            if m == 0 && n <= 0 {
                return;
            }
            let x = m as f64 * b[0][0] + n as f64 * b[1][0];
            let y = m as f64 * b[0][1] + n as f64 * b[1][1];
            if x.hypot(y) <= radius {
                orders.extend([m, n, -m, -n]);
            }
        };
        for n in lower.max(0)..=upper {
            emit(n);
        }
        for n in (lower..=upper.min(-1)).rev() {
            emit(n);
        }
    }
    Ok(orders)
}

/// Reduce a scalar wavevector to (-b/2,b/2], for positive reciprocal pitch b.
pub fn first_brillouin_1d(k: f64, b: f64) -> Result<f64> {
    if !k.is_finite() || !b.is_finite() || b <= 0.0 {
        return Err(Error::InvalidInput(
            "require finite wavevector and positive reciprocal pitch".into(),
        ));
    }
    let mut result = k - b * (k / b).round_ties_even();
    if result > 0.5 * b {
        result -= b;
    }
    if result <= -0.5 * b {
        result += b;
    }
    if !result.is_finite() {
        return Err(Error::InvalidInput("wavevector reduction overflow".into()));
    }
    Ok(result)
}

/// Iterative nearest-cell reduction, preserving the public treams iteration convention.
#[allow(clippy::float_cmp)] // Exact fixed-point termination of the discrete reduction.
pub fn first_brillouin(
    mut k: [f64; 3],
    b: [[f64; 3]; 3],
    dim: usize,
    iterations: usize,
) -> Result<[f64; 3]> {
    reciprocal(b, dim)?;
    if !(2..=3).contains(&dim) || k[..dim].iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(
            "require finite two- or three-dimensional wavevector".into(),
        ));
    }
    k[dim..].fill(0.0);
    let rows = std::array::from_fn::<_, 3, _>(|i| {
        Vector3::from_fn(|j, _| if i < dim && j < dim { b[i][j] } else { 0.0 })
    });
    let norms = rows.map(|v| v.norm_squared());
    if dim == 2
        && [
            (rows[0] + rows[1]).norm_squared(),
            (rows[0] - rows[1]).norm_squared(),
        ]
        .iter()
        .any(|&v| v < norms[0] - 1e-14 || v < norms[1] - 1e-14)
    {
        return Err(Error::InvalidInput(
            "lattice vectors are not of minimal length".into(),
        ));
    }
    let neighbours = cube(dim, 1, false)?;
    for _ in 0..=iterations {
        let previous = k;
        let mut point = Vector3::from(k);
        for j in 0..dim {
            point -= rows[j] * (point.dot(&rows[j]) / norms[j]).round_ties_even();
        }
        let mut best = point;
        for index in neighbours.chunks_exact(dim) {
            let mut candidate = point;
            for j in 0..dim {
                candidate += rows[j] * f64::from(i32::try_from(index[j]).unwrap_or_default());
            }
            if candidate.norm_squared() < best.norm_squared() {
                best = candidate;
            }
        }
        k = best.into();
        if k == previous {
            break;
        }
    }
    Ok(k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn reciprocal_duality(a in 0.3f64..3.0,b in 0.3f64..3.0,c in 0.3f64..3.0,s in -1.0f64..1.0,dim in 1usize..4) {
            let direct=[[a,s,0.0],[0.0,b,s],[0.0,0.0,c]];
            let reciprocal=reciprocal(direct,dim)?;
            for (i,row) in direct.iter().enumerate().take(dim) {for(j,dual)in reciprocal.iter().enumerate().take(dim) {
                let dot:f64=row.iter().zip(dual).take(dim).map(|(a,b)|a*b).sum();
                let expected=if i==j {TAU}else{0.0};prop_assert!((dot-expected).abs()<3e-14);
            }}
        }
        #[test]
        fn boundary_is_cube_difference(dim in 1usize..4,n in 1i64..8) {
            let full=cube(dim,n,false)?;let boundary=cube(dim,n,true)?;
            let expected:Vec<_>=full.chunks_exact(dim).filter(|r|r.iter().any(|x|x.abs()==n)).flatten().copied().collect();
            prop_assert_eq!(boundary,expected);
        }
        #[test]
        fn skew_diffraction_is_complete(skew in -3.0f64..3.0,radius in 0.0f64..4.0) {
            let b=[[1.0,skew],[0.0,1.0]];let orders=diffraction_orders(b,radius)?;
            let set:std::collections::BTreeSet<_>=orders.chunks_exact(2).map(|p|(p[0],p[1])).collect();
            let expected:std::collections::BTreeSet<_>=(-4..=4).flat_map(|m|(-16..=16).map(move|n|(m,n))).filter(|&(m,n)|f64::from(m).hypot(f64::from(m)*skew+f64::from(n))<=radius).map(|(m,n)|(i64::from(m),i64::from(n))).collect();
            prop_assert_eq!(set,expected);
        }
    }

    #[test]
    #[allow(clippy::indexing_slicing)]
    fn diffraction_cutoff_on_an_order_keeps_it() -> Result<()> {
        // A cutoff equal to an order's magnitude must keep that order, although the
        // row distance of the same order can round above the cutoff.
        let mut failures = Vec::new();
        for a in [
            0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0, 1.3, 1.5, 2.0,
        ] {
            let s = 3.0_f64.sqrt();
            for b in [
                [[TAU / a, 0.0], [0.0, TAU / a]],
                [[TAU / a, 0.0], [0.0, TAU / (1.7 * a)]],
                [[TAU / a, -TAU / (a * s)], [0.0, 2.0 * TAU / (a * s)]],
            ] {
                let magnitude = |m: i32, n: i32| {
                    let (m, n) = (f64::from(m), f64::from(n));
                    (m * b[0][0] + n * b[1][0]).hypot(m * b[0][1] + n * b[1][1])
                };
                for (m, n) in [(1, 0), (0, 1), (1, 1), (2, 0), (1, -1), (2, 1)] {
                    let radius = magnitude(m, n);
                    let orders = diffraction_orders(b, radius)?;
                    let set: std::collections::BTreeSet<_> =
                        orders.chunks_exact(2).map(|p| (p[0], p[1])).collect();
                    let expected: std::collections::BTreeSet<_> = (-8..=8)
                        .flat_map(|m| (-8..=8).map(move |n| (m, n)))
                        .filter(|&(m, n)| magnitude(m, n) <= radius)
                        .map(|(m, n)| (i64::from(m), i64::from(n)))
                        .collect();
                    if set != expected {
                        failures.push((a, b, (m, n)));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "orders dropped at the cutoff: {failures:?}"
        );
        Ok(())
    }

    #[test]
    #[allow(clippy::unwrap_used)]
    fn cube_matches_lean_model() {
        // `just formal` keeps this file equal to `Treams.Shells.cube`.
        let golden = include_str!("../../../formal/golden/cube.txt");
        for line in golden.lines() {
            let (case, points) = line.split_once(':').unwrap();
            let mut case = case.split_whitespace();
            let dim = case.next().unwrap().parse().unwrap();
            let n = case.next().unwrap().parse().unwrap();
            let edge = case.next() == Some("true");
            let expected: Vec<i64> = points
                .split_whitespace()
                .flat_map(|point| point.split(','))
                .map(|x| x.parse().unwrap())
                .collect();
            assert_eq!(cube(dim, n, edge).unwrap(), expected, "{line}");
        }
        assert_eq!(golden.lines().count(), 30);
    }
}
