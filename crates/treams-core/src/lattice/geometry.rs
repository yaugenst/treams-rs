//! Lattice geometry: cell volumes, reciprocal and reduced bases, integer cube points,
//! diffraction orders and reduction to the first Brillouin zone.
//!
//! Upstream: `volume`, `area`, `reciprocal`, `cube`, `cubeedge` and
//! `diffr_orders_circle` of `treams.lattice`, and `firstbrillouin1d`, `firstbrillouin2d`
//! and `firstbrillouin3d` of `treams.misc`. `reduce_basis` is a treams-rs extension
//! that the Ewald sums use.
#![allow(clippy::indexing_slicing)] // Dimensions are validated before fixed-array indexing.
use crate::{Error, Result};
use nalgebra::{Matrix3, Vector3};
use std::{
    f64::consts::TAU,
    ops::{Add, Mul, Sub},
};

/// Signed determinant for a one-, two- or three-dimensional cell.
///
/// Upstream: `treams.lattice.volume` and, in two dimensions, `treams.lattice.area`.
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
///
/// Upstream: `treams.lattice.reciprocal`.
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

/// Unimodular integer transforms `[U, W]` with `W = U^{-T}`: the rows `U a` of the
/// leading `dim` rows `a` are a reduced basis of the same lattice, and `W b` are their
/// reciprocal rows when `b` are those of `a`.
///
/// A row is reduced when neither adding a multiple of another row nor, in three
/// dimensions, adding the sum or difference of the two others shortens it by more than
/// a relative `1e-8`. Only such strict decreases are applied and rows are never
/// permuted, so a reduced basis (for example a hexagonal one, whose ties do not
/// count) is kept. Returns `None` for a reduced basis, or when the reduction would
/// need coefficients beyond `2^20`, which leaves the given basis in use.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Coefficients are bounded by 2^20.
pub(crate) fn reduce_basis(rows: [[f64; 3]; 3], dim: usize) -> Option<[[[i64; 3]; 3]; 2]> {
    const LIMIT: i64 = 1 << 20;
    /// Rows `v = U a` with the transforms `U` and `W = U^{-T}`.
    struct Basis {
        v: [[f64; 3]; 3],
        u: [[i64; 3]; 3],
        w: [[i64; 3]; 3],
    }
    fn norm(v: [f64; 3]) -> f64 {
        v.iter().map(|x| x * x).sum()
    }
    impl Basis {
        /// Add `c` times row `i` to row `k` for each `(i, c)` when that shortens row `k`.
        fn shorten(&mut self, k: usize, terms: &[(usize, i64)]) -> Option<bool> {
            let mut candidate = self.v[k];
            for &(i, c) in terms {
                if c.abs() > LIMIT {
                    return None;
                }
                for (x, y) in candidate.iter_mut().zip(self.v[i]) {
                    *x += c as f64 * y;
                }
            }
            if norm(candidate) >= (1.0 - 1e-8) * norm(self.v[k]) {
                return Some(false);
            }
            // Row k of U gains c row i; row i of W = U^{-T} loses c row k.
            for &(i, c) in terms {
                for j in 0..3 {
                    self.u[k][j] += c * self.u[i][j];
                    self.w[i][j] -= c * self.w[k][j];
                }
            }
            if self
                .u
                .iter()
                .chain(&self.w)
                .flatten()
                .any(|x| x.abs() > LIMIT)
            {
                return None;
            }
            self.v[k] = candidate;
            Some(true)
        }
    }
    let identity: [[i64; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| i64::from(i == j)));
    let mut basis = Basis {
        v: std::array::from_fn(|i| {
            std::array::from_fn(|j| if i < dim && j < dim { rows[i][j] } else { 0.0 })
        }),
        u: identity,
        w: identity,
    };
    for _ in 0..64 {
        let mut changed = false;
        for k in 0..dim {
            for i in (0..dim).filter(|&i| i != k) {
                let (a, b) = (basis.v[k], basis.v[i]);
                let mu = (a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / norm(b)).round();
                if !mu.is_finite() || mu.abs() > LIMIT as f64 {
                    return None;
                }
                if mu != 0.0 {
                    changed |= basis.shorten(k, &[(i, -mu as i64)])?;
                }
            }
            if dim == 3 {
                let (i, j) = ((k + 1) % 3, (k + 2) % 3);
                for (s, t) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                    changed |= basis.shorten(k, &[(i, s), (j, t)])?;
                }
            }
        }
        if !changed {
            break;
        }
    }
    (basis.u != identity).then_some([basis.u, basis.w])
}

/// All integer cube points, or just its boundary, in lexicographic row order.
///
/// Upstream: `treams.lattice.cube` (`edge = false`) and `treams.lattice.cubeedge`
/// (`edge = true`), flattened to `dim` columns per point.
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
///
/// Upstream: `treams.lattice.diffr_orders_circle`.
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

/// Reduce a scalar wavevector to `(-b/2, b/2]`, for positive reciprocal pitch `b`.
///
/// Upstream: `treams.misc.firstbrillouin1d`.
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

/// Iterative nearest-cell reduction of a 2D or 3D wavevector.
///
/// Each of up to `iterations + 1` passes rounds `k` against every row and then takes
/// the shortest of `k` and its translates to the `3^dim - 1` neighbouring cells, as
/// upstream does. A pass that leaves `k` unchanged ends the reduction.
///
/// Upstream: `treams.misc.firstbrillouin2d` and `treams.misc.firstbrillouin3d`.
///
/// # Errors
///
/// Fails with "lattice vectors are not of minimal length" for a 2D basis whose rows the
/// sum or difference of the two shortens beyond rounding, and for an invalid basis,
/// dimension or wavevector.
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
    // Relative to the rows, so rounding of hexagonal ties at any length passes. The
    // floor 1e-14 is the absolute tolerance of upstream `firstbrillouin2d`, so every
    // basis that upstream accepts passes too.
    let tolerance = (1e-13 * norms[0].max(norms[1])).max(1e-14);
    if dim == 2
        && [
            (rows[0] + rows[1]).norm_squared(),
            (rows[0] - rows[1]).norm_squared(),
        ]
        .iter()
        .any(|&v| v < norms[0] - tolerance || v < norms[1] - tolerance)
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
    use std::{f64::consts::TAU, result::Result};

    use proptest::{prelude::*, test_runner::TestCaseError};

    use super::{
        cube, diffraction_orders, first_brillouin, first_brillouin_1d, reciprocal, reduce_basis,
        volume,
    };
    use crate::test_support::{ALGEBRA_CASES, log_uniform, prop_assert_close, table};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

        #[test]
        fn reciprocal_duality(
            a in 0.3f64..3.0,
            b in 0.3f64..3.0,
            c in 0.3f64..3.0,
            s in -1.0f64..1.0,
            dim in 1usize..4,
        ) {
            check_reciprocal_duality([[a, s, 0.0], [0.0, b, s], [0.0, 0.0, c]], dim)?;
        }

        #[test]
        fn boundary_is_cube_difference(dim in 1usize..4, n in 1i64..8) {
            check_cube_boundary(dim, n)?;
        }

        #[test]
        fn diffraction_orders_are_complete_paired_and_unique(
            basis in reciprocal_basis(-3.0..3.0, 0.5..2.0),
            radius in 0.0f64..4.0,
        ) {
            check_diffraction_orders(basis, radius)?;
        }

        #[test]
        fn brillouin_reduction_is_a_covariant_voronoi_projection(
            basis in reciprocal_basis(-0.5..0.5, 1.0..2.0),
            hexagonal in any::<bool>(),
            k in prop::array::uniform2(-5.0f64..5.0),
            cells in prop::array::uniform2(-6i32..=6),
        ) {
            check_brillouin_2d(basis, hexagonal, k, cells)?;
        }

        #[test]
        fn brillouin_reduction_in_three_dimensions_is_periodic(
            skew in prop::array::uniform3(-0.2f64..0.2),
            k in prop::array::uniform3(-5.0f64..5.0),
            cells in prop::array::uniform3(-6i32..=6),
        ) {
            check_brillouin_3d(skew, k, cells)?;
        }

        #[test]
        fn brillouin_reduction_in_one_dimension_is_periodic(
            k in -100.0f64..100.0,
            b in log_uniform(-3.0..8.0),
            cells in -20i32..=20,
        ) {
            check_brillouin_1d(k, b, cells)?;
        }

        #[test]
        fn basis_reduction_finds_the_successive_minima(
            dim in 2usize..4,
            pitch in prop::array::uniform3(1.0f64..2.0),
            skew in prop::array::uniform3(-1.0f64..1.0),
            shear in prop::array::uniform3(-9i64..=9),
        ) {
            check_basis_reduction(dim, pitch, skew, shear)?;
        }
    }

    /// The boundary of a cube of half-width `n` is the set of its points with a
    /// coordinate of modulus `n`, in the same order.
    fn check_cube_boundary(dim: usize, n: i64) -> Result<(), TestCaseError> {
        let full = cube(dim, n, false)?;
        let boundary = cube(dim, n, true)?;
        let expected: Vec<_> = full
            .chunks_exact(dim)
            .filter(|point| point.iter().any(|x| x.abs() == n))
            .flatten()
            .copied()
            .collect();
        prop_assert_eq!(boundary, expected);
        Ok(())
    }

    /// Two-dimensional reciprocal rows `s R(theta) [[1, 0], [shear, aspect]]`, optionally
    /// reflected, at scales `s` over eleven decades, with the scale.
    fn reciprocal_basis(
        shear: std::ops::Range<f64>,
        aspect: std::ops::Range<f64>,
    ) -> impl Strategy<Value = ([[f64; 2]; 2], f64)> {
        (
            shear,
            aspect,
            0.0..TAU,
            any::<bool>(),
            log_uniform(-3.0..8.0),
        )
            .prop_map(|(shear, aspect, theta, reflect, scale)| {
                let (sin, cos) = theta.sin_cos();
                let flip = if reflect { -1.0 } else { 1.0 };
                let row = |x: f64, y: f64| {
                    [
                        scale * (cos * x - sin * y),
                        scale * flip * (sin * x + cos * y),
                    ]
                };
                ([row(1.0, 0.0), row(shear, aspect)], scale)
            })
    }

    /// Reciprocal rows are dual to the direct rows, the reciprocal of the reciprocal
    /// rows is the direct lattice, and the cell volumes multiply to `(2 pi)^dim`.
    fn check_reciprocal_duality(direct: [[f64; 3]; 3], dim: usize) -> Result<(), TestCaseError> {
        let reciprocal = reciprocal(direct, dim)?;
        for (i, row) in direct.iter().enumerate().take(dim) {
            for (j, dual) in reciprocal.iter().enumerate().take(dim) {
                let dot: f64 = row.iter().zip(dual).take(dim).map(|(a, b)| a * b).sum();
                let expected = if i == j { TAU } else { 0.0 };
                prop_assert_close!(dot, expected, 3e-14);
            }
        }
        let involution = super::reciprocal(reciprocal, dim)?;
        for (row, expected) in involution.iter().zip(direct).take(dim) {
            for (x, e) in row.iter().zip(expected).take(dim) {
                prop_assert_close!(*x, e, 1e-14, "{involution:?} {direct:?}");
            }
        }
        let product = volume(direct, dim)? * volume(reciprocal, dim)?;
        let expected = TAU.powi(i32::try_from(dim).unwrap());
        prop_assert_close!(product, expected, 1e-14 * expected);
        Ok(())
    }

    /// Diffraction orders start at `(0, 0)`, list every other order next to its
    /// opposite, contain no order twice and are exactly the orders inside the circle,
    /// for oblique, reflected and scaled bases.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Small order bounds.
    fn check_diffraction_orders(
        (b, scale): ([[f64; 2]; 2], f64),
        radius: f64,
    ) -> Result<(), TestCaseError> {
        let radius = radius * scale;
        let orders = diffraction_orders(b, radius)?;
        prop_assert_eq!(&orders[..2], &[0, 0]);
        let pairs: Vec<_> = orders.chunks_exact(2).map(|p| (p[0], p[1])).collect();
        for pair in pairs[1..].chunks_exact(2) {
            prop_assert_eq!(pair[1], (-pair[0].0, -pair[0].1));
        }
        let set: std::collections::BTreeSet<_> = pairs.iter().copied().collect();
        prop_assert_eq!(set.len(), pairs.len(), "duplicate orders");
        let det = (b[0][0] * b[1][1] - b[0][1] * b[1][0]).abs();
        let bound = |row: [f64; 2]| (radius * row[0].hypot(row[1]) / det).ceil() as i64 + 1;
        let magnitude = |m: i64, n: i64| {
            let (m, n) = (m as f64, n as f64);
            (m * b[0][0] + n * b[1][0]).hypot(m * b[0][1] + n * b[1][1])
        };
        let (mmax, nmax) = (bound(b[1]), bound(b[0]));
        let expected: std::collections::BTreeSet<_> = (-mmax..=mmax)
            .flat_map(|m| (-nmax..=nmax).map(move |n| (m, n)))
            .filter(|&(m, n)| magnitude(m, n) <= radius)
            .collect();
        prop_assert_eq!(set, expected);
        Ok(())
    }

    /// Two reductions of the same class `k + G` agree up to ties: they are equal to
    /// `1e-12 scale`, or differ by a lattice vector while having equal lengths.
    fn same_reduction(
        a: [f64; 3],
        b: [f64; 3],
        dual: &[[f64; 3]; 3],
        scale: f64,
    ) -> Result<(), TestCaseError> {
        let d: [f64; 3] = std::array::from_fn(|j| a[j] - b[j]);
        let norm = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>();
        if norm(d).sqrt() <= 1e-12 * scale {
            return Ok(());
        }
        for row in dual {
            let cells = row.iter().zip(d).map(|(x, y)| x * y).sum::<f64>() / TAU;
            prop_assert!(
                (cells - cells.round()).abs() < 1e-9,
                "{a:?} and {b:?} differ"
            );
        }
        prop_assert!(
            (norm(a) - norm(b)).abs() <= 1e-10 * scale * scale,
            "{a:?} {b:?}"
        );
        Ok(())
    }

    /// Two-dimensional Brillouin reduction accepts reduced bases at every scale,
    /// including exact hexagonal ties, and returns the shortest vector of `k + G`
    /// (Voronoi minimality against the eight neighbouring cells), independent of the
    /// representative `k`, idempotent and covariant under scaling.
    fn check_brillouin_2d(
        (b, scale): ([[f64; 2]; 2], f64),
        hexagonal: bool,
        k: [f64; 2],
        cells: [i32; 2],
    ) -> Result<(), TestCaseError> {
        let b = if hexagonal {
            // Rotate the hexagonal basis [[1, 0], [-1/2, sqrt 3 / 2]] like `b`.
            let [x, y] = [b[0][0], b[0][1]];
            let s3 = 3.0_f64.sqrt() / 2.0;
            let flip = (b[0][0] * b[1][1] - b[0][1] * b[1][0]).signum();
            [[x, y], [-0.5 * x - flip * s3 * y, -0.5 * y + flip * s3 * x]]
        } else {
            b
        };
        let rows = [[b[0][0], b[0][1], 0.0], [b[1][0], b[1][1], 0.0], [0.0; 3]];
        let dual = reciprocal(rows, 2)?;
        let k = [k[0] * scale, k[1] * scale, 0.0];
        let result = first_brillouin(k, rows, 2, 10)?;
        let norm = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>();
        for m in -1..=1_i32 {
            for n in -1..=1_i32 {
                let (m, n) = (f64::from(m), f64::from(n));
                let neighbour: [f64; 3] =
                    std::array::from_fn(|j| result[j] + m * rows[0][j] + n * rows[1][j]);
                prop_assert!(
                    norm(result) <= norm(neighbour) * (1.0 + 1e-12),
                    "{result:?} is longer than {neighbour:?}"
                );
            }
        }
        let shifted: [f64; 3] = std::array::from_fn(|j| {
            k[j] + f64::from(cells[0]) * rows[0][j] + f64::from(cells[1]) * rows[1][j]
        });
        let size = scale * (1.0 + norm(k).sqrt() / scale);
        same_reduction(first_brillouin(shifted, rows, 2, 10)?, result, &dual, size)?;
        same_reduction(first_brillouin(result, rows, 2, 10)?, result, &dual, size)?;
        let unit = first_brillouin(
            k.map(|x| x / scale),
            rows.map(|r| r.map(|x| x / scale)),
            2,
            10,
        )?;
        same_reduction(unit.map(|x| x * scale), result, &dual, size)?;
        Ok(())
    }

    /// Three-dimensional Brillouin reduction of nearly orthogonal bases is
    /// independent of the representative `k` and idempotent.
    fn check_brillouin_3d(
        skew: [f64; 3],
        k: [f64; 3],
        cells: [i32; 3],
    ) -> Result<(), TestCaseError> {
        let rows = [
            [1.0, skew[0], skew[1]],
            [skew[2], 1.3, skew[0]],
            [skew[1], skew[2], 1.6],
        ];
        let dual = reciprocal(rows, 3)?;
        let result = first_brillouin(k, rows, 3, 10)?;
        let shifted: [f64; 3] = std::array::from_fn(|j| {
            k[j] + (0..3)
                .map(|i| f64::from(cells[i]) * rows[i][j])
                .sum::<f64>()
        });
        same_reduction(first_brillouin(shifted, rows, 3, 10)?, result, &dual, 10.0)?;
        same_reduction(first_brillouin(result, rows, 3, 10)?, result, &dual, 10.0)?;
        Ok(())
    }

    /// One-dimensional reduction lands in `(-b/2, b/2]` and is periodic in `k`.
    fn check_brillouin_1d(k: f64, b: f64, cells: i32) -> Result<(), TestCaseError> {
        let k = k * b;
        let result = first_brillouin_1d(k, b)?;
        prop_assert!(
            -0.5 * b < result && result <= 0.5 * b,
            "{result} for b = {b}"
        );
        let shifted = first_brillouin_1d(k + f64::from(cells) * b, b)?;
        // Equivalent representatives may lie on opposite sides of a rounded boundary.
        let difference = (result - shifted).abs();
        prop_assert!(
            difference < 1e-12 * b || (difference - b).abs() < 1e-12 * b,
            "{result} {shifted}"
        );
        Ok(())
    }

    #[test]
    fn hexagonal_brillouin_zones_are_accepted_at_every_scale() {
        // An absolute tolerance on squared lengths, as in upstream `firstbrillouin2d`,
        // rejects some of these bases.
        let s = 3.0_f64.sqrt();
        for pitch in [0.3, 0.5, 0.8, 1.0, 20.0, 1e-7, 1e9] {
            let rows = [
                [pitch, 0.0, 0.0],
                [0.5 * pitch, 0.5 * s * pitch, 0.0],
                [0.0; 3],
            ];
            let b = reciprocal(rows, 2).unwrap();
            let k = [0.1 / pitch, 0.2 / pitch, 0.0];
            assert!(first_brillouin(k, b, 2, 2).is_ok(), "pitch {pitch}");
        }
        let skewed = [[1.0, 0.0, 0.0], [0.9, 0.3, 0.0], [0.0; 3]];
        assert!(first_brillouin([0.1, 0.2, 0.0], skewed, 2, 2).is_err());
    }

    /// A basis and a sheared copy of it reduce, by unimodular transforms with exact
    /// integer inverses, to bases whose rows no sum or difference of rows shortens.
    /// Such bases are Minkowski reduced for `dim <= 3`, so their sorted row lengths are
    /// the successive minima of the lattice and agree; reduced bases are kept.
    fn check_basis_reduction(
        dim: usize,
        pitch: [f64; 3],
        skew: [f64; 3],
        shear: [i64; 3],
    ) -> Result<(), TestCaseError> {
        let a = [
            [pitch[0], 0.0, 0.0],
            [skew[0], pitch[1], 0.0],
            [skew[1], skew[2], pitch[2]],
        ];
        let identity: [[i64; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| i64::from(i == j)));
        let shear = [[1, 0, 0], [shear[0], 1, 0], [shear[1], shear[2], 1]];
        let combine = |u: &[[i64; 3]; 3], rows: &[[f64; 3]; 3]| -> [[f64; 3]; 3] {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| {
                    if i < dim && j < dim {
                        (0..dim)
                            .map(|k| f64::from(i32::try_from(u[i][k]).unwrap()) * rows[k][j])
                            .sum()
                    } else {
                        0.0
                    }
                })
            })
        };
        let norm = |v: &[f64; 3]| v.iter().map(|x| x * x).sum::<f64>();
        let mut minima = Vec::new();
        for rows in [combine(&identity, &a), combine(&shear, &a)] {
            let [u, w] = reduce_basis(rows, dim).unwrap_or([identity; 2]);
            for (i, row) in u.iter().enumerate() {
                for (j, dual) in w.iter().enumerate() {
                    let product: i64 = row.iter().zip(dual).map(|(a, b)| a * b).sum();
                    prop_assert_eq!(product, i64::from(i == j), "U W^T = I");
                }
            }
            let v = combine(&u, &rows);
            for k in 0..dim {
                for c in [
                    [1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1],
                ] {
                    let (i, j) = ((k + 1) % dim, (k + 2) % dim);
                    if dim == 2 && c[1] != 0 {
                        continue;
                    }
                    let other: [f64; 3] = std::array::from_fn(|x| {
                        v[k][x] + f64::from(c[0]) * v[i][x] + f64::from(c[1]) * v[j][x]
                    });
                    prop_assert!(
                        norm(&v[k]) <= (1.0 + 1e-7) * norm(&other),
                        "row {k} of {v:?} shortened by {c:?}"
                    );
                }
            }
            prop_assert!(reduce_basis(v, dim).is_none(), "{v:?} is reduced");
            let mut lengths: Vec<f64> = v[..dim].iter().map(norm).collect();
            lengths.sort_by(f64::total_cmp);
            minima.push(lengths);
        }
        for (a, b) in minima[0].iter().zip(&minima[1]) {
            prop_assert!((a - b).abs() <= 1e-7 * a, "{minima:?}");
        }
        Ok(())
    }

    #[test]
    fn reduced_bases_are_kept() {
        let s = 3.0_f64.sqrt();
        for (rows, dim) in [
            ([[1.0, 0.0, 0.0], [0.5, 0.5 * s, 0.0], [0.0; 3]], 2),
            ([[1.0, 0.0, 0.0], [-0.5, 0.5 * s, 0.0], [0.0; 3]], 2),
            // A shorter second row is not permuted first.
            ([[1.6, 0.0, 0.0], [0.3, 1.4, 0.0], [0.0; 3]], 2),
            (
                [
                    [1.0, 0.0, 0.0],
                    [0.5, 0.5 * s, 0.0],
                    [0.5, 0.5 / s, (2.0_f64 / 3.0).sqrt()],
                ],
                3,
            ),
            ([[2.0, 0.0, 0.0], [0.0; 3], [0.0; 3]], 1),
        ] {
            assert!(reduce_basis(rows, dim).is_none(), "{rows:?}");
        }
        let sheared = [[1.6, 0.0, 0.0], [13.1, 1.4, 0.0], [0.0; 3]];
        assert_eq!(
            reduce_basis(sheared, 2),
            Some([
                [[1, 0, 0], [-8, 1, 0], [0, 0, 1]],
                [[1, 8, 0], [0, 1, 0], [0, 0, 1]]
            ])
        );
    }

    #[test]
    fn diffraction_cutoff_on_an_order_keeps_it() -> crate::Result<()> {
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
    fn cube_matches_lean_model() {
        // `just formal` keeps this file equal to `Treams.Shells.cube`.
        let cases = table::<String, i64>(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../formal/golden/cube.txt"
        )));
        // The cases of `formal/Golden.lean`, in its order: a duplicated, missing or
        // reordered line fails even when the line count is unchanged.
        let grid: Vec<_> = (1..=3)
            .flat_map(|dim| (0..=4).flat_map(move |n| [true, false].map(|edge| (dim, n, edge))))
            .collect();
        assert_eq!(cases.len(), grid.len());
        for ((key, expected), (dim, n, edge)) in cases.iter().zip(grid) {
            assert_eq!(key, &[dim.to_string(), n.to_string(), edge.to_string()]);
            assert_eq!(&cube(dim, n, edge).unwrap(), expected, "{key:?}");
        }
    }
}
