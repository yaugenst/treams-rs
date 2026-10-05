//! Dense linear-algebra tangents and adjoints.

use nalgebra::{DMatrix, DVector};
use proptest::{prelude::*, test_runner::TestCaseError};

use crate::{
    Complex, linalg,
    test_support::{ALGEBRA_CASES, central, complex_matrix, prop_assert_close, re_dot},
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

    #[test]
    fn linear_solve_pushforward_satisfies_implicit_equation_and_adjoint(
        (a, b, da, db, g) in (1_usize..=6, 1_usize..=4).prop_flat_map(|(n, columns)| (
            separated(n),
            complex_matrix(n, columns, 1.0),
            complex_matrix(n, n, 0.5),
            complex_matrix(n, columns, 0.5),
            complex_matrix(n, columns, 0.5),
        )),
    ) {
        check_linear_solve_pushforward(&a, &b, &da, &db, &g)?;
    }

    #[test]
    fn singular_values_frobenius_gradient(
        (a, direction, weights) in (1_usize..=5, 1_usize..=5).prop_flat_map(|(m, n)| (
            complex_matrix(m, n, 1.0),
            complex_matrix(m, n, 1.0),
            prop::collection::vec(-1.0_f64..1.0, m.min(n)),
        )),
        scale in 0.5_f64..2.0,
    ) {
        check_singular_values(&a, scale, &direction, &weights)?;
    }

    #[test]
    fn general_eigensystem_scale_and_shift_adjoint(
        (a, direction, gw, gv) in (2_usize..=6).prop_flat_map(|n| (
            separated(n),
            complex_matrix(n, n, 1.0),
            complex_matrix(n, 1, 0.5),
            complex_matrix(n, n, 0.5),
        )),
    ) {
        check_general_eigensystem(&a, &direction, gw.as_slice(), &gv)?;
    }
}

/// Eigenvalues are smooth at a tied eigenvector phase pivot, although the
/// phase-fixed eigenvectors are not. Their trace and adjoint stay well-defined.
#[test]
fn eigenvalue_derivatives_ignore_eigenvector_phase_ties() {
    let matrix = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 2.0]).map(Complex::from);
    let direction = DMatrix::from_row_slice(2, 2, &[0.2, 0.3, -0.1, 0.5]).map(Complex::from);
    let residual = linalg::eigvals(&matrix).unwrap();
    let tangent = residual.pushforward_values(&direction).unwrap();
    assert!(residual.pushforward(&direction).is_err());
    for (&value, &change) in residual.values().iter().zip(&tangent) {
        let expected = if value.re < 2.0 { 0.25 } else { 0.45 };
        assert!((change - Complex::from(expected)).norm() < 1e-14);
    }
    assert!((tangent.iter().sum::<Complex>() - direction.trace()).norm() < 1e-14);
    let weights = [Complex::new(0.2, 0.1), Complex::new(-0.3, 0.2)];
    let gradient = residual.pullback(&weights, DMatrix::zeros(2, 2)).unwrap();
    assert!((re_dot(&gradient, &direction) - re_dot(weights, tangent)).abs() < 1e-14);
}

/// The tangent satisfies the differentiated equation `A dX + dA X = dB`, agrees
/// with a central difference, and transposes to the recorded adjoint.
fn check_linear_solve_pushforward(
    a: &DMatrix<Complex>,
    b: &DMatrix<Complex>,
    da: &DMatrix<Complex>,
    db: &DMatrix<Complex>,
    g: &DMatrix<Complex>,
) -> Result<(), TestCaseError> {
    let residual = linalg::solve(a, b.clone()).unwrap();
    let value = residual.value().clone();
    let tangent = residual.pushforward(da, db.clone()).unwrap();
    prop_assert_close!(a * &tangent + da * value, db, 1e-12);
    let gradient = residual.pullback(g.clone()).unwrap();
    let pairing = re_dot(&gradient.operator, da) + re_dot(&gradient.rhs, db);
    prop_assert_close!(re_dot(g, &tangent), pairing, 1e-12);
    let loss = |t: f64| {
        let moved = linalg::solve(&(a + da * Complex::from(t)), b + db * Complex::from(t)).unwrap();
        re_dot(g, moved.value())
    };
    prop_assert_close!(central(1e-6, loss), pairing, 1e-7 * (1.0 + pairing.abs()));
    Ok(())
}

/// `n × n` matrices with diagonal `(1 + 0.7 j) + 0.1 i j` and off-diagonal entries
/// below `0.2 / n` in modulus: Gershgorin discs of radius `0.2` around centres `0.7`
/// apart separate the eigenvalues, and each eigenvector has a unique largest component.
fn separated(n: usize) -> impl Strategy<Value = DMatrix<Complex>> {
    let size = f64::from(u32::try_from(n).unwrap());
    complex_matrix(n, n, 0.2 / size / std::f64::consts::SQRT_2).prop_map(move |mut a| {
        for (j, position) in (0..n).zip(0_u32..) {
            let j_f = f64::from(position);
            a[(j, j)] = Complex::new(1.0 + 0.7 * j_f, 0.1 * j_f);
        }
        a
    })
}

/// Singular values are sorted, `Σ σ² = ‖A‖²_F` with gradient `2A` through the
/// pullback, and they are homogeneous. Where they are simple and positive, the
/// pullback of random weights matches a central difference along a random direction.
fn check_singular_values(
    a: &DMatrix<Complex>,
    scale: f64,
    direction: &DMatrix<Complex>,
    weights: &[f64],
) -> Result<(), TestCaseError> {
    let residual = linalg::svdvals(a).unwrap();
    let values = residual.values().to_vec();
    prop_assert!(values.windows(2).all(|pair| pair[0] >= pair[1]));
    let norm: f64 = values.iter().map(|s| s * s).sum();
    prop_assert_close!(norm, a.norm_squared(), 1e-12);
    let doubled: Vec<_> = values.iter().map(|s| 2.0 * s).collect();
    let gradient = residual.pullback(&doubled).unwrap();
    prop_assert_close!(gradient, a * Complex::new(2.0, 0.0), 1e-12);
    let scaled = linalg::svdvals(&(a * Complex::new(scale, 0.0))).unwrap();
    let scaled_norm: f64 = scaled.values().iter().map(|s| (s / scale).powi(2)).sum();
    prop_assert_close!(scaled_norm, norm, 1e-12);

    let simple = values.windows(2).all(|pair| pair[0] - pair[1] > 0.05);
    if simple && values.last().is_some_and(|&s| s > 0.05) {
        let loss = |t: f64| {
            let values = linalg::svdvals(&(a + direction * Complex::from(t))).unwrap();
            values
                .values()
                .iter()
                .zip(weights)
                .map(|(s, w)| s * w)
                .sum::<f64>()
        };
        let gradient = linalg::svdvals(a).unwrap().pullback(weights).unwrap();
        let analytic = re_dot(&gradient, direction);
        let tangent = linalg::svdvals(a).unwrap().pushforward(direction).unwrap();
        let pairing: f64 = tangent.iter().zip(weights).map(|(d, w)| d * w).sum();
        prop_assert_close!(pairing, analytic, 1e-12);
        // Differentiating Σ σ² = ‖A‖²_F gives 2 Σ σ dσ = 2 Re⟨A, dA⟩.
        let norm_tangent: f64 = tangent.iter().zip(&values).map(|(d, s)| 2.0 * s * d).sum();
        prop_assert_close!(norm_tangent, 2.0 * re_dot(a, direction), 1e-12);
        prop_assert_close!(central(1e-6, loss), analytic, 1e-7 * (1.0 + analytic.abs()));
    }
    Ok(())
}

/// Eigenpairs satisfy `A V = V W` with unit vectors whose largest component is real
/// and positive (to rounding). Eigenvalue cotangents pair with `A` as with `W` (scale covariance),
/// the trace of the gradient is the eigenvalue cotangent sum (shift covariance),
/// whatever the eigenvector cotangent, and the complete pullback matches a central
/// difference along a random direction. Perturbed eigenpairs are matched to the
/// nearest unperturbed eigenvalue, so no solver ordering is assumed.
fn check_general_eigensystem(
    a: &DMatrix<Complex>,
    direction: &DMatrix<Complex>,
    gw: &[Complex],
    gv: &DMatrix<Complex>,
) -> Result<(), TestCaseError> {
    let residual = linalg::eig(a).unwrap();
    let w = residual.values().to_vec();
    let vectors = residual.vectors().clone();
    let diagonal = DMatrix::from_diagonal(&DVector::from_vec(w.clone()));
    prop_assert_close!(a * &vectors, &vectors * diagonal, 1e-12);
    for vector in vectors.column_iter() {
        prop_assert_close!(vector.norm(), 1.0, 1e-14);
        let pivot = vector
            .iter()
            .max_by(|x, y| x.norm().total_cmp(&y.norm()))
            .unwrap();
        prop_assert!(pivot.re > 0.0 && pivot.im.abs() < 1e-15, "pivot {}", pivot);
    }
    let gradient = residual.pullback(gw, gv.clone()).unwrap();
    prop_assert_close!(re_dot(&gradient, a), re_dot(gw, &w), 1e-12);
    prop_assert_close!(gradient.trace(), gw.iter().sum::<Complex>(), 1e-12);
    let (dw, dv) = linalg::eig(a).unwrap().pushforward(direction).unwrap();
    let tangent_pairing = re_dot(gw, &dw) + re_dot(gv, &dv);
    prop_assert_close!(tangent_pairing, re_dot(&gradient, direction), 1e-12);
    let w_diagonal = DMatrix::from_diagonal(&DVector::from_vec(w.clone()));
    let dw_diagonal = DMatrix::from_diagonal(&DVector::from_vec(dw));
    prop_assert_close!(
        direction * &vectors + a * &dv,
        &dv * w_diagonal + &vectors * dw_diagonal,
        1e-12
    );
    for (vector, tangent) in vectors.column_iter().zip(dv.column_iter()) {
        prop_assert_close!(vector.dotc(&tangent).re, 0.0, 1e-14);
        let pivot = (0..vector.len())
            .max_by(|&i, &j| vector[i].norm().total_cmp(&vector[j].norm()))
            .unwrap();
        prop_assert_close!(tangent[pivot].im, 0.0, 1e-14);
    }

    let loss = |t: f64| {
        let moved = linalg::eig(&(a + direction * Complex::from(t))).unwrap();
        let nearest = |value: Complex| {
            let distance = |k: &usize| (moved.values()[*k] - value).norm();
            (0..w.len())
                .min_by(|x, y| distance(x).total_cmp(&distance(y)))
                .unwrap()
        };
        (0..w.len())
            .map(|j| {
                let k = nearest(w[j]);
                (gw[j].conj() * moved.values()[k]).re
                    + re_dot(gv.column(j).iter(), moved.vectors().column(k).iter())
            })
            .sum::<f64>()
    };
    let analytic = re_dot(&gradient, direction);
    prop_assert_close!(central(1e-6, loss), analytic, 1e-7 * (1.0 + analytic.abs()));
    Ok(())
}
