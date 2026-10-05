//! Directional derivatives of matrix-free illumination and surface integrals.

use std::sync::Arc;

use nalgebra::DMatrix;

use crate::{
    Complex,
    cluster::IterativeSphereCluster,
    ebcm::{Surface, qmat},
    linalg::GmresOptions,
    special::Radial,
    sw,
    test_support::{dot, patterned, re_dot},
};

/// Differentiating the converged equation matches both a perturbed solve and the
/// independent adjoint solve, including nonzero geometry, media and illumination
/// tangents. A uniform translation has exactly zero tangent.
#[test]
fn iterative_pushforward_matches_perturbed_solutions_and_adjoint() {
    let radii = [0.2, 0.24];
    let epsilon = [Complex::new(2.3, 0.04), Complex::new(3.1, 0.02)];
    let positions = [[0.1, 0.0, -0.1], [1.2, 0.2, 0.4]];
    let dr = [0.03, -0.02];
    let de = [Complex::new(0.2, -0.03), Complex::new(-0.1, 0.04)];
    let dp = [[0.03, 0.02, -0.04], [-0.02, 0.01, 0.03]];
    let dk = 0.07;
    let options = GmresOptions {
        rtol: 2e-13,
        ..GmresOptions::default()
    };
    for columns in [1, 3, 11] {
        let incident = patterned(32, columns, 0.3);
        let mut db = patterned(32, columns, -0.2) * Complex::new(0.03, 0.01);
        if columns > 8 {
            db.column_mut(8).fill(Complex::default());
        }
        let record = |step: f64| {
            let radii: Vec<_> = radii.iter().zip(dr).map(|(r, d)| r + step * d).collect();
            let epsilon: Vec<_> = epsilon.iter().zip(de).map(|(e, d)| e + step * d).collect();
            let positions: Vec<_> = positions
                .iter()
                .zip(dp)
                .map(|(p, d)| std::array::from_fn(|axis| p[axis] + step * d[axis]))
                .collect();
            let operator = Arc::new(
                IterativeSphereCluster::new(2, 1.3 + step * dk, &radii, &epsilon, &positions)
                    .unwrap(),
            );
            operator
                .record(&incident + &db * Complex::from(step), options)
                .unwrap()
        };
        let residual = record(0.0);
        let tangent = residual.pushforward(dk, &dr, &de, &dp, &db).unwrap();
        for report in &tangent.convergence {
            assert!(report.residual_norm <= options.rtol * report.rhs_norm);
        }
        let h = 1e-5;
        let difference =
            (&record(h).solution().value - &record(-h).solution().value) / Complex::from(2.0 * h);
        assert!((&difference - &tangent.value).norm() < 2e-8 * tangent.value.norm());
        let cotangent = patterned(32, columns, 0.7);
        let gradient = residual.pullback(&cotangent).unwrap();
        let expected = gradient.cluster.k0 * dk
            + dot(&gradient.cluster.radii, dr)
            + re_dot(&gradient.cluster.epsilon, de)
            + dot(
                gradient.cluster.positions.iter().flatten(),
                dp.iter().flatten(),
            )
            + re_dot(&gradient.incident, &db);
        let actual = re_dot(&cotangent, &tangent.value);
        assert!((actual - expected).abs() < 2e-11 * (1.0 + expected.abs()));
        // One retained solution supports mixed derivative actions without changing
        // either its coefficients or its solver state.
        let repeated = residual.pushforward(dk, &dr, &de, &dp, &db).unwrap();
        assert_eq!(repeated.value, tangent.value);
        let translation = residual
            .pushforward(
                0.0,
                &[0.0; 2],
                &[Complex::default(); 2],
                &[[0.3, -0.2, 0.1]; 2],
                &DMatrix::zeros(32, columns),
            )
            .unwrap();
        assert_eq!(translation.value.norm().to_bits(), 0.0_f64.to_bits());
        assert!(
            translation
                .convergence
                .iter()
                .all(|report| report.iterations == 0)
        );
    }
}

/// Mixed real surface and complex medium directions match finite differences and
/// VJP pairings for both radial functions and both surface element conventions.
/// Scaling all lengths while inversely scaling wavenumbers changes Q only by its
/// surface-element power, providing a physical identity independent of pullbacks.
#[test]
fn ebcm_pushforward_matches_differences_adjoint_and_length_scaling() {
    let base = || {
        super::gauss_legendre_surface(24, |theta| {
            (
                0.3 * (1.0 + 0.15 * theta.cos().powi(2)),
                -0.09 * theta.cos() * theta.sin(),
            )
        })
    };
    let ks = [
        [Complex::new(1.7, 0.1), Complex::new(1.8, 0.1)],
        [Complex::new(1.2, 0.02), Complex::new(1.3, 0.02)],
    ];
    let dk = [
        [Complex::new(0.1, 0.03), Complex::new(-0.2, 0.01)],
        [Complex::new(-0.1, -0.02), Complex::new(0.15, -0.01)],
    ];
    let zs = [Complex::new(0.8, 0.01), Complex::new(1.1, -0.02)];
    let dz = [Complex::new(0.07, 0.02), Complex::new(-0.04, 0.01)];
    let dr: Vec<_> = base().theta.iter().map(|t| 0.02 * t.cos()).collect();
    let ds: Vec<_> = base().theta.iter().map(|t| -0.02 * t.sin()).collect();
    for radial in [Radial::Regular, Radial::Singular] {
        for area in [false, true] {
            let record = |step: f64| {
                let surface = base();
                let surface = Surface {
                    radii: surface
                        .radii
                        .iter()
                        .zip(&dr)
                        .map(|(r, d)| r + step * d)
                        .collect(),
                    slopes: surface
                        .slopes
                        .iter()
                        .zip(&ds)
                        .map(|(r, d)| r + step * d)
                        .collect(),
                    ..surface
                };
                qmat(
                    sw::modes(2).unwrap(),
                    sw::modes(1).unwrap(),
                    surface,
                    std::array::from_fn(|i| std::array::from_fn(|j| ks[i][j] + step * dk[i][j])),
                    std::array::from_fn(|i| zs[i] + step * dz[i]),
                    radial,
                    area,
                )
                .unwrap()
            };
            let (value, residual) = record(0.0);
            let tangent = residual.pushforward(&dr, &ds, dk, dz).unwrap();
            let h = 1e-5;
            let difference = (record(h).0 - record(-h).0) / Complex::from(2.0 * h);
            assert!((&difference - &tangent).norm() < 3e-8 * tangent.norm());
            let cotangent = patterned(16, 6, 0.7);
            let gradient = residual.pullback(&cotangent).unwrap();
            let expected = dot(&gradient.radii, &dr)
                + dot(&gradient.slopes, &ds)
                + re_dot(gradient.ks.iter().flatten(), dk.iter().flatten())
                + re_dot(gradient.zs, dz);
            let actual = re_dot(&cotangent, &tangent);
            assert!((actual - expected).abs() < 2e-12 * (1.0 + expected.abs()));
            assert_eq!(residual.pushforward(&dr, &ds, dk, dz).unwrap(), tangent);
            let surface = base();
            let tangent = residual
                .pushforward(
                    &surface.radii,
                    &surface.slopes,
                    ks.map(|side| side.map(|k| -k)),
                    [Complex::default(); 2],
                )
                .unwrap();
            let power = if area { 2.0 } else { 1.0 };
            assert!((&tangent - &value * Complex::from(power)).norm() < 2e-12 * value.norm());
        }
    }
}
