//! Forward derivatives of dense cluster solves and their physical invariances.

use std::sync::Arc;

use nalgebra::DMatrix;
use proptest::{prelude::*, test_runner::TestCaseError};

use crate::{
    Complex, cluster, cw, sw,
    test_support::{DEFAULT_CASES, EXPENSIVE_CASES, dot, patterned, prop_assert_close, re_dot},
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

    #[test]
    fn interaction_tangents_match_implicit_equations_and_adjoint(seed in -2.0_f64..2.0) {
        check_interaction(seed)?;
    }

    #[test]
    fn particle_cluster_tangents_match_differences_and_translation_invariance(
        seed in -2.0_f64..2.0,
        cylindrical in any::<bool>(),
    ) {
        check_particles(seed, cylindrical)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(EXPENSIVE_CASES))]

    #[test]
    fn sphere_cluster_tangents_match_differences_adjoint_and_scale_invariance(
        lmax in 1_u32..=2,
        k0 in 0.7_f64..1.4,
        seed in -2.0_f64..2.0,
    ) {
        check_spheres(lmax, k0, seed)?;
    }
}

/// Full and requested-column tangents satisfy the differentiated multiple-scattering
/// equation and its reverse pairing, including complex local/coupling directions.
fn check_interaction(seed: f64) -> Result<(), TestCaseError> {
    let blocks = vec![
        patterned(1, 1, seed) * Complex::from(0.03),
        patterned(2, 2, seed + 0.2) * Complex::from(0.03),
    ];
    let directions = vec![patterned(1, 1, seed + 0.8), patterned(2, 2, seed + 0.9)];
    let coupling = patterned(3, 3, seed + 0.3) * Complex::from(0.05);
    let dc = patterned(3, 3, seed + 0.4);
    let incident = patterned(3, 2, seed + 0.5);
    let db = patterned(3, 2, seed + 0.6);
    let weight = patterned(3, 2, seed + 0.7);
    let dense = |blocks: &[DMatrix<Complex>]| {
        let mut result = DMatrix::zeros(3, 3);
        result[(0, 0)] = blocks[0][(0, 0)];
        result.view_mut((1, 1), (2, 2)).copy_from(&blocks[1]);
        result
    };
    let local = dense(&blocks);
    let dt = dense(&directions);
    let residual = cluster::interaction(local.clone(), coupling.clone()).unwrap();
    let dx = residual.pushforward(&dt, &dc).unwrap();
    let response = DMatrix::identity(3, 3) + &coupling * residual.value();
    let equation = (DMatrix::identity(3, 3) - &local * &coupling) * &dx;
    prop_assert_close!(
        equation,
        &dt * response + &local * &dc * residual.value(),
        2e-13
    );
    let blocked = cluster::interaction_blocks(blocks.clone(), coupling.clone()).unwrap();
    prop_assert_close!(
        &blocked.pushforward_blocks(&directions, &dc).unwrap(),
        &dx,
        2e-13
    );
    let factor = Arc::new(
        cluster::InteractionFactor::from_blocks(blocks.clone(), coupling.clone()).unwrap(),
    );
    let recorded = factor.record(incident.clone()).unwrap();
    let tangent = recorded.pushforward(&directions, &dc, &db).unwrap();
    prop_assert_close!(&tangent, &(&dx * &incident + residual.value() * &db), 2e-13);
    let gradients = recorded.pullback(&weight).unwrap();
    let expected = gradients
        .local
        .iter()
        .zip(&directions)
        .map(|(g, d)| re_dot(g, d))
        .sum::<f64>()
        + re_dot(&gradients.coupling, &dc)
        + re_dot(&gradients.incident, &db);
    prop_assert_close!(re_dot(&weight, &tangent), expected, 2e-12);
    let gradient = residual.pullback(&patterned(3, 3, seed)).unwrap();
    prop_assert_close!(
        re_dot(&patterned(3, 3, seed), &dx),
        re_dot(&gradient.local, &dt) + re_dot(&gradient.coupling, &dc),
        2e-12
    );
    // Both reverse solves leave the saved forward factors and fields available
    // for another direction from the very same linearization.
    prop_assert_close!(&residual.pushforward(&dt, &dc).unwrap(), &dx, 2e-13);
    prop_assert_close!(
        &recorded.pushforward(&directions, &dc, &db).unwrap(),
        &tangent,
        2e-13
    );
    let step = 1e-6;
    let sample = |h: f64| {
        let shifted = blocks
            .iter()
            .zip(&directions)
            .map(|(t, d)| t + d * Complex::from(h))
            .collect();
        cluster::InteractionFactor::from_blocks(shifted, &coupling + &dc * Complex::from(h))
            .unwrap()
            .solve(&(&incident + &db * Complex::from(h)))
            .unwrap()
    };
    prop_assert_close!(
        tangent,
        (sample(step) - sample(-step)) / Complex::from(2.0 * step),
        2e-8
    );
    Ok(())
}

/// A vacuum cluster has zero derivative when lengths shrink while frequency grows;
/// all combined geometry and material derivatives obey the pullback pairing and FD.
fn check_spheres(lmax: u32, k0: f64, seed: f64) -> Result<(), TestCaseError> {
    let radii = [0.24, 0.31];
    let epsilon = [Complex::new(2.1, 0.15), Complex::new(3.0, 0.2)];
    let positions = [[0.1, -0.2, 0.3], [1.5, 0.6, -0.4]];
    let dk = 0.2;
    let dr = [0.07, -0.05];
    let de = [Complex::new(0.1, -0.2), Complex::new(-0.3, 0.1)];
    let dp = [[0.1, 0.2, -0.15], [-0.2, 0.03, 0.1]];
    let residual = cluster::sphere_cluster(lmax, k0, &radii, &epsilon, &positions).unwrap();
    let tangent = residual.pushforward(dk, &dr, &de, &dp).unwrap();
    let scale = residual.value().norm();
    let invariant = residual
        .pushforward(
            k0,
            &radii.map(|r| -r),
            &[Complex::default(); 2],
            &positions.map(|p| p.map(|x| -x)),
        )
        .unwrap();
    prop_assert_close!(invariant.norm(), 0.0, 3e-12 * scale);
    let translated = residual
        .pushforward(
            0.0,
            &[0.0; 2],
            &[Complex::default(); 2],
            &[[0.2, -0.4, 0.3]; 2],
        )
        .unwrap();
    prop_assert_close!(translated.norm(), 0.0, 1e-14);
    let weight = patterned(residual.shape().0, residual.shape().1, seed);
    let gradient = residual.pullback(&weight).unwrap();
    prop_assert_close!(
        &residual.pushforward(dk, &dr, &de, &dp).unwrap(),
        &tangent,
        2e-13
    );
    let pairing = gradient.k0 * dk
        + dot(&gradient.radii, dr)
        + re_dot(&gradient.epsilon, de)
        + dot(gradient.positions.iter().flatten(), dp.iter().flatten());
    prop_assert_close!(re_dot(&weight, &tangent), pairing, 3e-12 * (1.0 + scale));
    let sample = |h: f64| {
        cluster::sphere_cluster(
            lmax,
            k0 + h * dk,
            &std::array::from_fn::<_, 2, _>(|i| radii[i] + h * dr[i]),
            &std::array::from_fn::<_, 2, _>(|i| epsilon[i] + h * de[i]),
            &std::array::from_fn::<_, 2, _>(|i| {
                std::array::from_fn(|j| positions[i][j] + h * dp[i][j])
            }),
        )
        .unwrap()
        .value()
        .clone()
    };
    let h = 2e-6;
    prop_assert_close!(
        tangent,
        (sample(h) - sample(-h)) / Complex::from(2.0 * h),
        2e-8 * (1.0 + scale)
    );
    Ok(())
}

/// Arbitrary local T blocks in either wave basis share the geometry JVP and remain
/// invariant under a common translation of every particle.
fn check_particles(seed: f64, cylindrical: bool) -> Result<(), TestCaseError> {
    let positions = vec![[0.1, -0.2, 0.3], [1.7, 0.4, -0.5]];
    let ks = [Complex::new(1.1, 0.05), Complex::new(1.4, 0.08)];
    let local = vec![
        patterned(2, 2, seed) * Complex::from(0.02),
        patterned(2, 2, seed + 0.1) * Complex::from(0.03),
    ];
    let dt = vec![patterned(2, 2, seed + 0.2), patterned(2, 2, seed + 0.3)];
    let dp = [[0.1, 0.2, -0.3], [-0.2, 0.1, 0.4]];
    let dk = [Complex::new(0.2, -0.1), Complex::new(-0.15, 0.05)];
    let solve = |blocks: Vec<DMatrix<Complex>>, positions: Vec<[f64; 3]>, ks| {
        if cylindrical {
            let modes = (0..2)
                .flat_map(|p| {
                    [0, 1].map(move |pol| {
                        (
                            p,
                            cw::Mode {
                                kz: 0.2,
                                m: i32::from(pol),
                                pol,
                            },
                        )
                    })
                })
                .collect();
            cluster::cylindrical_particle_cluster(blocks, cw::Basis { modes, positions }, ks, true)
                .unwrap()
        } else {
            let modes = (0..2)
                .flat_map(|p| {
                    [0, 1].map(move |pol| {
                        (
                            p,
                            sw::Mode {
                                l: 1,
                                m: i32::from(pol),
                                pol,
                            },
                        )
                    })
                })
                .collect();
            cluster::particle_cluster(blocks, sw::Basis { modes, positions }, ks, true).unwrap()
        }
    };
    let residual = solve(local.clone(), positions.clone(), ks);
    let tangent = residual.pushforward(&dt, &dp, dk).unwrap();
    let zero = vec![DMatrix::zeros(2, 2); 2];
    prop_assert_close!(
        residual
            .pushforward(&zero, &[[0.1, -0.4, 0.2]; 2], [Complex::default(); 2])
            .unwrap()
            .norm(),
        0.0,
        1e-14
    );
    let weight = patterned(4, 4, seed + 0.4);
    let gradient = residual.pullback(&weight).unwrap();
    let pairing = gradient
        .local
        .iter()
        .zip(&dt)
        .map(|(g, d)| re_dot(g, d))
        .sum::<f64>()
        + dot(gradient.positions.iter().flatten(), dp.iter().flatten())
        + re_dot(gradient.ks, dk);
    prop_assert_close!(re_dot(&weight, &tangent), pairing, 1e-12);
    let sample = |h: f64| {
        solve(
            local
                .iter()
                .zip(&dt)
                .map(|(t, d)| t + d * Complex::from(h))
                .collect(),
            positions
                .iter()
                .zip(dp)
                .map(|(p, d)| std::array::from_fn(|i| p[i] + h * d[i]))
                .collect(),
            std::array::from_fn(|i| ks[i] + h * dk[i]),
        )
        .value()
        .clone()
    };
    let h = 1e-6;
    prop_assert_close!(
        tangent,
        (sample(h) - sample(-h)) / Complex::from(2.0 * h),
        2e-8
    );
    Ok(())
}
