//! Directional derivative references, adjoint pairings and conservation identities
//! for the cached multilayer particle solves.

use proptest::{prelude::*, test_runner::TestCaseError};

use crate::{
    Complex,
    coeffs::{Material, Matrix2, mie, mie_cyl},
    test_support::{DEFAULT_CASES, central, dot, prop_assert_close, re_dot},
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

    #[test]
    fn particle_pushforwards_match_references_and_adjoints(
        (radii, materials) in super::layers(1.0, 1.0),
        embedding in super::chiral(1.0, 1.0),
        degree in 1_u32..=5,
        seed in -1.0_f64..1.0,
    ) {
        check_particle_pushforwards(&radii, materials, embedding, degree, seed)?;
    }

    #[test]
    fn lossless_mie_pushforward_conserves_energy(
        size in 0.2_f64..2.0,
        epsilon in 1.2_f64..5.0,
        degree in 1_u32..=5,
    ) {
        check_lossless_pushforward(size, epsilon, degree)?;
    }
}

/// Simultaneous real geometry and complex material directions give the same real
/// pairing as the VJP and a central directional difference, for both geometries.
fn check_particle_pushforwards(
    radii: &[f64],
    mut materials: Vec<Material>,
    embedding: Material,
    degree: u32,
    seed: f64,
) -> Result<(), TestCaseError> {
    materials.push(embedding);
    let dr: Vec<_> = radii.iter().map(|r| 0.1 * r).collect();
    let dm: Vec<_> = materials
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            Material {
                epsilon: Complex::new(0.17 + seed * 0.04, -0.06) * sign,
                mu: Complex::new(-0.08, 0.09 - seed * 0.03) * sign,
                kappa: Complex::new(0.03, 0.04) * sign,
            }
        })
        .collect();
    let shifted = |step: f64| {
        let radii: Vec<_> = radii.iter().zip(&dr).map(|(r, d)| r + step * d).collect();
        let materials: Vec<_> = materials
            .iter()
            .zip(&dm)
            .map(|(m, d)| Material {
                epsilon: m.epsilon + step * d.epsilon,
                mu: m.mu + step * d.mu,
                kappa: m.kappa + step * d.kappa,
            })
            .collect();
        (radii, materials)
    };
    let g = Matrix2::new(
        Complex::new(0.4, seed),
        Complex::new(-0.6, 0.2),
        Complex::new(0.3, -0.1),
        Complex::new(seed, 0.7),
    );
    let material_pairing = |epsilon: &[Complex], mu: &[Complex], kappa: &[Complex]| {
        re_dot(epsilon, dm.iter().map(|d| d.epsilon))
            + re_dot(mu, dm.iter().map(|d| d.mu))
            + re_dot(kappa, dm.iter().map(|d| d.kappa))
    };
    let sphere = mie(degree, radii, &materials).unwrap();
    let dt = sphere.pushforward(&dr, &dm).unwrap();
    let gradient = sphere.pullback(&g).unwrap();
    prop_assert_eq!(sphere.pushforward(&dr, &dm).unwrap(), dt);
    let jvp = re_dot(&g, &dt);
    let vjp = dot(&gradient.sizes, &dr)
        + material_pairing(&gradient.epsilon, &gradient.mu, &gradient.kappa);
    let reference = central(1e-5, |step| {
        let (radii, materials) = shifted(step);
        re_dot(&g, mie(degree, &radii, &materials).unwrap().value())
    });
    let scale = dt.norm() * g.norm() + 1e-12;
    prop_assert_close!(jvp, vjp, 2e-11 * scale);
    prop_assert_close!(jvp, reference, 2e-6 * scale);

    let (kz, k0, dkz, dk0) = (0.23, 1.1, -0.07, 0.13);
    let order = i32::try_from(degree).unwrap();
    let cylinder = mie_cyl(kz, order, k0, radii, &materials).unwrap();
    let dt = cylinder.pushforward(dkz, dk0, &dr, &dm).unwrap();
    let gradient = cylinder.pullback(&g).unwrap();
    prop_assert_eq!(cylinder.pushforward(dkz, dk0, &dr, &dm).unwrap(), dt);
    let jvp = re_dot(&g, &dt);
    let vjp = dkz * gradient.kz
        + dk0 * gradient.k0
        + dot(&gradient.layers.radii, &dr)
        + material_pairing(
            &gradient.layers.epsilon,
            &gradient.layers.mu,
            &gradient.layers.kappa,
        );
    let reference = central(1e-5, |step| {
        let (radii, materials) = shifted(step);
        re_dot(
            &g,
            mie_cyl(kz + step * dkz, order, k0 + step * dk0, &radii, &materials)
                .unwrap()
                .value(),
        )
    });
    let scale = dt.norm() * g.norm() + 1e-12;
    prop_assert_close!(jvp, vjp, 2e-11 * scale);
    prop_assert_close!(jvp, reference, 2e-6 * scale);
    Ok(())
}

/// In vacuum, differentiating `T^H + T + 2 T^H T = 0` along lossless changes of
/// radius and permittivity gives the linearized optical theorem.
fn check_lossless_pushforward(size: f64, epsilon: f64, degree: u32) -> Result<(), TestCaseError> {
    let materials = [
        Material {
            epsilon: epsilon.into(),
            ..Material::default()
        },
        Material::default(),
    ];
    let residual = mie(degree, &[size], &materials).unwrap();
    let zero = Material {
        epsilon: Complex::default(),
        mu: Complex::default(),
        kappa: Complex::default(),
    };
    let direction = [
        Material {
            epsilon: Complex::new(0.13, 0.0),
            ..zero
        },
        zero,
    ];
    let t = *residual.value();
    let dt = residual.pushforward(&[0.19], &direction).unwrap();
    let defect = dt.adjoint() + dt + (dt.adjoint() * t + t.adjoint() * dt) * Complex::new(2.0, 0.0);
    prop_assert_close!(defect, Matrix2::zeros(), 2e-11 * dt.norm() + 1e-15);
    Ok(())
}

#[test]
fn metric_derivatives_reuse_saved_gradients_and_errors() {
    use crate::tmatrix::{Metric, metric};
    use nalgebra::DMatrix;

    let matrix = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex::new(0.1, 0.2),
            Complex::new(0.02, 0.03),
            Complex::new(0.03, 0.04),
            Complex::new(0.3, 0.1),
        ],
    );
    let (_, residual) = metric(&matrix, &[0, 1], [1.0, 1.0], Metric::Chirality).unwrap();
    let expected = residual.pullback(1.0).unwrap();
    let tangent = residual.pushforward(&matrix, [0.0; 2]).unwrap();
    let replay = residual.pullback(1.0).unwrap();
    assert_eq!(replay.matrix, expected.matrix);
    assert_eq!(replay.ks.map(f64::to_bits), expected.ks.map(f64::to_bits));
    assert_eq!(
        residual.pushforward(&matrix, [0.0; 2]).unwrap().to_bits(),
        tangent.to_bits(),
    );

    let identity = DMatrix::identity(2, 2);
    let (_, residual) = metric(&identity, &[0, 1], [1.0, 1.0], Metric::Chirality).unwrap();
    let error = residual.pullback(1.0).unwrap_err().to_string();
    assert_eq!(
        residual
            .pushforward(&identity, [0.0; 2])
            .unwrap_err()
            .to_string(),
        error
    );
    assert_eq!(residual.pullback(1.0).unwrap_err().to_string(), error);
    assert_eq!(
        residual
            .pushforward(&DMatrix::zeros(2, 2), [0.0; 2])
            .unwrap()
            .to_bits(),
        0.0_f64.to_bits(),
    );
    assert_eq!(residual.pullback(0.0).unwrap().matrix, DMatrix::zeros(2, 2));
}
