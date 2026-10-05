//! Tangent identities of complete expansion and rotation matrices.

use crate::{Complex, cw, rotation, special::Radial, sw};

#[test]
fn expansion_rigid_translation_and_scale_directions_vanish() {
    let destination = sw::Basis {
        modes: sw::modes(2).unwrap().into_iter().map(|m| (0, m)).collect(),
        positions: vec![[0.8, -0.3, 0.6]],
    };
    let source = sw::Basis {
        modes: sw::modes(1).unwrap().into_iter().map(|m| (0, m)).collect(),
        positions: vec![[-0.4, 0.2, -0.1]],
    };
    let ks = [Complex::new(1.2, 0.05), Complex::new(1.4, 0.03)];
    for radial in [Radial::Regular, Radial::Singular] {
        let (_, residual) =
            sw::expansion(destination.clone(), source.clone(), ks, true, radial).unwrap();
        let rigid = [[0.3, -0.7, 0.4]];
        assert_eq!(
            residual
                .pushforward(&rigid, &rigid, [Complex::default(); 2])
                .unwrap()
                .norm()
                .to_bits(),
            0.0_f64.to_bits()
        );
        let scale = residual
            .pushforward(&destination.positions, &source.positions, ks.map(|k| -k))
            .unwrap();
        assert!(scale.norm() < 2e-12, "scale tangent: {}", scale.norm());
    }
    let cylindrical = |positions| cw::Basis {
        modes: [-1, 0, 1]
            .into_iter()
            .flat_map(|m| [0, 1].map(move |pol| (0, cw::Mode { kz: 0.3, m, pol })))
            .collect(),
        positions,
    };
    let (_, residual) = cw::expansion(
        cylindrical(destination.positions.clone()),
        cylindrical(source.positions.clone()),
        ks,
        Radial::Regular,
    )
    .unwrap();
    let scale = residual
        .pushforward_axial(
            &destination.positions,
            &source.positions,
            ks.map(|k| -k),
            &[-0.3],
        )
        .unwrap();
    assert!(
        scale.norm() < 2e-12,
        "cylindrical scale tangent: {}",
        scale.norm()
    );
}

#[test]
fn rotation_direction_differentiates_unitarity() {
    let basis = sw::Basis {
        modes: sw::modes(3)
            .unwrap()
            .into_iter()
            .map(|mode| (0, mode))
            .collect(),
        positions: vec![[0.0; 3]],
    };
    let residual = rotation::sw_rotation(&basis, &basis, [0.3, 0.7, -0.2]).unwrap();
    let direction = [0.4, -0.2, 0.3];
    let tangent = residual.pushforward(direction).unwrap();
    let cotangent = crate::test_support::patterned(basis.modes.len(), basis.modes.len(), 0.2);
    let gradient = residual.pullback(&cotangent).unwrap();
    let repeated = residual.pushforward(direction).unwrap();
    assert_eq!(repeated, tangent);
    let jvp_pairing: f64 = cotangent
        .iter()
        .zip(&tangent)
        .map(|(g, d)| (g.conj() * d).re)
        .sum();
    let vjp_pairing: f64 = gradient
        .into_iter()
        .zip(direction)
        .map(|(g, d)| g * d)
        .sum();
    assert!((jvp_pairing - vjp_pairing).abs() < 2e-12);
    let identity = tangent.adjoint() * residual.value() + residual.value().adjoint() * &tangent;
    assert!(
        identity.norm() < 3e-14,
        "unitarity tangent: {}",
        identity.norm()
    );
}
