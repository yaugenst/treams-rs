//! Directional references, adjoint identities and physical null directions of ports.

use super::*;

const STEP: f64 = 1e-5;

fn perturb<const N: usize>(value: [Complex; N], tangent: [Complex; N], step: f64) -> [Complex; N] {
    std::array::from_fn(|i| value[i] + step * tangent[i])
}

fn perturb_real<const N: usize>(value: [f64; N], tangent: [f64; N], step: f64) -> [f64; N] {
    std::array::from_fn(|i| value[i] + step * tangent[i])
}

fn matrix_direction(rows: usize, columns: usize, scale: f64) -> DMatrix<Complex> {
    DMatrix::from_fn(rows, columns, |i, j| {
        Complex::new(
            scale * (1.0 + f64::from(u32::try_from(i + j).unwrap())),
            scale * (0.5 - f64::from(u32::try_from(2 * i + j).unwrap())),
        )
    })
}

fn block_reference(value: impl Fn(f64) -> Blocks) -> Blocks {
    let plus = value(STEP);
    let minus = value(-STEP);
    std::array::from_fn(|b| (&plus[b] - &minus[b]) / Complex::from(2.0 * STEP))
}

fn block_pair(a: &Blocks, b: &Blocks) -> f64 {
    a.iter().zip(b).map(|(a, b)| re_dot(a, b)).sum()
}

#[test]
fn normal_incidence_and_diffraction_threshold_keep_their_derivative_contracts()
-> Result<(), TestCaseError> {
    let ks = [[Complex::from(1.2); 2], [Complex::from(1.6); 2]];
    let zs = [Complex::from(0.9), Complex::from(1.1)];
    let (_, boundary) = smatrix::interface(ks, zs, [0.0; 2], 2, false).unwrap();
    let zero_blocks: Blocks = std::array::from_fn(|_| DMatrix::zeros(2, 2));
    prop_assert_close!(
        boundary
            .pushforward(
                [[Complex::default(); 2]; 2],
                [Complex::default(); 2],
                [0.3, -0.2]
            )
            .unwrap(),
        zero_blocks,
        1e-14,
    );
    let (_, chirality) = smatrix::oriented_chirality(
        vec![[0.0; 2]],
        vec![Complex::new(1.2, 0.03)],
        vec![1],
        2,
        [0.0, 0.4],
    )
    .unwrap();
    prop_assert_close!(
        chirality
            .pushforward(&[[0.3, -0.2]], &[Complex::default()], [0.0; 2])
            .unwrap(),
        DMatrix::zeros(3, 1),
        1e-14,
    );
    let (_, grazing) = smatrix::interface([ks[0]; 2], [zs[0]; 2], [1.2, 0.0], 2, true).unwrap();
    prop_assert!(grazing.pushforward(ks, zs, [0.0; 2]).is_err());
    let (_, grazing) =
        smatrix::fresnel([ks[0]; 2], [[Complex::default(); 2]; 2], [zs[0]; 2]).unwrap();
    prop_assert!(grazing.pushforward(ks, ks, zs).is_err());
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

    #[test]
    fn fresnel_direction_matches_reference_adjoint_and_scale_identity(k in 1.1_f64..2.5) {
        check_fresnel(k)?;
    }

    #[test]
    fn interface_direction_matches_reference_adjoint_and_transparency(
        k in 1.1_f64..2.5, axis in 0_usize..3, fixed_q in any::<bool>(),
    ) {
        check_interface(k, axis, fixed_q)?;
    }

    #[test]
    fn propagation_direction_matches_reference_adjoint_and_inverse_scaling(k in 0.7_f64..2.0) {
        check_propagation(k)?;
    }

    #[test]
    fn chirality_directions_match_reference_adjoint_and_scale_identity(
        k in 0.7_f64..2.0, axis in 0_usize..3, stop in 0.001_f64..1.0,
    ) {
        check_chirality(k, axis, stop)?;
    }

    #[test]
    fn power_direction_matches_reference_adjoint_and_amplitude_identity(
        axis in 0_usize..3, direction in 0_usize..2,
        helicity in any::<bool>(), fixed_q in any::<bool>(),
    ) {
        check_power(axis, direction, helicity, fixed_q)?;
    }
}

fn check_fresnel(k: f64) -> Result<(), TestCaseError> {
    let ks = [
        [Complex::new(k, 0.1), Complex::new(k + 0.2, 0.1)],
        [Complex::new(k + 0.5, 0.2), Complex::new(k + 0.8, 0.2)],
    ];
    let kz = ks.map(|row| row.map(|k| (k * k - 0.2).sqrt()));
    let z = [Complex::new(0.9, 0.03), Complex::new(1.1, 0.04)];
    let dk = [
        [Complex::new(0.1, -0.2), Complex::new(-0.3, 0.15)],
        [Complex::new(0.2, 0.1), Complex::new(0.05, -0.13)],
    ];
    let dn = dk.map(|row| row.map(|v| Complex::i() * v));
    let dz = [Complex::new(0.1, -0.05), Complex::new(-0.07, 0.03)];
    let (_, residual) = smatrix::fresnel(ks, kz, z).unwrap();
    let tangent = residual.pushforward(dk, dn, dz).unwrap();
    let reference = block_reference(|step| {
        smatrix::fresnel(
            std::array::from_fn(|i| perturb(ks[i], dk[i], step)),
            std::array::from_fn(|i| perturb(kz[i], dn[i], step)),
            perturb(z, dz, step),
        )
        .unwrap()
        .0
    });
    prop_assert_close!(tangent.clone(), reference, 2e-9);
    let g: Blocks = std::array::from_fn(|_| matrix_direction(2, 2, 0.2));
    let gradient = residual.pullback(&g).unwrap();
    let pair = re_dot(gradient.ks.iter().flatten(), dk.iter().flatten())
        + re_dot(gradient.kz.iter().flatten(), dn.iter().flatten())
        + re_dot(gradient.z, dz);
    prop_assert_close!(block_pair(&g, &tangent), pair, 2e-12);
    let (_, residual) = smatrix::fresnel(ks, kz, z).unwrap();
    let zero: Blocks = std::array::from_fn(|_| DMatrix::zeros(2, 2));
    prop_assert_close!(
        residual
            .pushforward(ks, kz, [Complex::default(); 2])
            .unwrap(),
        zero.clone(),
        2e-12
    );
    prop_assert_close!(
        residual
            .pushforward(
                [[Complex::default(); 2]; 2],
                [[Complex::default(); 2]; 2],
                z
            )
            .unwrap(),
        zero,
        2e-12
    );
    Ok(())
}

fn check_interface(k: f64, axis: usize, fixed_q: bool) -> Result<(), TestCaseError> {
    let ks = [
        [Complex::new(k, 0.1), Complex::new(k + 0.2, 0.1)],
        [Complex::new(k + 0.5, 0.2), Complex::new(k + 0.8, 0.2)],
    ];
    let z = [Complex::new(0.9, 0.03), Complex::new(1.1, 0.04)];
    let q = [0.23, -0.14];
    let dk = [
        [Complex::new(0.1, -0.2), Complex::new(-0.3, 0.15)],
        [Complex::new(0.2, 0.1), Complex::new(0.05, -0.13)],
    ];
    let dz = [Complex::new(0.1, -0.05), Complex::new(-0.07, 0.03)];
    let dq = [0.09, -0.05];
    let (_, residual) = smatrix::interface(ks, z, q, axis, fixed_q).unwrap();
    let tangent = residual.pushforward(dk, dz, dq).unwrap();
    let reference = block_reference(|step| {
        smatrix::interface(
            std::array::from_fn(|i| perturb(ks[i], dk[i], step)),
            perturb(z, dz, step),
            perturb_real(q, if fixed_q { [0.0; 2] } else { dq }, step),
            axis,
            fixed_q,
        )
        .unwrap()
        .0
    });
    prop_assert_close!(tangent.clone(), reference, 2e-9);
    let g: Blocks = std::array::from_fn(|_| matrix_direction(2, 2, 0.2));
    let gradient = residual.pullback(&g).unwrap();
    let pair = re_dot(gradient.ks.iter().flatten(), dk.iter().flatten())
        + re_dot(gradient.z, dz)
        + dot(gradient.q, dq);
    prop_assert_close!(block_pair(&g, &tangent), pair, 2e-12);
    let (_, matched) = smatrix::interface([ks[0]; 2], [z[0]; 2], q, axis, fixed_q).unwrap();
    let zero: Blocks = std::array::from_fn(|_| DMatrix::zeros(2, 2));
    prop_assert_close!(
        matched.pushforward([dk[0]; 2], [dz[0]; 2], dq).unwrap(),
        zero,
        2e-12
    );
    Ok(())
}

fn check_propagation(k: f64) -> Result<(), TestCaseError> {
    let vectors = vec![
        [
            Complex::new(0.3, 0.0),
            Complex::new(-0.2, 0.0),
            Complex::new(k, 0.1),
        ],
        [
            Complex::new(0.1, 0.0),
            Complex::new(0.4, 0.0),
            Complex::new(k + 0.2, 0.2),
        ],
    ];
    let dv = vec![
        [
            Complex::new(0.1, 0.03),
            Complex::new(-0.02, 0.07),
            Complex::new(0.11, -0.03)
        ];
        2
    ];
    let distance = [0.1, -0.2, 0.8];
    let dd = [-0.2, 0.07, 0.13];
    let (_, residual) = smatrix::propagation(vectors.clone(), distance).unwrap();
    let tangent = residual.pushforward(&dv, dd).unwrap();
    let reference = block_reference(|step| {
        smatrix::propagation(
            vectors
                .iter()
                .zip(&dv)
                .map(|(&v, &d)| perturb(v, d, step))
                .collect(),
            perturb_real(distance, dd, step),
        )
        .unwrap()
        .0
    });
    prop_assert_close!(tangent.clone(), reference, 2e-10);
    let g: Blocks = std::array::from_fn(|_| matrix_direction(2, 2, 0.2));
    let invariant = residual
        .pushforward(&vectors, distance.map(|v| -v))
        .unwrap();
    prop_assert_close!(
        invariant,
        std::array::from_fn::<_, 4, _>(|_| DMatrix::zeros(2, 2)),
        1e-14
    );
    let gradient = residual.pullback(&g).unwrap();
    prop_assert_close!(
        block_pair(&g, &tangent),
        re_dot(gradient.vectors.iter().flatten(), dv.iter().flatten()) + dot(gradient.distance, dd),
        2e-12
    );
    Ok(())
}

fn check_chirality(k: f64, axis: usize, stop: f64) -> Result<(), TestCaseError> {
    let ks = vec![Complex::new(k, 0.08), Complex::new(k + 0.3, 0.15)];
    let normal = vec![Complex::new(k - 0.2, 0.09), Complex::new(k + 0.1, 0.11)];
    let transverse = vec![[0.23, -0.17], [0.11, 0.33]];
    let interval = [0.0, stop];
    let dk = vec![Complex::new(0.07, -0.11), Complex::new(-0.09, 0.13)];
    let dn = vec![Complex::new(0.03, 0.17), Complex::new(0.1, -0.07)];
    let dq = vec![[0.13, -0.03], [-0.2, 0.06]];
    let dz = [-0.03, 0.07];
    let shifted = |v: &[Complex], d: &[Complex], step| {
        v.iter()
            .zip(d)
            .map(|(&v, &d)| v + step * d)
            .collect::<Vec<_>>()
    };
    let g = matrix_direction(3, 2, 0.2);
    let (_, density) = smatrix::chirality_density(ks.clone(), normal.clone(), interval).unwrap();
    let tangent = density.pushforward(&dk, &dn, dz).unwrap();
    let plus = smatrix::chirality_density(
        shifted(&ks, &dk, STEP),
        shifted(&normal, &dn, STEP),
        perturb_real(interval, dz, STEP),
    )
    .unwrap()
    .0;
    let minus = smatrix::chirality_density(
        shifted(&ks, &dk, -STEP),
        shifted(&normal, &dn, -STEP),
        perturb_real(interval, dz, -STEP),
    )
    .unwrap()
    .0;
    prop_assert_close!(
        tangent.clone(),
        (plus - minus) / Complex::from(2.0 * STEP),
        2e-9
    );
    prop_assert_close!(
        density
            .pushforward(&ks, &normal, interval.map(|v| -v))
            .unwrap(),
        DMatrix::zeros(3, 2),
        2e-12
    );
    let gradient = density.pullback(&g).unwrap();
    prop_assert_close!(
        re_dot(&g, &tangent),
        re_dot(&gradient.ks, &dk) + re_dot(&gradient.normal, &dn) + dot(gradient.interval, dz),
        2e-12
    );
    let oriented = |step| {
        smatrix::oriented_chirality(
            transverse
                .iter()
                .zip(&dq)
                .map(|(&v, &d)| perturb_real(v, d, step))
                .collect(),
            shifted(&normal, &dn, step),
            vec![0, 1],
            axis,
            perturb_real(interval, dz, step),
        )
        .unwrap()
    };
    let (_, residual) = oriented(0.0);
    let tangent = residual.pushforward(&dq, &dn, dz).unwrap();
    prop_assert_close!(
        tangent.clone(),
        (oriented(STEP).0 - oriented(-STEP).0) / Complex::from(2.0 * STEP),
        2e-9
    );
    prop_assert_close!(
        residual
            .pushforward(&transverse, &normal, interval.map(|v| -v))
            .unwrap(),
        DMatrix::zeros(3, 2),
        2e-12
    );
    let gradient = residual.pullback(&g).unwrap();
    prop_assert_close!(
        re_dot(&g, &tangent),
        dot(gradient.transverse.iter().flatten(), dq.iter().flatten())
            + re_dot(&gradient.normal, &dn)
            + dot(gradient.interval, dz),
        2e-12
    );
    Ok(())
}

fn check_power(
    axis: usize,
    direction: usize,
    helicity: bool,
    fixed_q: bool,
) -> Result<(), TestCaseError> {
    let n = 4;
    let matrices: Blocks = std::array::from_fn(|b| {
        DMatrix::from_fn(n, n, |i, j| {
            Complex::new(
                if i == j {
                    if b == 0 || b == 3 { 0.8 } else { 0.1 }
                } else {
                    0.02
                },
                0.01,
            )
        })
    });
    let incident = matrix_direction(n, 2, 0.13);
    let dm: Blocks = std::array::from_fn(|_| matrix_direction(n, n, 0.013));
    let di = matrix_direction(n, 2, -0.017);
    let ks = [
        [
            Complex::new(1.3, 0.02),
            Complex::new(if helicity { 1.5 } else { 1.3 }, 0.02),
        ],
        [
            Complex::new(1.2, 0.01),
            Complex::new(if helicity { 1.4 } else { 1.2 }, 0.01),
        ],
    ];
    let zs = [Complex::new(0.9, 0.01), Complex::new(1.1, -0.01)];
    let q = [[0.2, 0.13], [0.31, -0.17]];
    let dk = [
        [Complex::new(0.07, -0.03); 2],
        [Complex::new(-0.03, 0.02); 2],
    ];
    let dz = [Complex::new(0.03, -0.07), Complex::new(-0.01, 0.06)];
    let dq = [[0.03, -0.01], [-0.07, 0.04]];
    let ports = |step| smatrix::TrPorts {
        ks: std::array::from_fn(|i| perturb(ks[i], dk[i], step)),
        zs: perturb(zs, dz, step),
        q: q.iter()
            .zip(dq)
            .map(|(&q, dq)| perturb_real(q, if fixed_q { [0.0; 2] } else { dq }, step))
            .collect(),
        modes: vec![(0, 0), (0, 1), (1, 0), (1, 1)],
        axis,
        helicity,
        direction,
    };
    let indices = [2 * direction + direction, 2 * (1 - direction) + direction];
    let residual = smatrix::tr(
        indices.map(|i| view(&matrices[i])),
        incident.clone(),
        ports(0.0),
        fixed_q,
    )
    .unwrap();
    let tangent = residual.pushforward(&dm, &di, dk, dz, &dq).unwrap();
    let value = |step| {
        let matrices: [DMatrix<Complex>; 2] =
            indices.map(|i| &matrices[i] + &dm[i] * Complex::from(step));
        smatrix::tr_value(
            matrices.each_ref().map(view),
            &(&incident + &di * Complex::from(step)),
            &ports(step),
        )
        .unwrap()
    };
    prop_assert_close!(
        tangent.clone(),
        (value(STEP) - value(-STEP)) / (2.0 * STEP),
        2e-9
    );
    let zero = std::array::from_fn(|_| DMatrix::zeros(n, n));
    let invariant = residual
        .pushforward(
            &zero,
            &incident,
            [[Complex::default(); 2]; 2],
            [Complex::default(); 2],
            &[[0.0; 2]; 2],
        )
        .unwrap();
    prop_assert_close!(invariant, DMatrix::zeros(2, 2), 2e-12);
    let g = DMatrix::from_row_slice(2, 2, &[0.3, -0.2, 0.1, 0.4]);
    let gradient = residual.pullback(&g).unwrap();
    let pair = block_pair(&gradient.matrices, &dm)
        + re_dot(&gradient.incident, &di)
        + re_dot(gradient.ks.iter().flatten(), dk.iter().flatten())
        + re_dot(gradient.zs, dz)
        + dot(gradient.q.iter().flatten(), dq.iter().flatten());
    prop_assert_close!(dot(&g, &tangent), pair, 2e-12);
    Ok(())
}
