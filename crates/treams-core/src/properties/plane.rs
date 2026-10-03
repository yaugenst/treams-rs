//! Plane waves, their expansions and diffraction channels.

use nalgebra::DMatrix;
use proptest::{prelude::*, test_runner::TestCaseError};

use crate::{
    Complex, basis, channels, cw, fields, pw,
    special::{self, Radial},
    test_support::{
        DEFAULT_CASES, complex_vec, cylindrical_basis, dot, patterned, prop_assert_close, re_dot,
        spherical_basis,
    },
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

    #[test]
    fn plane_permutation_inverse_and_homogeneous_adjoint(
        x in -0.2_f64..0.2, helicity in any::<bool>(),
    ) {
        check_plane_permutation(x, helicity)?;
    }

    #[test]
    fn plane_translation_composition_and_scale_adjoint(
        points in prop::collection::vec(prop::array::uniform3(-1.0_f64..1.0), 0..80),
        vectors in prop::collection::vec(prop::array::uniform3(decaying()), 0..80),
        k in 0.1_f64..2.0, seed in -2.0_f64..2.0,
    ) {
        check_plane_translation(&points, vectors, k, seed)?;
    }

    #[test]
    fn cylindrical_illumination_survives_vector_normalization(
        phi in -3.1_f64..3.1, kz in 0.01_f64..1.0, decades in -6.0_f64..6.0,
    ) {
        check_cylindrical_illumination_normalization(phi, kz, 10_f64.powf(decades))?;
    }

    #[test]
    fn plane_expansion_scale_adjoint(
        kx in 0.1_f64..0.7,
        kz in -1.0_f64..1.0,
        x in -0.3_f64..0.3,
        helicity in any::<bool>(),
        complex_transverse in any::<bool>(),
    ) {
        check_plane_expansion_scale(kx, kz, x, helicity, complex_transverse)?;
    }

    #[test]
    fn plane_expansion_translation_covariance(
        positions in prop::array::uniform3(prop::array::uniform3(-1.0_f64..1.0)),
        transverse in prop::collection::vec(prop::array::uniform2(-0.9_f64..0.9), 1..4),
        cylindrical in any::<bool>(), helicity in any::<bool>(), order in 1_u32..5,
        seed in -2.0_f64..2.0,
    ) {
        check_expansion_translation(positions, &transverse, cylindrical, helicity, order, seed)?;
    }

    #[test]
    fn plane_field_operator_and_scale_adjoint(
        kx in 0.1_f64..0.7, x in -0.5_f64..0.5, helicity in any::<bool>(),
    ) {
        check_plane_field_operator(kx, x, helicity)?;
    }

    #[test]
    fn channel_scale_and_cell_adjoint(
        k in 1.0_f64..2.0,
        qx in 0.1_f64..0.4,
        area in 2.0_f64..4.0,
        scale in 0.5_f64..2.0,
        helicity in any::<bool>(),
        period in 1.3_f64..2.2,
        x in -0.3_f64..0.3,
    ) {
        check_spherical_channel_scale(k, qx, area, scale, helicity)?;
        check_cylindrical_channels_scale(k, period, x)?;
    }

    #[test]
    fn plane_wave_maxwell_and_origin_reconstruction(
        transverse in prop_oneof![prop::array::uniform2(-1.0_f64..1.0), Just([0.0; 2])],
        z in prop_oneof![0.5_f64..2.0, -2.0_f64..-0.5],
        pol in 0_u8..2,
        helicity in any::<bool>(),
    ) {
        check_plane_wave_maxwell([transverse[0], transverse[1], z], pol, helicity)?;
    }

    #[test]
    fn spherical_channels_are_reciprocal_plane_expansions(
        positions in prop::array::uniform2(prop::array::uniform3(-0.4_f64..0.4)),
        q in prop::array::uniform2(-0.8_f64..0.8),
        k in (1.0_f64..2.0, 0.0_f64..0.2),
        area in 2.0_f64..4.0,
        pol in 0_u8..2,
        helicity in any::<bool>(),
        g in complex_vec(4 * 60 * 3, 1.0),
    ) {
        check_spherical_channels(positions, q, Complex::new(k.0, k.1), area, pol, helicity, &g)?;
    }

    #[test]
    fn cylindrical_channels_are_reciprocal_plane_expansions(
        positions in prop::array::uniform2(prop::array::uniform3(-0.4_f64..0.4)),
        kz in 0.1_f64..0.6,
        kx in -0.6_f64..0.6,
        k in (1.0_f64..2.0, 0.0_f64..0.2),
        period in 1.5_f64..3.0,
        pol in 0_u8..2,
        g in complex_vec(4 * 56 * 3, 1.0),
    ) {
        check_cylindrical_channels(positions, [kz, kx], Complex::new(k.0, k.1), period, pol, &g)?;
    }
}

/// Changing the plane-wave axis and changing it back is the identity, and the
/// permutation is homogeneous of degree zero in the wave vector.
fn check_plane_permutation(x: f64, helicity: bool) -> Result<(), TestCaseError> {
    let k = [
        Complex::new(0.4 + x, 0.1),
        Complex::new(0.3, -0.05),
        Complex::new(0.8, 0.1),
    ];
    let (forward, residual) = pw::permutation(vec![k, k], vec![0, 1], 1, helicity).unwrap();
    let rotated = [k[2], k[0], k[1]];
    let (inverse, _) = pw::permutation(vec![rotated, rotated], vec![0, 1], 2, helicity).unwrap();
    prop_assert_close!(&inverse * &forward, DMatrix::identity(2, 2), 1e-12);
    let g = DMatrix::from_fn(2, 2, |i, j| {
        Complex::new(if i == j { 0.7 } else { -0.1 }, 0.2)
    });
    let gradient = residual.pullback(&g).unwrap();
    let contraction: Complex = gradient
        .iter()
        .flat_map(|row| row.iter().zip(k).map(|(g, v)| g.conj() * v))
        .sum();
    prop_assert_close!(contraction, Complex::default(), 1e-12);
    Ok(())
}

/// Expansion coefficients at displaced copies of one mode set differ only by the
/// relative plane-wave phase, `a_p = exp(i k.(r_p - r_0)) a_0`; the position gradient is
/// the sum of the coefficient terms times `i k`, and it is the same with fixed
/// wavevectors.
fn check_expansion_translation(
    positions: [[f64; 3]; 3],
    transverse: &[[f64; 2]],
    cylindrical: bool,
    helicity: bool,
    order: u32,
    seed: f64,
) -> Result<(), TestCaseError> {
    let (k, kz) = (1.3, 0.4);
    let vectors: Vec<_> = transverse
        .iter()
        .map(|&[x, y]| {
            let z = if cylindrical {
                Complex::new(kz, 0.0)
            } else {
                Complex::new(k * k - x * x - y * y, 0.0).sqrt()
            };
            [x.into(), y.into(), z]
        })
        .collect();
    let polarizations: Vec<u8> = (0..vectors.len()).map(|j| u8::from(j % 2 == 0)).collect();
    let basis: basis::MultipoleBasis = if cylindrical {
        let modes = cylindrical_basis(i32::try_from(order).unwrap(), kz, [0.0; 3]).modes;
        cw::Basis {
            modes: (0..3)
                .flat_map(|p| modes.iter().map(move |&(_, m)| (p, m)))
                .collect(),
            positions: positions.to_vec(),
        }
        .into()
    } else {
        let modes = spherical_basis(order, [0.0; 3]).modes;
        crate::sw::Basis {
            modes: (0..3)
                .flat_map(|p| modes.iter().map(move |&(_, m)| (p, m)))
                .collect(),
            positions: positions.to_vec(),
        }
        .into()
    };
    let expand = |fixed_vectors| {
        pw::expansion(
            basis.clone(),
            vectors.clone(),
            polarizations.clone(),
            helicity,
            fixed_vectors,
        )
        .unwrap()
    };
    let (value, residual) = expand(false);
    let (_, fixed) = expand(true);
    let per_position = value.nrows() / 3;
    for (j, &vector) in vectors.iter().enumerate() {
        let reference = value.view((0, j), (per_position, 1));
        for (p, &position) in positions.iter().enumerate().skip(1) {
            let shift = pw::translate(vector, position).unwrap()
                / pw::translate(vector, positions[0]).unwrap();
            let expected = reference.map(|a| a * shift);
            let actual = value
                .view((p * per_position, j), (per_position, 1))
                .clone_owned();
            prop_assert_close!(actual, expected, 1e-12 * (1.0 + reference.norm()));
        }
    }
    let g = patterned(value.nrows(), value.ncols(), seed);
    let gradient = residual.pullback(&g).unwrap();
    let fixed = fixed.pullback(&g).unwrap();
    let expected: Vec<[f64; 3]> = (0..3)
        .map(|p| {
            std::array::from_fn(|a| {
                let rows = p * per_position..(p + 1) * per_position;
                rows.flat_map(|i| (0..vectors.len()).map(move |j| (i, j)))
                    .map(|(i, j)| {
                        (g[(i, j)].conj() * value[(i, j)] * Complex::i() * vectors[j][a]).re
                    })
                    .sum()
            })
        })
        .collect();
    let flat = |v: &[[f64; 3]]| v.iter().flatten().copied().collect::<Vec<_>>();
    let tolerance = 1e-12 * (1.0 + g.norm() * value.norm());
    prop_assert_close!(flat(&gradient.positions), flat(&expected), tolerance);
    prop_assert_close!(flat(&fixed.positions), flat(&expected), tolerance);
    prop_assert!(
        fixed
            .vectors
            .iter()
            .flatten()
            .all(|&v| v == Complex::default())
    );
    Ok(())
}

/// Wavevector components with real parts in `[-2, 2)` and decay rates in `[0, 0.2)`.
fn decaying() -> impl Strategy<Value = Complex> {
    (-2.0_f64..2.0, 0.0_f64..0.2).prop_map(|(re, im)| Complex::new(re, im))
}

/// Plane translation phases compose, `exp(i k.2r) = exp(i k.r)^2`, obey the Euler
/// identity of joint displacement and wavevector scaling, and their pullback equals
/// the forward phases times `i k` and `i r`, summed entry by entry, for
/// every shape, including the blocked parallel reductions above 4096 entries. The
/// wavevectors include the axial `(0, 0, k + 0.1i)`, which has no transverse direction.
fn check_plane_translation(
    points: &[[f64; 3]],
    mut vectors: Vec<[Complex; 3]>,
    k: f64,
    seed: f64,
) -> Result<(), TestCaseError> {
    vectors.push([Complex::default(), Complex::default(), Complex::new(k, 0.1)]);
    let vectors = vectors.as_slice();
    let (value, residual) = pw::phases(points.to_vec(), vectors.to_vec()).unwrap();
    let doubled = points.iter().map(|p| p.map(|v| 2.0 * v)).collect();
    let (twice, _) = pw::phases(doubled, vectors.to_vec()).unwrap();
    prop_assert_close!(
        value.component_mul(&value),
        twice,
        1e-12 * (1.0 + value.norm())
    );
    let g = patterned(points.len(), vectors.len(), seed);
    let gradient = residual.pullback(&g).unwrap();
    let spatial = dot(gradient.points.iter().flatten(), points.iter().flatten());
    let spectral = re_dot(gradient.vectors.iter().flatten(), vectors.iter().flatten());
    let scale = 1.0 + g.norm() * value.norm();
    prop_assert_close!(spatial, spectral, 1e-12 * scale);
    let weight = g.zip_map(&value, |g, v| g.conj() * Complex::i() * v);
    let expected_points: Vec<[f64; 3]> = (0..points.len())
        .map(|i| {
            std::array::from_fn(|a| {
                (0..vectors.len())
                    .map(|j| (weight[(i, j)] * vectors[j][a]).re)
                    .sum()
            })
        })
        .collect();
    let expected_vectors: Vec<[Complex; 3]> = (0..vectors.len())
        .map(|j| {
            std::array::from_fn(|a| {
                (0..points.len())
                    .map(|i| weight[(i, j)].conj() * points[i][a])
                    .sum()
            })
        })
        .collect();
    let flat = |v: &[[f64; 3]]| v.iter().flatten().copied().collect::<Vec<_>>();
    prop_assert_close!(
        flat(&gradient.points),
        flat(&expected_points),
        1e-12 * scale
    );
    let flat = |v: &[[Complex; 3]]| v.iter().flatten().copied().collect::<Vec<_>>();
    prop_assert_close!(
        flat(&gradient.vectors),
        flat(&expected_vectors),
        1e-12 * scale
    );
    Ok(())
}

/// A cylindrical plane-wave expansion keeps unit coefficients when the wave vector
/// is normalized with a one-ulp axial error, at any overall scale; its gradient
/// along the (real) wave vector vanishes, and a relative axial offset of 1e-8
/// selects no mode.
fn check_cylindrical_illumination_normalization(
    phi: f64,
    kz: f64,
    scale: f64,
) -> Result<(), TestCaseError> {
    let radius = (1.69 - kz * kz).sqrt();
    let direction = [radius * phi.cos(), radius * phi.sin(), kz];
    let norm = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut vector = direction.map(|v| Complex::new(v / norm * 1.3 * scale, 0.0));
    vector[2].re = vector[2].re.next_up();
    let basis = cylindrical_basis(3, kz * scale, [0.0; 3]);
    let (value, residual) =
        pw::expansion(basis.clone(), vec![vector], vec![0], true, false).unwrap();
    prop_assert_eq!(value.len(), basis.modes.len());
    for (coefficient, &(_, mode)) in value.iter().zip(&basis.modes) {
        let expected = if mode.pol == 0 { 1.0 } else { 0.0 };
        prop_assert_close!(coefficient.norm(), expected, 1e-12, "{mode:?}");
    }
    let gradient = residual.pullback(&value).unwrap();
    for g in gradient.vectors.iter().flatten() {
        prop_assert_close!(g.re * scale, 0.0, 1e-10);
    }
    let other = cw::Mode {
        kz: kz * scale * (1.0 + 1e-8),
        m: 0,
        pol: 0,
    };
    prop_assert!(!pw::cylindrical_mode_matches(other, vector[2], 0));
    Ok(())
}

/// Cylindrical diffraction channels are scale invariant; their period gradient is
/// fixed by the emitted-flux normalization and the Euler identity of joint scaling.
fn check_cylindrical_channels_scale(k: f64, period: f64, x: f64) -> Result<(), TestCaseError> {
    let basis = cylindrical_basis(3, 0.0, [x, 0.1, 0.2]);
    let q = vec![[0.0, 0.2], [0.0, 2.5]];
    let ks = [Complex::new(k, 0.1); 2];
    let (value, forward) = channels::cylindrical_channels(
        basis.clone(),
        ks,
        q.clone(),
        vec![0, 1],
        period,
        true,
        false,
    )
    .unwrap();
    let g = DMatrix::from_element(value.nrows(), 2, Complex::new(0.2, 0.1));
    let d = basis.modes.len();
    let emitted = g.rows(2 * d, 2 * d).dotc(&value.rows(2 * d, 2 * d)).re;
    let scaled = cw::Basis {
        positions: basis.positions.iter().map(|p| p.map(|x| x * 1.7)).collect(),
        ..basis.clone()
    };
    let (other, _) = channels::cylindrical_channels(
        scaled,
        ks.map(|k| k / 1.7),
        q.iter().map(|q| q.map(|v| v / 1.7)).collect(),
        vec![0, 1],
        period * 1.7,
        true,
        false,
    )
    .unwrap();
    prop_assert_close!(&value, &other, 1e-10);
    let gradient = forward.pullback(&g).unwrap();
    prop_assert_close!(gradient.measure * period, -emitted, 1e-10);
    let spatial = dot(
        gradient.positions.iter().flatten(),
        basis.positions.iter().flatten(),
    );
    let spectral = re_dot(gradient.ks, ks);
    let transverse = dot(gradient.q.iter().flatten(), q.iter().flatten());
    prop_assert_close!(
        spatial - spectral - transverse + period * gradient.measure,
        0.0,
        1e-10
    );
    Ok(())
}

/// Plane-wave expansions obey the Euler identity of joint scaling and are scale
/// invariant: in cylindrical waves (`kz = 0` modes) for the wave vector
/// `(kx + 0.1i, 0.3 + 0.2i, 0)`, and in spherical waves, where they also reconstruct
/// the field at the basis position, for a propagating wave vector
/// `(kx, 0.2, kz + 0.1i)` or a fixed one with complex transverse components.
fn check_plane_expansion_scale(
    kx: f64,
    kz: f64,
    x: f64,
    helicity: bool,
    complex_transverse: bool,
) -> Result<(), TestCaseError> {
    let position = [x, 0.1, 0.2];
    let spherical = if complex_transverse {
        [
            Complex::new(kx, 1.0),
            Complex::new(0.2, 0.3),
            Complex::new(1.3, -0.8),
        ]
    } else {
        [
            Complex::new(kx, 0.0),
            Complex::new(0.2, 0.0),
            Complex::new(kz, 0.1),
        ]
    };
    let cylindrical = [
        Complex::new(kx, 0.1),
        Complex::new(0.3, 0.2),
        Complex::default(),
    ];
    for (basis, vector) in [
        (
            basis::MultipoleBasis::from(spherical_basis(4, position)),
            spherical,
        ),
        (cylindrical_basis(3, 0.0, position).into(), cylindrical),
    ] {
        let vectors = vec![vector];
        let (value, residual) =
            pw::expansion(basis.clone(), vectors.clone(), vec![1], helicity, false).unwrap();
        let positions = basis.positions().to_vec();
        if let basis::MultipoleBasis::Spherical(_) = basis {
            let k = vector.iter().map(|v| v * v).sum::<Complex>().sqrt();
            let (reconstructed, _) = fields::field(
                basis.clone(),
                value.as_slice().to_vec(),
                positions.clone(),
                [k, k],
                helicity,
                Radial::Regular,
            )
            .unwrap();
            let unit = Some(vec![Complex::new(1.0, 0.0)]);
            let (expected, _) = pw::field(
                vectors.clone(),
                vec![1],
                positions.clone(),
                unit,
                helicity,
                false,
            )
            .unwrap();
            let reconstructed: Vec<_> = reconstructed.iter().flatten().copied().collect();
            prop_assert_close!(reconstructed, expected.as_slice().to_vec(), 1e-12);
        }
        let g = DMatrix::from_element(value.nrows(), 1, Complex::new(0.2, 0.1));
        let gradient = residual.pullback(&g).unwrap();
        let spatial = dot(
            gradient.positions.iter().flatten(),
            positions.iter().flatten(),
        );
        let spectral = re_dot(gradient.vectors.iter().flatten(), vectors.iter().flatten());
        prop_assert_close!(spatial, spectral, 1e-10);
        let shrunk = vectors.iter().map(|v| v.map(|k| k / 1.7)).collect();
        let (other, _) =
            pw::expansion(scaled(basis, 1.7), shrunk, vec![1], helicity, false).unwrap();
        prop_assert_close!(value, other, 1e-10);
    }
    Ok(())
}

/// `basis` with every position scaled by `factor`.
fn scaled(basis: basis::MultipoleBasis, factor: f64) -> basis::MultipoleBasis {
    let scale = |positions: Vec<[f64; 3]>| positions.into_iter().map(|p| p.map(|x| x * factor));
    match basis {
        basis::MultipoleBasis::Spherical(b) => basis::MultipoleBasis::Spherical(crate::sw::Basis {
            positions: scale(b.positions).collect(),
            ..b
        }),
        basis::MultipoleBasis::Cylindrical(b) => basis::MultipoleBasis::Cylindrical(cw::Basis {
            positions: scale(b.positions).collect(),
            ..b
        }),
    }
}

/// The plane-wave field operator times the amplitudes is the weighted field;
/// both pullbacks agree, obey the Euler identity and reproduce the amplitude pairing.
fn check_plane_field_operator(kx: f64, x: f64, helicity: bool) -> Result<(), TestCaseError> {
    let vectors = vec![
        [
            Complex::new(kx, 0.0),
            Complex::new(0.2, 0.0),
            Complex::new(1.3, 0.1),
        ],
        [
            Complex::new(1.5, 0.0),
            Complex::new(-0.1, 0.0),
            Complex::new(0.0, 0.2),
        ],
    ];
    let points = vec![[x, 0.2, 0.1], [0.3, -0.1, 0.2]];
    let coefficients = vec![Complex::new(0.7, 0.1), Complex::new(-0.2, 0.3)];
    let field = |coefficients| {
        pw::field(
            vectors.clone(),
            vec![0, 1],
            points.clone(),
            coefficients,
            helicity,
            false,
        )
        .unwrap()
    };
    let (matrix, operator) = field(None);
    let (value, weighted) = field(Some(coefficients.clone()));
    let contracted = &matrix * DMatrix::from_column_slice(2, 1, &coefficients);
    prop_assert_close!(contracted, value.clone(), 1e-12);
    let g = DMatrix::from_element(6, 1, Complex::new(0.2, 0.1));
    let full_g = DMatrix::from_fn(6, 2, |i, j| g[(i, 0)] * coefficients[j].conj());
    let a = operator.pullback(&full_g).unwrap();
    let b = weighted.pullback(&g).unwrap();
    prop_assert_close!(&a.points, &b.points, 1e-12);
    prop_assert_close!(&a.vectors, &b.vectors, 1e-12);
    let spatial = dot(b.points.iter().flatten(), points.iter().flatten());
    let spectral = re_dot(b.vectors.iter().flatten(), vectors.iter().flatten());
    prop_assert_close!(spatial, spectral, 1e-12);
    prop_assert_close!(
        re_dot(&b.coefficients, &coefficients),
        re_dot(&g, &value),
        1e-12
    );
    Ok(())
}

/// Spherical diffraction channels are invariant under joint scaling with the unit
/// cell area, and their pullback obeys the corresponding Euler identity.
fn check_spherical_channel_scale(
    k: f64,
    qx: f64,
    area: f64,
    scale: f64,
    helicity: bool,
) -> Result<(), TestCaseError> {
    let basis = spherical_basis(3, [0.1, 0.2, -0.3]);
    let ks = [Complex::new(k, 0.1); 2];
    let q = vec![[qx, 0.2], [3.2, -0.1]];
    let (value, forward) = channels::spherical_channels(
        basis.clone(),
        ks,
        q.clone(),
        vec![0, 1],
        area,
        helicity,
        false,
    )
    .unwrap();
    let scaled_basis = crate::sw::Basis {
        positions: basis
            .positions
            .iter()
            .map(|r| r.map(|x| x * scale))
            .collect(),
        ..basis.clone()
    };
    let (scaled, _) = channels::spherical_channels(
        scaled_basis,
        ks.map(|k| k / scale),
        q.iter().map(|q| q.map(|x| x / scale)).collect(),
        vec![0, 1],
        area * scale * scale,
        helicity,
        false,
    )
    .unwrap();
    let tolerance = 1e-10 * (1.0 + value.norm());
    prop_assert_close!(&value, &scaled, tolerance);
    let g = DMatrix::from_fn(value.nrows(), 2, |i, j| {
        Complex::new(if i % 3 == j { 0.3 } else { -0.2 }, 0.1)
    });
    let gradient = forward.pullback(&g).unwrap();
    let spatial = dot(
        gradient.positions.iter().flatten(),
        basis.positions.iter().flatten(),
    );
    let spectral = re_dot(gradient.ks, ks);
    let transverse = dot(gradient.q.iter().flatten(), q.iter().flatten());
    prop_assert_close!(
        spatial - spectral - transverse + 2.0 * area * gradient.measure,
        0.0,
        1e-8
    );
    Ok(())
}

/// Plane-wave polarizations are transverse, helicity polarizations are curl
/// eigenvectors, and the regular spherical expansion reproduces the polarization
/// at the origin, for upward, downward and axial wave vectors.
fn check_plane_wave_maxwell(
    direction: [f64; 3],
    pol: u8,
    helicity: bool,
) -> Result<(), TestCaseError> {
    let vector = direction.map(|v| v * Complex::new(1.0, 0.1));
    let k = vector.iter().map(|v| v * v).sum::<Complex>().sqrt();
    let polarization = pw::polarization(vector, pol, helicity).unwrap();
    let transversality: Complex = polarization.iter().zip(vector).map(|(e, k)| e * k).sum();
    prop_assert_close!(transversality, Complex::default(), 1e-12);
    if helicity {
        let [x, y, z] = vector;
        let [ex, ey, ez] = polarization;
        let cross = [y * ez - z * ey, z * ex - x * ez, x * ey - y * ex];
        let sign = special::helicity_sign(pol);
        for (curl, e) in cross.iter().zip(polarization) {
            prop_assert_close!(Complex::i() * curl, sign * k * e, 1e-12);
        }
    }
    let basis = spherical_basis(1, [0.0; 3]);
    let coefficients = pw::reference_spherical_expansion(&basis, vector, pol, helicity).unwrap();
    let (field, _) = fields::field(
        basis,
        coefficients,
        vec![[0.0; 3]],
        [k, k],
        helicity,
        Radial::Regular,
    )
    .unwrap();
    for (actual, expected) in field[0].iter().zip(polarization) {
        prop_assert_close!(*actual, expected, 1e-11);
    }
    Ok(())
}

/// The outgoing (`Im >= 0`, then `Re >= 0`) square root of `k^2 - |q|^2`.
fn normal_wavenumber(k: Complex, q: [f64; 2]) -> Complex {
    let normal = (k * k - q[0] * q[0] - q[1] * q[1]).sqrt();
    if normal.im < 0.0 || (normal.im == 0.0 && normal.re < 0.0) {
        -normal
    } else {
        normal
    }
}

/// Checks that channel columns (rows pack incident and emitted amplitudes, each for
/// the two sides) depend on the multipole positions only through `exp(+i k.r)`
/// (incident) and `exp(-i k.r)` (emitted): the position cotangents of any `g` are
/// `Re sum conj(g) (+-i k_a) value` over the rows of that position.
fn prop_assert_position_phases(
    value: &DMatrix<Complex>,
    g: &DMatrix<Complex>,
    pidxs: &[usize],
    vector: impl Fn(usize, usize) -> [Complex; 3],
    positions: &[[f64; 3]],
) -> Result<(), TestCaseError> {
    let d = pidxs.len();
    let mut expected = vec![[0.0; 3]; positions.len()];
    let mut scale = 1.0;
    for j in 0..value.ncols() {
        for side in 0..2 {
            let v = vector(j, side);
            for (i, &p) in pidxs.iter().enumerate() {
                for (row, sign) in [(side * d + i, 1.0), ((2 + side) * d + i, -1.0)] {
                    let weight = g[(row, j)].conj() * value[(row, j)] * sign * Complex::i();
                    for (e, k) in expected[p].iter_mut().zip(v) {
                        *e += (weight * k).re;
                        scale += (weight * k).norm();
                    }
                }
            }
        }
    }
    prop_assert_close!(positions.to_vec(), expected, 1e-12 * scale);
    Ok(())
}

/// Spherical channels of a 2D array are plane-wave expansions: their incident rows
/// equal `pw::reference_spherical_expansion` for `(q, +-k_z)` (including an evanescent
/// order), the emitted rows are reciprocal to them, `S_out(l, m; q, side) area k k_z
/// (-1)^m = -S_in(l, -m; -q, other side) / 2`, and positions enter only through phases.
fn check_spherical_channels(
    positions: [[f64; 3]; 2],
    q: [f64; 2],
    k: Complex,
    area: f64,
    pol: u8,
    helicity: bool,
    g: &[Complex],
) -> Result<(), TestCaseError> {
    let modes = crate::sw::modes(3).unwrap();
    let basis = crate::sw::Basis {
        modes: (0..2)
            .flat_map(|p| modes.iter().map(move |&mode| (p, mode)))
            .collect(),
        positions: positions.to_vec(),
    };
    let d = basis.modes.len();
    let columns = vec![q, q.map(|x| -x), [q[0] + 3.1, q[1]]];
    let polarizations = vec![pol, pol, 1 - pol];
    let (value, residual) = channels::spherical_channels(
        basis.clone(),
        [k; 2],
        columns.clone(),
        polarizations.clone(),
        area,
        helicity,
        false,
    )
    .unwrap();
    let vector = |j: usize, side: usize| -> [Complex; 3] {
        let normal = normal_wavenumber(k, columns[j]);
        let q = columns[j];
        [
            q[0].into(),
            q[1].into(),
            if side == 0 { normal } else { -normal },
        ]
    };
    for (j, &pol) in polarizations.iter().enumerate() {
        for side in 0..2 {
            let expected =
                pw::reference_spherical_expansion(&basis, vector(j, side), pol, helicity).unwrap();
            let actual: Vec<_> = value.column(j).rows(side * d, d).iter().copied().collect();
            let tolerance = 1e-12 * (1.0 + expected.iter().map(|x| x.norm()).sum::<f64>());
            prop_assert_close!(actual, expected, tolerance, "column {j}, side {side}");
        }
    }
    let kz = normal_wavenumber(k, q);
    for side in 0..2 {
        for (i, &(p, mode)) in basis.modes.iter().enumerate() {
            let mirror = basis
                .modes
                .iter()
                .position(|&(o, m)| o == p && m.l == mode.l && m.m == -mode.m && m.pol == mode.pol)
                .unwrap();
            let sign = if mode.m % 2 == 0 { 1.0 } else { -1.0 };
            let emitted = value[((2 + side) * d + i, 0)] * area * k * kz * sign;
            let incident = -0.5 * value[((1 - side) * d + mirror, 1)];
            prop_assert_close!(emitted, incident, 1e-12 * (1.0 + incident.norm()));
        }
    }
    let g = DMatrix::from_column_slice(4 * d, 3, g);
    let gradient = residual.pullback(&g).unwrap();
    let pidxs: Vec<_> = basis.modes.iter().map(|&(p, _)| p).collect();
    prop_assert_position_phases(&value, &g, &pidxs, vector, &gradient.positions)
}

/// Cylindrical channels of a 1D array along x are plane-wave expansions: incident
/// rows equal `pw::reference_cylindrical_expansion` for `(k_x, +-k_y, k_z)` with
/// `q = (k_z, k_x)` (including an evanescent order), emitted rows are reciprocal to them,
/// `S_out(k_z, m; q, side) period k_y (-1)^m = 2 S_in(-k_z, -m; -q, other side)`, and
/// positions enter only through phases.
#[allow(clippy::float_cmp)] // Axial wavenumbers are exact mode labels.
fn check_cylindrical_channels(
    positions: [[f64; 3]; 2],
    q: [f64; 2],
    k: Complex,
    period: f64,
    pol: u8,
    g: &[Complex],
) -> Result<(), TestCaseError> {
    let basis = cw::Basis {
        modes: (0..2)
            .flat_map(|p| {
                [q[0], -q[0]].into_iter().flat_map(move |kz| {
                    (-3..=3).flat_map(move |m| [1, 0].map(|pol| (p, cw::Mode { kz, m, pol })))
                })
            })
            .collect(),
        positions: positions.to_vec(),
    };
    let d = basis.modes.len();
    let columns = vec![q, q.map(|x| -x), [q[0], q[1] + 3.1]];
    let polarizations = vec![pol, pol, 1 - pol];
    let (value, residual) = channels::cylindrical_channels(
        basis.clone(),
        [k; 2],
        columns.clone(),
        polarizations.clone(),
        period,
        true,
        false,
    )
    .unwrap();
    let vector = |j: usize, side: usize| -> [Complex; 3] {
        let [kz, kx] = columns[j];
        let normal = normal_wavenumber(k, columns[j]);
        [
            kx.into(),
            if side == 0 { normal } else { -normal },
            kz.into(),
        ]
    };
    for (j, &pol) in polarizations.iter().enumerate() {
        for side in 0..2 {
            let expected =
                pw::reference_cylindrical_expansion(&basis, vector(j, side), pol).unwrap();
            let actual: Vec<_> = value.column(j).rows(side * d, d).iter().copied().collect();
            let tolerance = 1e-12 * (1.0 + expected.iter().map(|x| x.norm()).sum::<f64>());
            prop_assert_close!(actual, expected, tolerance, "column {j}, side {side}");
        }
    }
    let ky = normal_wavenumber(k, q);
    for side in 0..2 {
        for (i, &(p, mode)) in basis.modes.iter().enumerate() {
            if mode.kz != q[0] {
                continue;
            }
            let mirror = basis
                .modes
                .iter()
                .position(|&(o, m)| {
                    o == p && m.kz == -mode.kz && m.m == -mode.m && m.pol == mode.pol
                })
                .unwrap();
            let sign = if mode.m % 2 == 0 { 1.0 } else { -1.0 };
            let emitted = value[((2 + side) * d + i, 0)] * period * ky * sign;
            let incident = 2.0 * value[((1 - side) * d + mirror, 1)];
            prop_assert_close!(emitted, incident, 1e-12 * (1.0 + incident.norm()));
        }
    }
    let g = DMatrix::from_column_slice(4 * d, 3, g);
    let gradient = residual.pullback(&g).unwrap();
    let pidxs: Vec<_> = basis.modes.iter().map(|&(p, _)| p).collect();
    prop_assert_position_phases(&value, &g, &pidxs, vector, &gradient.positions)
}
