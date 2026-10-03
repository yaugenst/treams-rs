//! Spherical and cylindrical vector waves, translations, rotations and field evaluation.

use std::f64::consts::FRAC_1_SQRT_2;

use nalgebra::{DMatrix, DVector};
use proptest::{prelude::*, test_runner::TestCaseError};

use crate::{
    Complex, basis, cw, fields,
    numerics::parity,
    rotation,
    special::{self, Radial},
    sw::{Mode, cartesian_translation},
    test_support::{
        self, ALGEBRA_CASES, DEFAULT_CASES, EXPENSIVE_CASES, central, complex, complex_vec,
        degree_order, dot, five_point, log_uniform, patterned, prop_assert_close, radial, re_dot,
        spherical_basis,
    },
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

    #[test]
    fn conversion_field_and_adjoint(
        m in -2_i32..3, kz in -0.7_f64..0.7, x in -0.3_f64..0.3, helicity in any::<bool>(),
    ) {
        check_conversion_field(m, kz, x, helicity)?;
    }

    #[test]
    fn rotation_group_and_adjoint(
        l in 1_i32..8, theta in -6.0_f64..6.0, phi in -3.0_f64..3.0, psi in -3.0_f64..3.0,
    ) {
        check_rotation_group(l, [phi, theta, psi])?;
    }

    #[test]
    fn rotation_blocks_match_wigner_elements(
        angles in prop::array::uniform3(-3.2_f64..3.2),
        rows in Just((0..32).collect::<Vec<usize>>()).prop_shuffle(),
        count in 1_usize..=32,
        seed in -3.0_f64..3.0,
    ) {
        check_rotation_blocks(angles, &rows[..count], seed)?;
    }

    #[test]
    fn cylindrical_rotation_phases(
        phi in -3.2_f64..3.2,
        psi in -3.2_f64..3.2,
        rows in Just((0..40).collect::<Vec<usize>>()).prop_shuffle(),
        count in 1_usize..=40,
        seed in -3.0_f64..3.0,
    ) {
        check_cylindrical_rotation(phi, psi, &rows[..count], seed)?;
    }

    #[test]
    fn translation_scale_derivative_identity(
        (l, m) in degree_order(1..5),
        (lambda, mu) in degree_order(1..5),
        pol in 0_u8..2,
        to_pol in 0_u8..2,
        position in prop_oneof![
            prop::array::uniform3(-1.5_f64..1.5),
            prop_oneof![0.5_f64..2.0, -2.0_f64..-0.5].prop_map(|z| [0.0, 0.0, z]),
        ],
        k in complex(0.2).prop_map(|k| k + 1.2),
        helicity in any::<bool>(),
        radial in radial(),
    ) {
        let to = Mode { l: lambda, m: mu, pol: to_pol };
        let from = Mode { l, m, pol };
        check_translation_scale(to, from, k, position, helicity, radial, 1.3, 1e-12)?;
    }

    #[test]
    fn translation_scale_invariance_and_euler_identity(
        ((l, m), (lambda, mu)) in (degree_order(1..41), degree_order(1..41)),
        (pol, flip) in (0_u8..2, any::<bool>()),
        helicity in any::<bool>(),
        radial in radial(),
        k in 0.7_f64..2.0,
        position in prop_oneof![
            (-1.0_f64..1.0, -1.0_f64..1.0, 0.5_f64..2.0).prop_map(<[f64; 3]>::from),
            (0.5_f64..2.0, any::<bool>()).prop_map(|(z, up)| [0.0, 0.0, if up { z } else { -z }]),
        ],
        exponent in -12_i32..=12,
    ) {
        // Helicity modes couple only equal polarizations, so `flip` applies to parity modes.
        let to_pol = if flip && !helicity { 1 - pol } else { pol };
        let to = Mode { l: lambda, m: mu, pol: to_pol };
        let from = Mode { l, m, pol };
        let (k, s) = (Complex::new(k, 0.05), 10_f64.powi(exponent));
        check_translation_scale(to, from, k, position, helicity, radial, s, 1e-10)?;
    }

    #[test]
    fn vector_wave_maxwell_and_scaling(
        (l, m) in degree_order(1..7),
        order in -7_i32..8,
        kz in -0.7_f64..0.7,
        pol in 0_u8..2,
        x in -1.0_f64..1.0,
        y in -1.0_f64..1.0,
        z in 0.5_f64..2.0,
        rho_y in 0.2_f64..1.2,
        radial in radial(),
    ) {
        check_vector_wave_maxwell(Mode { l, m, pol }, [x, y, z], radial)?;
        check_cylindrical_field_maxwell(cw::Mode { kz, m: order, pol }, x, rho_y, radial)?;
    }

    #[test]
    fn cylindrical_translation_scale_identity(
        m in -7_i32..8, mu in -7_i32..8, x in -1.0_f64..1.0, y in 0.5_f64..2.0, radial in radial(),
    ) {
        check_cylindrical_translation_scale(m, mu, [x, y, 0.4], radial)?;
    }

    #[test]
    fn translations_are_rotation_equivariant(
        angles in prop::array::uniform3(-3.2_f64..3.2),
        displacement in prop_oneof![
            prop::array::uniform3(-1.5_f64..1.5),
            (-1.5_f64..1.5).prop_map(|z| [0.0, 0.0, z]),
        ],
        helicity in any::<bool>(),
        radial in radial(),
    ) {
        check_rotation_equivariance(angles, displacement, helicity, radial)?;
    }

    #[test]
    fn cylindrical_translations_obey_graf_addition(
        m in -6_i32..=6,
        mu in -6_i32..=6,
        a in prop::array::uniform3(-0.15_f64..0.15),
        b_rho in 1.0_f64..1.5,
        b_phi in -3.2_f64..3.2,
        b_z in -1.0_f64..1.0,
        radial in radial(),
    ) {
        check_graf_addition(m, mu, a, [b_rho, b_phi, b_z], radial)?;
    }

    #[test]
    fn field_evaluation_contract(
        (case, inputs) in field_case().prop_flat_map(|case| {
            let inputs = FieldInputs::strategy(&case);
            (Just(case), inputs)
        }),
        keys in prop::collection::vec(0_u32..1000, 64),
    ) {
        check_field_evaluation(&case, &inputs, &keys)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(EXPENSIVE_CASES))]

    #[test]
    fn regular_translations_compose(
        a in prop::array::uniform3(-0.2_f64..0.2),
        b in prop::array::uniform3(-0.2_f64..0.2),
        helicity in any::<bool>(),
    ) {
        check_regular_composition(a, b, helicity)?;
    }

    #[test]
    fn outgoing_translation_addition_theorem(
        a in prop::array::uniform3(-0.03_f64..0.03),
        distance in 1.0_f64..1.5,
        theta in prop_oneof![Just(0.0), Just(std::f64::consts::PI), 0.0..std::f64::consts::PI],
        phi in -3.2_f64..3.2,
        helicity in any::<bool>(),
    ) {
        check_outgoing_addition(a, [distance, theta, phi], helicity)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

    #[test]
    fn spherical_harmonic_ladder_identities(
        (p, m) in degree_order(0..11),
        radius in prop_oneof![log_uniform(-3.0..0.2), Just(0.0)],
        cosine in prop_oneof![Just(1.0), Just(-1.0), -1.0_f64..1.0],
        phi in -3.2_f64..3.2,
        k in complex(0.2).prop_map(|k| k + 1.2),
        radial in radial(),
    ) {
        check_harmonic_ladders(p, m, k, [radius, cosine, phi], radial)?;
    }

    #[test]
    fn translations_under_spatial_inversion(
        (l, m) in degree_order(1..9),
        (lambda, mu) in degree_order(1..9),
        pol in 0_u8..2,
        to_pol in 0_u8..2,
        displacement in prop_oneof![
            prop::array::uniform3(-2.0_f64..2.0),
            (-2.0_f64..2.0).prop_map(|z| [0.0, 0.0, z]),
            Just([0.0; 3]),
        ],
        k in complex(0.2).prop_map(|k| k + 1.2),
        helicity in any::<bool>(),
        radial in radial(),
    ) {
        let to = Mode { l: lambda, m: mu, pol: to_pol };
        check_translation_inversion(to, Mode { l, m, pol }, displacement, k, helicity, radial)?;
    }
}

/// A converted cylindrical wave reproduces the source field, the conversion is
/// translation invariant, and its position pullback matches a central difference.
fn check_conversion_field(m: i32, kz: f64, x: f64, helicity: bool) -> Result<(), TestCaseError> {
    let destination = spherical_basis(8, [x, 0.1, -0.2]);
    let source = cw::Basis {
        modes: vec![(0, cw::Mode { kz, m, pol: 1 })],
        positions: vec![[0.2, -0.1, 0.1]],
    };
    let ks = [Complex::new(1.3, 0.1); 2];
    let convert = |destination: crate::sw::Basis| {
        cw::to_sw_matrix(destination, source.clone(), ks, helicity).unwrap()
    };
    let (converted, residual) = convert(destination.clone());
    let points = vec![[x + 0.1, 0.2, -0.1]];
    let (actual, _) = fields::field(
        destination.clone(),
        converted.as_slice().to_vec(),
        points.clone(),
        ks,
        helicity,
        Radial::Regular,
    )
    .unwrap();
    let (expected, _) = fields::field(
        source.clone(),
        vec![Complex::new(1.0, 0.0)],
        points,
        ks,
        helicity,
        Radial::Regular,
    )
    .unwrap();
    prop_assert_close!(actual, expected, 1e-10);
    let g = DMatrix::from_element(converted.nrows(), 1, Complex::new(0.2, 0.1));
    let gradient = residual.pullback(&g).unwrap();
    for (d, s) in gradient.destination[0].iter().zip(gradient.source[0]) {
        prop_assert_close!(*d, -s, 1e-12);
    }
    let numeric = central(1e-5, |step| {
        let mut shifted = destination.clone();
        shifted.positions[0][0] += step;
        re_dot(&g, &convert(shifted).0)
    });
    prop_assert_close!(gradient.destination[0][0], numeric, 1e-7);
    Ok(())
}

/// Rotations compose with their inverse to the identity, their Euler-angle pullback
/// matches a central difference, and Wigner d composes additively in the polar angle.
fn check_rotation_group(l: i32, angles: [f64; 3]) -> Result<(), TestCaseError> {
    let basis = crate::sw::Basis {
        modes: (-l..=l).map(|m| (0, Mode { l, m, pol: 1 })).collect(),
        positions: vec![[0.0; 3]],
    };
    let rotate = |angles| rotation::sw_rotation(&basis, &basis, angles).unwrap();
    let [phi, theta, psi] = angles;
    let forward = rotate(angles);
    let inverse = rotate([-psi, -theta, -phi]);
    let n = forward.value().nrows();
    prop_assert_close!(
        forward.value() * inverse.value(),
        DMatrix::identity(n, n),
        1e-11
    );
    let g = DMatrix::from_fn(n, n, |i, j| {
        Complex::new(if i == j { 0.3 } else { -0.2 }, 0.1)
    });
    let direction = [0.2, -0.1, 0.3];
    let numeric = central(1e-5, |step| {
        let angles = std::array::from_fn(|i| angles[i] + step * direction[i]);
        re_dot(&g, rotate(angles).value())
    });
    let analytic = forward.pullback(&g).unwrap();
    prop_assert_close!(numeric, dot(analytic, direction), 1e-8);
    let a = special::wigner_small_d_matrix(l, theta).unwrap();
    let b = special::wigner_small_d_matrix(l, phi).unwrap();
    let ab = special::wigner_small_d_matrix(l, theta + phi).unwrap();
    prop_assert_close!(a * b, ab, 1e-11);
    Ok(())
}

/// A spherical rotation is block diagonal in position, degree and polarization with
/// the Wigner D elements `D^l_(mu m)(phi, theta, psi)` as blocks, in any mode order,
/// and its Euler-angle pullback is the sum of the elementwise Wigner pullbacks. The
/// blocks come from the eigendecomposed small-d matrix and its ladder derivative, the
/// elements from the Jacobi recurrence.
fn check_rotation_blocks(angles: [f64; 3], rows: &[usize], seed: f64) -> Result<(), TestCaseError> {
    let modes = crate::sw::modes(2).unwrap();
    let source = crate::sw::Basis {
        modes: (0..2)
            .flat_map(|pidx| modes.iter().map(move |&mode| (pidx, mode)))
            .collect(),
        positions: vec![[0.0; 3], [1.0, 0.0, 0.0]],
    };
    let destination = crate::sw::Basis {
        modes: rows.iter().map(|&i| source.modes[i]).collect(),
        positions: source.positions.clone(),
    };
    let rotation = rotation::sw_rotation(&destination, &source, angles).unwrap();
    let g = patterned(rows.len(), source.modes.len(), seed);
    let mut labels = Vec::new();
    let mut cotangent = Vec::new();
    let mut expected = DMatrix::zeros(rows.len(), source.modes.len());
    for (i, &(p, to)) in destination.modes.iter().enumerate() {
        for (j, &(q, from)) in source.modes.iter().enumerate() {
            if (p, to.l, to.pol) == (q, from.l, from.pol) {
                labels.push([to.l, to.m, from.m]);
                cotangent.push(g[(i, j)]);
                expected[(i, j)] =
                    special::wigner_d(to.l, to.m, from.m, angles.map(Complex::from)).unwrap();
            }
        }
    }
    prop_assert_close!(rotation.value(), &expected, 1e-13);
    let (_, elements) =
        special::wigner_d_array(labels, angles.map(|angle| vec![Complex::from(angle)])).unwrap();
    let expected = elements.pullback(&cotangent).unwrap().map(|g| g[0].re);
    prop_assert_close!(rotation.pullback(&g).unwrap(), expected, 1e-12);
    Ok(())
}

/// A cylindrical rotation is diagonal in position, axial wavenumber, order and
/// polarization with the phases `exp(-i m (phi + psi))`, in any mode order; the axial
/// labels `-0.0` and `0.0` are equal. Its pullback is `d/dphi = d/dpsi = -i m D`
/// (theta is fixed at zero).
fn check_cylindrical_rotation(
    phi: f64,
    psi: f64,
    rows: &[usize],
    seed: f64,
) -> Result<(), TestCaseError> {
    let source = cw::Basis {
        modes: (0..2)
            .flat_map(|pidx| [0.0, 0.3].map(|kz| (pidx, kz)))
            .flat_map(|(pidx, kz)| (-2..=2).map(move |m| (pidx, kz, m)))
            .flat_map(|(pidx, kz, m)| [1, 0].map(|pol| (pidx, cw::Mode { kz, m, pol })))
            .collect(),
        positions: vec![[0.0; 3], [1.0, 0.0, 0.0]],
    };
    // Destination rows are source modes, with the axial label 0.0 written as -0.0.
    let destination = cw::Basis {
        modes: rows
            .iter()
            .map(|&i| {
                let (pidx, mode) = source.modes[i];
                let kz = if mode.kz > 0.0 { mode.kz } else { -0.0 };
                (pidx, cw::Mode { kz, ..mode })
            })
            .collect(),
        positions: source.positions.clone(),
    };
    let rotation = rotation::cw_rotation(&destination, &source, [phi, 0.0, psi]).unwrap();
    let g = patterned(rows.len(), source.modes.len(), seed);
    let mut expected = DMatrix::zeros(rows.len(), source.modes.len());
    let mut gradient = 0.0;
    // The source modes are distinct, so row `i` couples only to column `rows[i]`.
    for (i, &j) in rows.iter().enumerate() {
        let m = f64::from(source.modes[j].1.m);
        let value = (-Complex::i() * m * (phi + psi)).exp();
        expected[(i, j)] = value;
        gradient += (g[(i, j)].conj() * (-Complex::i()) * m * value).re;
    }
    prop_assert_close!(rotation.value(), &expected, 1e-15);
    prop_assert_close!(
        rotation.pullback(&g).unwrap(),
        [gradient, 0.0, gradient],
        1e-12
    );
    Ok(())
}

/// The expansion matrix of `source` modes about `from` in `destination` modes about `to`.
fn expansion(
    (destination, to): (&[Mode], [f64; 3]),
    (source, from): (&[Mode], [f64; 3]),
    helicity: bool,
    radial: Radial,
) -> DMatrix<Complex> {
    let basis = |modes: &[Mode], position| crate::sw::Basis {
        modes: modes.iter().map(|&mode| (0, mode)).collect(),
        positions: vec![position],
    };
    // A chiral medium in the helicity basis, an achiral one in the parity basis.
    let ks = if helicity {
        [Complex::new(1.2, 0.05), Complex::new(1.4, 0.05)]
    } else {
        [Complex::new(1.3, 0.05); 2]
    };
    crate::sw::expansion(
        basis(destination, to),
        basis(source, from),
        ks,
        helicity,
        radial,
    )
    .unwrap()
    .0
}

/// Regular translations form a group: translating by `b` and then by `a` through
/// the modes up to degree 10 equals translating by `a + b`, in a chiral medium for
/// helicity modes. The neglected degrees decay like `(k |b| / 2)^p / p!`.
fn check_regular_composition(
    a: [f64; 3],
    b: [f64; 3],
    helicity: bool,
) -> Result<(), TestCaseError> {
    let outer = crate::sw::modes(2).unwrap();
    let middle = crate::sw::modes(10).unwrap();
    let sum = std::array::from_fn(|i| a[i] + b[i]);
    let direct = expansion((&outer, sum), (&outer, [0.0; 3]), helicity, Radial::Regular);
    let composed = expansion((&outer, sum), (&middle, b), helicity, Radial::Regular)
        * expansion((&middle, b), (&outer, [0.0; 3]), helicity, Radial::Regular);
    prop_assert_close!(&composed, &direct, 1e-12 * (1.0 + direct.norm()));
    Ok(())
}

/// Addition theorem of outgoing waves: for `|a| < |b|`, an outgoing translation by
/// `b` followed by a regular translation by `a` through the modes up to degree 14
/// equals the outgoing translation by `a + b`, also with `b` on the polar axis. With
/// `|a| <= 0.052` and `|b| >= 1` the neglected degrees contribute below `1e-13`.
fn check_outgoing_addition(
    a: [f64; 3],
    [distance, theta, phi]: [f64; 3],
    helicity: bool,
) -> Result<(), TestCaseError> {
    let outer = crate::sw::modes(2).unwrap();
    let middle = crate::sw::modes(14).unwrap();
    let sine = if theta % std::f64::consts::PI == 0.0 {
        0.0
    } else {
        theta.sin()
    };
    let b = [
        distance * sine * phi.cos(),
        distance * sine * phi.sin(),
        distance * theta.cos(),
    ];
    let sum = std::array::from_fn(|i| a[i] + b[i]);
    let direct = expansion(
        (&outer, sum),
        (&outer, [0.0; 3]),
        helicity,
        Radial::Singular,
    );
    let composed = expansion((&outer, sum), (&middle, b), helicity, Radial::Regular)
        * expansion((&middle, b), (&outer, [0.0; 3]), helicity, Radial::Singular);
    prop_assert_close!(&composed, &direct, 1e-12 * direct.norm());
    Ok(())
}

/// Translations are rotation equivariant: with `D` the rotation of the Euler angles
/// and `R = Rz(phi) Ry(theta) Rz(psi)`, `D T(d) D^H = T(R d)`, also for displacements
/// on the polar axis. This fixes the rotation direction relative to the translation.
fn check_rotation_equivariance(
    [phi, theta, psi]: [f64; 3],
    displacement: [f64; 3],
    helicity: bool,
    radial: Radial,
) -> Result<(), TestCaseError> {
    let modes = crate::sw::modes(3).unwrap();
    let basis = crate::sw::Basis {
        modes: modes.iter().map(|&mode| (0, mode)).collect(),
        positions: vec![[0.0; 3]],
    };
    let rotation = rotation::sw_rotation(&basis, &basis, [phi, theta, psi])
        .unwrap()
        .value()
        .clone();
    let rotated = test_support::rotate([phi, theta, psi], displacement);
    let origin = [0.0; 3];
    let translated = expansion((&modes, displacement), (&modes, origin), helicity, radial);
    let expected = expansion((&modes, rotated), (&modes, origin), helicity, radial);
    let actual = &rotation * translated * rotation.adjoint();
    prop_assert_close!(&actual, &expected, 1e-12 * (1.0 + expected.norm()));
    Ok(())
}

/// Graf's addition theorem for cylindrical waves: the translations by `b` and then by
/// `a` summed over the orders `-40..=40` equal the translation by `a + b`; for
/// outgoing waves the second translation is regular and `|a_rho| < |b_rho|`. `b` is
/// given in cylindrical coordinates `[rho, phi, z]`.
fn check_graf_addition(
    m: i32,
    mu: i32,
    a: [f64; 3],
    [b_rho, b_phi, b_z]: [f64; 3],
    radial: Radial,
) -> Result<(), TestCaseError> {
    let b = [b_rho * b_phi.cos(), b_rho * b_phi.sin(), b_z];
    let k = Complex::new(1.3, 0.05);
    let mode = |m| cw::Mode { kz: 0.4, m, pol: 1 };
    let sum = std::array::from_fn(|i| a[i] + b[i]);
    let expected = cw::cartesian_translation(mode(mu), mode(m), k, sum, radial).unwrap();
    let composed: Complex = (-40..=40)
        .map(|q| {
            let first = cw::cartesian_translation(mode(q), mode(m), k, b, radial).unwrap();
            let second =
                cw::cartesian_translation(mode(mu), mode(q), k, a, Radial::Regular).unwrap();
            second.value * first.value
        })
        .sum();
    prop_assert_close!(
        composed,
        expected.value,
        1e-12 * (1.0 + expected.value.norm())
    );
    Ok(())
}

/// Ladder identities of the scalar waves `psi_p^m = z_p(k r) P_p^m(cos theta)
/// exp(i m phi)` of `sw::harmonic`, with Condon-Shortley phase and `psi = 0` for
/// `|m| > p`:
/// - `d/dz psi_p^m = k / (2p + 1) [(p + m) psi_(p-1)^m - (p - m + 1) psi_(p+1)^m]`,
/// - `(d/dx + i d/dy) psi_p^m = k / (2p + 1) [psi_(p-1)^(m+1) + psi_(p+1)^(m+1)]`,
/// - `(d/dx - i d/dy) psi_p^m = -k / (2p + 1) [a psi_(p-1)^(m-1) + b psi_(p+1)^(m-1)]`
///   with `a = (p + m)(p + m - 1)` and `b = (p - m + 1)(p - m + 2)`,
/// - `k d/dk psi = r . grad psi`.
///
/// They check every derivative component exactly, on the polar axis, at the origin
/// (whose degree-one gradients are tabulated) and on both sides of the radial series,
/// at the point of spherical coordinates `[radius, cos theta, phi]`.
fn check_harmonic_ladders(
    p: i32,
    m: i32,
    k: Complex,
    [radius, cosine, phi]: [f64; 3],
    radial: Radial,
) -> Result<(), TestCaseError> {
    // Outgoing waves are singular at the origin.
    let radius = if radial == Radial::Singular && radius == 0.0 {
        0.7
    } else {
        radius
    };
    let sine = (1.0 - cosine * cosine).sqrt();
    let position = [
        radius * sine * phi.cos(),
        radius * sine * phi.sin(),
        radius * cosine,
    ];
    let r = position.iter().map(|v| v * v).sum::<f64>().sqrt();
    let psi = |p: i32, m: i32| {
        if p < 0 || m.abs() > p {
            return crate::sw::CartesianTranslation::default();
        }
        let jet = special::spherical_radial(p.unsigned_abs(), k * r, radial).unwrap();
        crate::sw::harmonic(p, m, k, special::direction(position), jet)
    };
    let wave = psi(p, m);
    let (n, order) = (f64::from(p), f64::from(m));
    let c = k / (2.0 * n + 1.0);
    let i = Complex::i();
    let ladders = [
        (
            wave.position[2],
            [
                c * (n + order) * psi(p - 1, m).value,
                -c * (n - order + 1.0) * psi(p + 1, m).value,
            ],
        ),
        (
            wave.position[0] + i * wave.position[1],
            [c * psi(p - 1, m + 1).value, c * psi(p + 1, m + 1).value],
        ),
        (
            wave.position[0] - i * wave.position[1],
            [
                -c * (n + order) * (n + order - 1.0) * psi(p - 1, m - 1).value,
                -c * (n - order + 1.0) * (n - order + 2.0) * psi(p + 1, m - 1).value,
            ],
        ),
    ];
    // The combinations of Cartesian components may cancel far below the components.
    let gradient = wave.position.iter().map(|d| d.norm()).sum::<f64>();
    for (derivative, [lower, upper]) in ladders {
        let scale = gradient + lower.norm() + upper.norm();
        prop_assert_close!(derivative, lower + upper, 1e-12 * scale + f64::MIN_POSITIVE);
    }
    let spatial: Complex = (0..3).map(|a| position[a] * wave.position[a]).sum();
    let scale = r * gradient + (k * wave.k).norm();
    prop_assert_close!(k * wave.k, spatial, 1e-12 * scale + f64::MIN_POSITIVE);
    Ok(())
}

/// Spatial inversion `d -> -d` maps parity modes to themselves with the sign
/// `s = (-1)^(l + lambda)`, flipped for M-N couplings, and exchanges the helicities
/// with `s = (-1)^(l + lambda)`: `T(-d) = s T'(d)`, `grad T(-d) = -s grad T'(d)` and
/// `d/dk T(-d) = s d/dk T'(d)`, where `T'` couples the image modes.
fn check_translation_inversion(
    to: Mode,
    from: Mode,
    displacement: [f64; 3],
    k: Complex,
    helicity: bool,
    radial: Radial,
) -> Result<(), TestCaseError> {
    let (image, sign) = if helicity {
        let flip = |mode: Mode| Mode {
            pol: 1 - mode.pol,
            ..mode
        };
        ((flip(to), flip(from)), parity(to.l + from.l))
    } else {
        let cross = i32::from(to.pol != from.pol);
        ((to, from), parity(to.l + from.l + cross))
    };
    let inverted =
        cartesian_translation(to, from, k, displacement.map(|d| -d), helicity, radial).unwrap();
    let expected =
        cartesian_translation(image.0, image.1, k, displacement, helicity, radial).unwrap();
    let tolerance = 1e-14 * (1.0 + expected.value.norm() + expected.k.norm());
    prop_assert_close!(inverted.value, sign * expected.value, tolerance);
    prop_assert_close!(inverted.k, sign * expected.k, tolerance);
    let gradient = expected.position.map(|d| -sign * d);
    prop_assert_close!(inverted.position, gradient, tolerance);
    Ok(())
}

/// Cylindrical helicity waves solve Maxwell's equations, on the axis for regular waves.
fn check_cylindrical_field_maxwell(
    mode: cw::Mode,
    x: f64,
    y: f64,
    radial: Radial,
) -> Result<(), TestCaseError> {
    let k = Complex::new(1.3, 0.1);
    let axis = if radial == Radial::Singular {
        [x, y, 0.3]
    } else {
        [0.0, 0.0, 0.3]
    };
    for position in [[x, y, 0.3], axis] {
        let wave = fields::cylindrical_wave(mode, k, position, true, radial).unwrap();
        check_helicity_maxwell(&wave, k, mode.pol, 1e-10)?;
    }
    Ok(())
}

/// Euler identity `r · ∇A = k ∂A/∂k` of spherical translation coefficients, also on the
/// polar axis, and their invariance under `(r, k) -> (s r, k / s)`, with `∇A / s` and
/// `s ∂A/∂k`. Degrees reach 80 and `s` 1e-12 to 1e12, where `|s r|^p` under- or
/// overflows unless the angular part is evaluated on the unit sphere.
fn check_translation_scale(
    to: Mode,
    from: Mode,
    k: Complex,
    position: [f64; 3],
    helicity: bool,
    radial: Radial,
    s: f64,
    tolerance: f64,
) -> Result<(), TestCaseError> {
    let translation =
        |k, position| cartesian_translation(to, from, k, position, helicity, radial).unwrap();
    let result = translation(k, position);
    let scale = 1.0
        + result.value.norm()
        + (k * result.k).norm()
        + result.position.iter().map(|g| g.norm()).sum::<f64>();
    let spatial: Complex = result
        .position
        .iter()
        .zip(position)
        .map(|(g, r)| g * r)
        .sum();
    prop_assert_close!(spatial, k * result.k, tolerance * scale);
    let scaled = translation(k / s, position.map(|r| r * s));
    prop_assert_close!(scaled.value, result.value, tolerance * scale);
    prop_assert_close!(
        scaled.position.map(|g| g * s),
        result.position,
        tolerance * scale
    );
    prop_assert_close!(k * scaled.k / s, k * result.k, tolerance * scale);
    Ok(())
}

/// Checks the source-free Maxwell equations of a helicity wave with wave number `k`:
/// `∇×E = (2 pol - 1) k E` and `∇·E = 0`, to `tolerance (1 + |E|)`.
fn check_helicity_maxwell(
    wave: &fields::VectorWave,
    k: Complex,
    pol: u8,
    tolerance: f64,
) -> Result<(), TestCaseError> {
    // `position[c][a]` is `∂E_c / ∂x_a`: the gradients of E_x, E_y and E_z.
    let [ex, ey, ez] = wave.position;
    let curl = [ez[1] - ey[2], ex[2] - ez[0], ey[0] - ex[1]];
    let scale = wave.value.iter().map(Complex::norm_sqr).sum::<f64>().sqrt();
    let tolerance = tolerance * (1.0 + scale);
    let helicity = special::helicity_sign(pol);
    for (curl, value) in curl.into_iter().zip(wave.value) {
        prop_assert_close!(curl, helicity * k * value, tolerance);
    }
    prop_assert_close!(ex[0] + ey[1] + ez[2], Complex::default(), tolerance);
    Ok(())
}

/// Spherical helicity waves solve Maxwell's equations in both hemispheres, on both
/// poles and, for regular waves, at the origin, and obey the Euler identity
/// `r · ∇E = k ∂E/∂k`.
fn check_vector_wave_maxwell(
    mode: Mode,
    [x, y, z]: [f64; 3],
    radial: Radial,
) -> Result<(), TestCaseError> {
    let k = Complex::new(1.2, 0.1);
    let origin = if radial == Radial::Singular { z } else { 0.0 };
    let points = [
        [x, y, z],
        [x, y, -z],
        [0.0, 0.0, z],
        [0.0, 0.0, -z],
        [0.0, 0.0, origin],
    ];
    for position in points {
        let wave = fields::spherical_wave(mode, k, position, true, radial).unwrap();
        check_helicity_maxwell(&wave, k, mode.pol, 1e-10)?;
        let scale = 1.0 + wave.value.iter().map(Complex::norm_sqr).sum::<f64>().sqrt();
        for (jacobian, dk) in wave.position.iter().zip(wave.k) {
            let spatial: Complex = jacobian.iter().zip(position).map(|(d, r)| d * r).sum();
            prop_assert_close!(spatial, k * dk, 1e-10 * scale);
        }
    }
    Ok(())
}

/// Euler identity and scale invariance of cylindrical translation coefficients.
fn check_cylindrical_translation_scale(
    m: i32,
    mu: i32,
    position: [f64; 3],
    radial: Radial,
) -> Result<(), TestCaseError> {
    let k = Complex::new(1.2, 0.1);
    let kz = 0.3;
    let to = cw::Mode { kz, m: mu, pol: 1 };
    let source = cw::Mode { kz, m, pol: 1 };
    let jet = cw::cartesian_translation(to, source, k, position, radial).unwrap();
    let scale = 1.0 + jet.value.norm();
    let spatial: Complex = jet.position.iter().zip(position).map(|(g, p)| g * p).sum();
    prop_assert_close!(spatial, k * jet.k + kz * jet.kz, 1e-9 * scale);
    let axial = |mode| cw::Mode {
        kz: kz / 1.3,
        ..mode
    };
    let scaled = cw::cartesian_translation(
        axial(to),
        axial(source),
        k / 1.3,
        position.map(|v| v * 1.3),
        radial,
    )
    .unwrap();
    prop_assert_close!(scaled.value, jet.value, 1e-10 * scale);
    Ok(())
}

/// Inputs of a weighted field evaluation in either wave family.
#[derive(Clone, Debug)]
struct FieldCase {
    basis: basis::MultipoleBasis,
    points: Vec<[f64; 3]>,
    ks: [Complex; 2],
    helicity: bool,
    radial: Radial,
}

/// Spherical (`lmax` 1 to 3) or cylindrical (`|m| <= 2`, two axial wavenumbers)
/// bases at one or two positions, with both polarizations of each mode adjacent. The
/// points add the poles of the first position (spherical), and for regular waves the
/// position itself or a point on its axis, which take the small-argument branches.
/// Chiral wavenumbers occur only in the helicity basis. All wavenumbers are lossy,
/// which keeps finite differences of the cylindrical transverse wavenumber off the
/// branch cut of its square root.
fn field_case() -> impl Strategy<Value = FieldCase> {
    (
        any::<bool>(),
        0_i32..3,
        prop::collection::vec(prop::array::uniform3(-0.5_f64..0.5), 1..=2),
        prop::array::uniform2(-0.4_f64..0.4),
        prop::collection::vec(prop::array::uniform3(-2.0_f64..2.0), 1..=3),
        (
            1.0_f64..1.5,
            0.01_f64..0.1,
            prop_oneof![Just(0.0), 0.05_f64..0.3],
        ),
        radial(),
        any::<bool>(),
    )
        .prop_map(
            |(
                spherical,
                order,
                positions,
                kzs,
                mut points,
                (k, loss, chirality),
                radial,
                helicity,
            )| {
                let [x, y, z] = positions[0];
                let basis = if spherical {
                    let modes = crate::sw::modes(order.unsigned_abs() + 1).unwrap();
                    points.extend([[x, y, z + 0.9], [x, y, z - 0.9]]);
                    if radial == Radial::Regular {
                        points.push(positions[0]);
                    }
                    basis::MultipoleBasis::Spherical(crate::sw::Basis {
                        modes: (0..positions.len())
                            .flat_map(|pidx| modes.iter().map(move |&mode| (pidx, mode)))
                            .collect(),
                        positions,
                    })
                } else {
                    if radial == Radial::Regular {
                        points.push([x, y, 0.7]);
                    }
                    basis::MultipoleBasis::Cylindrical(cw::Basis {
                        modes: (0..positions.len())
                            .flat_map(|pidx| {
                                kzs.into_iter().flat_map(move |kz| {
                                    (-order..=order).flat_map(move |m| {
                                        [1, 0].map(|pol| (pidx, cw::Mode { kz, m, pol }))
                                    })
                                })
                            })
                            .collect(),
                        positions,
                    })
                };
                let chirality = if helicity { chirality } else { 0.0 };
                FieldCase {
                    basis,
                    points,
                    ks: [-chirality, chirality].map(|c| Complex::new(k + c, loss)),
                    helicity,
                    radial,
                }
            },
        )
}

impl FieldCase {
    fn field(&self, coefficients: &[Complex]) -> (Vec<[Complex; 3]>, fields::FieldResidual) {
        let basis = self.basis.clone();
        let points = self.points.clone();
        fields::field(
            basis,
            coefficients.to_vec(),
            points,
            self.ks,
            self.helicity,
            self.radial,
        )
        .unwrap()
    }

    fn operator(&self) -> (DMatrix<Complex>, fields::OperatorResidual) {
        let (basis, points) = (self.basis.clone(), self.points.clone());
        fields::operator(basis, points, self.ks, self.helicity, self.radial).unwrap()
    }

    /// Complete pullback; cylindrical bases add the per-mode axial gradients.
    fn pullback(
        &self,
        field: fields::FieldResidual,
        g: &[[Complex; 3]],
    ) -> (fields::FieldGradient, Vec<f64>) {
        match self.basis {
            basis::MultipoleBasis::Spherical(_) => (field.pullback(g).unwrap(), Vec::new()),
            basis::MultipoleBasis::Cylindrical(_) => field.pullback_axial(g).unwrap(),
        }
    }

    /// Axial wavenumber of each mode; empty for spherical bases.
    fn kzs(&self) -> Vec<f64> {
        match &self.basis {
            basis::MultipoleBasis::Spherical(_) => Vec::new(),
            basis::MultipoleBasis::Cylindrical(basis) => {
                basis.modes.iter().map(|m| m.1.kz).collect()
            }
        }
    }

    /// The same case with its modes reordered: mode `j` is the original `order[j]`.
    fn permuted(&self, order: &[usize]) -> Self {
        let basis = match &self.basis {
            basis::MultipoleBasis::Spherical(basis) => {
                basis::MultipoleBasis::Spherical(crate::sw::Basis {
                    modes: order.iter().map(|&i| basis.modes[i]).collect(),
                    positions: basis.positions.clone(),
                })
            }
            basis::MultipoleBasis::Cylindrical(basis) => {
                basis::MultipoleBasis::Cylindrical(cw::Basis {
                    modes: order.iter().map(|&i| basis.modes[i]).collect(),
                    positions: basis.positions.clone(),
                })
            }
        };
        Self {
            basis,
            ..self.clone()
        }
    }

    /// The case moved by `t` along the real directions of points, positions and
    /// per-mode axial wavenumbers and the complex direction of the wavenumbers.
    fn moved(&self, t: f64, direction: &FieldDirection) -> Self {
        let shift = |values: &[[f64; 3]], d: &[[f64; 3]]| -> Vec<[f64; 3]> {
            values
                .iter()
                .zip(d)
                .map(|(v, d)| std::array::from_fn(|a| v[a] + t * d[a]))
                .collect()
        };
        let basis = match &self.basis {
            basis::MultipoleBasis::Spherical(basis) => {
                basis::MultipoleBasis::Spherical(crate::sw::Basis {
                    modes: basis.modes.clone(),
                    positions: shift(&basis.positions, &direction.positions),
                })
            }
            basis::MultipoleBasis::Cylindrical(basis) => {
                basis::MultipoleBasis::Cylindrical(cw::Basis {
                    modes: basis
                        .modes
                        .iter()
                        .zip(&direction.kzs)
                        .map(|(&(pidx, mode), d)| {
                            (
                                pidx,
                                cw::Mode {
                                    kz: mode.kz + t * d,
                                    ..mode
                                },
                            )
                        })
                        .collect(),
                    positions: shift(&basis.positions, &direction.positions),
                })
            }
        };
        Self {
            basis,
            points: shift(&self.points, &direction.points),
            ks: std::array::from_fn(|i| self.ks[i] + t * direction.ks[i]),
            ..self.clone()
        }
    }
}

/// A perturbation direction of every differentiable field input.
#[derive(Debug)]
struct FieldDirection {
    points: Vec<[f64; 3]>,
    positions: Vec<[f64; 3]>,
    ks: [Complex; 2],
    kzs: Vec<f64>,
    coefficients: Vec<Complex>,
}

/// Amplitudes, an output cotangent and a perturbation direction of a field case.
#[derive(Debug)]
struct FieldInputs {
    coefficients: Vec<Complex>,
    cotangent: Vec<[Complex; 3]>,
    direction: FieldDirection,
}

impl FieldInputs {
    /// Independent entries of modulus at most `√2` sized to `case`. The direction
    /// keeps equal wavenumbers equal, since parity bases require an achiral medium.
    fn strategy(case: &FieldCase) -> impl Strategy<Value = Self> + use<> {
        let (modes, samples) = (case.basis.len(), case.points.len());
        let real = |rows| prop::collection::vec(prop::array::uniform3(-1.0_f64..1.0), rows);
        let chiral = case.ks[0] != case.ks[1];
        (
            complex_vec(modes, 1.0),
            prop::collection::vec(prop::array::uniform3(complex(1.0)), samples),
            real(samples),
            real(case.basis.positions().len()),
            prop::array::uniform2(complex(1.0)),
            prop::collection::vec(-1.0_f64..1.0, modes),
            complex_vec(modes, 1.0),
        )
            .prop_map(
                move |(coefficients, cotangent, points, positions, [k0, k1], kzs, directions)| {
                    Self {
                        coefficients,
                        cotangent,
                        direction: FieldDirection {
                            points,
                            positions,
                            ks: [k0, if chiral { k1 } else { k0 }],
                            kzs,
                            coefficients: directions,
                        },
                    }
                },
            )
    }
}

/// The weighted field of spherical or cylindrical waves, and its operator:
///
/// - the operator applied to the amplitudes is the field, and each column is the
///   single wave evaluated on its own;
/// - reordering the modes (which separates the polarizations of a mode and defeats
///   the shared per-sample evaluations) reorders the operator columns exactly and
///   leaves the field and its geometry gradients unchanged;
/// - in an achiral medium, helicity amplitudes `(a₀, a₁)` give the parity field with
///   N amplitude `(a₁ + a₀)/√2` and M amplitude `(a₁ - a₀)/√2`;
/// - the pullback reproduces the loss through the amplitudes, sums to zero over
///   points and positions (translation invariance), obeys the Euler identity of joint
///   coordinate and (axial) wavenumber scaling, equals the operator pullback with
///   the outer-product cotangent, and matches a fourth-order finite difference
///   along a random direction of every input.
fn check_field_evaluation(
    case: &FieldCase,
    inputs: &FieldInputs,
    keys: &[u32],
) -> Result<(), TestCaseError> {
    let cylindrical = matches!(case.basis, basis::MultipoleBasis::Cylindrical(_));
    for point in &case.points {
        for position in case.basis.positions() {
            let offset: [f64; 3] = std::array::from_fn(|a| point[a] - position[a]);
            let distance = if cylindrical {
                offset[0].hypot(offset[1])
            } else {
                offset[0].hypot(offset[1]).hypot(offset[2])
            };
            // Outgoing waves are singular at their position (axis).
            prop_assume!(case.radial == Radial::Regular || distance > 0.4);
        }
    }
    let modes = case.basis.len();
    let samples = case.points.len();
    let FieldInputs {
        coefficients,
        cotangent: g,
        direction,
    } = inputs;
    let (matrix, operator) = case.operator();
    let (field, _) = case.field(coefficients);
    let value: Vec<_> = field.iter().flatten().copied().collect();
    let scale = 1.0 + matrix.norm() * DVector::from_column_slice(coefficients).norm();
    let contracted = &matrix * DVector::from_column_slice(coefficients);
    prop_assert_close!(contracted.as_slice(), value.as_slice(), 1e-14 * scale);
    for (i, column) in matrix.column_iter().enumerate() {
        for (sample, point) in case.points.iter().enumerate() {
            let wave = match &case.basis {
                basis::MultipoleBasis::Spherical(basis) => {
                    let (pidx, mode) = basis.modes[i];
                    let position = std::array::from_fn(|a| point[a] - basis.positions[pidx][a]);
                    let k = case.ks[usize::from(mode.pol)];
                    fields::spherical_wave(mode, k, position, case.helicity, case.radial)
                }
                basis::MultipoleBasis::Cylindrical(basis) => {
                    let (pidx, mode) = basis.modes[i];
                    let position = std::array::from_fn(|a| point[a] - basis.positions[pidx][a]);
                    let k = case.ks[usize::from(mode.pol)];
                    fields::cylindrical_wave(mode, k, position, case.helicity, case.radial)
                }
            };
            let expected = wave.unwrap().value;
            let actual: [Complex; 3] = std::array::from_fn(|c| column[3 * sample + c]);
            let magnitude = expected.iter().map(|z| z.norm()).sum::<f64>();
            prop_assert_close!(actual, expected, 1e-14 * (1.0 + magnitude), "mode {}", i);
        }
    }

    // Reordered modes: exact operator columns, equal fields and gradients.
    let mut order: Vec<usize> = (0..modes).collect();
    order.sort_by_key(|&i| (case.basis.position_pol(i).1, keys[i]));
    let permuted = case.permuted(&order);
    let (permuted_matrix, _) = permuted.operator();
    for (j, &i) in order.iter().enumerate() {
        prop_assert_eq!(permuted_matrix.column(j), matrix.column(i), "mode {}", i);
    }
    let permuted_coefficients: Vec<_> = order.iter().map(|&i| coefficients[i]).collect();
    let (permuted_field, permuted_residual) = permuted.field(&permuted_coefficients);
    prop_assert_close!(&permuted_field, &field, 1e-14 * scale);
    let (permuted_gradient, permuted_axial) = permuted.pullback(permuted_residual, g);

    let loss = re_dot(g.iter().flatten(), &value);
    let (gradient, axial) = case.pullback(case.field(coefficients).1, g);
    let gradient_scale = scale * (1.0 + g.iter().flatten().map(|z| z.norm()).sum::<f64>());
    prop_assert_close!(
        re_dot(&gradient.coefficients, coefficients),
        loss,
        1e-14 * gradient_scale
    );
    let tolerance = 1e-13 * gradient_scale;
    for (j, &i) in order.iter().enumerate() {
        prop_assert_close!(
            permuted_gradient.coefficients[j],
            gradient.coefficients[i],
            tolerance
        );
        if cylindrical {
            prop_assert_close!(permuted_axial[j], axial[i], tolerance);
        }
    }
    prop_assert_close!(&permuted_gradient.points, &gradient.points, tolerance);
    prop_assert_close!(&permuted_gradient.positions, &gradient.positions, tolerance);
    prop_assert_close!(permuted_gradient.ks, gradient.ks, tolerance);

    // Operator pullback with the cotangent g ⊗ conj(c).
    let outer = DMatrix::from_fn(3 * samples, modes, |r, j| {
        g[r / 3][r % 3] * coefficients[j].conj()
    });
    let (operator_gradient, operator_axial) = if cylindrical {
        operator.pullback_axial(&outer).unwrap()
    } else {
        (operator.pullback(&outer).unwrap(), Vec::new())
    };
    prop_assert!(operator_gradient.coefficients.is_empty());
    prop_assert_close!(&operator_gradient.points, &gradient.points, tolerance);
    prop_assert_close!(&operator_gradient.positions, &gradient.positions, tolerance);
    prop_assert_close!(operator_gradient.ks, gradient.ks, tolerance);
    prop_assert_close!(&operator_axial, &axial, tolerance);

    // Translation invariance and the Euler identity of joint scaling.
    let geometry = || gradient.points.iter().chain(&gradient.positions).flatten();
    for axis in 0..3 {
        let total: f64 = gradient
            .points
            .iter()
            .chain(&gradient.positions)
            .map(|g| g[axis])
            .sum();
        prop_assert_close!(total, 0.0, tolerance, "axis {}", axis);
    }
    let coordinates: Vec<f64> = case
        .points
        .iter()
        .chain(case.basis.positions())
        .flatten()
        .copied()
        .collect();
    let kzs = case.kzs();
    let spatial = dot(geometry(), &coordinates);
    let spectral = re_dot(gradient.ks, case.ks) + dot(&axial, &kzs);
    let magnitude = dot(
        geometry().map(|g| g.abs()),
        coordinates.iter().map(|x| x.abs()),
    ) + gradient
        .ks
        .iter()
        .zip(case.ks)
        .map(|(g, k)| g.norm() * k.norm())
        .sum::<f64>()
        + dot(axial.iter().map(|g| g.abs()), kzs.iter().map(|kz| kz.abs()));
    prop_assert_close!(spatial, spectral, 1e-13 * (1.0 + magnitude));

    // Helicity to parity amplitudes; polarization 1 precedes 0 in each pair.
    if case.ks[0] == case.ks[1] && case.helicity {
        let parity: Vec<_> = coefficients
            .chunks_exact(2)
            .flat_map(|pair| [pair[0] + pair[1], pair[0] - pair[1]].map(|c| c * FRAC_1_SQRT_2))
            .collect();
        let parity_case = FieldCase {
            helicity: false,
            ..case.clone()
        };
        prop_assert_close!(&parity_case.field(&parity).0, &field, 1e-14 * scale);
    }

    // Fourth-order difference along a random direction of every input.
    let numeric = five_point(1e-4, |t| {
        let moved: Vec<_> = coefficients
            .iter()
            .zip(&direction.coefficients)
            .map(|(c, d)| c + t * d)
            .collect();
        let (value, _) = case.moved(t, direction).field(&moved);
        re_dot(g.iter().flatten(), value.iter().flatten())
    });
    let analytic = dot(
        gradient.points.iter().flatten(),
        direction.points.iter().flatten(),
    ) + dot(
        gradient.positions.iter().flatten(),
        direction.positions.iter().flatten(),
    ) + re_dot(gradient.ks, direction.ks)
        + re_dot(&gradient.coefficients, &direction.coefficients)
        + if cylindrical {
            dot(&axial, &direction.kzs)
        } else {
            0.0
        };
    let noise = g
        .iter()
        .flatten()
        .zip(&value)
        .map(|(g, v)| g.norm() * v.norm())
        .sum::<f64>();
    prop_assert_close!(
        numeric,
        analytic,
        1e-8 * (1.0 + analytic.abs()) + 1e-10 * noise
    );
    Ok(())
}
