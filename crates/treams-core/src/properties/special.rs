//! Legendre, Wigner and Bessel identities and their adjoints.

use std::f64::consts::PI;

use proptest::{prelude::*, test_runner::TestCaseError};

use nalgebra::DMatrix;

use crate::{
    Complex, rotation,
    special::{
        self, Angular, Bessel, Radial, SERIES_RADIUS, angular_array, angular_value, bessel,
        bessel_array, spherical_hankels, spherical_radial, spherical_radial_sequence,
        wigner_d_array, wigner_small_d, wigner3j,
    },
    sw::{self, Mode},
    test_support::{
        ALGEBRA_CASES, EXPENSIVE_CASES, degree_order, log_polar, log_uniform, prop_assert_close,
    },
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(ALGEBRA_CASES))]

    #[test]
    fn real_degree_legendre_recurrence_and_derivative(
        (floor, m) in degree_order(1..29), fraction in 0.05_f64..0.95, x in -0.99_f64..0.99,
    ) {
        check_real_degree_legendre(f64::from(floor + 1) + fraction, m, x)?;
    }

    #[test]
    fn real_degree_legendre_order_reflection(
        (floor, m) in (0_i32..40).prop_flat_map(|n| (Just(n), 1..=(n + 3).min(30))),
        fraction in 0.05_f64..0.95,
        x in -0.999_f64..0.999,
    ) {
        check_real_degree_reflection(f64::from(floor) + fraction, m, x)?;
    }

    #[test]
    fn real_degree_legendre_branch_continuity(
        (floor, m) in (0_i32..60).prop_flat_map(|n| (Just(n), -(n + 2).min(40)..=(n + 2).min(40))),
        fraction in 0.05_f64..0.95,
    ) {
        check_real_degree_branches(f64::from(floor) + fraction, m)?;
    }

    #[test]
    fn wigner_generator_and_euler_adjoint(
        (l, m, k) in degree_order(0..16).prop_flat_map(|(l, m)| (Just(l), Just(m), -l..=l)),
        x in -3.0_f64..3.0,
        y in -0.3_f64..0.3,
    ) {
        check_wigner_generator(l, m, k, Complex::new(x, y))?;
    }

    #[test]
    fn angular_legendre_recurrence_adjoint(
        (l, m) in degree_order(1..12), x in -0.8_f64..0.8, y in -0.3_f64..0.3,
    ) {
        check_angular_legendre(l, m, Complex::new(x, y))?;
    }

    #[test]
    fn bessel_wronskian_adjoint(
        z in log_polar(-3.0..0.7, -PI..PI),
        order in prop_oneof![(0_i32..=10).prop_map(f64::from), 0.0_f64..10.0],
        spherical in any::<bool>(),
        pair in prop_oneof![
            Just([Bessel::J, Bessel::Y]),
            Just([Bessel::H1, Bessel::H2]),
            Just([Bessel::J, Bessel::H1]),
        ],
    ) {
        check_bessel_wronskian(z, order, spherical, pair)?;
    }

    #[test]
    fn bessel_kinds_equation_and_reflection(
        z in log_polar(-3.0..1.2, -PI..PI),
        order in prop_oneof![(-20_i32..=20).prop_map(f64::from), -12.0_f64..12.0],
        spherical in any::<bool>(),
    ) {
        check_bessel_kinds(z, order, spherical)?;
    }

    #[test]
    fn spherical_series_and_wronskian(
        l in 0_u32..=40,
        // Just below and above the series switch.
        z in prop_oneof![
            log_polar(-3.0..(0.996 * SERIES_RADIUS).log10(), -PI..PI),
            log_polar(SERIES_RADIUS.log10()..1.5, -PI..PI),
        ],
    ) {
        check_spherical_series_and_wronskian(l, z)?;
    }

    #[test]
    fn spherical_bessel_parity_on_the_negative_real_axis(
        l in 0_u32..=30,
        x in log_uniform(-3.0..1.5),
        zero in prop_oneof![Just(0.0), Just(-0.0)],
    ) {
        check_spherical_parity(l, x, zero)?;
    }

    #[test]
    fn wigner_small_d_matrix_symmetries_and_unitarity(
        l in 0_i32..=20, theta in -6.0_f64..6.0, phi in -3.0_f64..3.0, psi in -3.0_f64..3.0,
    ) {
        check_wigner_small_d_matrix(l, [phi, theta, psi])?;
    }

    #[test]
    fn wigner3j_permutation_and_reflection_symmetry(symbol in wigner3j_symbol(90)) {
        check_wigner3j_symmetries(symbol)?;
    }

    #[test]
    fn wigner3j_orthogonality(
        (symbol, other) in wigner3j_symbol(60).prop_flat_map(|symbol| {
            let [j1, j2, _, _, _, m3] = symbol;
            (Just(symbol), (j1 - j2).abs().max(m3.abs())..=j1 + j2)
        }),
    ) {
        check_wigner3j_orthogonality(symbol, other)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(EXPENSIVE_CASES))]

    #[test]
    fn wigner_small_d_matches_the_generator_exponential(
        l in prop_oneof![0_i32..=30, 31_i32..=128], theta in -6.3_f64..6.3,
    ) {
        check_wigner_algorithms(l, theta)?;
    }
}

/// An order of degree `j`, biased to the extremes `±j` and `±(j - 1)`, where the
/// classically forbidden regions of the 3j recurrence are widest.
fn extreme_order(j: i32) -> impl Strategy<Value = i32> {
    let inner = (j - 1).max(0);
    prop_oneof![-j..=j, Just(j), Just(-j), Just(inner), Just(-inner)]
}

/// Admissible symbols `[j1, j2, j3, m1, m2, m3]` with `j1, j2 <= max`, extreme orders and
/// `j3` often within ten of its lower bound.
fn wigner3j_symbol(max: i32) -> impl Strategy<Value = [i32; 6]> {
    (0..=max, 0..=max)
        .prop_flat_map(|(j1, j2)| (Just(j1), Just(j2), extreme_order(j1), extreme_order(j2)))
        .prop_flat_map(|(j1, j2, m1, m2)| {
            let (lower, upper) = ((j1 - j2).abs().max((m1 + m2).abs()), j1 + j2);
            prop_oneof![lower..=upper.min(lower + 10), lower..=upper]
                .prop_map(move |j3| [j1, j2, j3, m1, m2, -m1 - m2])
        })
}

/// Three-term recurrence and derivative identity of real-degree Legendre functions;
/// `|m| < degree - 1` keeps both neighbouring degrees in the same order domain.
fn check_real_degree_legendre(degree: f64, m: i32, x: f64) -> Result<(), TestCaseError> {
    let (value, derivative) = special::ferrers_real_degree::<true>(degree, m, x).unwrap();
    let (lower, _) = special::ferrers_real_degree::<false>(degree - 1.0, m, x).unwrap();
    let (upper, _) = special::ferrers_real_degree::<false>(degree + 1.0, m, x).unwrap();
    let m = f64::from(m);
    let recurrence =
        (degree - m + 1.0) * upper - (2.0 * degree + 1.0) * x * value + (degree + m) * lower;
    let expected = (degree + m) * lower - degree * x * value;
    let scale = 1.0 + upper.abs() + lower.abs() + degree * value.abs();
    prop_assert_close!(recurrence, 0.0, 2e-11 * scale);
    let scale = 1.0 + expected.abs() + degree * value.abs();
    prop_assert_close!((1.0 - x * x) * derivative, expected, 1e-10 * scale);
    Ok(())
}

/// Order reflection `P_v^-m = (-1)^m P_v^m / prod_(j = 1-m)^m (v + j)` of real-degree
/// Ferrers functions, which ties the negative-order code to the positive one, also for
/// orders above the degree. The scale includes `(1 - x^2) P'/(v + 1)`, the local size of
/// the function next to its nodes.
fn check_real_degree_reflection(degree: f64, m: i32, x: f64) -> Result<(), TestCaseError> {
    let (positive, _) = special::ferrers_real_degree::<false>(degree, m, x).unwrap();
    let (negative, derivative) = special::ferrers_real_degree::<true>(degree, -m, x).unwrap();
    let product: f64 = (1 - m..=m).map(|j| degree + f64::from(j)).product();
    let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
    let expected = sign * positive / product;
    let scale =
        negative.abs() + expected.abs() + ((1.0 - x * x) * derivative).abs() / (degree + 1.0);
    prop_assert_close!(negative, expected, 1e-12 * scale);
    Ok(())
}

/// Real-degree Ferrers functions switch between the hypergeometric series and its
/// continuation at `x = -0.35`, and to the logarithmic series in `(1 + x)/2` where
/// `(v + |m| + 1)^2 (1 + x) = 4`; both sides agree at adjacent floating-point arguments.
fn check_real_degree_branches(degree: f64, m: i32) -> Result<(), TestCaseError> {
    let boundary = 4.0 / (degree + f64::from(m.abs()) + 1.0).powi(2) - 1.0;
    for x in [-0.35, boundary] {
        if x <= -1.0 || x > -0.35 {
            continue;
        }
        let below = f64::from_bits(x.to_bits() + 1);
        let (value, derivative) = special::ferrers_real_degree::<true>(degree, m, x).unwrap();
        let (other, _) = special::ferrers_real_degree::<false>(degree, m, below).unwrap();
        let scale = value.abs() + other.abs() + ((1.0 - x * x) * derivative).abs();
        prop_assert_close!(other, value, 1e-11 * scale, "x = {}", x);
    }
    Ok(())
}

/// The Euler-angle pullback of Wigner D equals the Lie-algebra generators applied to D.
fn check_wigner_generator(l: i32, m: i32, k: i32, theta: Complex) -> Result<(), TestCaseError> {
    let zero = vec![Complex::default()];
    let (value, residual) =
        wigner_d_array(vec![[l, m, k]], [zero.clone(), vec![theta], zero]).unwrap();
    let value = value[0];
    let ladder = |m: i32| 0.5 * f64::from(l * (l + 1) - m * (m + 1)).max(0.0).sqrt();
    let raised = wigner_small_d(l, m + 1, k, theta).unwrap();
    let lowered = wigner_small_d(l, m - 1, k, theta).unwrap();
    let derivatives = [
        -Complex::i() * f64::from(m) * value,
        ladder(m) * raised - ladder(m - 1) * lowered,
        -Complex::i() * f64::from(k) * value,
    ];
    let g = Complex::new(0.4, 0.2);
    let gradient = residual.pullback(&[g]).unwrap();
    prop_assert_eq!(gradient.len(), 3);
    for (actual, expected) in gradient.iter().zip(derivatives) {
        let tolerance = 1e-11 * (1.0 + expected.norm());
        prop_assert_close!(actual[0], g * expected.conj(), tolerance);
    }
    Ok(())
}

/// Degree recurrence and derivative identity of complex-argument Legendre functions, and
/// the vector-wave angular functions pi and tau in terms of them.
fn check_angular_legendre(l: i32, m: i32, z: Complex) -> Result<(), TestCaseError> {
    let (degree, order) = (f64::from(l), f64::from(m));
    let (p, residual) =
        angular_array(vec![degree], vec![order], vec![z], Angular::Legendre).unwrap();
    let p = p[0];
    let lower = angular_value(f64::from(l - 1), order, z, Angular::Legendre).unwrap();
    let upper = angular_value(f64::from(l + 1), order, z, Angular::Legendre).unwrap();
    let scale = 1.0 + p.norm() + lower.norm() + upper.norm();
    let recurrence =
        f64::from(l - m + 1) * upper - f64::from(2 * l + 1) * z * p + f64::from(l + m) * lower;
    prop_assert_close!(recurrence, Complex::default(), 1e-12 * scale);
    let g = Complex::new(0.4, 0.2);
    let derivative = residual.pullback(&[g]).unwrap()[0].conj() / g.conj();
    let expected = f64::from(l + m) * lower - f64::from(l) * z * p;
    prop_assert_close!((1.0 - z * z) * derivative, expected, 1e-12 * scale);
    // pi = m P / sin(theta) and tau = dP/d(theta) = -sin(theta) dP/dz.
    let sine = crate::numerics::complex_sqrt(1.0 - z * z);
    let pi = angular_value(degree, order, z, Angular::Pi).unwrap();
    let tau = angular_value(degree, order, z, Angular::Tau).unwrap();
    prop_assert_close!(pi * sine, order * p, 1e-12 * scale * (1.0 + order.abs()));
    let expected = -sine * derivative;
    prop_assert_close!(tau, expected, 1e-12 * (scale + expected.norm()));
    Ok(())
}

/// Bessel Wronskians `W(J, Y)`, `W(H1, H2)` and `W(J, H1)` of integer and real orders
/// from the series and AMOS branches, at every phase up to `|z| = 5`, and their argument
/// derivative through all four pullbacks, which evaluate second derivatives.
/// `W(H1, H2)` cancels between its products at small arguments, so tolerances scale
/// with the products.
fn check_bessel_wronskian(
    z: Complex,
    order: f64,
    spherical: bool,
    [first, second]: [Bessel; 2],
) -> Result<(), TestCaseError> {
    let call = |kind, derivative| {
        let (value, residual) =
            bessel_array(vec![order], vec![z], kind, spherical, derivative).unwrap();
        (value[0], residual)
    };
    let (a, a_residual) = call(first, 0);
    let (bp, bp_residual) = call(second, 1);
    let (ap, ap_residual) = call(first, 1);
    let (b, b_residual) = call(second, 0);
    let unit = if spherical {
        1.0 / z.powu(2)
    } else {
        2.0 / (PI * z)
    };
    let expected = unit
        * match (first, second) {
            (Bessel::J, Bessel::Y) => Complex::new(1.0, 0.0),
            (Bessel::H1, Bessel::H2) => Complex::new(0.0, -2.0),
            _ => Complex::i(),
        };
    let scale = expected.norm().max((a * bp).norm() + (ap * b).norm());
    prop_assert_close!(a * bp - ap * b, expected, 3e-12 * scale);
    let g = Complex::new(0.4, 0.3);
    let gradient = [
        a_residual.pullback(&[g * bp.conj()]),
        bp_residual.pullback(&[g * a.conj()]),
        ap_residual.pullback(&[-g * b.conj()]),
        b_residual.pullback(&[-g * ap.conj()]),
    ]
    .into_iter()
    .flat_map(Result::unwrap)
    .sum::<Complex>();
    let derivative = -(if spherical { 2.0 } else { 1.0 }) * expected / z;
    prop_assert_close!(gradient, g * derivative.conj(), 2e-10 * scale);
    Ok(())
}

/// Every kind and derivative order of [`bessel`]: `H1 = J + iY` and `H2 = J - iY`,
/// which ties the Hankel-sequence derivatives to the order differences of J and Y;
/// Bessel's equation for the cylindrical second derivatives of J, H1 and H2, which are
/// not formed from it; and `C_-n = (-1)^n C_n` for integer cylindrical orders.
///
/// Y enters the equation through the Hankel identities: AMOS Y alone is only accurate
/// to about 4e-12 at negative orders close to half-integers. The equation also skips
/// negative non-integer orders, where the order differences evaluate at the rounded
/// orders `v -+ 2` and `J_v` varies with `v` like `pi Y_|v|` near negative integers.
fn check_bessel_kinds(z: Complex, order: f64, spherical: bool) -> Result<(), TestCaseError> {
    let evaluate = |order| {
        [Bessel::J, Bessel::Y, Bessel::H1, Bessel::H2]
            .map(|kind| [0, 1, 2].map(|d| bessel(order, z, kind, spherical, d).unwrap()))
    };
    let [j, y, h1, h2] = evaluate(order);
    for d in 0..3 {
        let tolerance = 1e-13 * (j[d].norm() + y[d].norm());
        prop_assert_close!(
            h1[d],
            j[d] + Complex::i() * y[d],
            tolerance,
            "derivative {}",
            d
        );
        prop_assert_close!(
            h2[d],
            j[d] - Complex::i() * y[d],
            tolerance,
            "derivative {}",
            d
        );
    }
    if spherical {
        // Spherical second derivatives come from the differential equation itself.
        return Ok(());
    }
    if order >= 0.0 || order.fract() == 0.0 {
        for (f, kind) in [j, h1, h2].iter().zip(["J", "H1", "H2"]) {
            let terms = [z * z * f[2], z * f[1], (z * z - order * order) * f[0]];
            let scale = terms.iter().map(|t| t.norm()).sum::<f64>() + (order * order * f[0]).norm();
            let residual = terms.iter().sum::<Complex>();
            prop_assert_close!(residual, Complex::default(), 1e-12 * scale, "{}", kind);
        }
    }
    if order.fract() == 0.0 {
        let sign = if order.rem_euclid(2.0) == 0.0 {
            1.0
        } else {
            -1.0
        };
        let reflected = evaluate(-order);
        for (f, g) in [j, y, h1, h2].iter().zip(reflected) {
            for d in 0..3 {
                prop_assert_close!(g[d], sign * f[d], 1e-13 * f[d].norm(), "derivative {}", d);
            }
        }
    }
    Ok(())
}

/// Below `|z| = SERIES_RADIUS` (0.5) the regular spherical jet is a power series: its
/// value and first derivative match AMOS `sqrt(pi / 2z) J_(l+1/2)`, and its second
/// derivative solves the spherical Bessel equation. On both branches the Wronskian with
/// the outgoing jet is `j h' - j' h = i / z^2`.
fn check_spherical_series_and_wronskian(l: u32, z: Complex) -> Result<(), TestCaseError> {
    let j = spherical_radial(l, z, Radial::Regular).unwrap();
    let h = spherical_radial(l, z, Radial::Singular).unwrap();
    let scale = (j.value * h.first).norm() + (j.first * h.value).norm();
    prop_assume!(scale.is_finite());
    prop_assert_close!(
        j.value * h.first - j.first * h.value,
        Complex::i() / (z * z),
        1e-13 * scale
    );
    if z.norm() >= SERIES_RADIUS {
        return Ok(());
    }
    let prefactor = (PI / (2.0 * z)).sqrt();
    let amos = |l: u32| prefactor * complex_bessel::besselj(f64::from(l) + 0.5, z).unwrap();
    let value = amos(l);
    let first = f64::from(l) * value / z - amos(l + 1);
    prop_assert_close!(j.value, value, 2e-13 * value.norm());
    prop_assert_close!(j.first, first, 2e-13 * first.norm());
    let terms = [
        z * z * j.second,
        2.0 * z * j.first,
        (z * z - f64::from(l * (l + 1))) * j.value,
    ];
    let scale =
        terms.iter().map(|t| t.norm()).sum::<f64>() + f64::from(l * (l + 1)) * j.value.norm();
    prop_assert_close!(
        terms.iter().sum::<Complex>(),
        Complex::default(),
        1e-13 * scale
    );
    Ok(())
}

/// The spherical Bessel functions are single-valued, with `j_l(-z) = (-1)^l j_l(z)`,
/// `y_l(-z) = -(-1)^l y_l(z)` and `h1_l(-z) = (-1)^l h2_l(z)`; each derivative adds a
/// factor -1. On the negative real axis, with either sign of a zero imaginary part,
/// every kind and derivative of [`bessel`], the radial jets of [`spherical_radial`] and
/// `spherical_radial_sequence`, and the Hankel rows of `spherical_hankels` (which singular
/// polar translations use, starting at any degree) satisfy them against the positive
/// axis, both inside the regular power series (`x < SERIES_RADIUS`, which is 0.5) and
/// where AMOS takes the limit from above the axis. Tolerances scale with the jet's size,
/// which covers values next to a zero; `h1` has no real zeros, so each Hankel value gets
/// its own.
fn check_spherical_parity(l: u32, x: f64, zero: f64) -> Result<(), TestCaseError> {
    let (positive, negative) = (Complex::new(x, 0.0), Complex::new(-x, zero));
    let order = f64::from(l);
    let parity = if l.is_multiple_of(2) { 1.0 } else { -1.0 };
    let kinds = [
        (Bessel::J, Bessel::J, parity),
        (Bessel::Y, Bessel::Y, -parity),
        (Bessel::H1, Bessel::H2, parity),
        (Bessel::H2, Bessel::H1, parity),
    ];
    for (kind, mirror, sign) in kinds {
        // y_l and the Hankel functions overflow at small x and large l.
        let Ok(reflected) = [0, 1, 2]
            .map(|d| bessel(order, positive, mirror, true, d))
            .into_iter()
            .collect::<crate::Result<Vec<_>>>()
        else {
            continue;
        };
        let expected = [sign, -sign, sign]
            .iter()
            .zip(&reflected)
            .map(|(sign, f)| sign * f)
            .collect::<Vec<_>>();
        let actual = [0, 1, 2].map(|d| bessel(order, negative, kind, true, d).unwrap());
        let size = expected.iter().map(|f| f.norm()).sum::<f64>();
        prop_assert_close!(actual.to_vec(), expected, 1e-13 * size, "{:?}", kind);
    }
    for radial in [Radial::Regular, Radial::Singular] {
        let Ok(reflected) = spherical_radial(l, positive, radial) else {
            continue;
        };
        // On the positive axis the incoming function h2 is the conjugate of h1.
        let mirror = |z: Complex| match radial {
            Radial::Regular => parity * z,
            Radial::Singular => parity * z.conj(),
        };
        let expected = [reflected.value, -reflected.first, reflected.second].map(mirror);
        let size = expected.iter().map(|f| f.norm()).sum::<f64>();
        let sequence = spherical_radial_sequence(l, negative, radial).unwrap();
        for jet in [
            spherical_radial(l, negative, radial).unwrap(),
            sequence[sequence.len() - 1],
        ] {
            let actual = [jet.value, jet.first, jet.second];
            prop_assert_close!(actual, expected, 1e-13 * size, "{:?}", radial);
        }
    }
    // Like y_l, h1_l overflows at small x and large l.
    let Ok(reflected) = (0..=l)
        .map(|p| bessel(f64::from(p), positive, Bessel::H1, true, 0))
        .collect::<crate::Result<Vec<_>>>()
    else {
        return Ok(());
    };
    for first in [0, l / 2] {
        let expected = (first..=l)
            .zip(reflected.iter().skip(usize::try_from(first).unwrap()))
            .map(|(p, h)| {
                if p.is_multiple_of(2) {
                    h.conj()
                } else {
                    -h.conj()
                }
            })
            .collect::<Vec<_>>();
        let actual = spherical_hankels(first, l, negative).unwrap();
        prop_assert_eq!(actual.len(), expected.len());
        for (p, (h, expected)) in (first..).zip(actual.into_iter().zip(expected)) {
            let tolerance = 1e-13 * expected.norm();
            prop_assert_close!(h, expected, tolerance, "h1 of degree {} from {}", p, first);
        }
    }
    Ok(())
}

/// Wigner small-d symmetries `d_(m m') = (-1)^(m - m') d_(m' m) = d_(-m', -m)`, and
/// unitarity `D D^H = I` of the rotation `D = e^(-i m phi) d e^(-i m' psi)`.
fn check_wigner_small_d_matrix(l: i32, angles: [f64; 3]) -> Result<(), TestCaseError> {
    let d = special::wigner_small_d_matrix(l, angles[1]).unwrap();
    let index = |m: i32| usize::try_from(l + m).unwrap();
    for m in -l..=l {
        for k in -l..=l {
            let value = d[(index(m), index(k))];
            let sign = if (m - k) % 2 == 0 { 1.0 } else { -1.0 };
            let transposed = sign * d[(index(k), index(m))];
            prop_assert_close!(transposed, value, 1e-13, "({}, {})", m, k);
            prop_assert_close!(d[(index(-k), index(-m))], value, 1e-13, "({}, {})", m, k);
        }
    }
    if l == 0 {
        // Degree 0 is no wave mode; its d is the scalar 1.
        prop_assert_close!(d[(0, 0)], 1.0, 1e-15);
        return Ok(());
    }
    let basis = sw::Basis {
        modes: (-l..=l).map(|m| (0, Mode { l, m, pol: 1 })).collect(),
        positions: vec![[0.0; 3]],
    };
    let rotation = rotation::sw_rotation(&basis, &basis, angles).unwrap();
    let n = rotation.value().nrows();
    prop_assert_close!(
        rotation.value() * rotation.value().adjoint(),
        DMatrix::identity(n, n),
        1e-12
    );
    Ok(())
}

/// Two independent algorithms for the Wigner small-d matrix agree up to degree 128:
/// the Jacobi recurrence of [`wigner_small_d`] (the public ufunc) and the exponential of
/// the angular-momentum generator in `special::wigner_small_d_matrix` (multipole
/// rotations), to `2e-12 + 1e-10 |d|` per element. The recurrence's matrix is also
/// orthogonal to `2e-12` per element.
fn check_wigner_algorithms(l: i32, theta: f64) -> Result<(), TestCaseError> {
    let d = special::wigner_small_d_matrix(l, theta).unwrap();
    let n = d.nrows();
    let label = |i: usize| i32::try_from(i).unwrap() - l;
    let small = DMatrix::from_fn(n, n, |i, j| {
        wigner_small_d(l, label(i), label(j), Complex::from(theta)).unwrap()
    });
    for (i, j) in (0..n).flat_map(|i| (0..n).map(move |j| (i, j))) {
        let expected = Complex::from(d[(i, j)]);
        let tolerance = 2e-12 + 1e-10 * expected.norm();
        prop_assert_close!(
            small[(i, j)],
            expected,
            tolerance,
            "({}, {})",
            label(i),
            label(j)
        );
    }
    let gram = small.adjoint() * &small - DMatrix::identity(n, n);
    let deviation = gram.iter().map(|z| z.norm()).fold(0.0, f64::max);
    prop_assert!(
        deviation < 2e-12,
        "largest deviation from orthogonality {deviation}"
    );
    Ok(())
}

/// Wigner 3j symbols are invariant under cyclic column permutations and change by
/// `(-1)^(j1 + j2 + j3)` under odd permutations and under `m -> -m`. Each permutation
/// runs the recurrence over a different column, so an unstable recurrence breaks the
/// symmetry. Symbols outside the selection rules are exact zeros.
fn check_wigner3j_symmetries([j1, j2, j3, m1, m2, m3]: [i32; 6]) -> Result<(), TestCaseError> {
    let value = wigner3j(j1, j2, j3, m1, m2, m3);
    let sign = if (j1 + j2 + j3) % 2 == 0 { 1.0 } else { -1.0 };
    let images = [
        (wigner3j(j2, j3, j1, m2, m3, m1), 1.0),
        (wigner3j(j3, j1, j2, m3, m1, m2), 1.0),
        (wigner3j(j2, j1, j3, m2, m1, m3), sign),
        (wigner3j(j1, j3, j2, m1, m3, m2), sign),
        (wigner3j(j3, j2, j1, m3, m2, m1), sign),
        (wigner3j(j1, j2, j3, -m1, -m2, -m3), sign),
    ];
    for (index, (image, factor)) in images.into_iter().enumerate() {
        let tolerance = 1e-10 * image.abs().max(value.abs()) + 1e-12;
        prop_assert_close!(image, factor * value, tolerance, "image {}", index);
    }
    prop_assert!(wigner3j(j1, j2, j3, m1, m2, m3 + 1) == 0.0);
    prop_assert!(wigner3j(j1, j2, j1 + j2 + 1, m1, m2, m3) == 0.0);
    Ok(())
}

/// Both orthogonality relations of Wigner 3j symbols:
/// `(2 j3 + 1) Σ_(m1' + m2' = -m3) (j1 j2 j3; m1' m2' m3) (j1 j2 j3'; m1' m2' m3) = δ(j3, j3')`
/// and `Σ_j3 (2 j3 + 1) (j1 j2 j3; m1 m2 m3)² = 1`, whose row runs through both
/// classically forbidden regions of the recurrence.
fn check_wigner3j_orthogonality(
    [j1, j2, j3, m1, m2, m3]: [i32; 6],
    other: i32,
) -> Result<(), TestCaseError> {
    let sum: f64 = (-j1..=j1)
        .map(|m1| {
            let m2 = -m1 - m3;
            wigner3j(j1, j2, j3, m1, m2, m3) * wigner3j(j1, j2, other, m1, m2, m3)
        })
        .sum();
    let expected = if j3 == other { 1.0 } else { 0.0 };
    prop_assert_close!(sum * f64::from(2 * j3 + 1), expected, 1e-12);
    let row: f64 = ((j1 - j2).abs()..=j1 + j2)
        .map(|j3| f64::from(2 * j3 + 1) * wigner3j(j1, j2, j3, m1, m2, m3).powi(2))
        .sum();
    prop_assert_close!(row, 1.0, 1e-12);
    Ok(())
}
