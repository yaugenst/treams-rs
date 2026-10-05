//! Directional derivatives of multipole fields and their sampled operators.

use nalgebra::DVector;
use proptest::{prelude::*, test_runner::TestCaseError};

use super::{FieldCase, FieldInputs, field_case};
use crate::{
    basis::MultipoleBasis,
    special::Radial,
    test_support::{DEFAULT_CASES, dot, five_point, patterned, prop_assert_close, re_dot},
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(DEFAULT_CASES))]

    #[test]
    fn field_pushforwards_match_adjoint_finite_difference_and_product_rule(
        (case, inputs) in field_case().prop_flat_map(|case| {
            FieldInputs::strategy(&case).prop_map(move |inputs| (case.clone(), inputs))
        }),
    ) {
        check_field_pushforwards(&case, &inputs)?;
    }
}

/// Simultaneous coefficient, geometry, medium and (for cylinders) axial directions
/// reproduce independent finite differences, transpose the existing pullbacks and
/// obey `d(A c) = dA c + A dc` and joint-translation invariance.
fn check_field_pushforwards(case: &FieldCase, inputs: &FieldInputs) -> Result<(), TestCaseError> {
    let cylindrical = matches!(case.basis, MultipoleBasis::Cylindrical(_));
    for point in &case.points {
        for position in case.basis.positions() {
            let [x, y, z] = std::array::from_fn(|axis| point[axis] - position[axis]);
            let distance = if cylindrical {
                x.hypot(y)
            } else {
                x.hypot(y).hypot(z)
            };
            prop_assume!(case.radial == Radial::Regular || distance > 0.4);
        }
    }
    let d = &inputs.direction;
    let (primal, field_residual) = case.field(&inputs.coefficients);
    let push_field = || {
        if cylindrical {
            field_residual.pushforward_axial(&d.coefficients, &d.points, &d.positions, d.ks, &d.kzs)
        } else {
            field_residual.pushforward(&d.coefficients, &d.points, &d.positions, d.ks)
        }
        .unwrap()
    };
    let tangent = push_field();
    let (matrix, operator_residual) = case.operator();
    let push_operator = || {
        if cylindrical {
            operator_residual.pushforward_axial(&d.points, &d.positions, d.ks, &d.kzs)
        } else {
            operator_residual.pushforward(&d.points, &d.positions, d.ks)
        }
        .unwrap()
    };
    let operator_tangent = push_operator();

    let coefficients = DVector::from_column_slice(&inputs.coefficients);
    let coefficient_tangent = DVector::from_column_slice(&d.coefficients);
    let product = &operator_tangent * &coefficients + &matrix * &coefficient_tangent;
    let tangent = DVector::from_column_slice(tangent.as_flattened());
    let scale = 1.0
        + tangent.norm()
        + operator_tangent.norm() * coefficients.norm()
        + matrix.norm() * coefficient_tangent.norm();
    prop_assert_close!(product.as_slice(), tangent.as_slice(), 2e-13 * scale);

    let (gradient, axial) = if cylindrical {
        field_residual.pullback_axial(&inputs.cotangent).unwrap()
    } else {
        (
            field_residual.pullback(&inputs.cotangent).unwrap(),
            Vec::new(),
        )
    };
    // Both derivative directions borrow the same saved work: a pullback leaves
    // the subsequent pushforward exactly reproducible without another record.
    let repeated_field = push_field();
    prop_assert_eq!(repeated_field.as_flattened(), tangent.as_slice());
    let input_pairing = re_dot(&gradient.coefficients, &d.coefficients)
        + dot(gradient.points.iter().flatten(), d.points.iter().flatten())
        + dot(
            gradient.positions.iter().flatten(),
            d.positions.iter().flatten(),
        )
        + re_dot(gradient.ks, d.ks)
        + if cylindrical {
            dot(&axial, &d.kzs)
        } else {
            0.0
        };
    let field_pairing = re_dot(inputs.cotangent.iter().flatten(), tangent.iter());
    prop_assert_close!(input_pairing, field_pairing, 2e-12 * scale);

    let operator_cotangent = patterned(matrix.nrows(), matrix.ncols(), 0.73);
    let (gradient, axial) = if cylindrical {
        operator_residual
            .pullback_axial(&operator_cotangent)
            .unwrap()
    } else {
        (
            operator_residual.pullback(&operator_cotangent).unwrap(),
            Vec::new(),
        )
    };
    let repeated_operator = push_operator();
    prop_assert_eq!(repeated_operator.as_slice(), operator_tangent.as_slice());
    let input_pairing = dot(gradient.points.iter().flatten(), d.points.iter().flatten())
        + dot(
            gradient.positions.iter().flatten(),
            d.positions.iter().flatten(),
        )
        + re_dot(gradient.ks, d.ks)
        + if cylindrical {
            dot(&axial, &d.kzs)
        } else {
            0.0
        };
    let operator_pairing = re_dot(operator_cotangent.iter(), operator_tangent.iter());
    prop_assert_close!(
        input_pairing,
        operator_pairing,
        2e-12 * (1.0 + matrix.norm() + operator_tangent.norm())
    );

    let numeric = five_point(1e-4, |step| {
        let moved_coefficients: Vec<_> = inputs
            .coefficients
            .iter()
            .zip(&d.coefficients)
            .map(|(value, direction)| value + step * direction)
            .collect();
        let value = case.moved(step, d).field(&moved_coefficients).0;
        re_dot(inputs.cotangent.iter().flatten(), value.iter().flatten())
    });
    let primal_norm = DVector::from_column_slice(primal.as_flattened()).norm();
    prop_assert_close!(
        numeric,
        field_pairing,
        1e-8 * (1.0 + field_pairing.abs()) + 1e-10 * primal_norm
    );
    let numeric = five_point(1e-4, |step| {
        re_dot(
            operator_cotangent.iter(),
            case.moved(step, d).operator().0.iter(),
        )
    });
    prop_assert_close!(
        numeric,
        operator_pairing,
        1e-8 * (1.0 + operator_pairing.abs()) + 1e-10 * matrix.norm()
    );

    let displacement = [0.17, -0.23, 0.31];
    let invariant = operator_residual
        .pushforward(
            &vec![displacement; case.points.len()],
            &vec![displacement; case.basis.positions().len()],
            [crate::Complex::default(); 2],
        )
        .unwrap();
    prop_assert_eq!(invariant.norm().to_bits(), 0.0_f64.to_bits());
    Ok(())
}
