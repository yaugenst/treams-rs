"""Gradient benchmark objectives, real-coordinate packing and evidence checks."""

import functools
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from _scripts import load

bench = load("benchmark_gradients")


def arguments(tmp_path, family, parameters):
    return SimpleNamespace(
        family=family,
        parameters=parameters,
        particles=2,
        lmax=1,
        samples=3,
        seed=12,
        fd_steps=[1e-4, 1e-5, 1e-6],
        full_sweep_limit=32,
        directions=2,
        gradient_rtol=2e-5,
        array_output=tmp_path / "gradient-check.npz",
    )


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize(
    "family,parameters",
    [
        ("cluster", "physical"),
        ("sphere", "physical"),
        ("layers", "physical"),
        ("field", "physical"),
        ("cylindrical-field", "physical"),
        ("field", "coefficients"),
        ("cylindrical-field", "coefficients"),
    ],
)
def test_full_real_coordinate_gradient_matches_upstream(tmp_path, family, parameters):
    args = arguments(tmp_path, family, parameters)
    result = bench.validate(args)
    assert result["passed"]
    assert max(result["gradient_block_relative_errors"].values()) < 2e-5
    with np.load(args.array_output) as arrays:
        assert len(arrays["input_coordinates"]) == result["parameter_count"]
        np.testing.assert_allclose(
            arrays["gradient"], arrays["fd_gradient"], rtol=2e-5, atol=1e-9
        )
    if parameters == "coefficients":
        assert result["linear_adjoint_relative_error"] < 3e-9


@pytest.mark.gradients
@settings(max_examples=5)
@given(amplitude=st.floats(0.2, 3, allow_nan=False, allow_infinity=False))
def test_field_quadratic_loss_scales_gradient_without_changing_complex_pairing(
    amplitude,
):
    args = SimpleNamespace(
        family="field",
        parameters="coefficients",
        particles=1,
        lmax=1,
        samples=2,
        seed=45,
    )
    problem = bench.Problem(args, "rust")
    value, context = problem.record(problem.x)
    gradient = problem.reverse(context, bench.seed(value, args.family), problem.x)
    scaled, context = problem.record(amplitude * problem.x)
    scaled_gradient = problem.reverse(
        context, bench.seed(scaled, args.family), amplitude * problem.x
    )
    np.testing.assert_allclose(scaled, amplitude * value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        scaled_gradient, amplitude * gradient, rtol=1e-12, atol=1e-12
    )
    assert (
        abs(np.dot(gradient, problem.x) - 2 * bench.objective(value, args.family))
        < 1e-12
    )


@pytest.mark.interface
def test_nonfinite_numerical_results_cannot_pass_error_gate():
    with pytest.raises(AssertionError, match="finite"):
        bench.relative_error(np.asarray([np.nan]), np.asarray([1.0]))


FAMILIES = [
    ("cluster", "physical"),
    ("sphere", "physical"),
    ("layers", "physical"),
    ("field", "physical"),
    ("cylindrical-field", "physical"),
]


@functools.cache
def problem(family, parameters):
    """A small recorded problem and the complex value shape of its objective."""
    args = SimpleNamespace(
        family=family,
        parameters=parameters,
        particles=2,
        lmax=1,
        samples=2,
        seed=7,
    )
    instance = bench.Problem(args, "rust")
    value, _ = instance.record(instance.x)
    return instance, np.shape(value)


def complex_arrays(shape):
    parts = st.floats(-1e3, 1e3, allow_subnormal=False)
    return hnp.arrays(np.float64, (2, *shape), elements=parts).map(
        lambda pair: pair[0] + 1j * pair[1]
    )


@pytest.mark.interface
@settings(max_examples=40)
@given(case=st.sampled_from(FAMILIES), data=st.data())
def test_real_coordinate_packing_round_trips_every_parameter(case, data):
    """Complex parameters are two independent real coordinates, in a fixed order."""
    instance, _ = problem(*case)
    x = data.draw(
        hnp.arrays(np.float64, instance.x.shape, elements=st.floats(-1e3, 1e3))
    )
    np.testing.assert_array_equal(instance.pack(instance.unpack(x)), x)
    values = {}
    for name in instance.names:
        template = np.asarray(instance.data[name])
        if np.iscomplexobj(template):
            values[name] = data.draw(complex_arrays(template.shape))
        else:
            values[name] = data.draw(
                hnp.arrays(np.float64, template.shape, elements=st.floats(-1e3, 1e3))
            )
    unpacked = instance.unpack(instance.pack(values))
    for name, value in values.items():
        np.testing.assert_array_equal(unpacked[name], value)
        assert np.shape(unpacked[name]) == np.shape(instance.data[name])


@pytest.mark.interface
@settings(max_examples=40)
@given(case=st.sampled_from(FAMILIES), data=st.data())
def test_objective_seed_is_its_exact_complex_cotangent(case, data):
    """(L(v + d) - L(v - d)) / 2 = Re(vdot(seed(v), d)) for the quadratic loss.

    This includes the layers objective's selection of one scattering block.
    """
    instance, shape = problem(*case)
    family = instance.family
    value, direction = (
        data.draw(complex_arrays(shape)),
        data.draw(complex_arrays(shape)),
    )
    plus = bench.objective(value + direction, family)
    minus = bench.objective(value - direction, family)
    pairing = np.vdot(bench.seed(value, family), direction).real
    assert abs((plus - minus) / 2 - pairing) <= 1e-12 * (plus + minus + 1e-300)
