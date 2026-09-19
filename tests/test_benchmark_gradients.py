"""Gradient benchmark objectives, real-coordinate packing and evidence checks."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

BENCH = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "benchmark_gradients.py")
)


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


@pytest.mark.ad_contract
@pytest.mark.oracle_numerical
@pytest.mark.filterwarnings("ignore:'where' used without 'out'.*:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:`scipy.special.sph_harm` is deprecated.*:DeprecationWarning"
)
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
    result = BENCH["validate"](args)
    assert result["passed"]
    assert max(result["gradient_block_relative_errors"].values()) < 2e-5
    with np.load(args.array_output) as arrays:
        assert len(arrays["input_coordinates"]) == result["parameter_count"]
        np.testing.assert_allclose(
            arrays["gradient"], arrays["fd_gradient"], rtol=2e-5, atol=1e-9
        )
    if parameters == "coefficients":
        assert result["linear_adjoint_relative_error"] < 3e-9


@pytest.mark.ad_contract
@settings(max_examples=5, deadline=None)
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
    problem = BENCH["Problem"](args, "rust")
    value, context = problem.record(problem.x)
    gradient = problem.reverse(context, BENCH["seed"](value, args.family), problem.x)
    scaled, context = problem.record(amplitude * problem.x)
    scaled_gradient = problem.reverse(
        context, BENCH["seed"](scaled, args.family), amplitude * problem.x
    )
    np.testing.assert_allclose(scaled, amplitude * value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        scaled_gradient, amplitude * gradient, rtol=1e-12, atol=1e-12
    )
    assert (
        abs(np.dot(gradient, problem.x) - 2 * BENCH["objective"](value, args.family))
        < 1e-12
    )


def test_nonfinite_numerical_results_cannot_pass_error_gate():
    with pytest.raises(AssertionError, match="finite"):
        BENCH["relative_error"](np.asarray([np.nan]), np.asarray([1.0]))
