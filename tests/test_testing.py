"""Public derivative checks detect contract errors and actual wrong derivatives."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from treams_rs import diff
from treams_rs.testing import check_gradient, check_pullback


@given(
    real=st.lists(st.floats(-2, 2), min_size=1, max_size=8),
    imaginary=st.floats(-1, 1),
    seed=st.integers(0, 1000),
)
def test_complex_real_pairing_for_nonholomorphic_objective(real, imaginary, seed):
    value = np.asarray(real) + 1j * imaginary
    check_gradient(
        lambda x: np.vdot(x, x).real,
        lambda x: 2 * x,
        value,
        seed=seed,
    )
    with pytest.raises(AssertionError, match=r"parameter 0.*absolute error"):
        check_gradient(
            lambda x: np.vdot(x, x).real,
            lambda x: 2 * x + np.ones_like(x),
            value,
            directions=(np.ones_like(value),),
        )


@given(scale=st.floats(0.3, 4), coupling=st.floats(-0.3, 0.3))
@pytest.mark.parametrize("complex_inputs", [False, True])
def test_native_solve_one_shot_context_and_both_parameters(
    scale, coupling, complex_inputs
):
    a = np.array([[scale + 1, coupling], [-coupling, 2 * scale + 1]])
    b = np.array([[0.4, 0.6], [0.7, 0.1]])
    if complex_inputs:
        a = a + np.array([[0.1j, 0.2j], [-0.3j, 0.05j]])
        b = b + 0.2j
    check_pullback(diff.solve, a, b, rtol=2e-7, atol=2e-9)


def test_tuple_outputs_and_mixed_real_complex_parameters():
    calls = []

    class Context:
        def __init__(self, x, y):
            self.x, self.y = x, y
            self.consumed = False

        def pullback(self, g, h):
            assert not self.consumed
            self.consumed = True
            calls.append(1)
            return g * self.y.conj() + h, (g * self.x.conj()).real

    def record(x, y):
        return (x * y, x), Context(x, y)

    check_pullback(
        record,
        np.array([0.3 + 0.4j, -0.2 + 0.1j]),
        np.array([0.8, 1.2]),
        directions=(np.array([1j, 0.2j]), np.array([0.2, -0.4])),
        cotangents=(np.array([0.2 + 0.6j, 0.7j]), np.array([0.3j, 0.8])),
    )
    assert calls == [1]


def test_each_parameter_checked_separately_so_bad_gradients_cannot_cancel():
    with pytest.raises(AssertionError, match="parameter 0"):
        check_gradient(
            lambda x, y: x + y,
            lambda x, y: (np.asarray(2.0), np.asarray(0.0)),
            np.asarray(0.4),
            np.asarray(0.8),
            directions=(np.asarray(1.0), np.asarray(1.0)),
        )


def test_reject_wrong_complex_convention_in_imaginary_direction():
    with pytest.raises(AssertionError, match="Re\\(vdot\\)"):
        check_pullback(
            lambda x: (x * x, lambda g: 2 * x * g),
            np.asarray(0.4 + 0.3j),
            directions=(np.asarray(1j),),
            cotangents=np.asarray(1.0 + 0j),
        )


def test_list_is_one_array_gradient_and_static_parameters_use_closure():
    check_pullback(
        lambda x: (3 * x, lambda g: (3 * g).tolist()),
        [0.2, 0.6],
    )
    check_pullback(lambda z: diff.bessel([0, 1, 2], z), np.asarray(0.8 + 0.3j))


@pytest.mark.parametrize(
    ("record", "error"),
    [
        (lambda x: (x, lambda g: np.asarray(1.0)), "gradient for parameter 0.*shape"),
        (lambda x: (x, lambda g: (g, g)), "2 gradients for 1 parameters"),
        (lambda x: (x, lambda g: g * np.nan), "gradient for parameter 0.*finite"),
        (lambda x: (x * np.nan, lambda g: g), "output 0.*finite"),
    ],
)
def test_invalid_gradient_and_output_contracts(record, error):
    with pytest.raises(ValueError, match=error):
        check_pullback(record, np.array([0.2, 0.4]))


def test_perturbed_output_shape_or_structure_must_not_change():
    for changed in (np.ones(3), (np.ones(2), np.ones(2))):
        with pytest.raises(ValueError, match="perturbed output"):
            check_pullback(
                lambda x, changed=changed: (
                    x if np.all(x == 0) else changed,
                    lambda g: g,
                ),
                np.zeros(2),
            )


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"directions": [np.ones(2)]}, "directions must be a tuple"),
        ({"directions": ()}, "one item per parameter"),
        ({"directions": (np.ones(1),)}, "direction 0.*expected shape"),
        ({"directions": (np.zeros(2),)}, "direction must be nonzero"),
        ({"directions": (1j * np.ones(2),)}, "real value requires a real probe"),
        ({"cotangents": (np.ones(2),)}, "single-array/tuple structure"),
        ({"cotangents": np.zeros(2)}, "cotangents must not all be zero"),
        ({"step": 0}, "step must be finite and positive"),
        ({"rtol": -1}, "rtol and atol must be finite and nonnegative"),
    ],
)
def test_invalid_probe_contracts(options, error):
    with pytest.raises(ValueError, match=error):
        check_pullback(lambda x: (x, lambda g: g), np.ones(2), **options)


def test_tuple_cotangent_count_checked():
    with pytest.raises(ValueError, match="one item per output"):
        check_pullback(
            lambda x: ((x, x), lambda g, h: g + h),
            np.ones(2),
            cotangents=(np.ones(2),),
        )


@pytest.mark.parametrize("output", [np.ones(1), np.asarray(1 + 0j)])
def test_gradient_requires_real_scalar_objective(output):
    with pytest.raises(ValueError, match="real scalar objective"):
        check_gradient(lambda x: output, lambda x: x, np.asarray(1.0))


def test_seed_is_reproducible():
    cotangents = []

    def record(x):
        def pullback(g):
            cotangents.append(g.copy())
            return g

        return x, pullback

    for _ in range(2):
        check_pullback(record, np.ones(5), seed=18)
    np.testing.assert_array_equal(*cotangents)


def test_native_multiple_outputs_eigensystem():
    matrix = np.array([[2 + 0.1j, 0.1], [0.2, 3 - 0.3j]])
    check_pullback(diff.eig, matrix)


def test_shifted_real_output_cannot_silently_discard_imaginary_change():
    def record(x):
        return x if np.all(x == 0) else 1j * x, lambda g: g * 0

    with pytest.raises(ValueError, match="real probe/output"):
        check_pullback(record, np.zeros(2))
