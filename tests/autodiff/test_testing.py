"""Public derivative checks detect contract errors and actual wrong derivatives."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from treams_rs import diff
from treams_rs.testing import check_gradient, check_pullback, check_pushforward

from _support import complex_normal

pytestmark = pytest.mark.gradients


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


@given(
    m=st.integers(1, 4),
    n=st.integers(1, 4),
    complex_input=st.booleans(),
    seed=st.integers(0, 2**16),
)
def test_linear_record_passes_only_with_the_adjoint(m, n, complex_input, seed):
    # Imaginary parts bounded away from zero separate A^T from A^H.
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], (m, n))
    a = rng.normal(size=(m, n)) + 1j * signs * rng.uniform(0.5, 1.0, (m, n))
    x = complex_normal(rng, n) if complex_input else rng.normal(size=n)
    check_pullback(lambda v: (a @ v, lambda g: a.conj().T @ g), x, seed=seed)
    for wrong in (lambda g: a.T @ g, lambda g: -(a.conj().T @ g)):
        with pytest.raises(AssertionError, match="parameter 0"):
            check_pullback(lambda v, wrong=wrong: (a @ v, wrong), x, seed=seed)


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
        (lambda x: (x, lambda g: np.asarray(1.0)), r"parameter\[0\] expected shape"),
        (lambda x: (x, lambda g: (g, g)), "expected 1 gradients, received 2"),
        (lambda x: (x, lambda g: g * np.nan), "gradient for parameter 0.*finite"),
        # A real parameter ignores the imaginary part, but not a non-finite one.
        (
            lambda x: (x, lambda g: g + complex(0, np.inf)),
            "gradient for parameter 0.*finite",
        ),
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


@pytest.mark.parametrize("vary_real_input", [False, True])
def test_pushforward_checks_mixed_tuple_outputs_with_fresh_contexts(vary_real_input):
    calls = []

    class Context:
        def __init__(self, z, x):
            self.z, self.x = z, x
            self.consumed = False

        def pushforward(self, dz, dx):
            assert not self.consumed
            self.consumed = True
            calls.append("pushforward")
            return (
                dz * self.x + self.z * dx,
                2 * (self.z.conj() * dz).real,
                dx,
            )

        def pullback(self, g, h, k):
            assert not self.consumed
            self.consumed = True
            calls.append("pullback")
            return g * self.x + 2 * h * self.z, (self.z.conj() * g).real + k

    def record(z, x):
        return (z * x, abs(z) ** 2, x), Context(z, x)

    check_pushforward(
        record,
        np.array([0.3 + 0.4j, -0.2 + 0.1j]),
        np.array([0.8, 1.2]),
        directions=(
            np.array([0.2j, 0.1 + 0.3j]),
            np.array([0.2, -0.4]) if vary_real_input else np.zeros(2),
        ),
        seed=17,
    )
    assert calls == ["pushforward", "pullback"]


@pytest.mark.parametrize(
    ("derivative", "message"),
    [("pushforward", "finite differences"), ("pullback", "adjoint identity")],
)
def test_pushforward_checker_detects_wrong_derivative_on_either_side(
    derivative, message
):
    class Context:
        def pushforward(self, direction):
            return (3 if derivative == "pushforward" else 2) * direction

        def pullback(self, cotangent):
            return (3 if derivative == "pullback" else 2) * cotangent

    with pytest.raises(AssertionError, match=message):
        check_pushforward(
            lambda x: (2 * x, Context()),
            np.array([0.2, 0.4]),
            directions=(np.ones(2),),
            cotangents=np.ones(2),
        )


def test_pushforward_checker_rejects_nonfinite_output_tangent():
    class Context:
        def pushforward(self, direction):
            return np.full_like(direction, np.nan)

        def pullback(self, cotangent):
            return cotangent

    with pytest.raises(ValueError, match=r"tangent for output 0.*finite"):
        check_pushforward(lambda x: (x, Context()), np.array([0.2, 0.4]))


def test_adjoint_check_uses_tighter_tolerance_than_finite_differences():
    class Context:
        def pushforward(self, direction):
            return 2 * direction

        def pullback(self, cotangent):
            return (2 + 1e-8) * cotangent

    def record(x):
        return 2 * x, Context()

    parameters = (np.array([0.2, 0.4]),)
    probes = {"directions": (np.ones(2),), "cotangents": np.ones(2)}
    with pytest.raises(AssertionError, match="adjoint identity"):
        check_pushforward(record, *parameters, rtol=1e-3, **probes)
    check_pushforward(record, *parameters, adjoint_rtol=1e-7, **probes)


@pytest.mark.parametrize("tolerance", [-1.0, np.nan, np.inf])
def test_pushforward_checker_rejects_invalid_adjoint_tolerance(tolerance):
    with pytest.raises(ValueError, match="adjoint tolerances"):
        check_pushforward(diff.svdvals, np.eye(2), adjoint_atol=tolerance)
