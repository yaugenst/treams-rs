"""Wigner d, D and 3j symbols (treams_rs.special): upstream and exact Racah references,
group and orthogonality identities, and the pullbacks over all Euler angles."""

import math
from fractions import Fraction

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams.special as oracle
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad
from treams_rs import diff, special
from treams_rs.testing import check_gradient

from _support import assert_reusable_context, degree_order, sum_to


@pytest.mark.reference
@pytest.mark.parametrize("theta", [0, 0.3, -0.7, np.pi, 1.3 + 0.2j, 0.3 - 0.4j])
def test_public_wigner_broadcast_reference(theta):
    labels = [
        (degree, m, k)
        for degree in range(9)
        for m in range(-degree, degree + 1)
        for k in range(-degree, degree + 1)
    ]
    degree, m, k = np.array(labels).T
    assert_allclose(
        special.wignersmalld(degree, m, k, theta),
        oracle.wignersmalld(degree, m, k, theta),
        rtol=1e-10,
        atol=2e-12,
    )
    assert_allclose(
        special.wignerd(degree, m, k, 0.2, theta, -0.3),
        oracle.wignerd(degree, m, k, 0.2, theta, -0.3),
        rtol=1e-10,
        atol=2e-12,
    )
    assert_allclose(special.wignersmalld(3, [-4, 0, 4], [0, 4, 0], theta), 0)


@pytest.mark.physics
@given(degree=st.integers(0, 15), theta=st.floats(-5, 5), phi=st.floats(-4, 4))
def test_public_wigner_group_and_unitarity(degree, theta, phi):
    labels = np.arange(-degree, degree + 1)
    a = special.wignersmalld(degree, labels[:, None], labels, theta)
    b = special.wignersmalld(degree, labels[:, None], labels, phi)
    c = special.wignersmalld(degree, labels[:, None], labels, theta + phi)
    assert_allclose(a @ b, c, rtol=2e-11, atol=2e-12)
    assert_allclose(a.conj().T @ a, np.eye(2 * degree + 1), atol=2e-12)
    # Inversion and order reflection: d(-theta) = d(theta)^T and
    # d_{-m,-k} = (-1)^(m-k) d_{mk}.
    inverse = special.wignersmalld(degree, labels[:, None], labels, -theta)
    assert_allclose(inverse, a.T, atol=1e-14)
    assert_allclose(a[::-1, ::-1], (-1.0) ** (labels[:, None] - labels) * a, atol=1e-14)


@pytest.mark.reference
def test_large_public_wigner_agrees_with_matrix_algorithm():
    # The native wigner_small_matches_the_generator_exponential property compares
    # the two algorithms up to degree 128; this checks the public boundary.
    degree = 30
    modes = tr.SphericalBasis([(degree, m, 1) for m in range(-degree, degree + 1)])
    labels = np.arange(-degree, degree + 1)
    actual = special.wignersmalld(degree, labels[:, None], labels, 1.3)
    expected = tr.operators.rotate(0, 1.3, 0, basis=modes)
    assert_allclose(actual, expected, rtol=1e-10, atol=2e-12)
    assert_allclose(actual.conj().T @ actual, np.eye(len(modes)), atol=2e-12)


@pytest.mark.interface
def test_wigner_tiny_angle_and_masked_invalid_labels():
    for theta in [1e-10, -1e-10, 1e-100]:
        assert_allclose(
            special.wignersmalld(1, -1, 0, theta), theta / np.sqrt(2), rtol=3e-15
        )
    # A masked-out invalid label is never evaluated.
    out = np.full(2, 7 + 0j)
    special.wignersmalld([-1, 2], 0, 0, 0.2, out=out, where=[False, True])
    assert out[0] == 7
    with pytest.raises(ValueError, match="integer"):
        special.wignersmalld(1.5, 0, 0, 0.2)


@pytest.mark.physics
@pytest.mark.reference
@given(first=degree_order(0, 15), second=degree_order(0, 15))
@settings(max_examples=50)
def test_public_wigner3j_symmetry_and_normalization(first, second):
    (a, m), (b, n) = first, second
    degrees = np.arange(abs(a - b), a + b + 1)
    value = special.wigner3j(a, b, degrees, m, n, -m - n)
    assert_allclose(
        value, oracle.wigner3j(a, b, degrees, m, n, -m - n), rtol=1e-11, atol=2e-14
    )
    # An odd permutation or flipping every order multiplies by the sign. Cyclic
    # permutations are pinned by test_wigner3j_in_every_argument_order.
    sign = (-1.0) ** (a + b + degrees)
    for permuted in (
        special.wigner3j(b, a, degrees, n, m, -m - n),
        special.wigner3j(a, b, degrees, -m, -n, m + n),
    ):
        assert_allclose(permuted, sign * value, rtol=1e-11, atol=2e-14)
    assert_allclose(np.sum((2 * degrees + 1) * value**2), 1, atol=2e-13)


@pytest.mark.reference
@pytest.mark.parametrize(
    "arguments",
    [(15, 15, 23, 15, -15, 0), (23, 15, 15, 0, 15, -15), (15, 23, 15, -15, 0, 15)],
)
def test_wigner3j_in_every_argument_order(arguments):
    # The even permutations of one symbol, exact value from SymPy's Racah formula
    # at 30 digits; for the last two the recurrence enters a forbidden region.
    assert_allclose(special.wigner3j(*arguments), 7.776528999879744e-06, rtol=1e-12)


@pytest.mark.reference
def test_public_wigner3j_is_accurate_in_the_forbidden_regions():
    # Exact Racah values in rational arithmetic. treams recurs downward into the lower
    # classically forbidden region for the first three and returns -2.06e-8, -1.41e-9
    # and -2.16e-3, and upward into the upper one for the last three and returns
    # -6.912476e-8, -7.85e-10 and -2.77e-9
    # (docs/coming-from-treams/differences.md). The native recurrence continues
    # in each direction while the symbols grow.
    symbols = [
        [47, 60, 39, -47, 8],
        [53, 69, 46, -52, 10],
        [88, 66, 56, -10, 66],
        [128, 64, 95, -128, 64],
        [245, 195, 144, -245, 194],
        [260, 130, 190, -260, 129],
    ]
    j1, j2, j3, m1, m2 = np.array(symbols).T
    exact = [
        1.3295559888160243e-11,
        -1.2301818424524703e-09,
        5.787402366675233e-17,
        -6.912539260950578e-08,
        -1.8847971705081455e-10,
        -1.5123512820829832e-12,
    ]
    assert_allclose(special.wigner3j(j1, j2, j3, m1, m2, -m1 - m2), exact, rtol=1e-12)
    for (a, b, c, m, n), value in zip(symbols, exact, strict=True):
        assert _racah_wigner3j(a, b, c, m, n, -m - n) == pytest.approx(value, rel=1e-15)


def _racah_wigner3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """The Wigner 3j symbol from the Racah formula in exact rational arithmetic."""
    f = math.factorial
    square = Fraction(
        f(j1 + j2 - j3)
        * f(j1 - j2 + j3)
        * f(j2 + j3 - j1)
        * f(j1 + m1)
        * f(j1 - m1)
        * f(j2 + m2)
        * f(j2 - m2)
        * f(j3 + m3)
        * f(j3 - m3),
        f(j1 + j2 + j3 + 1),
    )
    first = max(0, j2 - j3 - m1, j1 - j3 + m2)
    last = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    total = sum(
        Fraction(
            (-1) ** k,
            f(k)
            * f(j1 + j2 - j3 - k)
            * f(j1 - m1 - k)
            * f(j2 + m2 - k)
            * f(j3 - j2 + m1 + k)
            * f(j3 - j1 - m2 + k),
        )
        for k in range(first, last + 1)
    )
    square *= total * total
    # isqrt(square 4^shift) has at least 64 bits, so its float rounds correctly.
    shift = max(
        0, 66 - (square.numerator.bit_length() - square.denominator.bit_length()) // 2
    )
    root = math.isqrt((square.numerator << (2 * shift)) // square.denominator)
    sign = (-1) ** (j1 - j2 - m3) * (1 if total >= 0 else -1)
    return sign * math.ldexp(root, -shift)


@st.composite
def _extreme_wigner3j_symbol(draw):
    """Labels `(j1, j2, j3, m1, m2)` up to 260 with orders biased to ±j and ±(j - 1),
    where the classically forbidden regions are widest, and `j3` often next to treams'
    switch between upward and downward recurrence."""

    def order(j):
        extremes = st.sampled_from([j, -j, max(j - 1, 0), -max(j - 1, 0)])
        return draw(extremes | st.integers(-j, j))

    j1, j2 = draw(st.integers(0, 260)), draw(st.integers(0, 260))
    m1, m2 = order(j1), order(j2)
    lower, upper = max(abs(j1 - j2), abs(m1 + m2)), min(j1 + j2, 260)
    assume(lower <= upper)
    degrees = st.integers(lower, upper)
    switch = abs(j1 - j2) + (j1 + j2 - abs(j1 - j2)) // 4
    near = (max(lower, switch - 8), min(upper, switch + 8))
    if near[0] <= near[1]:
        degrees |= st.integers(*near)
    return j1, j2, draw(degrees), m1, m2


@pytest.mark.reference
@given(symbol=_extreme_wigner3j_symbol())
@settings(max_examples=100)
def test_public_wigner3j_matches_exact_racah_values(symbol):
    # Errors are absolute on the scale of the row, whose (2 j3 + 1) symbol^2 sum to
    # one. Sampled symbols stay within 5e-13 on it; the closed-form start values, sums
    # of lgamma up to 4e3, can lose a few more ulps. Recurring upward into the upper
    # forbidden region, as treams does below its switch, fails for one symbol in twenty.
    j1, j2, j3, m1, m2 = symbol
    value = special.wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
    exact = _racah_wigner3j(j1, j2, j3, m1, m2, -m1 - m2)
    assert abs(value - exact) * math.sqrt(2 * j3 + 1) < 2e-12


@pytest.mark.physics
def test_wigner3j_tables_are_orthogonal():
    # The selection rule zeros every symbol with m1 + m2 + m3 != 0. For fixed
    # j1, j2 and m = m1 + m2, the symbols sqrt(2 j + 1) (j1 j2 j; m1 m2 -m)
    # with rows j and columns m1 therefore form a square orthogonal block:
    # both orthogonality relations of the 3j symbols at once.
    errors = []
    for j1 in range(9):
        for j2 in range(9):
            low, high = abs(j1 - j2), j1 + j2
            for m in range(-high, high + 1):
                j = np.arange(max(low, abs(m)), high + 1)[:, None]
                m1 = np.arange(max(-j1, m - j2), min(j1, m + j2) + 1)
                block = np.sqrt(2 * j + 1) * special.wigner3j(j1, j2, j, m1, m - m1, -m)
                assert block.shape == (len(m1), len(m1))
                identity = np.eye(len(m1))
                error = max(
                    abs(block @ block.T - identity).max(),
                    abs(block.T @ block - identity).max(),
                )
                errors.append((error, j1, j2, m))
    worst = max(errors)
    assert worst[0] <= 1e-12, worst
    # The selection rule itself, on one full table.
    j, m, m1, m2 = np.ix_(range(1, 6), range(-5, 6), range(-3, 4), range(-2, 3))
    values = special.wigner3j(3, 2, j, m1, m2, -m)
    assert np.all(values[np.broadcast_to(m != m1 + m2, values.shape)] == 0)


@st.composite
def _matrix_element(draw, high=8):
    """A degree with a row and a column order of its small-d matrix."""
    degree, row = draw(degree_order(0, high))
    return degree, row, draw(st.integers(-degree, degree))


@pytest.mark.physics
@given(
    first=_matrix_element(),
    second=_matrix_element(),
    beta=st.floats(-3, 3),
    imaginary=st.one_of(st.just(0.0), st.floats(-0.5, 0.5)),
)
def test_wigner_clebsch_gordan_series(first, second, beta, imaginary):
    # d^a_{mk}(beta) d^b_{nq}(beta) = sum_j (2j + 1) (-1)^(M+K)
    #   (a b j; m n -M) (a b j; k q -K) d^j_{MK}(beta), with M = m + n and
    # K = k + q, ties the small-d matrices to the 3j symbols; it holds for the
    # analytic continuation to complex beta too.
    (a, m, k), (b, n, q) = first, second
    beta = complex(beta, imaginary)
    total, column = m + n, k + q
    j = np.arange(max(abs(a - b), abs(total), abs(column)), a + b + 1)
    series = np.sum(
        (2 * j + 1)
        * (-1.0) ** (total + column)
        * special.wigner3j(a, b, j, m, n, -total)
        * special.wigner3j(a, b, j, k, q, -column)
        * special.wignersmalld(j, total, column, beta)
    )
    product = special.wignersmalld(a, m, k, beta) * special.wignersmalld(b, n, q, beta)
    assert_allclose(series, product, rtol=1e-11, atol=1e-13)


@pytest.mark.gradients
@pytest.mark.parametrize("scalar", [True, False])
def test_wigner_all_euler_adjoints_and_owned_broadcast(scalar):
    labels = (
        (5, 1, -2)
        if scalar
        else (np.array([3, 4, 5])[:, None], 1, np.array([-2, 0, 2])[:, None])
    )
    angles = [
        np.array(0.2 + 0.1j),
        np.array(0.7 + 0.2j) if scalar else np.array([0.7 + 0.2j, 0.8 - 0.1j]),
        np.array(0.3),
    ]
    value, context = diff.wignerd(*labels, *angles)
    assert_allclose(value, special.wignerd(*labels, *angles), rtol=2e-13)
    g = np.full_like(value, 0.3 + 0.2j)
    gradient = context.pullback(g)
    # D = exp(-i m phi) d(theta) exp(-i k psi), and the small-d ladder gives
    # d'_{mk} = c(m) d_{m+1,k} - c(m-1) d_{m-1,k}, c(m) = sqrt(l(l+1) - m(m+1)) / 2.
    degree, row, column = (np.asarray(label) for label in labels)
    phi, theta, psi = angles

    def ladder(m):
        return 0.5 * np.sqrt(np.maximum(degree * (degree + 1) - m * (m + 1), 0))

    small = ladder(row) * special.wignersmalld(degree, row + 1, column, theta)
    small -= ladder(row - 1) * special.wignersmalld(degree, row - 1, column, theta)
    derivatives = [
        -1j * row * value,
        np.exp(-1j * row * phi) * small * np.exp(-1j * column * psi),
        -1j * column * value,
    ]
    for actual, derivative, angle in zip(gradient, derivatives, angles, strict=True):
        assert actual.shape == angle.shape
        expected = sum_to(g * derivative.conj(), angle.shape)
        assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)
    _, context = diff.wignerd(*labels, *angles)
    for a in angles:
        a[...] = 0
    assert_reusable_context(context, g, gradient)


@pytest.mark.gradients
def test_wigner_advect_composition_parallel_and_empty():
    phi = np.array(0.2)
    theta = np.linspace(0.3, 0.8, 1024).astype(complex) + 0.1j
    psi = np.array(-0.1)

    def objective(p, t, s):
        v = ad.wignerd(p, t, s, degree=5, row=2, column=-3)
        return anp.sum(anp.real(v) ** 2)

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2)),
        phi,
        theta,
        psi,
        directions=(np.array(0.1), np.full(theta.shape, 0.2 + 0.1j), np.array(-0.3)),
        step=1e-6,
        rtol=1e-7,
        atol=0,
    )
    value, context = diff.wignerd(np.empty((0, 2)), 0, 0, np.ones((1, 1)), 0.3, 0.1)
    gradients = context.pullback(np.empty_like(value))
    assert gradients[0].shape == (1, 1)
    for g in gradients:
        assert_allclose(g, 0)
    # At zero angle off-diagonal values vanish but their derivatives do not.
    _, context = diff.wignerd(1, -1, 0, 0, 0, 0)
    assert_allclose(context.pullback(np.array(1 + 0j))[1], 1 / np.sqrt(2))
