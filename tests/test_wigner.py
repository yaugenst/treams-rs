import numpy as np
import pytest
import treams.special as oracle
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import special


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


@given(degree=st.integers(0, 15), theta=st.floats(-5, 5), phi=st.floats(-4, 4))
@settings(max_examples=30)
def test_public_wigner_group_and_unitarity(degree, theta, phi):
    labels = np.arange(-degree, degree + 1)
    a = special.wignersmalld(degree, labels[:, None], labels, theta)
    b = special.wignersmalld(degree, labels[:, None], labels, phi)
    c = special.wignersmalld(degree, labels[:, None], labels, theta + phi)
    assert_allclose(a @ b, c, rtol=2e-11, atol=2e-12)
    assert_allclose(a.conj().T @ a, np.eye(2 * degree + 1), atol=2e-12)


@pytest.mark.parametrize("degree", [30, 60, 128])
def test_large_public_wigner_agrees_with_matrix_algorithm(degree):
    modes = tr.SphericalWaveBasis([(degree, m, 1) for m in range(-degree, degree + 1)])
    labels = np.arange(-degree, degree + 1)
    actual = special.wignersmalld(degree, labels[:, None], labels, 1.3)
    expected = tr.rotate(0, 1.3, 0, basis=modes)
    assert_allclose(actual, expected, rtol=1e-10, atol=2e-12)
    assert_allclose(actual.conj().T @ actual, np.eye(len(modes)), atol=2e-12)


def test_wigner_tiny_angle_and_masked_strides():
    for theta in [1e-10, -1e-10, 1e-100]:
        assert_allclose(
            special.wignersmalld(1, -1, 0, theta), theta / np.sqrt(2), rtol=3e-15
        )
    theta = np.linspace(0.2, 0.9, 2048).astype(complex)
    expected = special.wignersmalld(3, 1, -2, theta)
    special.wignersmalld(3, 1, -2, theta, out=theta)
    assert_allclose(theta, expected)
    assert_allclose(
        special.wignersmalld(3, 1, -2, theta[::-1]),
        special.wignersmalld(3, 1, -2, theta)[::-1],
    )
    out = np.full(2, 7 + 0j)
    special.wignersmalld([-1, 2], 0, 0, 0.2, out=out, where=[False, True])
    assert out[0] == 7
    with pytest.raises(ValueError, match="integer"):
        special.wignersmalld(1.5, 0, 0, 0.2)


@given(a=st.integers(0, 15), b=st.integers(0, 15), seed=st.integers(0, 30))
@settings(max_examples=50)
def test_public_wigner3j_symmetry_and_normalization(a, b, seed):
    m = seed % (2 * a + 1) - a
    n = seed % (2 * b + 1) - b
    degrees = np.arange(abs(a - b), a + b + 1)
    value = special.wigner3j(a, b, degrees, m, n, -m - n)
    assert_allclose(
        value, oracle.wigner3j(a, b, degrees, m, n, -m - n), rtol=1e-11, atol=2e-14
    )
    assert_allclose(
        special.wigner3j(b, a, degrees, n, m, -m - n),
        (-1.0) ** (a + b + degrees) * value,
        rtol=1e-11,
        atol=2e-14,
    )
    assert_allclose(np.sum((2 * degrees + 1) * value**2), 1, atol=2e-13)


def test_public_ewald_integral_broadcasts_and_identities():
    n = np.arange(-6, 8)[:, None] / 2
    z = np.array([0.3 + 0.1j, 1.3 - 0.2j, 4.1 + 0.3j])
    actual = special.incgamma(n, z)
    assert_allclose(actual, oracle.incgamma(n, z), rtol=3e-10, atol=1e-13)
    assert_allclose(
        special.incgamma(n + 1, z),
        n * actual + z**n * np.exp(-z),
        rtol=3e-11,
        atol=1e-13,
    )
    orders = np.arange(-7, 5)[:, None]
    eta = np.array([0.5, 0.7 - 0.1j, 0.9])
    assert_allclose(
        special.intkambe(orders, z, eta),
        oracle.intkambe(orders, z, eta),
        rtol=3e-8,
        atol=1e-12,
    )
    out = np.full((len(n), 3), 9 + 0j)
    special.incgamma(n, z, out=out, where=[True, False, True])
    assert_allclose(out[:, [0, 2]], actual[:, [0, 2]])
    assert_allclose(out[:, 1], 9)


@pytest.mark.parametrize("scalar", [True, False])
def test_wigner_all_euler_adjoints_and_owned_broadcast(scalar):
    from treams_rs import diff

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
    value, context = diff.wigner(*labels, *angles)
    assert_allclose(value, special.wignerd(*labels, *angles), rtol=2e-13)
    g = np.full_like(value, 0.3 + 0.2j)
    gradient = context.pullback(g)
    steps = [0.17 + 0.03j, -0.2 + 0.1j, 0.1 - 0.05j]
    h = 1e-6
    for axis, step in enumerate(steps):
        assert gradient[axis].shape == angles[axis].shape
        plus, minus = list(angles), list(angles)
        plus[axis] = plus[axis] + h * step
        minus[axis] = minus[axis] - h * step
        numerical = (
            special.wignerd(*labels, *plus) - special.wignerd(*labels, *minus)
        ) / (2 * h)
        assert_allclose(
            np.sum(gradient[axis].conj() * step).real,
            np.vdot(g, numerical).real,
            rtol=1e-7,
            atol=1e-10,
        )
    _, context = diff.wigner(*labels, *angles)
    for a in angles:
        a[...] = 0
    with pytest.raises(ValueError, match="shape"):
        context.pullback(np.zeros((30, 30), complex))
    for actual, expected in zip(context.pullback(g), gradient, strict=True):
        assert_allclose(actual, expected)
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


def test_wigner_advect_composition_parallel_and_empty():
    import advect
    import advect.numpy as anp

    from treams_rs import advect as ad
    from treams_rs import diff

    phi = np.array(0.2)
    theta = np.linspace(0.3, 0.8, 1024).astype(complex) + 0.1j
    psi = np.array(-0.1)

    def objective(p, t, s):
        v = ad.wigner(p, t, s, degree=5, row=2, column=-3)
        return anp.sum(anp.real(v) ** 2)

    gradient = advect.grad(objective, argnums=(0, 1, 2))(phi, theta, psi)
    h = 1e-6
    expected = (
        objective(phi + h * 0.1, theta + h * (0.2 + 0.1j), psi - h * 0.3)
        - objective(phi - h * 0.1, theta - h * (0.2 + 0.1j), psi + h * 0.3)
    ) / (2 * h)
    actual = (
        gradient[0] * 0.1
        + np.sum(gradient[1].conj() * (0.2 + 0.1j)).real
        - gradient[2] * 0.3
    )
    assert_allclose(actual, expected, rtol=1e-7)
    value, context = diff.wigner(np.empty((0, 2)), 0, 0, np.ones((1, 1)), 0.3, 0.1)
    gradients = context.pullback(np.empty_like(value))
    assert gradients[0].shape == (1, 1)
    for g in gradients:
        assert_allclose(g, 0)
    # At zero angle off-diagonal values vanish but their derivatives do not.
    _, context = diff.wigner(1, -1, 0, 0, 0, 0)
    assert_allclose(context.pullback(np.array(1 + 0j))[1], 1 / np.sqrt(2))


def test_parallel_integral_ufunc_aliasing_and_unaligned_storage():
    n = 2048
    raw = bytearray(16 * n + 1)
    z = np.ndarray((n,), dtype=np.complex128, buffer=raw, offset=1)
    z[:] = np.linspace(0.3, 1.3, n) + 0.1j
    expected = oracle.incgamma(1.5, z.copy())
    special.incgamma(1.5, z, out=z)
    assert_allclose(z, expected, rtol=2e-12)
    z = np.linspace(0.3, 1.3, n) + 0.1j
    eta = np.full(n, 0.7 + 0.1j)
    expected = oracle.intkambe(-2, z, eta)
    special.intkambe(-2, z[::-1], eta[::-1], out=eta[::-1])
    assert_allclose(eta, expected, rtol=3e-10)
