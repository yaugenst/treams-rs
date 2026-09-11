import numpy as np
import pytest
import treams
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr

# Upstream explicitly zeros excluded entries after its masked ufunc call.
pytestmark = pytest.mark.filterwarnings(
    "ignore:'where' used without 'out'.*:UserWarning"
)


@pytest.mark.parametrize(
    "angles",
    [
        [0, 0, 0],
        [0.2, 0.7, -0.3],
        [0.2, -0.7, 0.3],
        [0.2, np.pi, 0.3],
    ],
)
def test_spherical_rotation_reference(angles):
    basis = tr.SphericalWaveBasis.default(5, 2)
    oracle = treams.SphericalWaveBasis(basis.modes, basis.positions)
    actual = tr.rotate(*angles, basis=basis)
    expected = treams.rotate(*angles, basis=oracle)
    assert_allclose(actual, expected, rtol=1e-10, atol=2e-12)
    assert_allclose(actual.conj().T @ actual, np.eye(len(basis)), atol=2e-13)


@pytest.mark.parametrize("theta", [1e-10, -1e-10, 1e-100])
def test_small_angle_generator(theta):
    # Upstream rounds cos(theta) to one here and loses these first-order entries.
    basis = tr.SphericalWaveBasis([(1, m, 1) for m in (-1, 0, 1)])
    actual = tr.rotate(0, theta, 0, basis=basis)
    assert_allclose(actual[0, 1], theta / np.sqrt(2), rtol=1e-12, atol=0)
    assert_allclose(actual[1, 0], -theta / np.sqrt(2), rtol=1e-12, atol=0)


@pytest.mark.parametrize("spherical", [True, False])
def test_partial_basis_rotation_pullback(spherical):
    if spherical:
        basis = tr.SphericalWaveBasis.default(3, 2)
        oracle = treams.SphericalWaveBasis
    else:
        basis = tr.CylindricalWaveBasis.default([0.1, 0.2], 3, 2)
        oracle = treams.CylindricalWaveBasis
    to = type(basis)(basis.modes[::3], basis.positions)
    source = type(basis)(basis.modes[1::2], basis.positions)
    angles = np.array([0.2, 0.7 if spherical else 0, -0.3])
    actual, context = tr.diff.rotation(angles, to, source)
    expected = treams.rotate(
        *angles,
        basis=(oracle(to.modes, to.positions), oracle(source.modes, source.positions)),
    )
    assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    rng = np.random.default_rng(42)
    g = rng.normal(size=actual.shape) + 1j * rng.normal(size=actual.shape)
    gradients = context.pullback(g)
    direction = np.array([0.1, 0.3 if spherical else 0, -0.2])
    h = 1e-5
    plus = tr.diff.rotation(angles + h * direction, to, source)[0]
    minus = tr.diff.rotation(angles - h * direction, to, source)[0]
    assert_allclose(
        np.dot(gradients, direction),
        np.vdot(g, (plus - minus) / (2 * h)).real,
        rtol=1e-8,
        atol=1e-9,
    )
    with pytest.raises(ValueError, match="consumed"):
        context.pullback(g)


@pytest.mark.parametrize("degree", [10, 30, 60])
def test_large_rotation_unitarity(degree):
    basis = tr.SphericalWaveBasis([(degree, m, 1) for m in range(-degree, degree + 1)])
    rotation = tr.rotate(0.2, 1.3, -0.4, basis=basis)
    assert_allclose(rotation.conj().T @ rotation, np.eye(len(basis)), atol=1e-12)


@given(phi=st.floats(-3, 3), theta=st.floats(-3, 3), psi=st.floats(-3, 3))
def test_advect_rotation_composition(phi, theta, psi):
    import advect
    import advect.numpy as anp

    from treams_rs import advect as ad

    basis = tr.SphericalWaveBasis.default(1)
    rng = np.random.default_rng(14)
    matrix = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))

    def objective(angles):
        rotation = ad.rotation(angles, destination=basis)
        inverse = ad.rotation(-angles[::-1], destination=basis)
        transformed = rotation @ matrix @ inverse
        return anp.sum(anp.real(transformed[0, :2] * anp.conj(transformed[0, :2])))

    angles = np.array([phi, theta, psi])
    actual = advect.grad(objective)(angles)
    direction = np.array([0.2, -0.1, 0.3])
    h = 1e-5
    assert_allclose(
        np.dot(actual, direction),
        (objective(angles + h * direction) - objective(angles - h * direction))
        / (2 * h),
        rtol=2e-6,
        atol=2e-8,
    )


def test_sphere_rotation_invariance_and_cylinder_axis_constraint():
    sphere = tr.TMatrix.sphere(4, 1.3, 0.2, [3, 1])
    assert_allclose(sphere.rotate(0.2, 0.7, -0.3).array, sphere.array, atol=1e-14)
    cylinder = tr.TMatrixC.cylinder([0.2], 3, 1.3, [0.2], [3, 1])
    assert_allclose(cylinder.rotate(0.3).array, cylinder.array, atol=1e-14)
    with pytest.raises(ValueError, match="theta"):
        cylinder.rotate(0.1, 0.2)
