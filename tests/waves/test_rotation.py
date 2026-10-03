"""Rotations of spherical and cylindrical bases: operators.rotate against upstream, the
small-angle generator, pullbacks, and rotated cluster responses."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad
from treams_rs.testing import check_gradient, check_pullback

from _support import complex_normal, to_oracle


@pytest.mark.physics
@pytest.mark.reference
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
    basis = tr.SphericalBasis.default(5, 2)
    actual = tr.operators.rotate(*angles, basis=basis)
    expected = treams.rotate(*angles, basis=to_oracle(basis))
    assert_allclose(actual, expected, rtol=1e-10, atol=2e-12)
    assert_allclose(actual.conj().T @ actual, np.eye(len(basis)), atol=2e-13)


@pytest.mark.physics
@pytest.mark.parametrize("theta", [1e-10, -1e-10, 1e-100])
def test_small_angle_generator(theta):
    # Upstream rounds cos(theta) to one here and loses these first-order entries.
    basis = tr.SphericalBasis([(1, m, 1) for m in (-1, 0, 1)])
    actual = tr.operators.rotate(0, theta, 0, basis=basis)
    assert_allclose(actual[0, 1], theta / np.sqrt(2), rtol=1e-12, atol=0)
    assert_allclose(actual[1, 0], -theta / np.sqrt(2), rtol=1e-12, atol=0)


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize("spherical", [True, False])
def test_partial_basis_rotation_pullback(spherical):
    if spherical:
        basis = tr.SphericalBasis.default(3, 2)
    else:
        basis = tr.CylindricalBasis.default([0.1, 0.2], 3, 2)
    to = type(basis)(basis.modes[::3], basis.positions)
    source = type(basis)(basis.modes[1::2], basis.positions)
    angles = np.array([0.2, 0.7 if spherical else 0, -0.3])

    def record(angles):
        return tr.diff.rotation(angles, to, source)

    actual = record(angles)[0]
    expected = treams.rotate(*angles, basis=(to_oracle(to), to_oracle(source)))
    assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)
    rng = np.random.default_rng(42)
    g = complex_normal(rng, actual.shape)
    # Cylindrical waves only rotate about the z axis: theta stays zero.
    check_pullback(
        record,
        angles,
        directions=(np.array([0.1, 0.3 if spherical else 0, -0.2]),),
        cotangents=g,
        step=1e-5,
        rtol=1e-8,
        atol=1e-9,
    )


@pytest.mark.physics
@pytest.mark.gradients
@pytest.mark.parametrize("angles", [(0.3, -1.1, 2.0), (0.4, 0.0, -0.2)])
def test_advect_rotation_composition(angles):
    # Rust rotation_group_and_adjoint checks random angles; this composes the
    # Advect adapter, including at the pole theta = 0.
    basis = tr.SphericalBasis.default(1)
    matrix = complex_normal(np.random.default_rng(14), (6, 6))

    def objective(angles):
        rotation = ad.rotation(angles, destination=basis)
        inverse = ad.rotation(-angles[::-1], destination=basis)
        transformed = rotation @ matrix @ inverse
        return anp.sum(anp.real(transformed[0, :2] * anp.conj(transformed[0, :2])))

    check_gradient(
        objective,
        advect.grad(objective),
        np.array(angles),
        directions=(np.array([0.2, -0.1, 0.3]),),
        step=1e-5,
        rtol=2e-6,
        atol=2e-8,
    )


@pytest.mark.physics
@pytest.mark.interface
def test_sphere_rotation_invariance_and_cylinder_axis_constraint():
    sphere = tr.TMatrix.sphere(4, 1.3, 0.2, [3, 1])
    assert_allclose(sphere.rotate(0.2, 0.7, -0.3).array, sphere.array, atol=1e-14)
    cylinder = tr.CylindricalTMatrix.cylinder([0.2], 3, 1.3, [0.2], [3, 1])
    assert_allclose(cylinder.rotate(0.3).array, cylinder.array, atol=1e-14)
    with pytest.raises(ValueError, match="theta"):
        cylinder.rotate(0.1, 0.2)


def _euler(phi, theta, psi):
    def about_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    c, s = np.cos(theta), np.sin(theta)
    about_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return about_z(phi) @ about_y @ about_z(psi)


@pytest.mark.physics
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@settings(max_examples=15)
@given(
    phi=st.floats(-np.pi, np.pi),
    theta=st.floats(0, np.pi),
    psi=st.floats(-np.pi, np.pi),
)
def test_rotated_cluster_response_is_the_rotated_geometry(poltype, phi, theta, psi):
    # Rotating a solved sphere cluster's local channels (origins fixed) equals
    # solving the cluster with its positions rotated by R = Rz(phi)Ry(theta)Rz(psi).
    kappa = 0.1 if poltype == "helicity" else 0
    particles = [
        tr.TMatrix.sphere(2, 1.3, radius, [(3, 1.2, kappa), 1], poltype)
        for radius in (0.2, 0.25)
    ]
    positions = np.array([[0, 0, 0], [0.5, 0.3, 0.6]])
    solved = tr.Cluster(particles, positions=positions).solve()
    rotation = _euler(phi, theta, psi)
    moved = tr.Cluster(particles, positions=positions @ rotation.T).solve()
    rotated = solved.rotate(phi, theta, psi)
    assert_allclose(rotated.array, moved.array, rtol=0, atol=1e-13)
    assert_allclose(
        rotated.rotate(-psi, -theta, -phi).array, solved.array, rtol=0, atol=1e-13
    )
