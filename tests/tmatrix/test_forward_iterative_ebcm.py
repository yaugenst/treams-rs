"""Forward directions through matrix-free solves and sampled EBCM surfaces."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import _native
from treams_rs.iterative import SphereCluster
from treams_rs.testing import check_pushforward

from _support import complex_normal

pytestmark = pytest.mark.gradients


@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("radial_area_factor", [False, True])
def test_ebcm_pushforward_handles_all_dynamic_inputs(singular, radial_area_factor):
    nodes, weights = np.polynomial.legendre.leggauss(24)
    theta = (nodes + 1) * np.pi / 2
    radii = 0.3 * (1 + 0.2 * np.cos(theta) ** 2)
    slopes = -0.12 * np.cos(theta) * np.sin(theta)
    ks = np.array([[1.7 + 0.1j, 1.8 + 0.1j], [1.2 + 0.02j, 1.3 + 0.02j]])
    zs = np.array([0.8 + 0.01j, 1.1 - 0.02j])

    def record(*args):
        return tr.diff.ebcm_qmat(
            *args,
            theta=theta,
            weights=weights * np.pi / 2,
            destination=tr.SphericalBasis.default(2),
            source=tr.SphericalBasis.default(1),
            singular=singular,
            radial_area_factor=radial_area_factor,
        )

    check_pushforward(record, radii, slopes, ks, zs, seed=418)
    directions = [np.zeros_like(value) for value in (radii, slopes, ks, zs)]
    value, context = record(radii, slopes, ks, zs)
    with pytest.raises(ValueError, match="tangent"):
        context.pushforward(directions[0][:-1], *directions[1:])
    with pytest.raises(ValueError, match="tangent"):
        context.pushforward(directions[0] + 1j, *directions[1:])
    with pytest.raises(ValueError, match="tangent"):
        context.pushforward(*directions[:2], directions[2] + np.nan, directions[3])
    assert_allclose(context.pushforward(*directions), 0, atol=0, rtol=0)
    directions[0] = np.full_like(radii, 0.03)
    tangent = context.pushforward(*directions)
    cotangent = np.full_like(value, 0.4 + 0.1j)
    gradient = context.pullback(cotangent)
    assert_allclose(
        np.vdot(cotangent, tangent).real,
        sum(np.vdot(g, d).real for g, d in zip(gradient, directions, strict=True)),
    )
    assert_allclose(context.pushforward(*directions), tangent, atol=0, rtol=0)


@pytest.mark.parametrize(
    "columns,vector", [(1, False), (3, False), (11, False), (1, True)]
)
def test_iterative_pushforward_matches_dense_with_certified_solves(columns, vector):
    radii = np.array([0.2, 0.24])
    epsilon = np.array([2.3 + 0.04j, 3.1 + 0.02j])
    positions = np.array([[0.1, 0.0, -0.1], [1.2, 0.2, 0.4]])
    directions = (
        0.07,
        np.array([0.03, -0.02]),
        np.array([0.2 - 0.03j, -0.1 + 0.04j]),
        np.array([[0.03, 0.02, -0.04], [-0.02, 0.01, 0.03]]),
    )
    rng = np.random.default_rng(691)
    storage = np.asfortranarray(complex_normal(rng, (32, columns * 2)))
    incident = storage[:, ::2]
    dincident = np.asfortranarray(complex_normal(rng, incident.shape)) * 0.03
    if vector:
        incident = incident[:, 0]
        dincident = dincident[:, 0]
    operator = SphereCluster(2, 1.3, radii, epsilon, positions)
    solution, context = operator.record(incident, rtol=2e-12)
    dense, dense_context = _native.sphere_cluster(2, 1.3, radii, epsilon, positions)
    ddense = dense_context.pushforward(*directions)
    tangent = context.pushforward(*directions, dincident)
    assert_allclose(solution.coefficients, dense @ incident, rtol=2e-9, atol=1e-13)
    assert_allclose(
        tangent.coefficients,
        ddense @ incident + dense @ dincident,
        rtol=2e-9,
        atol=1e-13,
    )
    assert tangent.coefficients.shape == incident.shape
    assert len(tangent.convergence) == columns
    assert all(
        report.residual_norm <= 2e-12 * report.rhs_norm
        for report in tangent.convergence
    )
    cotangent = complex_normal(rng, incident.shape)
    gradient = context.pullback(cotangent)
    assert_allclose(
        np.vdot(cotangent, tangent.coefficients).real,
        sum(
            np.vdot(g, d).real
            for g, d in zip(gradient[:-1], (*directions, dincident), strict=True)
        ),
        rtol=2e-9,
        atol=1e-13,
    )
    assert_allclose(
        context.pushforward(*directions, dincident).coefficients,
        tangent.coefficients,
        atol=0,
        rtol=0,
    )


@pytest.mark.interface
def test_iterative_rejects_bad_directions_and_context_remains_usable():
    operator = SphereCluster(
        1,
        1.3,
        [0.2, 0.24],
        [2.3 + 0.04j, 3.1 + 0.02j],
        [[0.1, 0.0, -0.1], [1.2, 0.2, 0.4]],
    )
    _, context = operator.record(np.ones(12, dtype=complex))
    directions = (0.0, np.zeros(2), np.zeros(2), np.zeros((2, 3)), np.zeros(12))
    for index, bad in [
        (0, np.nan),
        (1, np.zeros(3)),
        (2, np.full(2, np.inf)),
        (3, np.zeros((2, 2))),
        (4, np.zeros(13)),
    ]:
        invalid = list(directions)
        invalid[index] = bad
        with pytest.raises(ValueError, match="tangent"):
            context.pushforward(*invalid)
    tangent = context.pushforward(*directions)
    assert_allclose(tangent.coefficients, 0, atol=0, rtol=0)
    assert tangent.convergence[0] == (0, 0.0, 0.0)
