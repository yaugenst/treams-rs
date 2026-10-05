"""Lattice directions through broadcast sums, periodic couplings and custom tables."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs.testing import check_pushforward

pytestmark = pytest.mark.gradients


@pytest.mark.parametrize(
    "spherical,dim", [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2)]
)
@pytest.mark.parametrize("part", ["full", "real", "reciprocal", "direct"])
def test_lattice_sum_broadcast_pushforward(spherical, dim, part):
    coordinates = 3 if spherical else 2
    parameters = (
        np.array([[2.1 + 0.2j], [2.3 + 0.1j]]),
        np.linspace(0.1, 0.2, dim),
        np.diag(np.linspace(1.5, 1.7, dim)),
        np.array([[0.19, 0.11, 0.07], [0.21, -0.09, 0.05]])[:, :coordinates],
        np.array([[0.9 + 0.03j, 0.95 - 0.02j]]),
    )

    def record(*values):
        return tr.diff.lattice_sum(
            dim, [2, 3], -1, *values, spherical=spherical, part=part, shell=2
        )

    check_pushforward(record, *parameters, step=2e-6, rtol=4e-6, atol=3e-7)


@pytest.mark.physics
@pytest.mark.parametrize("spherical", [True, False])
def test_lattice_pushforward_preserves_split_and_scale_identities(spherical):
    parameters = (
        np.asarray(2.1 + 0.2j),
        np.array([0.1, 0.15]),
        np.array([[1.6, 0.1], [0.05, 1.7]]),
        np.array([0.19, 0.11, 0.07])[: 3 if spherical else 2],
        np.asarray(0.9 + 0.03j),
    )
    directions = tuple(np.full_like(value, 0.03) for value in parameters)

    def pushforward(part, tangents):
        _, context = tr.diff.lattice_sum(
            2, 2, -1, *parameters, spherical=spherical, part=part
        )
        return context.pushforward(*tangents)

    assert_allclose(
        pushforward("full", directions),
        pushforward("real", directions) + pushforward("reciprocal", directions),
        atol=1e-11,
        rtol=1e-11,
    )
    k, q, a, r, eta = parameters
    assert_allclose(
        pushforward("full", (-k, -q, a, r, np.zeros_like(eta))),
        0,
        atol=1e-11,
    )


@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_periodic_expansion_pushforward_including_axial_groups(cylindrical, shared):
    destination_positions = np.array([[0.12, 0.08, 0.15], [-0.17, 0.13, 0.07]])
    source_positions = np.array([[0.03, -0.09, -0.04]])
    ks = np.array([1.3 + 0.05j, 1.3 + 0.05j if shared else 1.5 + 0.07j])
    q = np.array([0.13, 0.07])
    a = np.array([[1.7, 0.08], [-0.02, 1.6]])

    def record(destination_positions, source_positions, ks, q, a, *axial):
        if cylindrical:
            kzs = axial[0]
            destination = tr.CylindricalBasis(
                [
                    (0, kzs[0], -1, 0),
                    (0, kzs[0], 1, 1),
                    (1, kzs[1], 0, 0),
                    (1, kzs[1], -1, 1),
                ],
                positions=destination_positions,
            )
            source = tr.CylindricalBasis(
                [(0, kz, 0, pol) for kz in kzs for pol in (0, 1)],
                positions=source_positions,
            )
        else:
            destination = tr.SphericalBasis(
                [(0, 1, -1, 0), (0, 1, 0, 1), (1, 1, 1, 0)],
                positions=destination_positions,
            )
            source = tr.SphericalBasis(
                [(0, 1, 0, 0), (0, 1, 1, 1)], positions=source_positions
            )
        value, context = tr.diff.lattice_expansion(
            destination, source, ks, q, a, eta=0.9
        )
        if cylindrical:
            context = SimpleNamespace(
                pushforward=context.pushforward_axial, pullback=context.pullback_axial
            )
        return value, context

    parameters = (destination_positions, source_positions, ks, q, a)
    if cylindrical:
        parameters += (np.array([-0.2, 0.35]),)
    check_pushforward(record, *parameters, step=2e-6, rtol=4e-6, atol=3e-7)


@pytest.mark.parametrize("poltype,channels", [("helicity", 2), ("parity", 1)])
def test_table_pushforward_is_the_same_linear_angular_map(poltype, channels):
    destination = tr.SphericalBasis(
        [(0, 1, -1, 0), (1, 2, 1, 1)], positions=[[0, 0, 0], [0.2, 0.3, 0.4]]
    )
    source = tr.SphericalBasis([(0, 2, -2, 0), (0, 1, 0, 1)])
    rng = np.random.default_rng(10)
    shape = (2, 1, channels, 25)
    table = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    tangent = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    tangent = tangent[..., ::-1].copy()[..., ::-1]

    def record(table):
        return tr.diff.lattice_expansion_from_table(
            table, destination, source, poltype=poltype
        )

    check_pushforward(record, table, directions=(tangent,))
    value, context = record(table)
    with pytest.raises(ValueError, match="tangent"):
        context.pushforward(tangent.reshape(-1))
    with pytest.raises(ValueError, match="tangent"):
        context.pushforward(np.full(shape, np.nan))
    assert_allclose(context.pushforward(tangent), record(tangent)[0], rtol=2e-13)
    cotangent = np.ones_like(value)
    gradient = context.pullback(cotangent)
    assert_allclose(context.pushforward(-tangent), -record(tangent)[0], rtol=2e-13)
    assert_allclose(context.pullback(-cotangent), -gradient)
    assert_allclose(context.pushforward(tangent), record(tangent)[0], rtol=2e-13)


@pytest.mark.interface
def test_lattice_pushforward_validates_real_tangents_and_preserves_context():
    parameters = (
        np.asarray(2.1 + 0.2j),
        np.array([0.1]),
        np.array([[1.6]]),
        np.array([0.19, 0.11, 0.07]),
        np.asarray(0.9),
    )
    _, context = tr.diff.lattice_sum(1, 2, -1, *parameters)
    directions = [np.zeros_like(value) for value in parameters]
    directions[1] = np.array([1j])
    with pytest.raises(ValueError, match="real"):
        context.pushforward(*directions)
    directions[1] = np.zeros(1)
    assert_allclose(context.pushforward(*directions), 0)
