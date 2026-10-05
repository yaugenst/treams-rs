"""Native wave JVPs: reference directions, adjoint duality and shape contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import diff
from treams_rs.testing import check_pushforward

from _support import assert_saved_context

pytestmark = pytest.mark.gradients


def _check(record, inputs, directions, *, axial=False):
    if not axial:
        check_pushforward(
            record, *inputs, directions=directions, step=2e-6, rtol=2e-7, atol=2e-8
        )
    value, context = record(*inputs)
    push = context.pushforward_axial if axial else context.pushforward
    actual = push(*directions)
    if axial:
        h = 2e-6
        plus = record(*(x + h * dx for x, dx in zip(inputs, directions, strict=True)))[
            0
        ]
        minus = record(*(x - h * dx for x, dx in zip(inputs, directions, strict=True)))[
            0
        ]
        assert_allclose(actual, (plus - minus) / (2 * h), rtol=2e-7, atol=2e-8)
    cotangent = np.arange(value.size).reshape(value.shape) * (0.07 + 0.03j) + 0.2j
    pull = context.pullback_axial if axial else context.pullback
    gradients = pull(cotangent)
    np.testing.assert_array_equal(push(*directions), actual)
    if hasattr(context, "_state"):
        assert_saved_context(
            context, directions, cotangent, suffix="_axial" if axial else ""
        )
    if len(inputs) == 1:
        gradients = (gradients,)
    expected = sum(
        np.vdot(g, d).real for g, d in zip(gradients, directions, strict=True)
    )
    assert_allclose(np.vdot(cotangent, actual).real, expected, rtol=2e-12, atol=2e-12)
    assert_allclose(push(*(-d for d in directions)), -actual)
    repeated = pull(-cotangent)
    if len(inputs) == 1:
        repeated = (repeated,)
    for gradient, reference in zip(repeated, gradients, strict=True):
        assert_allclose(gradient, -np.asarray(reference))
    assert_allclose(push(*directions), actual)


@pytest.mark.parametrize("family", ["spherical", "cylindrical", "conversion"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("singular", [False, True])
def test_expansion_pushforward(family, shared, singular):
    if family == "conversion" and singular:
        pytest.skip("cylindrical-to-spherical expansions are regular")
    destination = np.array([[0.8, -0.3, 0.6], [0.2, 0.9, -0.1]])
    source = np.array([[-0.4, 0.2, -0.1]])
    ks = np.array([1.2 + 0.05j, (1.2 + 0.05j) if shared else (1.4 + 0.03j)])
    inputs = (destination, source, ks)
    directions = (
        np.array([[0.2, 0.1, -0.2], [-0.1, 0.3, 0.2]]),
        np.array([[0.3, -0.2, 0.1]]),
        np.array([0.2 + 0.07j, -0.1 + 0.04j]),
    )

    def record(destination, source, ks):
        to = (
            tr.CylindricalBasis.default([0.3], 2, 2, positions=destination)
            if family == "cylindrical"
            else tr.SphericalBasis.default(2, 2, positions=destination)
        )
        from_ = (
            tr.SphericalBasis.default(1, positions=source)
            if family == "spherical"
            else tr.CylindricalBasis.default([0.3], 1, positions=source)
        )
        return diff.expansion(to, from_, ks, singular=singular)

    _check(record, inputs, directions)


def test_cylindrical_expansion_shared_axial_directions():
    destination = np.array([[0.8, -0.3, 0.6]])
    source = np.array([[-0.4, 0.2, -0.1]])
    ks = np.array([1.2 + 0.05j, 1.2 + 0.05j])
    kz = np.array([-0.3, 0.2])

    def record(destination, source, ks, kz):
        to = tr.CylindricalBasis.default(kz, 2, positions=destination)
        from_ = tr.CylindricalBasis.default(kz, 1, positions=source)
        return diff.expansion(to, from_, ks)

    _check(
        record,
        (destination, source, ks, kz),
        (
            destination * 0.1,
            source * -0.2,
            np.array([0.2 + 0.1j, -0.3 + 0.2j]),
            np.array([0.1, -0.2]),
        ),
        axial=True,
    )


@pytest.mark.parametrize("spherical", [False, True])
def test_rotation_pushforward(spherical):
    basis = (
        tr.SphericalBasis.default(3)
        if spherical
        else tr.CylindricalBasis.default([0.2], 3)
    )
    angles = np.array([0.2, 0.7 if spherical else 0.0, -0.3])
    direction = np.array([0.3, -0.2 if spherical else 0.0, 0.1])
    _check(lambda angles: diff.rotation(angles, basis), (angles,), (direction,))


def test_periodic_conversion_pushforward():
    destination = np.array([[0.8, -0.3, 0.6]])
    source = np.array([[-0.4, 0.2, -0.1]])
    ks = np.array([1.2 + 0.05j, 1.4 + 0.03j])
    kz = np.array([-0.3, 0.2])

    def record(destination, source, ks, kz, period):
        to = tr.CylindricalBasis([(0, kz[0], -1, 0), (0, kz[1], 1, 1)], destination)
        from_ = tr.SphericalBasis.default(2, positions=source)
        return diff.periodic_to_cw(to, from_, ks, float(period))

    _check(
        record,
        (destination, source, ks, kz, np.array(1.4)),
        (
            destination * 0.1,
            source * -0.2,
            np.array([0.2 + 0.1j, -0.3 + 0.2j]),
            np.array([0.1, -0.2]),
            np.array(0.3),
        ),
    )


@pytest.mark.parametrize("spherical", [False, True])
@pytest.mark.parametrize("singular", [False, True])
def test_translation_broadcast_pushforward(spherical, singular):
    kr = np.array([[1.1 + 0.1j], [1.5 - 0.05j]])
    phi = np.array([0.2, 0.4, 0.7])
    if spherical:

        def record(kr, theta, phi):
            return diff.spherical_translation(
                kr,
                theta,
                phi,
                destination=(2, 1, 1),
                source=(1, -1, 1),
                singular=singular,
            )

        inputs = (kr, np.array(0.6), phi)
    else:

        def record(kr, phi, z, kz):
            return diff.cylindrical_translation(
                kr, phi, z, kz, order=2, singular=singular
            )

        inputs = (kr, phi, np.array(0.3), np.array(0.2))
    inputs = tuple(np.asarray(x, dtype=np.complex128) for x in inputs)
    _check(record, inputs, tuple(np.ones_like(x) * (0.1 + 0.03j) for x in inputs))


def test_expansion_invalid_tangent_preserves_context():
    basis = tr.SphericalBasis.default(1)
    _, context = diff.expansion(basis, basis, [1.0, 1.0])
    with pytest.raises(ValueError, match="tangent"):
        context.pushforward(np.zeros((2, 3)), np.zeros((1, 3)), np.zeros(2))
    with pytest.raises(ValueError, match="real"):
        context.pushforward(np.ones((1, 3)) * 1j, np.zeros((1, 3)), np.zeros(2))
    assert_allclose(
        context.pushforward(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(2)), 0
    )
    gradient = context.pullback(np.ones((len(basis), len(basis))))
    assert all(np.isfinite(value).all() for value in gradient)
    assert_allclose(
        context.pushforward(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(2)), 0
    )
