"""Native radiation-channel JVPs, fixed labels, and reusable context validation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import treams_rs as tr

from _support import arrange


def channel_case(cylindrical, poltype="helicity", fixed_q=False):
    positions = np.array([[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1]])
    if cylindrical:
        basis = tr.CylindricalBasis.default([0.2, -0.3], 2, 2)
        q = np.array([[0.2, 0.1], [0.2, 2.5], [-0.3, -0.2], [-0.3, -2.4]])
        function = tr.diff.cylindrical_channels
    else:
        basis = tr.SphericalBasis.default(2, 2)
        q = np.array([[0.2, 0.1], [0.2, 2.5], [-0.3, -0.2], [-0.3, -2.4]])
        function = tr.diff.spherical_channels
    ks = np.array([1.3 + 0.1j, 1.4 + 0.15j])
    if poltype == "parity":
        ks[:] = ks[0]

    def record(positions, ks, q, measure):
        return function(
            type(basis)(basis.modes, positions),
            ks,
            q,
            [0, 1, 1, 0],
            float(measure),
            poltype=poltype,
            fixed_q=fixed_q,
        )

    direction = (
        np.array([[0.1, -0.2, 0.03], [-0.12, 0.13, -0.04]]),
        np.full(2, 0.1 + 0.03j),
        np.array([[0.03, 0.1], [-0.02, -0.2], [0.05, 0.3], [-0.04, 0.2]]),
        0.07,
    )
    return record, (positions, ks, q, 1.7), direction


@pytest.mark.gradients
@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("fixed_q", [False, True])
def test_channel_pushforward_matches_directional_difference_and_adjoint(
    cylindrical, poltype, fixed_q
):
    record, values, direction = channel_case(cylindrical, poltype, fixed_q)
    value, context = record(*values)
    tangent = context.pushforward(*direction)
    # Fixed coordinates have zero derivatives even if their supplied tangent is
    # nonzero. Hold those values fixed when computing the numerical reference.
    effective_q = direction[2].copy()
    if cylindrical:
        effective_q[:, 0] = 0
    if fixed_q:
        effective_q[:] = 0
    effective = (*direction[:2], effective_q, direction[3])
    step = 1e-5
    plus = record(*(v + step * d for v, d in zip(values, effective, strict=True)))[0]
    minus = record(*(v - step * d for v, d in zip(values, effective, strict=True)))[0]
    assert tangent.shape == value.shape
    assert_allclose(tangent, (plus - minus) / (2 * step), rtol=2e-8, atol=2e-8)
    rng = np.random.default_rng(21)
    cotangent = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradient = context.pullback(cotangent)
    expected = sum(np.vdot(g, d).real for g, d in zip(gradient, direction, strict=True))
    assert_allclose(np.vdot(cotangent, tangent).real, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(context.pushforward(*(-d for d in direction)), -tangent)
    for actual, expected in zip(context.pullback(-cotangent), gradient, strict=True):
        assert_allclose(actual, -expected)
    assert_allclose(context.pushforward(*direction), tangent)


@pytest.mark.interface
@pytest.mark.parametrize("cylindrical", [False, True])
def test_channel_pushforward_rejected_tangents_leave_context_available(cylindrical):
    record, values, direction = channel_case(cylindrical)
    value, context = record(*values)
    for index in range(4):
        bad = list(direction)
        bad[index] = np.full_like(bad[index], np.nan)
        with pytest.raises(ValueError, match="tangent"):
            context.pushforward(*bad)
    with pytest.raises(ValueError, match="tangent"):
        context.pushforward(direction[0][:1], *direction[1:])
    with pytest.raises(ValueError, match="tangent"):
        context.pushforward(*direction[:2], direction[2][:1], direction[3])
    tangent = context.pushforward(*direction)
    assert tangent.shape == value.shape
    assert_allclose(context.pushforward(*direction), tangent)
    assert_allclose(context.pushforward(*(-d for d in direction)), -tangent)


@pytest.mark.interface
@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("layout", ["F", "strided", "reversed"])
def test_channel_pushforward_accepts_strided_tangents(cylindrical, layout):
    record, values, direction = channel_case(cylindrical)
    tangents = (*[arrange(value, layout) for value in direction[:3]], direction[3])
    actual = record(*values)[1].pushforward(*tangents)
    expected = record(*values)[1].pushforward(*direction)
    assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.gradients
@pytest.mark.parametrize("fixed_q", [False, True])
def test_spherical_channel_pushforward_at_fixed_normal_incidence(fixed_q):
    record, values, direction = channel_case(False, fixed_q=fixed_q)
    values = (*values[:2], np.zeros_like(values[2]), values[3])
    if not fixed_q:
        direction = (*direction[:2], np.zeros_like(direction[2]), direction[3])
    tangent = record(*values)[1].pushforward(*direction)
    step = 1e-5
    effective = (*direction[:2], np.zeros_like(direction[2]), direction[3])
    plus = record(*(v + step * d for v, d in zip(values, effective, strict=True)))[0]
    minus = record(*(v - step * d for v, d in zip(values, effective, strict=True)))[0]
    assert_allclose(tangent, (plus - minus) / (2 * step), rtol=2e-8, atol=2e-8)
