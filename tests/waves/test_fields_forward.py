"""Forward derivatives of multipole field samples and field operators."""

from types import SimpleNamespace

import numpy as np
import pytest

from treams_rs import CylindricalBasis, SphericalBasis, diff
from treams_rs.testing import check_pushforward

from _support import (
    arrange,
    assert_reusable_context,
    assert_saved_context,
    assert_tree_allclose,
)


def _record_case(family, operator, *, singular=False, poltype="helicity", native=False):
    axial = family == "cylindrical-axial"
    positions = np.array([[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]])
    basis = (
        SphericalBasis.default(2, 2, positions)
        if family == "spherical"
        else CylindricalBasis.default([0.2, -0.3], 2, 2, positions)
    )
    coefficients = np.arange(1, len(basis) + 1) * (0.03 + 0.01j)
    points = np.array([[0.8, 0.4, 0.7], [0.6, -0.5, 0.9]])
    if not singular:
        points[1] = positions[0]
    ks = np.array([1.2 + 0.1j, 1.4 + 0.05j])
    if poltype == "parity":
        ks[1] = ks[0]
    parameters = [points, positions, ks]
    if not operator:
        parameters.insert(0, coefficients)
    if axial:
        parameters.append(np.array([mode[1] for mode in basis.modes]))

    def record(*values):
        if axial:
            *values, kzs = values
            modes = [
                (p, kz, m, pol)
                for (p, _, m, pol), kz in zip(basis.modes, kzs, strict=True)
            ]
        else:
            modes = basis.modes
        if operator:
            points, origins, ks = values
        else:
            coefficients, points, origins, ks = values
        moved = type(basis)(modes, origins)
        kwargs = {"singular": singular, "poltype": poltype}
        if operator:
            value, context = diff.field_operator(points, moved, ks, **kwargs)
        else:
            value, context = diff.field(coefficients, points, moved, ks, **kwargs)
        if axial and not native:
            context = SimpleNamespace(
                pushforward=context.pushforward_axial,
                pullback=context.pullback_axial,
            )
        return value, context

    rng = np.random.default_rng(723)
    directions = [
        0.1
        * (
            rng.normal(size=value.shape)
            + (1j * rng.normal(size=value.shape) if np.iscomplexobj(value) else 0)
        )
        for value in parameters
    ]
    if poltype == "parity":
        directions[-2 if axial else -1][:] = directions[-2 if axial else -1][0]
    return record, parameters, directions


@pytest.mark.gradients
@pytest.mark.parametrize("family", ["spherical", "cylindrical", "cylindrical-axial"])
@pytest.mark.parametrize("operator", [False, True])
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_field_pushforward_matches_difference_and_adjoint(
    family, operator, singular, poltype
):
    record, parameters, directions = _record_case(
        family, operator, singular=singular, poltype=poltype
    )
    check_pushforward(
        record, *parameters, directions=tuple(directions), rtol=2e-6, atol=2e-8
    )


@pytest.mark.interface
@pytest.mark.parametrize("family", ["spherical", "cylindrical", "cylindrical-axial"])
@pytest.mark.parametrize("operator", [False, True])
@pytest.mark.parametrize("layout", ["F", "reversed"])
def test_field_pushforward_preserves_context_and_accepts_layouts(
    family, operator, layout
):
    record, parameters, directions = _record_case(family, operator)
    value, context = record(*parameters)
    expected = record(*parameters)[1].pushforward(*directions)
    assert_reusable_context(
        SimpleNamespace(pullback=context.pushforward),
        tuple(directions),
        expected,
        rtol=0,
    )
    for index, direction in enumerate(directions):
        if not np.iscomplexobj(direction):
            invalid = list(directions)
            invalid[index] = direction.astype(complex) + 1j
            with pytest.raises(ValueError, match="real"):
                context.pushforward(*invalid)
    actual = context.pushforward(
        *(arrange(direction, layout) for direction in directions)
    )
    assert actual.shape == value.shape
    np.testing.assert_array_equal(actual, expected)
    cotangent = np.ones_like(value)
    expected_gradients = record(*parameters)[1].pullback(cotangent)
    assert_tree_allclose(
        context.pullback(cotangent), expected_gradients, rtol=0, strict=True
    )
    np.testing.assert_array_equal(context.pushforward(*directions), expected)


@pytest.mark.interface
@pytest.mark.gradients
@pytest.mark.parametrize("family", ["spherical", "cylindrical", "cylindrical-axial"])
@pytest.mark.parametrize("operator", [False, True])
@pytest.mark.parametrize("singular", [False, True])
def test_field_saved_state_preserves_derivatives_and_has_fixed_shape(
    family, operator, singular
):
    record, parameters, directions = _record_case(
        family, operator, singular=singular, native=True
    )
    value, context = record(*parameters)
    context_type = type(context)
    points, positions = parameters[:2] if operator else parameters[1:3]
    modes = value.shape[-1] if operator else len(parameters[0])
    size = context_type._state_spec(
        modes, len(positions), len(points), family != "spherical"
    )
    state = assert_saved_context(
        context,
        directions,
        np.ones_like(value) * (0.3 + 0.2j),
        size=size,
        suffix="_axial" if family == "cylindrical-axial" else "",
    )
    shifted = [
        value + 0.03 * direction
        for value, direction in zip(parameters, directions, strict=True)
    ]
    assert record(*shifted)[1]._state().shape == state.shape
    bad_header = state.copy()
    bad_header[:8] = np.uint8(255)
    for malformed in (state[:-1], np.r_[state, np.uint8(0)], bad_header):
        with pytest.raises(ValueError, match="state"):
            context_type._from_state(malformed)
