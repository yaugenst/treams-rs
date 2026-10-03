"""Ordinary matrix methods preserve the same physics and analytic gradients."""

import functools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_transparent_api import engine as engine

import treams_rs as tr

pytestmark = pytest.mark.gradients


def _response():
    return tr.TMatrix(
        -0.1 * np.eye(6) + 0.003 * np.arange(36).reshape(6, 6) + 0.02j * np.eye(6),
        k0=1.2,
    )


def _objective(api, x, *, operation):
    if operation == "average_sphere":
        return api.sphere_tmatrix(
            k0=1.2, lmax=1, radius=x, material=3 + 0.1j
        ).average_cross_sections.scattering
    if operation == "average_cylinder":
        return api.cylinder_tmatrix(
            k0=1.2, kz=[0.0, 2.0], mmax=1, radius=x, material=3 + 0.1j
        ).average_cross_widths.extinction
    response = _response()
    if operation == "rotate":
        array = response.rotate(0.1, x, 0.3).array
        return abs(array[0, 1] + 0.2) ** 2
    if operation == "translate":
        array = response.translate([x, 0.02, 0.03]).array
        return abs(array[0, 1] + 0.2) ** 2
    if operation == "matmul":
        return sum(abs(response @ [x, 0, 0, 0, 0, 0]) ** 2)
    array = [[x * value.item() + 0.03j for value in row] for row in response.array]
    dynamic = api.TMatrix(array, k0=1.2)
    return getattr(dynamic, operation)


@pytest.mark.parametrize(
    "operation",
    [
        "average_sphere",
        "average_cylinder",
        "rotate",
        "translate",
        "matmul",
        "circular_dichroism",
        "duality_breaking",
        "electromagnetic_chirality",
    ],
)
def test_tmatrix_method_matches_numpy_and_finite_difference(engine, operation):
    function = functools.partial(_objective, operation=operation)
    x = 0.2
    value, gradient = engine.value_and_grad(function, x)
    step = 1e-6
    expected = (function(tr, x + step) - function(tr, x - step)) / (2 * step)
    assert_allclose(value, function(tr, x), rtol=1e-12, atol=1e-14)
    assert_allclose(gradient, expected, rtol=2e-5, atol=1e-8)
