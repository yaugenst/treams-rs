"""Public numerical namespaces carry full array JVPs through every adapter."""

import contextlib

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_array_dispatch import CASES
from test_framework_physics import Engine

from treams_rs import cw, pw, sw

from _support import jax_x64

pytestmark = [
    pytest.mark.gradients,
    pytest.mark.filterwarnings(
        "ignore:`torch.jit.script` is deprecated.*:FutureWarning:torch.jit._script"
    ),
]


@pytest.fixture(scope="module", params=["advect", "jax", "torch", "autograd"])
def engine(request):
    pytest.importorskip(request.param)
    with jax_x64() if request.param == "jax" else contextlib.nullcontext():
        yield Engine(request.param)


def _check_forward(engine, function, parameters):
    parameters = np.asarray(parameters, dtype=np.float64)
    direction = np.linspace(0.2, -0.1, parameters.size).reshape(parameters.shape)
    value, tangent = engine.jvp(lambda _tr, x: function(x), parameters, direction)
    step = 1e-5
    expected = (
        function(parameters + step * direction)
        - function(parameters - step * direction)
    ) / (2 * step)
    assert_allclose(value, function(parameters), rtol=2e-11, atol=2e-12)
    assert_allclose(tangent, expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("name", CASES)
def test_array_namespace_jvp_matches_finite_difference(engine, name):
    # Reuse the value/reverse coverage matrix so newly supported functions get
    # a forward check without maintaining a second capability list.
    function, parameter = CASES[name]
    _check_forward(engine, function, parameter)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(
            lambda x: sw.rotate(np.array([[1], [2]]), 1, 1, [1, 2], 0, 1, *x),
            id="spherical_rotation_mask",
        ),
        pytest.param(
            lambda x: cw.rotate(0.2, np.array([[1], [2]]), 1, 0.2, [1, 2], 1, x[0]),
            id="cylindrical_rotation_mask",
        ),
        pytest.param(
            lambda x: cw.translate(
                x[0], [1, 2], 1, x[0], 0, 1, 0.7 + x[0], x[1], x[2], singular=False
            ),
            id="cylindrical_translation_matching_axial_labels",
        ),
        pytest.param(
            lambda x: pw.translate(np.array([[0.2], [0.5]]), 0.3, 0.8, x, x[1], x[2]),
            id="plane_translation_broadcast",
        ),
        pytest.param(
            lambda x: pw.to_sw([1, 2], [0, 1], 1, *x, 1),
            id="plane_to_spherical",
        ),
        pytest.param(
            lambda x: pw.to_cw(x[2], [1, 2], 1, *x, 1),
            id="plane_to_cylindrical",
        ),
        pytest.param(
            lambda x: pw.to_cw(x[2] + 0.1, [1, 2], 1, *x, 0),
            id="plane_to_cylindrical_unmatched",
        ),
        pytest.param(
            lambda x: pw.permute_xyz(*x, [0, 1], 0, poltype="parity"),
            id="plane_permutation_selection",
        ),
        pytest.param(
            lambda x: sw.periodic_to_cw(
                0.2 * x[0], [0, 1], 1, [1, 2], [0, 1], 1, 1.2 + x[1], 2.0 + x[2]
            ),
            id="periodic_spherical_to_cylindrical",
        ),
        pytest.param(
            lambda x: cw.to_sw([1, 2], [0, 1], 1, 0.2, [0, 1], 1, x[:, None] + 1.2),
            id="cylindrical_to_spherical_broadcast",
        ),
    ],
)
def test_wave_namespace_jvp_matches_finite_difference(engine, function):
    _check_forward(engine, function, [0.6, 0.8, 1.1])
