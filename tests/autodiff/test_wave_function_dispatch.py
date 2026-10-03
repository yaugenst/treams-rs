"""Ordinary wave coefficient calls retain gradients through all adapters."""

import contextlib
import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

from treams_rs import cw, pw, sw

from _support import jax_x64

pytestmark = pytest.mark.gradients


@pytest.fixture(params=["advect", "jax", "torch", "autograd"])
def gradient(request):
    framework = pytest.importorskip(request.param)
    with jax_x64() if request.param == "jax" else contextlib.nullcontext():

        def evaluate(function, parameters):
            if request.param == "torch":
                tensor = framework.tensor(parameters, requires_grad=True)
                value = function(tensor)
                (gradient,) = framework.autograd.grad(value, tensor)
                return value.detach().numpy(), gradient.numpy()
            operation = framework.value_and_grad(function)
            if request.param == "jax":
                operation = framework.jit(operation)
            return tuple(np.asarray(value) for value in operation(parameters))

        yield evaluate


def _real_sum(value):
    if hasattr(value, "real"):
        return value.real.sum()
    # Autograd exposes real through its NumPy namespace, not ArrayBox.real.
    return importlib.import_module("autograd.numpy").real(value).sum()


def spherical_translation(x):
    return _real_sum(
        sw.translate([1, 2], [0, 1], 1, 1, 0, 1, x[0], x[1], x[2], singular=False)
    )


def spherical_rotation(x):
    return _real_sum(
        sw.rotate(np.array([[1], [2]]), 1, 1, [1, 2], 0, 1, x[0], x[1], x[2])
    )


def cylindrical_translation(x):
    return _real_sum(
        cw.translate(
            x[0], [1, 2], 1, x[0], 0, 1, 0.7 + x[0], x[1], x[2], singular=False
        )
    )


def cylindrical_rotation(x):
    return _real_sum(
        cw.rotate(0.2, np.array([[1], [2]]), 1, 0.2, np.array([[1], [2]]), 1, x)
    )


def plane_translation(x):
    return _real_sum(pw.translate(np.array([[0.2], [0.5]]), 0.3, 0.8, x, x[1], x[2]))


def plane_to_spherical(x):
    return _real_sum(pw.to_sw([1, 2], [0, 1], 1, x[0], x[1], x[2], 1))


def plane_to_cylindrical(x):
    return _real_sum(pw.to_cw(x[2], [1, 2], 1, x[0], x[1], x[2], 1))


def plane_permutation(x):
    return _real_sum(pw.permute_xyz(x[0], x[1], x[2], [0, 1], 0, poltype="parity"))


def spherical_to_periodic_cylindrical(x):
    return _real_sum(
        sw.periodic_to_cw(
            0.2 * x[0], [0, 1], 1, [1, 2], [0, 1], 1, 1.2 + x[1], 2.0 + x[2]
        )
    )


def cylindrical_to_spherical(x):
    return _real_sum(cw.to_sw([1, 2], [0, 1], 1, 0.2, [0, 1], 1, x[:, None] + 1.2))


def spherical_periodic_geometry(x):
    return _real_sum(
        sw.translate_periodic(
            1.3 + 0.1j,
            0.1 * x[0],
            1.7 + x[1],
            [0, 0, 0],
            ([1, 1], [0, 1], [1, 1]),
            rsin=[0.2, 0.1, 0.3 + 0.1 * x[2]],
            eta=0.7,
        )
    )


def cylindrical_periodic_geometry(x):
    return _real_sum(
        cw.translate_periodic(
            1.3 + 0.1j,
            0.1 * x[0],
            1.7 + x[1],
            [0, 0, 0],
            ([0.2, 0.2], [0, 1], [1, 1]),
            rsin=[0.2, 0.1, 0.3 + 0.1 * x[2]],
            eta=0.7,
        )
    )


@pytest.mark.parametrize(
    "function",
    [
        spherical_translation,
        spherical_rotation,
        cylindrical_translation,
        cylindrical_rotation,
        plane_translation,
        plane_to_spherical,
        plane_to_cylindrical,
        plane_permutation,
        spherical_to_periodic_cylindrical,
        cylindrical_to_spherical,
        spherical_periodic_geometry,
        cylindrical_periodic_geometry,
    ],
)
def test_wave_values_and_gradients_match_numpy(gradient, function):
    parameters = np.array([0.6, 0.8, 1.1])
    value, actual = gradient(function, parameters)
    step = 1e-5
    expected = np.array(
        [
            (function(parameters + direction) - function(parameters - direction))
            / (2 * step)
            for direction in step * np.eye(3)
        ]
    )
    assert_allclose(value, function(parameters), rtol=2e-11, atol=2e-12)
    assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)


def test_plain_numpy_keeps_output_argument_and_ufunc_attributes():
    output = np.empty(2, dtype=np.complex128)
    assert sw.translate(1, 0, 1, 1, 0, 1, [0.2, 0.3], 0.4, 0.2, out=output) is output
    assert pw.to_cw.nin == 7
    assert cw.periodic_to_pw.nin == 8


def test_autodiff_rejects_output_mutation():
    advect = importlib.import_module("advect")
    output = np.empty((), dtype=np.complex128)
    with pytest.raises(TypeError, match="out or where"):
        advect.grad(
            lambda radius: (
                sw.translate(1, 0, 1, 1, 0, 1, radius, 0.3, 0.2, out=output).real
            )
        )(0.4)


def test_missing_native_derivative_has_explicit_error():
    advect = importlib.import_module("advect")
    with pytest.raises(NotImplementedError, match="axial mode label"):
        advect.grad(lambda axial: cw.to_sw(1, 0, 1, axial, 0, 1, 1.3).real)(0.2)
    with pytest.raises(NotImplementedError, match="periodic_to_pw"):
        advect.grad(
            lambda axial: sw.periodic_to_pw(0.2, 0.3, axial, 1, 1, 0, 1, 2.0).real
        )(1.2)
