"""Ordinary lattice sums retain the selected framework's first derivatives."""

import contextlib
import importlib

import advect as ad
import numpy as np
import pytest
from numpy.testing import assert_allclose

from treams_rs import lattice
from treams_rs.testing import check_gradient

from _support import jax_x64

pytestmark = pytest.mark.gradients


@pytest.fixture(params=["advect", "jax", "torch", "autograd"])
def value_and_grad(request):
    framework = pytest.importorskip(request.param)
    scope = jax_x64() if request.param == "jax" else contextlib.nullcontext()

    def evaluate(function, value):
        if request.param == "torch":
            tensor = framework.tensor(
                value, dtype=framework.float64, requires_grad=True
            )
            result = function(tensor)
            (gradient,) = framework.autograd.grad(result, tensor)
            return result.detach().numpy(), gradient.detach().numpy()
        operation = framework.value_and_grad(function)
        if request.param == "jax":
            operation = framework.jit(operation)
        result, gradient = operation(np.asarray(value, dtype=np.float64))
        return np.asarray(result), np.asarray(gradient)

    with scope:
        yield evaluate


@pytest.mark.parametrize("prefix", ["lsum", "realsum", "recsum", "dsum"])
@pytest.mark.parametrize(
    "suffix,labels,kpar,a,r",
    [
        ("sw1d", (2,), 0.13, 1.7, 0.19),
        ("sw1d_shift", (2, 1), 0.13, 1.7, [0.19, 0.11, 0.07]),
        ("sw2d", (2, 1), [0.13, 0.1], [[1.7, 0.0], [0.0, 1.8]], [0.19, 0.11]),
        (
            "sw2d_shift",
            (2, 1),
            [0.13, 0.1],
            [[1.7, 0.0], [0.0, 1.8]],
            [0.19, 0.11, 0.07],
        ),
        (
            "sw3d",
            (2, 1),
            [0.13, 0.1, 0.12],
            np.diag([1.7, 1.8, 1.9]),
            [0.19, 0.11, 0.07],
        ),
        ("cw1d", (1,), 0.13, 1.7, 0.19),
        ("cw1d_shift", (1,), 0.13, 1.7, [0.19, 0.11]),
        ("cw2d", (1,), [0.13, 0.1], [[1.7, 0.0], [0.0, 1.8]], [0.19, 0.11]),
    ],
)
def test_each_named_sum_preserves_values_and_gradients(
    prefix, suffix, labels, kpar, a, r
):
    function = getattr(lattice, prefix + suffix)
    parameter = 2 if prefix == "dsum" else 0.9

    def objective(k):
        value = function(*labels, k + 0.2j, kpar, a, r, parameter)
        return abs(value) ** 2

    point = np.asarray(2.1)
    value, gradient = ad.value_and_grad(objective)(point)
    assert_allclose(value, objective(point), rtol=2e-12)
    # Lattice derivatives vary strongly with frequency near diffraction orders;
    # the native convergence tolerance requires this finite-difference tolerance.
    check_gradient(objective, lambda _: gradient, point, rtol=5e-5, atol=3e-7)


@pytest.mark.parametrize("prefix", ["lsum", "realsum", "recsum", "dsum"])
def test_nested_geometry_and_broadcast_gradients(value_and_grad, prefix):
    function = getattr(lattice, prefix + "cw")

    def objective(x):
        parameter = 2 if prefix == "dsum" else x[6]
        value = function(
            2,
            [1, 2],
            x[0] + 0.2j,
            [x[1], 0.1],
            [x[2], x[3]],
            [x[4], x[5]],
            parameter,
        )
        return (abs(value) ** 2).sum()

    point = np.array([2.1, 0.13, 1.7, 1.8, 0.19, 0.11, 0.9])
    value, gradient = value_and_grad(objective, point)
    assert_allclose(value, objective(point), rtol=2e-12)
    check_gradient(objective, lambda _: gradient, point, rtol=5e-5, atol=3e-7)
    if prefix in ("lsum", "dsum"):
        assert_allclose(gradient[-1], 0, atol=1e-12)


def test_static_lattice_metadata_with_a_moving_bloch_vector(value_and_grad):
    from treams_rs import Lattice

    def objective(kpar):
        value = lattice.lsumsw1d_shift(
            2, 1, 2.1 + 0.2j, kpar, Lattice(1.7), [0.19, 0.11, 0.07], 0.9
        )
        return (abs(value) ** 2).sum()

    point = np.asarray(0.13)
    value, gradient = value_and_grad(objective, point)
    assert_allclose(value, objective(point), rtol=2e-12)
    check_gradient(objective, lambda _: gradient, point, rtol=5e-5, atol=3e-7)


def test_static_wavevector_with_a_moving_period_and_scalar_shift(value_and_grad):
    from treams_rs import WaveVector

    def objective(x):
        value = lattice.lsumsw1d(2, 2.1 + 0.2j, WaveVector(0.13), x[0], x[1], 0.9)
        return abs(value) ** 2

    point = np.asarray([1.7, 0.19])
    value, gradient = value_and_grad(objective, point)
    assert_allclose(value, objective(point), rtol=2e-12)
    check_gradient(objective, lambda _: gradient, point, rtol=5e-5, atol=3e-7)


@pytest.mark.interface
def test_numpy_output_and_native_attributes_are_preserved():
    native = importlib.import_module("treams_rs._native")
    output = np.zeros(2, dtype=complex)
    result = lattice.dsumsw1d(
        2, [2.1, 2.3], 0.13, 1.7, 0.19, 2, out=output, where=[True, False]
    )
    assert result is output
    assert result[1] == 0
    assert_allclose(result[0], native.dsumsw1d(2, 2.1, 0.13, 1.7, 0.19, 2))
    assert lattice.dsumsw1d.nin == native.dsumsw1d.nin
    assert lattice.dsumsw1d.types == native.dsumsw1d.types
    assert type(lattice.dsumcw1d(1, 2.1, 0.13, 1.7, 0.19, 2)) is complex


@pytest.mark.interface
def test_framework_output_mutation_is_rejected(value_and_grad):
    def objective(k):
        return abs(lattice.lsumcw1d(1, k, 0.13, 1.7, 0.19, out=np.zeros(()))) ** 2

    with pytest.raises(TypeError, match="out"):
        value_and_grad(objective, np.asarray(2.1))
