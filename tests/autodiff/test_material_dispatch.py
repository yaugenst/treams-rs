"""Material construction and derived optical constants preserve framework values."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_framework_physics import engine as engine
from test_framework_physics import real

import treams_rs as tr

pytestmark = pytest.mark.gradients


def refractive_index(_tr, epsilon):
    return real(tr.Material(epsilon).n)


def impedance(_tr, epsilon):
    return real(tr.Material(epsilon).impedance)


def helicity_wavenumbers(_tr, epsilon):
    return real(tr.Material((epsilon, 1.0, 0.2)).ks(1.2)).sum()


def axial_wavenumber(_tr, transverse):
    return real(tr.Material(3.0).kzs(1.2, transverse, 0.1)).sum()


@pytest.mark.parametrize(
    ("objective", "point", "value", "gradient"),
    [
        (refractive_index, 3.0, np.sqrt(3), 0.5 / np.sqrt(3)),
        (impedance, 3.0, 1 / np.sqrt(3), -0.5 / 3**1.5),
        (helicity_wavenumbers, 3.0, 2.4 * np.sqrt(3), 1.2 / np.sqrt(3)),
        (
            axial_wavenumber,
            0.2,
            2 * np.sqrt(1.2**2 * 3 - 0.2**2 - 0.1**2),
            -0.4 / np.sqrt(1.2**2 * 3 - 0.2**2 - 0.1**2),
        ),
    ],
)
def test_material_optical_constants_match_analytic_derivatives(
    engine, objective, point, value, gradient
):
    actual, derivative = engine.value_and_grad(objective, point)
    assert_allclose(actual, value, rtol=1e-12)
    assert_allclose(derivative, gradient, rtol=1e-12)


def test_complex_transverse_wavenumbers_preserve_values_and_gradients(engine):
    def objective(_tr, components):
        kx = components[0] + 1j * components[1]
        ky = components[2] + 1j * components[3]
        axial = tr.Material(3.0).kzs(1.2, kx, ky)
        return real((1 - 0.37j) * axial).sum()

    point = np.array([0.2, 0.3, 0.1, -0.15])
    actual, gradient = engine.value_and_grad(objective, point)
    step = 1e-6
    expected = np.array(
        [
            (objective(tr, point + delta) - objective(tr, point - delta)) / (2 * step)
            for delta in step * np.eye(len(point))
        ]
    )
    assert_allclose(actual, objective(tr, point), rtol=1e-12)
    assert_allclose(gradient, expected, rtol=2e-7, atol=1e-9)


@pytest.mark.parametrize("container", [list, tuple, np.asarray])
def test_autograd_material_sequences_retain_each_component_gradient(container):
    autograd = pytest.importorskip("autograd")
    anp = pytest.importorskip("autograd.numpy")
    parameters = container([3.0, 1.0, 0.0])
    value, gradient = autograd.value_and_grad(
        lambda values: anp.real(tr.Material(values).n)
    )(parameters)
    assert_allclose(value, np.sqrt(3), rtol=1e-12)
    assert_allclose(gradient, [0.5 / np.sqrt(3), 0.5 * np.sqrt(3), 0], rtol=1e-12)
