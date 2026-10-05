"""Ordinary interaction helpers retain derivatives of the response and illumination."""

import functools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_transparent_api import engine as engine

import treams_rs as tr

pytestmark = pytest.mark.gradients


def _local(api, scale=1.0, k0=1.2):
    single = api.SphericalBasis.default(1)
    basis = api.SphericalBasis(
        [(particle, *mode[1:]) for particle in range(2) for mode in single.modes],
        [[0, 0, 0], [0.8, 0.2, 0.1]],
    )
    diagonal = [-0.02 + 0.03j] * 6 + [-0.03 + 0.02j] * 6
    array = [
        [scale * value if i == j else 0 for j in range(12)]
        for i, value in enumerate(diagonal)
    ]
    return api.TMatrix(array, basis=basis, k0=k0)


def _objective(api, x, *, operation):
    matrix = _local(
        api,
        scale=x if operation in ("operator", "solve", "illuminate", "factor") else 1,
        k0=x if operation == "frequency" else 1.2,
    )
    if operation == "operator":
        value = matrix.interaction()
    elif operation == "solve":
        value = matrix.interaction.solve().array
    elif operation == "wave":
        incident = api.plane_wave([0, 0, 1], [x, 0.3j], k0=1.2)
        value = matrix.interaction.illuminate(incident)
    else:
        amplitude = x if operation == "constant_illumination" else 1.0
        columns = [[amplitude * (i + 1) / 12, 0.1j] for i in range(12)]
        if operation == "factor":
            factor = matrix.interaction.factor()
            columns = np.asarray(columns)
            value = factor.solve(columns) + factor.solve(0.3 * columns)
        else:
            value = matrix.interaction.illuminate(columns)
    return (abs(value) ** 2).sum()


@pytest.mark.parametrize(
    "operation",
    [
        "operator",
        "solve",
        "illuminate",
        "factor",
        "frequency",
        "constant_illumination",
        "wave",
    ],
)
def test_interaction_methods_match_numpy_and_finite_difference(engine, operation):
    function = functools.partial(_objective, operation=operation)
    x, step = 0.8, 1e-6
    value, gradient = engine.value_and_grad(function, x)
    expected = (function(tr, x + step) - function(tr, x - step)) / (2 * step)
    assert_allclose(value, function(tr, x), rtol=1e-12, atol=1e-14)
    assert_allclose(gradient, expected, rtol=2e-5, atol=1e-8)


def _self_interaction(api, radius, *, solve):
    matrix = api.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3 + 0.1j)
    if solve:
        matrix = matrix.interaction.solve()
    return (abs(matrix.array) ** 2).sum()


def test_single_particle_has_no_self_interaction(engine):
    direct = engine.value_and_grad(
        functools.partial(_self_interaction, solve=False), 0.2
    )
    coupled = engine.value_and_grad(
        functools.partial(_self_interaction, solve=True), 0.2
    )
    assert_allclose(coupled, direct, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("changing", ["radius", "amplitude"])
def test_plane_ports_illuminate_interaction_with_gradients(engine, changing):
    basis = tr.PlaneWavePorts.default([0, 0])

    def objective(api, parameter):
        matrix = api.sphere_tmatrix(
            k0=1.2,
            lmax=1,
            radius=parameter if changing == "radius" else 0.2,
            material=3 + 0.1j,
        )
        incident = api.Wave(
            [parameter if changing == "amplitude" else 1.0, 0.3j],
            basis=basis,
            k0=1.2,
            kind="up",
        )
        return (abs(matrix.interaction.illuminate(incident)) ** 2).sum()

    parameter, step = 0.2, 1e-6
    value, gradient = engine.value_and_grad(objective, parameter)
    expected = (objective(tr, parameter + step) - objective(tr, parameter - step)) / (
        2 * step
    )
    assert_allclose(value, objective(tr, parameter), rtol=1e-12)
    assert_allclose(gradient, expected, rtol=2e-5, atol=1e-9)


def test_constant_cluster_factor_differentiates_repeated_illumination(engine):
    particle = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3 + 0.1j)
    cluster = tr.Cluster([particle, particle], positions=[[0, 0, 0], [0.8, 0, 0]])
    factor = cluster.factor()

    def objective(api, amplitude):
        incident = api.plane_wave([0, 0, 1], [amplitude, 0], k0=1.2)
        scattered = factor.scatter(incident)
        return (abs(scattered.coefficients) ** 2).sum()

    for amplitude in (0.4, 0.8):
        value, gradient = engine.value_and_grad(objective, amplitude)
        assert_allclose(value, objective(tr, amplitude), rtol=1e-12)
        assert_allclose(amplitude * gradient, 2 * value, rtol=1e-12)
        forward_value, tangent = engine.jvp(objective, amplitude, 0.3)
        assert_allclose(forward_value, value, rtol=1e-12)
        assert_allclose(tangent, 0.3 * gradient, rtol=1e-12)


def _cluster_objective(api, parameter, *, changing):
    particle = api.sphere_tmatrix(
        k0=1.2,
        lmax=1,
        radius=parameter if changing == "radius" else 0.2,
        material=3 + 0.1j,
    )
    cluster = api.Cluster(
        [particle, particle],
        positions=[[0, 0, 0], [parameter if changing == "position" else 0.8, 0, 0]],
    )
    factor = cluster.factor()
    incident = api.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    first = factor.scatter(incident).coefficients
    second = factor.scatter(np.ones((12, 2), dtype=complex)).coefficients
    return (abs(first) ** 2).sum() + (abs(second) ** 2).sum()


@pytest.mark.parametrize("changing", ["radius", "position"])
def test_dynamic_cluster_factor_retains_geometry_gradients(engine, changing):
    function = functools.partial(_cluster_objective, changing=changing)
    parameter = 0.2 if changing == "radius" else 0.8
    value, gradient = engine.value_and_grad(function, parameter)
    step = 1e-6
    expected = (function(tr, parameter + step) - function(tr, parameter - step)) / (
        2 * step
    )
    assert_allclose(value, function(tr, parameter), rtol=1e-12)
    assert_allclose(gradient, expected, rtol=2e-5, atol=1e-9)
