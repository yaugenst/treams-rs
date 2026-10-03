"""Constant NumPy physics objects compose with each differentiation backend."""

import contextlib
import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

import treams_rs as core

from _support import jax_x64

pytestmark = pytest.mark.gradients


class Engine:
    def __init__(self, name, framework):
        self.name = name
        self.framework = framework
        self.tr = importlib.import_module(f"treams_rs.{name}")

    def value_and_grad(self, function, value):
        if self.name == "torch":
            x = self.framework.tensor(
                value, dtype=self.framework.float64, requires_grad=True
            )
            result = function(x)
            (gradient,) = self.framework.autograd.grad(result, x)
            return result.detach().numpy(), gradient.numpy()
        function = self.framework.value_and_grad(function)
        if self.name == "jax":
            function = self.framework.jit(function)
        return tuple(np.asarray(x) for x in function(np.asarray(value)))


@pytest.fixture(params=["advect", "jax", "torch", "autograd"])
def engine(request):
    framework = pytest.importorskip(request.param)
    scope = jax_x64() if request.param == "jax" else contextlib.nullcontext()
    with scope:
        yield Engine(request.param, framework)


def _assert_same_gradient(engine, mixed, explicit, value):
    expected = engine.value_and_grad(explicit, value)
    actual = engine.value_and_grad(mixed, value)
    assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)
    assert abs(actual[1]) > 1e-10


@pytest.mark.parametrize("parity", [False, True])
def test_constant_plane_illumination_keeps_radius_gradient(engine, parity):
    incident = core.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    if parity:
        incident = incident.with_polarization("parity")

    def objective(radius, wave):
        tm = engine.tr.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3 + 0.1j)
        return tm.cross_sections(wave).scattering

    _assert_same_gradient(
        engine,
        lambda radius: objective(radius, incident),
        lambda radius: objective(
            radius, engine.tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        ),
        0.2,
    )


@pytest.mark.parametrize("solve", [False, True])
def test_cluster_promotes_constant_first_particle_and_source(engine, solve):
    def objective(radius, constants):
        particles = [
            constants.sphere_tmatrix(k0=1.2, lmax=1, radius=0.15, material=2),
            engine.tr.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3 + 0.1j),
        ]
        cluster = engine.tr.Cluster(particles, positions=[[0, 0, 0], [0, 0, 1]])
        response = cluster.solve() if solve else cluster
        incident = constants.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        field = response.scatter(incident).efield([[0.2, 0.4, 2]])
        return (abs(field) ** 2).sum()

    _assert_same_gradient(
        engine, lambda x: objective(x, core), lambda x: objective(x, engine.tr), 0.2
    )


def test_cluster_selects_backend_from_nested_positions(engine):
    def objective(distance, constants):
        particles = [
            constants.sphere_tmatrix(k0=1.2, lmax=1, radius=r, material=3 + 0.1j)
            for r in (0.15, 0.2)
        ]
        response = engine.tr.Cluster(
            particles, positions=[[0, 0, 0], [0.3, 0, distance]]
        ).solve()
        incident = constants.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        return response.cross_sections(incident).scattering

    _assert_same_gradient(
        engine, lambda x: objective(x, core), lambda x: objective(x, engine.tr), 1.0
    )


def test_stack_promotes_constant_layers_and_infers_incident_side(engine):
    ports = core.PlaneWavePorts.default([0, 0])

    def objective(thickness, constants):
        layers = [
            constants.propagation(k0=1.2, basis=ports, distance=0.2),
            engine.tr.slab(k0=1.2, basis=ports, thickness=thickness, material=3),
            constants.slab(k0=1.2, basis=ports, thickness=0.15, material=2),
        ]
        network = engine.tr.stack(layers)
        incident = constants.plane_wave([0, 0, -1], "positive_helicity", k0=1.2)
        return network.power(incident).transmission

    _assert_same_gradient(
        engine, lambda x: objective(x, core), lambda x: objective(x, engine.tr), 0.3
    )


def test_constant_port_wave_illuminates_framework_system(engine):
    ports = core.PlaneWavePorts.default([0, 0])

    def objective(thickness, constants):
        lower = constants.slab(k0=1.2, basis=ports, thickness=0.2, material=2)
        wave = lower.scatter(negative=[1.0, 0.0]).positive
        upper = engine.tr.slab(k0=1.2, basis=ports, thickness=thickness, material=3)
        field = upper.scatter(negative=wave).positive.efield([[0.2, 0.3, 1.5]])
        return upper.power(wave).transmission + (abs(field) ** 2).sum()

    _assert_same_gradient(
        engine, lambda x: objective(x, core), lambda x: objective(x, engine.tr), 0.3
    )


def test_periodic_lattice_promotes_constant_particle(engine):
    def objective(period, constants):
        particle = constants.sphere_tmatrix(
            k0=1.2, lmax=1, radius=0.2, material=3 + 0.1j
        )
        response = engine.tr.solve_periodic(
            particle, lattice=[[period, 0.0], [0.0, 2.0]], kpar=[0.1, 0.05]
        )
        return response.to_smatrix(orders=[[0, 0]]).power([1.0, 0.0]).transmission

    _assert_same_gradient(
        engine, lambda x: objective(x, core), lambda x: objective(x, engine.tr), 2.0
    )


@pytest.mark.parametrize("polarization", ["helicity", "parity"])
def test_cartesian_polarization_matches_numpy_fields_and_gradient(engine, polarization):
    def field(amplitude, tr):
        wave = tr.plane_wave(
            [0.2, 0.3, 1],
            [amplitude, 0.2j, 0.1],
            k0=1.2,
            polarization=polarization,
        )
        return wave.efield([[0.2, 0.4, 1.5]])

    def objective(amplitude):
        return (abs(field(amplitude, engine.tr)) ** 2).sum()

    value, gradient = engine.value_and_grad(objective, 0.4)
    step = 1e-5

    def reference(x):
        return (abs(field(x, core)) ** 2).sum()

    expected_gradient = (reference(0.4 + step) - reference(0.4 - step)) / (2 * step)
    assert_allclose(value, reference(0.4), rtol=1e-13)
    assert_allclose(gradient, expected_gradient, rtol=1e-9)


@pytest.mark.parametrize("complex_dtype", [False, True])
def test_real_direction_gradient_uses_native_angular_pullback(engine, complex_dtype):
    def field(direction, tr):
        if complex_dtype:
            direction = direction + 0j
        wave = tr.plane_wave([direction, 0.3, 1], "positive_helicity", k0=1.2)
        sphere = tr.sphere_tmatrix(k0=1.2, lmax=2, radius=0.2, material=3 + 0.1j)
        return sphere.scatter(wave).efield([[0.2, 0.4, 1.5]])

    def objective(direction):
        return engine.tr._backend.xp.real(field(direction, engine.tr)).sum()

    value, gradient = engine.value_and_grad(objective, 0.2)
    step = 1e-5
    expected = field(0.2, core).real.sum()
    expected_gradient = (
        field(0.2 + step, core) - field(0.2 - step, core)
    ).real.sum() / (2 * step)
    assert_allclose(value, expected, rtol=1e-12)
    assert_allclose(gradient, expected_gradient, rtol=1e-8)


def test_dynamic_plane_direction_rejects_nonzero_imaginary_components(engine):
    def objective(direction):
        wave = core.plane_wave([direction + 0.1j, 0.3, 1], [1, 0], k0=1.2)
        return engine.tr._backend.xp.real(wave.efield([[0.2, 0.4, 1.5]])).sum()

    # JAX reports errors raised by a compiled callback as a runtime error.
    with pytest.raises((ValueError, RuntimeError), match="require a real direction"):
        engine.value_and_grad(objective, 0.2)


def test_constant_cluster_receiver_accepts_differentiable_illumination(engine):
    def objective(amplitude, constants):
        particles = [
            constants.sphere_tmatrix(k0=1.2, lmax=1, radius=r, material=3 + 0.1j)
            for r in (0.15, 0.2)
        ]
        cluster = constants.Cluster(particles, positions=[[0, 0, 0], [0, 0, 1]])
        incident = engine.tr.plane_wave([0, 0, 1], [0, amplitude], k0=1.2)
        field = cluster.scatter(incident).efield([[0.2, 0.4, 2]])
        return (abs(field) ** 2).sum()

    _assert_same_gradient(
        engine, lambda x: objective(x, core), lambda x: objective(x, engine.tr), 0.7
    )


def test_constant_wave_receiver_accepts_differentiable_field_points(engine):
    def objective(height, constants):
        sphere = constants.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3 + 0.1j)
        incident = constants.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        wave = sphere.scatter(incident)
        field = wave.efield(r=[[0.2, 0.4, height]])
        return (abs(field) ** 2).sum()

    _assert_same_gradient(
        engine, lambda x: objective(x, core), lambda x: objective(x, engine.tr), 1.5
    )


@pytest.mark.parametrize("ports", [False, True])
@pytest.mark.parametrize("quantity", ["efield", "hfield"])
def test_coefficient_batch_fields_match_numpy_and_gradient(engine, ports, quantity):
    basis = (
        core.PlaneWavePorts.default([0.1, 0.2])
        if ports
        else core.SphericalBasis.default(1)
    )
    coefficients = np.arange(1, 2 * len(basis) + 1).reshape(len(basis), 2) * (
        0.1 + 0.03j
    )
    points = [[0.2, 0.4, 1.5], [0.3, 0.1, 1.2]]
    kind = "up" if ports else "singular"

    def objective(amplitude):
        wave = core.Wave(
            [[value * amplitude for value in row] for row in coefficients.tolist()],
            basis=basis,
            k0=1.2,
            medium=2.0,
            polarization="parity",
            kind=kind,
        )
        field = getattr(wave, quantity)(r=points)
        assert field.shape == (2, 3, 2)
        return (abs(field) ** 2).sum()

    value, gradient = engine.value_and_grad(objective, 0.7)
    expected = objective(0.7)
    assert_allclose(value, expected, rtol=2e-12)
    assert_allclose(gradient, 2 * expected / 0.7, rtol=2e-12)


@pytest.mark.parametrize("label", [1.0, np.float64(1), np.array(1.0), -1.0])
def test_static_scalar_polarization_label_with_traced_frequency(engine, label):
    def objective(k0, pol):
        wave = core.plane_wave([0, 0, 1], pol, k0=k0)
        return engine.tr._backend.xp.real(wave.efield([[0.2, 0.3, 1.2]])).sum()

    expected_label = 0 if label == -1 else 1
    _assert_same_gradient(
        engine,
        lambda k0: objective(k0, label),
        lambda k0: objective(k0, expected_label),
        1.2,
    )


def test_concrete_framework_direction_infers_incident_side(engine):
    direction = engine.tr._backend.array([0.0, 0.0, -1.0])
    wave = core.plane_wave(direction, "positive_helicity", k0=1.2)
    system = core.slab(
        k0=1.2,
        basis=core.PlaneWavePorts.default([0, 0]),
        thickness=0.2,
        material=3,
    )
    actual = system.power(wave).transmission
    if engine.name == "torch":
        actual = actual.detach().numpy()
    expected = system.power(
        core.plane_wave([0, 0, -1], "positive_helicity", k0=1.2)
    ).transmission
    assert_allclose(actual, expected, rtol=1e-13)


def test_differentiable_direction_requires_incident_side(engine):
    def objective(direction):
        wave = core.plane_wave([direction, 0.3, 1.0], "positive_helicity", k0=1.2)
        system = core.slab(
            k0=1.2,
            basis=core.PlaneWavePorts.default([0, 0]),
            thickness=0.2,
            material=3,
        )
        return system.power(wave).transmission

    with pytest.raises(ValueError, match="supply side"):
        engine.value_and_grad(objective, 0.2)


def test_plane_ports_expand_with_frequency_gradient(engine):
    ports = core.PlaneWavePorts.default([0.1, 0.2])

    def objective(k0):
        wave = core.Wave([1.0, 0.3j], basis=ports, k0=k0, kind="up")
        sphere = core.sphere_tmatrix(k0=k0, lmax=1, radius=0.2, material=3 + 0.1j)
        field = sphere.scatter(wave).efield([[0.2, 0.3, 1.5]])
        return (abs(field) ** 2).sum()

    value, gradient = engine.value_and_grad(objective, 1.2)
    step = 1e-5
    expected = objective(1.2)
    expected_gradient = (objective(1.2 + step) - objective(1.2 - step)) / (2 * step)
    assert_allclose(value, expected, rtol=2e-12)
    assert_allclose(gradient, expected_gradient, rtol=1e-8)


def test_wave_reexpansion_in_identical_multicentre_basis_is_identity(engine):
    basis = core.SphericalBasis(
        [(0, 1, 0, 0), (1, 1, 0, 0)],
        positions=[[0, 0, 0], [0, 0, 1]],
    )

    def objective(amplitude):
        wave = core.Wave([amplitude, 2 * amplitude], basis=basis, k0=1.2)
        expanded = wave.in_basis(basis).in_basis(basis, positions=wave.positions)
        return (abs(expanded.array) ** 2).sum()

    value, gradient = engine.value_and_grad(objective, 0.7)
    assert_allclose(value, 5 * 0.7**2, rtol=1e-13)
    assert_allclose(gradient, 10 * 0.7, rtol=1e-13)
