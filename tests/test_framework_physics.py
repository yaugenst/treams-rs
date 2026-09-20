"""Complete physical workflows retain native first-order derivatives in each backend."""

import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

import treams_rs as core


@pytest.fixture(params=["advect", "jax", "torch"])
def engine(request):
    name = request.param
    framework = pytest.importorskip(name)
    if name == "jax":
        framework.config.update("jax_enable_x64", True)
    tr = importlib.import_module("treams_rs." + name)

    def evaluate(function, value):
        if name == "torch":
            x = framework.tensor(value, dtype=framework.float64, requires_grad=True)
            y = function(x)
            y.backward()
            return y.detach().numpy(), x.grad.numpy()
        if name == "jax":
            return framework.jit(framework.value_and_grad(function))(value)
        return function(value), framework.grad(function)(value)

    return tr, evaluate


def check_direction(evaluate, objective, x):
    value, gradient = evaluate(objective, x)
    h = 1e-5
    plus, _ = evaluate(objective, x + h)
    minus, _ = evaluate(objective, x - h)
    assert_allclose(gradient, (plus - minus) / (2 * h), rtol=2e-5, atol=1e-9)
    return value


def test_sphere_scattering_fields_and_material_gradient(engine):
    tr, evaluate = engine

    def objective(epsilon):
        tm = tr.sphere_tmatrix(
            k0=1.2, lmax=1, radius=0.2, material=tr.Material(epsilon + 0.1j)
        )
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        field = tm.scatter(wave).efield([[0.4, 0.3, 1.7]])
        return (abs(field) ** 2).sum() + tm.cross_sections(wave).scattering

    actual = check_direction(evaluate, objective, 3.0)
    tm = core.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3 + 0.1j)
    inc = core.PlaneWave([0, 0, 1], 1, k0=1.2)
    expected = (
        abs(tm.scatter(inc).efield([[0.4, 0.3, 1.7]])) ** 2
    ).sum() + tm.cross_sections(inc).scattering
    assert_allclose(actual, expected, rtol=1e-12)


def test_plane_frequency_gradient_at_axis(engine):
    tr, evaluate = engine

    def objective(k0):
        field = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0).efield(
            [[0.1, 0.2, 1.3]]
        )
        return field.real.sum()

    check_direction(evaluate, objective, 1.2)


def test_cluster_requested_illumination_gradient(engine):
    tr, evaluate = engine

    def objective(radius):
        a = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3 + 0.1j)
        b = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.15, material=2.0)
        system = tr.Cluster([a, b], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        field = system.scatter(wave).efield([[0.2, 0.4, 2.0]])
        full = system.solve().scatter(wave).efield([[0.2, 0.4, 2.0]])
        return (abs(field) ** 2).sum() + (abs(full) ** 2).sum()

    check_direction(evaluate, objective, 0.2)


def test_periodic_to_power_gradient(engine):
    tr, evaluate = engine
    ports = core.PlaneWavePorts.default([[0.1, 0.05]])

    def objective(radius):
        tm = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3 + 0.1j)
        response = tr.solve_periodic(
            tm, lattice=[[2.0, 0.0], [0.0, 2.0]], kpar=[0.1, 0.05]
        )
        power = response.to_smatrix(ports).power([1.0, 0.0])
        return power.transmission + 0.3 * power.reflection

    check_direction(evaluate, objective, 0.2)


def test_layers_and_cylinder_gradient(engine):
    tr, evaluate = engine
    ports = core.PlaneWavePorts.default([[0.1, 0.05]])

    def objective(thickness):
        sm = tr.slab(
            k0=1.2, basis=ports, thickness=thickness, material=tr.Material(2.0)
        )
        return sm.power([1.0, 0.0]).reflection

    check_direction(evaluate, objective, 0.3)

    def cylinder(radius):
        tm = tr.cylinder_tmatrix(k0=1.2, kz=0.0, mmax=1, radius=radius, material=3.0)
        field = tm.scatter(
            tr.plane_wave([1.0, 0.0, 0.0], "positive_helicity", k0=1.2)
        ).efield([[2.0, 0.3, 0.0]])
        return (abs(field) ** 2).sum()

    check_direction(evaluate, cylinder, 0.2)


def test_periodic_lattice_and_bloch_gradients_follow_orders(engine):
    tr, evaluate = engine

    def objective(period):
        tm = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3 + 0.1j)
        response = tr.solve_periodic(
            tm, lattice=[[period, 0.0], [0.0, 2.0]], kpar=[0.1, 0.05]
        )
        return response.to_smatrix(orders=[[0, 0]]).power([1.0, 0.0]).transmission

    check_direction(evaluate, objective, 2.0)

    def bloch(q):
        tm = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3 + 0.1j)
        response = tr.solve_periodic(
            tm, lattice=[[2.0, 0.0], [0.0, 2.0]], kpar=[q, 0.05]
        )
        return response.to_smatrix(orders=[[0, 0]]).power([1.0, 0.0]).transmission

    check_direction(evaluate, bloch, 0.1)


def test_asymmetric_interface_flux_matches_normal_api(engine):
    tr, evaluate = engine
    ports = core.PlaneWavePorts.default([[0.1, 0.05]])

    def objective(epsilon):
        return (
            tr.interface(
                k0=1.2,
                basis=ports,
                negative_medium=1.0,
                positive_medium=tr.Material(epsilon),
            )
            .power([1.0, 0.0])
            .transmission
        )

    actual = check_direction(evaluate, objective, 4.0)
    expected = (
        core.interface(k0=1.2, basis=ports, negative_medium=1.0, positive_medium=4.0)
        .power([1.0, 0.0])
        .transmission
    )
    assert_allclose(actual, expected, rtol=2e-13)


@pytest.mark.parametrize("family", ["sphere", "cylinder", "plane"])
@pytest.mark.parametrize("polarization", ["helicity", "parity"])
def test_all_six_fields_match_normal_api(engine, family, polarization):
    tr, _ = engine
    points = [[1.2, 0.3, 0.7]]
    direction = [1.0, 0.0, 0.0]
    inc = tr.plane_wave(direction, "positive_helicity", k0=1.2)
    expected_inc = core.PlaneWave(direction, 1, k0=1.2)
    if family == "plane":
        actual, expected = inc, expected_inc
    elif family == "sphere":
        actual = tr.sphere_tmatrix(
            k0=1.2, lmax=1, radius=0.2, material=3 + 0.1j, polarization=polarization
        ).scatter(inc)
        expected = core.sphere_tmatrix(
            k0=1.2, lmax=1, radius=0.2, material=3 + 0.1j, polarization=polarization
        ).scatter(expected_inc.with_polarization(polarization))
    else:
        actual = tr.cylinder_tmatrix(
            k0=1.2,
            kz=0.0,
            mmax=1,
            radius=0.2,
            material=3 + 0.1j,
            polarization=polarization,
        ).scatter(inc)
        expected = core.cylinder_tmatrix(
            k0=1.2,
            kz=0.0,
            mmax=1,
            radius=0.2,
            material=3 + 0.1j,
            polarization=polarization,
        ).scatter(expected_inc.with_polarization(polarization))
    for name in ["efield", "hfield", "dfield", "bfield", "gfield", "ffield"]:
        args = (1, points) if name in ("gfield", "ffield") else (points,)
        value = getattr(actual, name)(*args)
        if hasattr(value, "detach"):
            value = value.detach().numpy()
        assert_allclose(value, getattr(expected, name)(*args), rtol=2e-12, atol=1e-14)


def test_typed_planar_scattering_fields_and_frequency_gradient(engine):
    tr, evaluate = engine
    ports = core.PlaneWavePorts.default([[0.0, 0.0]])

    def objective(k0):
        sm = tr.slab(basis=ports, k0=k0, thickness=0.3, material=2.0)
        incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
        outgoing = sm.scatter(negative=incident)
        field = outgoing.positive.efield([[0.1, 0.2, 0.7]])
        return (abs(field) ** 2).sum() + sm.power(incident).reflection

    check_direction(evaluate, objective, 1.2)
    actual = tr.interface(
        basis=ports, k0=1.2, negative_medium=1.0, positive_medium=4.0
    ).scatter(negative=tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2))
    expected = core.interface(
        basis=ports, k0=1.2, negative_medium=1.0, positive_medium=4.0
    ).scatter(negative=core.plane_wave([0, 0, 1], "positive_helicity", k0=1.2))
    for side in ["positive", "negative"]:
        for name in ["efield", "hfield", "dfield", "bfield", "gfield", "ffield"]:
            args = (
                (1, [[0.1, 0.2, 0.7]])
                if name in ("gfield", "ffield")
                else ([[0.1, 0.2, 0.7]],)
            )
            value = getattr(getattr(actual, side), name)(*args)
            if hasattr(value, "detach"):
                value = value.detach().numpy()
            assert_allclose(
                value,
                getattr(getattr(expected, side), name)(*args),
                rtol=2e-12,
                atol=1e-13,
            )


def test_composed_torch_fields_repeat_backward():
    torch = pytest.importorskip("torch")
    from treams_rs import torch as tr

    radius = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    tm = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3 + 0.1j)
    wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    value = (abs(tm.scatter(wave).efield([[0.2, 0.3, 1.5]])) ** 2).sum()
    first = torch.autograd.grad(value, radius, retain_graph=True)[0]
    second = torch.autograd.grad(value, radius)[0]
    assert_allclose(first, second, rtol=1e-13)


def test_framework_physical_mismatches_raise(engine):
    tr, _ = engine
    tm = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)
    with pytest.raises(Exception, match="matching k0"):
        tm.scatter(tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)).efield(
            [[0.2, 0.3, 1.5]]
        )
    ports = core.PlaneWavePorts.default([[0.0, 0.0]])
    sm = tr.interface(basis=ports, k0=1.2, negative_medium=1.0, positive_medium=1.0)
    with pytest.raises(ValueError, match="away"):
        sm.scatter(negative=tr.plane_wave([0, 0, -1], "positive_helicity", k0=1.2))


def test_public_types_and_supplied_outgoing_wave(engine):
    tr, _ = engine
    tm = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)
    assert isinstance(tm, tr.TMatrix)
    basis = core.SphericalBasis.default(1, positions=[[0.0, 0.0, 2.0]])
    coefficients = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=complex)
    source = tr.wave(coefficients, basis=basis, k0=1.2, kind="outgoing")
    scattered = tm.scatter(source)
    assert isinstance(scattered, tr.Wave)
    from treams_rs import diff

    expansion, _ = diff.expansion(tm.basis, basis, [1.2, 1.2], singular=True)
    expected = (
        np.asarray(core.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0).array)
        @ expansion
        @ coefficients
    )
    actual = scattered.coefficients
    if hasattr(actual, "detach"):
        actual = actual.detach().numpy()
    assert_allclose(actual, expected, rtol=2e-13)
    assert_allclose(tr.plane_wave([0, 0, 1], 1, k0=1.2).coefficients, [0, 1])


def test_propagation_cascade_and_parity(engine):
    tr, _ = engine
    ports = core.PlaneWavePorts.default([[0.1, 0.05]])
    a = tr.interface(
        basis=ports,
        k0=1.2,
        negative_medium=1.0,
        positive_medium=2.0,
        polarization="parity",
    )
    p = tr.propagation(
        basis=ports, k0=1.2, distance=0.3, medium=2.0, polarization="parity"
    )
    z = tr.interface(
        basis=ports,
        k0=1.2,
        negative_medium=2.0,
        positive_medium=1.0,
        polarization="parity",
    )
    actual = tr.stack([a, p, z])
    expected = core.slab(
        basis=ports, k0=1.2, thickness=0.3, material=2.0, polarization="parity"
    )
    assert actual.polarization == "parity"
    values = actual.array
    if hasattr(values, "detach"):
        values = values.detach().numpy()
    assert_allclose(values, expected.array, rtol=2e-13, atol=1e-14)


def test_jax_physical_scale_invariant():
    jax = pytest.importorskip("jax")
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from treams_rs import jax as tr

    jax.config.update("jax_enable_x64", True)

    def cross_section(k0, radius):
        tm = tr.sphere_tmatrix(k0=k0, lmax=1, radius=radius, material=3 + 0.1j)
        return tm.cross_sections(tr.plane_wave([0, 0, 1], 1, k0=k0)).scattering

    operation = jax.jit(jax.value_and_grad(cross_section, argnums=(0, 1)))

    @settings(max_examples=6, deadline=None)
    @given(
        st.floats(min_value=0.12, max_value=0.35),
        st.floats(min_value=0.8, max_value=1.5),
    )
    def check(radius, k0):
        value, (dk, dr) = operation(k0, radius)
        assert_allclose(-k0 * dk + radius * dr, 2 * value, rtol=2e-10, atol=1e-12)

    check()


def test_high_level_jax_rejects_reduced_precision_without_detaching():
    jax = pytest.importorskip("jax")
    from treams_rs import jax as tr

    jax.config.update("jax_enable_x64", True)
    operation = jax.jit(
        lambda radius: (
            tr.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3.0).array
        )
    )
    with pytest.raises(TypeError, match="float64"):
        operation(jax.numpy.asarray(0.2, dtype=jax.numpy.float32))
    with jax.enable_x64(False), pytest.raises(ValueError, match="jax_enable_x64"):
        tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)


def test_framework_has_the_same_static_geometry_vocabulary(engine):
    tr, evaluate = engine
    for name in (
        "PlaneWavePorts",
        "PlaneWaveBasis",
        "SphericalBasis",
        "CylindricalBasis",
        "Lattice",
    ):
        assert getattr(tr, name) is getattr(core, name)
    ports = tr.PlaneWavePorts.default([0, 0])

    def objective(thickness):
        return (
            tr.slab(k0=1.7, basis=ports, thickness=thickness, material=2.5)
            .power([0, 1])
            .reflection
        )

    check_direction(evaluate, objective, 0.2)
