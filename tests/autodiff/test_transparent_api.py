"""The ordinary physics API differentiates without importing treams adapters."""

import contextlib
import functools
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_framework_physics import SCENARIOS, Engine, _advect_reference, real

import treams_rs as core

from _support import jax_x64

pytestmark = pytest.mark.gradients
ENGINES = ("advect", "jax", "torch", "autograd")


@pytest.fixture(scope="module", params=ENGINES)
def engine(request):
    pytest.importorskip(request.param)
    scope = jax_x64() if request.param == "jax" else contextlib.nullcontext()
    with scope:
        yield Engine(request.param, root_api=True)


@pytest.mark.workflows
@pytest.mark.parametrize("name", SCENARIOS)
def test_physics_workflow_uses_ordinary_api(engine, name):
    function, x0, _, _ = SCENARIOS[name]
    # Evaluate through the root API before loading the explicit reference adapter.
    actual = engine.value_and_grad(function, x0)
    expected = _advect_reference(name)
    assert_allclose(actual[0], expected[0], rtol=1e-12)
    assert_allclose(
        actual[1], expected[1], rtol=1e-12, atol=1e-15 * max(1, abs(expected[1]))
    )


def _finite_difference(function, x, step=1e-5):
    x = np.asarray(x, dtype=float)
    directions = np.eye(x.size).reshape((x.size, *x.shape))
    return np.asarray(
        [
            (function(core, x + step * d) - function(core, x - step * d)) / (2 * step)
            for d in directions
        ]
    ).reshape(x.shape)


def layered_particle(tr, parameters, *, cylinder):
    radii = [parameters[0], 0.35]
    materials = [tr.Material(parameters[1] + 0.1j), (2.0, 1.0, 0.02)]
    if cylinder:
        response = tr.CylindricalTMatrix.cylinder([0.0], 1, 1.2, radii, [*materials, 1])
        incident = tr.plane_wave([1, 0, 0], "positive_helicity", k0=1.2)
        return response.cross_widths(incident).scattering
    response = tr.TMatrix.sphere(1, 1.2, radii, [*materials, 1])
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    return response.cross_sections(incident).scattering


@pytest.mark.parametrize("cylinder", [False, True])
def test_particle_classmethod_with_nested_materials(engine, cylinder):
    function = functools.partial(layered_particle, cylinder=cylinder)
    x = np.array([0.2, 3.0])
    value, gradient = engine.value_and_grad(function, x)
    assert_allclose(value, function(core, x), rtol=1e-12)
    assert_allclose(gradient, _finite_difference(function, x), rtol=2e-5, atol=1e-9)


def layered_network(tr, parameters, *, use_classmethod):
    basis = tr.PlaneWavePorts.default([[0.1, 0.05]])
    thicknesses = [parameters[0], 0.2]
    materials = [tr.Material(parameters[1] + 0.1j), 1.5]
    if use_classmethod:
        response = tr.SMatrix.slab(thicknesses, basis, 1.2, [1, *materials, 1])
    else:
        response = tr.multilayer_slab(
            basis=basis, k0=1.2, thicknesses=thicknesses, materials=materials
        )
    return response.power([1.0, 0.0]).transmission


@pytest.mark.parametrize("use_classmethod", [False, True])
def test_planar_layers_accept_nested_parameters(engine, use_classmethod):
    function = functools.partial(layered_network, use_classmethod=use_classmethod)
    x = np.array([0.3, 2.0])
    value, gradient = engine.value_and_grad(function, x)
    assert_allclose(value, function(core, x), rtol=1e-12)
    assert_allclose(gradient, _finite_difference(function, x), rtol=2e-5, atol=1e-9)


def constant_receiver(tr, parameter, *, kind):
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    if kind == "points":
        return real(incident.efield([[0.1, 0.2, parameter]])).sum()
    particle = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)
    if kind == "illumination":
        incident = tr.plane_wave([0, 0, 1], [parameter, 0.3j], k0=1.2)
        return (abs(particle.scatter(incident).efield([[0.3, 0.4, 1.5]])) ** 2).sum()
    scattered = particle.scatter(incident)
    return (abs(scattered.efield([[0.3, 0.4, parameter]])) ** 2).sum()


@pytest.mark.parametrize("kind", ["points", "illumination", "scattered_points"])
def test_numpy_receiver_promotes_when_method_argument_is_traced(engine, kind):
    function = functools.partial(constant_receiver, kind=kind)
    x = np.array(1.5)
    value, gradient = engine.value_and_grad(function, x)
    assert_allclose(value, function(core, x), rtol=1e-12)
    assert_allclose(gradient, _finite_difference(function, x), rtol=2e-5, atol=1e-9)


def amplitude_constructor(tr, amplitude, *, kind):
    def scaled(values):
        # PyTorch arithmetic requires tensor constants before entering treams.
        if type(amplitude).__module__ == "torch":
            import torch

            values = torch.tensor(values)
        return amplitude * values

    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    if kind == "plane":
        wave = tr.PlaneWave([0, 0, 1], [amplitude, 0.0], k0=1.2)
    elif kind == "wave":
        basis = tr.SphericalBasis.default(1)
        coefficients = incident.in_basis(basis).coefficients
        wave = tr.Wave(scaled(coefficients), basis=basis, k0=1.2)
    elif kind == "tmatrix":
        particle = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)
        response = tr.TMatrix(scaled(particle.array), k0=1.2)
        wave = response.scatter(incident)
    else:
        basis = tr.PlaneWavePorts.default([[0.0, 0.0]])
        network = tr.slab(basis=basis, k0=1.2, thickness=0.3, material=2.0)
        response = tr.SMatrix(scaled(network.array), basis=basis, k0=1.2)
        return response.power([1.0, 0.0]).transmission
    return (abs(wave.efield([[0.3, 0.4, 1.5]])) ** 2).sum()


@pytest.mark.parametrize("kind", ["plane", "wave", "tmatrix", "smatrix"])
def test_array_constructors_preserve_amplitude_gradient(engine, kind):
    function = functools.partial(amplitude_constructor, kind=kind)
    amplitude = 0.8
    value, gradient = engine.value_and_grad(function, amplitude)
    assert_allclose(value, function(core, amplitude), rtol=1e-12)
    assert_allclose(amplitude * gradient, 2 * value, rtol=1e-12)


def complex_material_and_unused_parameter(tr, parameters):
    particle = tr.sphere_tmatrix(
        k0=1.2,
        lmax=1,
        radius=0.2,
        material=tr.Material(parameters[0] + 1j * parameters[1]),
    )
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    return particle.cross_sections(incident).absorption


def test_complex_material_and_unused_input_gradient(engine):
    x = np.array([3.0, 0.1, 42.0])
    function = complex_material_and_unused_parameter
    value, gradient = engine.value_and_grad(function, x)
    assert_allclose(value, function(core, x), rtol=1e-12)
    assert_allclose(gradient, _finite_difference(function, x), rtol=2e-5, atol=1e-9)
    assert gradient[2] == 0


@pytest.mark.interface
def test_different_backends_in_nested_parameters_are_rejected():
    jax = pytest.importorskip("jax")
    torch = pytest.importorskip("torch")
    with jax_x64(), pytest.raises((TypeError, ValueError), match=r"[Mm]ix|backend"):
        core.multilayer_sphere_tmatrix(
            k0=1.2,
            lmax=1,
            radii=[torch.tensor(0.2, dtype=torch.float64), 0.35],
            materials=[core.Material(jax.numpy.asarray(3.0)), 2.0],
        )


def _run_fresh_python(source, *arguments):
    environment = os.environ | {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TREAMS_RS_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source), *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.interface
def test_numpy_workflow_does_not_import_optional_frameworks():
    _run_fresh_python("""
        import sys
        import numpy as np
        import treams_rs as tr

        optional = {"advect", "jax", "jaxlib", "torch", "autograd"}
        assert optional.isdisjoint(sys.modules)
        particle = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)
        incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        cross = particle.cross_sections(incident)
        assert type(particle) is tr.TMatrix
        assert type(particle.array) is np.ndarray
        assert np.isfinite(cross.scattering)
        assert optional.isdisjoint(sys.modules)
    """)


@pytest.mark.interface
def test_numpy_objects_stay_numpy_with_all_frameworks_loaded():
    for name in ENGINES:
        pytest.importorskip(name)
    _run_fresh_python("""
        import advect
        import autograd
        import jax
        import numpy as np
        import torch
        import treams_rs as tr
        from treams_rs._dispatch import backend_for

        material = tr.Material(3.0)
        particle = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=material)
        incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        constants = [material, particle, incident, {"points": [[0.2, 0.3, 1.0]]}]
        assert backend_for(constants) is None
        assert type(particle) is tr.TMatrix and type(particle.array) is np.ndarray
        assert isinstance(particle.cross_sections(incident).scattering, float)
        epsilon = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        moving = tr.Material(epsilon)
        assert backend_for(constants, {"nested": [moving]}).xp is torch
    """)


@pytest.mark.interface
@pytest.mark.parametrize("name", ENGINES)
def test_first_derivative_lazily_loads_only_its_treams_adapter(name):
    pytest.importorskip(name)
    _run_fresh_python(
        """
        import importlib
        import sys
        import numpy as np
        import treams_rs as tr

        name = sys.argv[1]
        framework = importlib.import_module(name)
        adapters = {f"treams_rs.{n}" for n in ("advect", "jax", "torch", "autograd")}
        assert adapters.isdisjoint(sys.modules)
        constant = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)
        assert type(constant) is tr.TMatrix and type(constant.array) is np.ndarray
        assert adapters.isdisjoint(sys.modules)

        def scattering(radius):
            particle = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3.0)
            incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
            return particle.cross_sections(incident).scattering

        if name == "torch":
            radius = framework.tensor(0.2, dtype=framework.float64, requires_grad=True)
            value = scattering(radius)
            (gradient,) = framework.autograd.grad(value, radius)
            value, gradient = value.detach().numpy(), gradient.numpy()
        elif name == "jax":
            framework.config.update("jax_enable_x64", True)
            value, gradient = framework.jit(framework.value_and_grad(scattering))(0.2)
        else:
            value, gradient = framework.value_and_grad(scattering)(np.asarray(0.2))
        assert np.isfinite(value) and np.isfinite(gradient) and gradient > 0
        assert adapters.intersection(sys.modules) == {f"treams_rs.{name}"}
    """,
        name,
    )
