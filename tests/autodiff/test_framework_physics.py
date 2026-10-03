"""Complete physical workflows retain native first-order derivatives in each backend.

The three engines compose the same native pullbacks. Advect, a hard test
dependency, is checked against central finite differences once per scenario;
JAX and PyTorch must then reproduce Advect's values and gradients to rounding,
and physical invariants are checked through each engine's own derivatives.
"""

import contextlib
import functools
import importlib
import pickle

import advect
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

# Scenarios take the namespace they run in as `tr`: treams_rs.advect, .jax or .torch,
# or treams_rs itself in the has_core checks, which import it as `core`.
import treams_rs as core
from treams_rs import diff

# Constructors and Operations list the shared framework functions that the
# pickling tests cover.
from treams_rs._framework import Constructors, Operations
from treams_rs.testing import check_gradient

from _support import jax_x64

pytestmark = pytest.mark.gradients

FIELDS = ("efield", "hfield", "dfield", "bfield", "gfield", "ffield")


class Engine:
    """A framework namespace and first-order evaluation of real objectives."""

    def __init__(self, name):
        self.name = name
        self.framework = importlib.import_module(name)
        self.tr = importlib.import_module(f"treams_rs.{name}")
        self._objectives = {}

    def value_and_grad(self, function, x):
        """Value and gradient of the real scalar ``function(tr, x)`` at float64 ``x``."""
        x = np.asarray(x, dtype=np.float64)
        if self.name == "torch":
            tensor = self.framework.tensor(x, requires_grad=True)
            value = function(self.tr, tensor)
            (gradient,) = self.framework.autograd.grad(value, tensor)
            return value.detach().numpy(), gradient.numpy()
        if self.name == "jax":
            # jit caches by function identity, so compile each objective once.
            if function not in self._objectives:
                jax = self.framework
                objective = functools.partial(function, self.tr)
                self._objectives[function] = jax.jit(jax.value_and_grad(objective))
            value, gradient = self._objectives[function](x)
            return np.asarray(value), np.asarray(gradient)
        value, gradient = advect.value_and_grad(functools.partial(function, self.tr))(x)
        return np.asarray(value), np.asarray(gradient)

    @staticmethod
    def numpy(value):
        return value.detach().numpy() if hasattr(value, "detach") else np.asarray(value)


@pytest.fixture(scope="module", params=["advect", "jax", "torch"])
def engine(request):
    pytest.importorskip(request.param)
    scope = jax_x64() if request.param == "jax" else contextlib.nullcontext()
    with scope:
        yield Engine(request.param)


# Scenarios: real objectives function(tr, x) of one parameter -----------------------

SCENARIOS = {}


def scenario(x0, *, name=None, has_core=True, rtol=1e-12, **options):
    """Register ``function(tr, x, **options)``, differentiated at ``x0``.

    ``has_core`` scenarios also run unchanged in the NumPy namespace
    ``treams_rs``, an independent evaluation of the complete workflow, compared
    at ``rtol``.
    """

    def register(function):
        objective = functools.partial(function, **options)
        SCENARIOS[name or function.__name__] = (objective, x0, has_core, rtol)
        return function

    return register


@scenario(3.0)
def sphere_permittivity(tr, epsilon):
    tm = tr.sphere_tmatrix(
        k0=1.2, lmax=1, radius=0.2, material=tr.Material(epsilon + 0.1j)
    )
    wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    field = tm.scatter(wave).efield([[0.4, 0.3, 1.7]])
    return (abs(field) ** 2).sum() + tm.cross_sections(wave).scattering


@scenario(1.2)
def plane_frequency_on_axis(tr, k0):
    wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
    return wave.efield([[0.1, 0.2, 1.3]]).real.sum()


@scenario(0.2, name="cluster_requested_illumination", solve=False)
@scenario(0.2, name="cluster_solved_response", solve=True)
def cluster_field(tr, radius, *, solve):
    a = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3 + 0.1j)
    b = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.15, material=2.0)
    system = tr.Cluster([a, b], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    scattered = (system.solve() if solve else system).scatter(wave)
    return (abs(scattered.efield([[0.2, 0.4, 2.0]])) ** 2).sum()


@scenario(1.0, name="cluster_position_solved", solve=True)
@scenario(1.0, name="cluster_position_requested", solve=False)
def cluster_position(tr, distance, *, solve):
    # The solved response pulls positions back through the native cluster.
    a = tr.sphere_tmatrix(k0=1.2, lmax=2, radius=0.2, material=3 + 0.1j)
    b = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.15, material=2.0)
    zero = 0 * distance
    positions = [[zero, zero, zero], [0.3 + zero, zero, distance]]
    system = tr.Cluster([a, b], positions=positions)
    wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    scattered = (system.solve() if solve else system).scatter(wave)
    return (abs(scattered.efield([[0.2, 0.4, 2.5]])) ** 2).sum()


@scenario(1.2, name="cluster_frequency_helicity", polarization="helicity")
@scenario(1.2, name="cluster_frequency_parity", polarization="parity")
def cluster_frequency(tr, k0, *, polarization):
    # k0 reaches the solve through the particle blocks and the wavenumbers.
    particles = [
        tr.sphere_tmatrix(
            k0=k0, lmax=1, radius=r, material=3 + 0.1j, polarization=polarization
        )
        for r in (0.2, 0.15, 0.1)
    ]
    positions = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.6], [0.4, 0.0, 1.3]]
    solved = tr.Cluster(particles, positions=positions).solve()
    wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
    return solved.with_polarization("helicity").cross_sections(wave).extinction


@scenario(0.2, name="periodic_radius_helicity", polarization="helicity")
@scenario(0.2, name="periodic_radius_parity", polarization="parity")
def periodic_power(tr, radius, *, polarization):
    tm = tr.sphere_tmatrix(
        k0=1.2, lmax=1, radius=radius, material=3 + 0.1j, polarization=polarization
    )
    response = tr.solve_periodic(tm, lattice=[[2.0, 0.0], [0.0, 2.0]], kpar=[0.1, 0.05])
    network = response.to_smatrix(core.PlaneWavePorts.default([[0.1, 0.05]]))
    assert network.polarization == polarization
    power = network.power([1.0, 0.0])
    return power.transmission + 0.3 * power.reflection


@scenario(1.0, name="complex_index_plane_x", direction=[1, 0, 0])
@scenario(1.0, name="complex_index_plane_z", direction=[0, 0, 1])
@scenario(1.0, name="complex_index_plane_oblique", direction=[1, 0.2, 0.3])
def complex_index_plane(tr, index, *, direction):
    medium = tr.Material(index + 0.1j, index + 0.1j)
    wave = tr.plane_wave(direction, "positive_helicity", k0=1.2, medium=medium)
    coefficients = wave.in_basis(core.SphericalBasis.default(1)).coefficients
    return wave.efield([[0.2, 0.4, 0.7]]).real.sum() + coefficients.real.sum()


@pytest.mark.interface
@pytest.mark.parametrize("direction", [[1, 0, 0], [0, 0, 1], [1, 0.2, 0.3]])
def test_negative_index_plane_is_outside_supported_material_branches(engine, direction):
    with pytest.raises(Exception, match="material branch"):
        engine.value_and_grad(
            functools.partial(complex_index_plane, direction=direction), -1.0
        )


@scenario(0.2)
def cylinder_width(tr, radius):
    cylinder = tr.cylinder_tmatrix(k0=1.2, kz=0.0, mmax=1, radius=radius, material=3.0)
    incident = tr.plane_wave([1, 0, 0], "positive_helicity", k0=1.2)
    return cylinder.cross_widths(incident).extinction


@scenario(0.2)
def cylinder_field(tr, radius):
    cylinder = tr.cylinder_tmatrix(k0=1.2, kz=0.0, mmax=1, radius=radius, material=3.0)
    incident = tr.plane_wave([1.0, 0.0, 0.0], "positive_helicity", k0=1.2)
    return (abs(cylinder.scatter(incident).efield([[2.0, 0.3, 0.0]])) ** 2).sum()


@scenario(0.3)
def slab_thickness_oblique(tr, thickness):
    ports = core.PlaneWavePorts.default([[0.1, 0.05]])
    sm = tr.slab(k0=1.2, basis=ports, thickness=thickness, material=tr.Material(2.0))
    return sm.power([1.0, 0.0]).reflection


@scenario(0.2)
def slab_thickness_normal(tr, thickness):
    # Normal incidence takes the fixed-q port path, with the other helicity.
    ports = core.PlaneWavePorts.default([0, 0])
    sm = tr.slab(k0=1.7, basis=ports, thickness=thickness, material=2.5)
    return sm.power([0, 1]).reflection


@scenario(1.2)
def slab_frequency_normal(tr, k0):
    ports = core.PlaneWavePorts.default([[0.0, 0.0]])
    sm = tr.slab(basis=ports, k0=k0, thickness=0.3, material=2.0)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=k0)
    field = sm.scatter(negative=incident).positive.efield([[0.1, 0.2, 0.7]])
    return (abs(field) ** 2).sum() + sm.power(incident).reflection


@scenario(4.0, rtol=2e-13)
def interface_permittivity(tr, epsilon):
    network = tr.interface(
        k0=1.2,
        basis=core.PlaneWavePorts.default([[0.1, 0.05]]),
        negative_medium=1.0,
        positive_medium=tr.Material(epsilon),
    )
    return network.power([1.0, 0.0]).transmission


# Three transverse groups whose ports interleave; layer blocks couple only
# ports of one group and follow each port's polarization.
INTERLEAVED_PORTS = core.PlaneWavePorts(
    [
        (0.1, 0.0, 1),
        (0.0, 0.2, 0),
        (0.1, 0.0, 0),
        (-0.1, 0.1, 1),
        (0.0, 0.2, 1),
        (-0.1, 0.1, 0),
    ]
)
PORT_AMPLITUDES = [0.0, 1.0, 0.3j, 0.0, 0.0, 0.2]


@scenario(0.3)
def interleaved_slab_thickness(tr, thickness):
    sm = tr.slab(
        basis=INTERLEAVED_PORTS, k0=1.2, thickness=thickness, material=2.0 + 0.1j
    )
    return sm.power([0.0, 1.0, 0.0, 0.0, 0.3j, 0.0]).reflection


@scenario(0.3, name="interleaved_parity_slab")
def interleaved_parity_slab(tr, thickness):
    # Parity changes pair each port with its partner wherever the groups interleave.
    sm = tr.slab(
        basis=INTERLEAVED_PORTS,
        k0=1.2,
        thickness=thickness,
        material=2.0 + 0.1j,
        polarization="parity",
    )
    return sm.power([0.0, 1.0, 0.0, 0.0, 0.3j, 0.0]).reflection


@scenario(0.4, name="partial_helicity_slab", polarization="helicity")
@scenario(0.4, name="partial_parity_slab", polarization="parity")
def partial_slab(tr, thickness, *, polarization):
    ports = core.PlaneWavePorts.default([[0.6, 0.2]])[:1]
    sm = tr.slab(
        basis=ports,
        k0=1.3,
        thickness=thickness,
        material=2.3,
        polarization=polarization,
    )
    return sm.power([1.0]).transmission


@scenario(-0.1, name="gain_interface", chirality=False)
@scenario(0.5, name="chiral_interface", chirality=True)
def unusual_interface(tr, parameter, *, chirality):
    medium = (
        tr.Material(1, 1, parameter) if chirality else tr.Material(2.3 + 1j * parameter)
    )
    sm = tr.interface(
        basis=core.PlaneWavePorts.default([[0.2, 0.3]]),
        k0=1.3,
        negative_medium=1,
        positive_medium=medium,
    )
    return (abs(sm.array) ** 2).sum()


@scenario(-0.1)
def gain_interface_power(tr, imaginary):
    sm = tr.interface(
        basis=core.PlaneWavePorts.default([[0.2, 0.3]]),
        k0=1.3,
        negative_medium=1,
        positive_medium=tr.Material(2.3 + 1j * imaginary),
    )
    power = sm.power([0.0, 1.0])
    return power.transmission + 0.2 * power.reflection


@scenario(-0.1, name="gain_fields_axial", direction=[0, 0, 1])
@scenario(-0.1, name="gain_fields_oblique", direction=[0.2, 0.3, 1])
def gain_fields(tr, imaginary, *, direction):
    wave = tr.plane_wave(
        direction, "positive_helicity", k0=1.3, medium=tr.Material(2.3 + 1j * imaginary)
    )
    return sum((abs(getattr(wave, f)([0.1, 0.2, 0.3])) ** 2).sum() for f in FIELDS[:4])


@pytest.mark.interface
def test_planar_material_branch_check_follows_framework_values(engine):
    def reflection(tr, kappa):
        sm = tr.interface(
            basis=core.PlaneWavePorts.default([[0.2, 0.3]]),
            k0=1.3,
            negative_medium=1,
            positive_medium=tr.Material(1, 1, kappa),
        )
        return (abs(sm.array) ** 2).sum()

    value, gradient = engine.value_and_grad(reflection, 0.5)
    assert np.isfinite(value) and np.isfinite(gradient)
    with pytest.raises(Exception, match="material branch"):
        engine.value_and_grad(reflection, 1.5)


@scenario(0.4, name="partial_parity_hfield", field="hfield")
@scenario(0.4, name="partial_parity_dfield", field="dfield")
@scenario(0.4, name="partial_parity_bfield", field="bfield")
@scenario(0.4, name="partial_parity_gfield", field="gfield")
@scenario(0.4, name="partial_parity_ffield", field="ffield")
def partial_parity_fields(tr, thickness, *, field):
    ports = core.PlaneWavePorts.default([[0.6, 0.2]])[:1]
    sm = tr.slab(
        basis=ports, k0=1.3, thickness=thickness, material=2.3, polarization="parity"
    )
    wave = sm.scatter(negative=[1.0]).positive
    arguments = ([0.1, 0.2, 0.3],) if field in FIELDS[:4] else (1, [0.1, 0.2, 0.3])
    return (abs(getattr(wave, field)(*arguments)) ** 2).sum()


@scenario(0.4)
def partial_parity_plane_incidence(tr, thickness):
    sm = tr.slab(
        basis=core.PlaneWavePorts.default([[0, 0]])[:1],
        k0=1.3,
        thickness=thickness,
        material=2.3,
        polarization="parity",
    )
    incident = tr.plane_wave([0, 0, 1], [np.sqrt(0.5), np.sqrt(0.5)], k0=1.3)
    # NumPy requires explicit convention agreement; frameworks convert incident waves.
    if tr is core:
        incident = incident.with_polarization("parity")
    return sm.power(incident).transmission


@pytest.mark.physics
def test_partial_helicity_port_wave_can_illuminate_parity_network(engine):
    tr = engine.tr
    full = core.PlaneWavePorts.default([[0.2, 0.3]])
    source = (
        tr.propagation(basis=full[:1], k0=1.3, distance=0.2)
        .scatter(negative=[1.0])
        .positive
    )
    target = tr.propagation(basis=full, k0=1.3, distance=0.4, polarization="parity")
    actual = target.scatter(negative=source).positive.efield([0.1, 0.2, 0.3])
    amplitude = Engine.numpy(source.coefficients)[0] * np.sqrt(0.5)
    expected = target.scatter(negative=[amplitude, amplitude]).positive.efield(
        [0.1, 0.2, 0.3]
    )
    assert_allclose(
        Engine.numpy(actual), Engine.numpy(expected), rtol=2e-13, atol=2e-13
    )


@scenario(0.4, name="partial_parity_propagation")
def partial_parity_propagation(tr, distance):
    sm = tr.propagation(
        basis=core.PlaneWavePorts.default([[0.6, 0.2]])[:1],
        k0=1.3,
        distance=distance,
        medium=2.3,
        polarization="parity",
    )
    return (sm.array.real**2).sum()


@scenario(0.3, name="interleaved_parity_propagation")
def interleaved_parity_propagation(tr, distance):
    network = tr.stack(
        [
            tr.propagation(
                distance=distance,
                basis=INTERLEAVED_PORTS,
                k0=1.2,
                medium=1.5 + 0.1j,
                polarization="parity",
            ),
            tr.slab(
                basis=INTERLEAVED_PORTS,
                k0=1.2,
                thickness=0.2,
                material=2.0,
                negative_medium=1.5 + 0.1j,
                polarization="parity",
            ),
        ]
    )
    return network.power([0.2, 1.0, 0.0, 0.3j, 0.0, 0.0]).transmission


def port_systems(api, thickness, amplitudes):
    """The outgoing port wave of one slab and a second slab it illuminates."""
    lower = api.slab(basis=INTERLEAVED_PORTS, k0=1.2, thickness=thickness, material=2.0)
    upper = api.slab(
        basis=INTERLEAVED_PORTS, k0=1.2, thickness=0.2, material=1.5 + 0.1j
    )
    return lower.scatter(negative=amplitudes).positive, upper


@scenario(0.3)
def port_wave_second_system(tr, thickness):
    wave, upper = port_systems(tr, thickness, PORT_AMPLITUDES)
    return upper.power(wave).transmission


# The NumPy namespace converts periodic responses to fixed ports only.
@scenario(2.0, name="lattice_period", has_core=False, parameter="period")
@scenario(0.1, name="lattice_bloch", has_core=False, parameter="bloch")
def lattice_order_power(tr, x, *, parameter):
    period, q = (x, 0.1) if parameter == "period" else (2.0, x)
    tm = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3 + 0.1j)
    response = tr.solve_periodic(
        tm, lattice=[[period, 0.0], [0.0, 2.0]], kpar=[q, 0.05]
    )
    return response.to_smatrix(orders=[[0, 0]]).power([1.0, 0.0]).transmission


@scenario(0.4, has_core=False)
def chiral_slab_bands(tr, thickness):
    # Distinct Bloch wavenumbers keep the eigen-derivatives well defined.
    ports = core.PlaneWavePorts.default([[0.1, 0.05]])
    material = tr.Material(2.0, 1, 0.1)
    sm = tr.slab(k0=1.2, basis=ports, thickness=thickness, material=material)
    return (abs(sm.bands(0.5).wavenumbers) ** 2).sum()


@functools.cache
def _advect_reference(name):
    """Finite-difference-checked Advect value and gradient of one scenario."""
    function, x0, _, _ = SCENARIOS[name]
    objective = functools.partial(function, importlib.import_module("treams_rs.advect"))
    check_gradient(
        objective,
        advect.grad(objective),
        np.asarray(x0),
        directions=(np.asarray(1.0),),
        step=1e-5,
        rtol=2e-5,
        atol=1e-9,
    )
    value, gradient = advect.value_and_grad(objective)(np.asarray(x0))
    return np.asarray(value), np.asarray(gradient)


@pytest.mark.workflows
@pytest.mark.parametrize("name", SCENARIOS)
def test_workflow_value_and_gradient(engine, name):
    function, x0, has_core, rtol = SCENARIOS[name]
    value, gradient = _advect_reference(name)
    if engine.name == "advect":
        if has_core:
            assert_allclose(value, function(core, x0), rtol=rtol)
        return
    actual = engine.value_and_grad(function, x0)
    assert_allclose(actual[0], value, rtol=1e-12)
    assert_allclose(actual[1], gradient, rtol=1e-12, atol=1e-15 * max(1, abs(gradient)))


@pytest.mark.workflows
def test_requested_illumination_equals_the_solved_response():
    requested = _advect_reference("cluster_requested_illumination")
    solved = _advect_reference("cluster_solved_response")
    assert_allclose(requested, solved, rtol=1e-12)


def heterogeneous_cluster_intensity(tr, radius, *, solved):
    # Different multipole orders make the local response blocks unequal.
    particles = [
        tr.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3 + 0.1j),
        tr.sphere_tmatrix(k0=1.2, lmax=2, radius=0.25, material=2.0),
        tr.sphere_tmatrix(
            k0=1.2, lmax=1, radius=0.2, material=tr.Material(2.5, 1.0, 0.1)
        ),
    ]
    positions = [[0.0, 0.0, 0.0], [0.9, 0.1, 0.2], [-0.3, 0.8, 1.1]]
    cluster = tr.Cluster(particles, positions=positions)
    wave = tr.plane_wave([0.3, 0.0, 1.0], "positive_helicity", k0=1.2)
    scattered = (cluster.solve() if solved else cluster).scatter(wave)
    return (abs(scattered.efield([[0.4, -0.2, 2.5], [1.5, 0.3, -1.0]])) ** 2).sum()


HETEROGENEOUS_CLUSTER = {
    solved: functools.partial(heterogeneous_cluster_intensity, solved=solved)
    for solved in (False, True)
}


@pytest.mark.workflows
def test_heterogeneous_cluster_scatter_matches_solved_response(engine):
    requested = engine.value_and_grad(HETEROGENEOUS_CLUSTER[False], 0.3)
    solved = engine.value_and_grad(HETEROGENEOUS_CLUSTER[True], 0.3)
    assert_allclose(requested[0], solved[0], rtol=1e-12)
    assert_allclose(requested[1], solved[1], rtol=1e-10)


# Physical invariants through each engine's own derivatives -------------------------


def cross_section(tr, x, *, kind, family):
    k0, radius, epsilon = x[0], x[1], x[2]
    if family == "sphere":
        particle = tr.sphere_tmatrix(
            k0=k0, lmax=2, radius=radius, material=tr.Material(epsilon)
        )
        incident = tr.plane_wave([0, 0.6, 0.8], "positive_helicity", k0=k0)
        return getattr(particle.cross_sections(incident), kind)
    particle = tr.cylinder_tmatrix(
        k0=k0, kz=0.0, mmax=2, radius=radius, material=tr.Material(epsilon)
    )
    incident = tr.plane_wave([1, 0, 0], "positive_helicity", k0=k0)
    return getattr(particle.cross_widths(incident), kind)


CROSS_SECTIONS = {
    (kind, family): functools.partial(cross_section, kind=kind, family=family)
    for kind in ("scattering", "absorption")
    for family in ("sphere", "cylinder")
}


@pytest.mark.physics
@pytest.mark.parametrize("family", ["sphere", "cylinder"])
@settings(max_examples=10)
@given(
    k0=st.floats(0.8, 1.5), radius=st.floats(0.12, 0.35), epsilon=st.floats(1.2, 5.0)
)
def test_lossless_cross_sections_scale_and_absorb_nothing(
    engine, family, k0, radius, epsilon
):
    x = np.array([k0, radius, epsilon])
    value, gradient = engine.value_and_grad(CROSS_SECTIONS["scattering", family], x)
    # Maxwell scaling k0 -> k0 / s, radius -> s radius multiplies areas by s**2
    # and widths by s: -k0 d/dk0 + radius d/dradius returns the degree.
    degree = 2 if family == "sphere" else 1
    assert_allclose(
        -k0 * gradient[0] + radius * gradient[1], degree * value, rtol=1e-12
    )
    # Extinction and scattering take separate paths; for a lossless particle
    # their difference and its derivatives vanish.
    loss, derivative = engine.value_and_grad(CROSS_SECTIONS["absorption", family], x)
    assert_allclose(loss, 0, atol=1e-11 * value)
    assert_allclose(derivative, 0, atol=1e-11 * np.abs(gradient).max())


def lossless_cross_section(tr, radius, *, kind, cluster):
    # One expansion position, or a solved two-position response with overlap.
    response = tr.sphere_tmatrix(k0=1.2, lmax=2, radius=radius, material=3.0)
    if cluster:
        other = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=2.0)
        positions = [[0.0, 0.0, 0.0], [0.3, 0.4, 0.9]]
        response = tr.Cluster([response, other], positions=positions).solve()
    wave = tr.plane_wave([0.2, 0.0, 1.0], "positive_helicity", k0=1.2)
    return getattr(response.cross_sections(wave), kind)


LOSSLESS_CROSS_SECTIONS = {
    (kind, cluster): functools.partial(
        lossless_cross_section, kind=kind, cluster=cluster
    )
    for kind in ("scattering", "absorption")
    for cluster in (False, True)
}


@pytest.mark.physics
@pytest.mark.parametrize("cluster", [False, True])
def test_lossless_cross_sections_satisfy_the_optical_theorem(engine, cluster):
    scattering = engine.value_and_grad(
        LOSSLESS_CROSS_SECTIONS["scattering", cluster], 0.3
    )
    absorption = engine.value_and_grad(
        LOSSLESS_CROSS_SECTIONS["absorption", cluster], 0.3
    )
    assert abs(absorption[0]) <= 1e-12 * scattering[0]
    assert abs(absorption[1]) <= 1e-12 * abs(scattering[1])
    expected = lossless_cross_section(core, 0.3, kind="scattering", cluster=cluster)
    assert_allclose(scattering[0], expected, rtol=1e-12)


@pytest.mark.interface
def test_parity_cross_sections_require_an_achiral_exterior(engine):
    # The power weights are helicity wavenumbers. A single position records no
    # overlap, so the guard itself must reject parity labels in a chiral medium.
    tr = engine.tr
    medium = tr.Material(1.2, 1.0, 0.1)
    common = {"k0": 1.0, "radius": 0.5, "material": 4.0, "medium": medium}
    sphere = {"lmax": 1, **common}
    cylinder = {"kz": 0.0, "mmax": 1, **common}
    for make, options, direction in (
        (tr.sphere_tmatrix, sphere, [0, 0, 1]),
        (tr.cylinder_tmatrix, cylinder, [1, 0, 0]),
    ):
        wave = tr.plane_wave(direction, "positive_helicity", k0=1.0, medium=medium)
        lossless = make(**options).cross_sections(wave)
        assert abs(lossless.absorption) <= 1e-12 * lossless.scattering
        with pytest.raises(Exception, match="achiral"):
            make(**options, polarization="parity").cross_sections(wave)


def lossless_slab_power(tr, x):
    thickness, k0, epsilon, kappa = x[0], x[1], x[2], x[3]
    ports = core.PlaneWavePorts.default([[0.1, 0.05]])
    material = tr.Material(epsilon, 1.0, kappa)
    network = tr.slab(k0=k0, basis=ports, thickness=thickness, material=material)
    power = network.power([1.0, 0.0])
    return power.transmission + power.reflection


@pytest.mark.physics
@pytest.mark.parametrize(
    "engine",
    [
        "advect",
        # About 4% of the draws have a cascade pivot that XLA's mode would flush
        # to zero, see test_jax.py::test_slab_cascade_with_unit_scale_pivot.
        "jax",
        "torch",
    ],
    indirect=True,
)
@settings(max_examples=10)
@given(
    thickness=st.floats(0.1, 0.6),
    k0=st.floats(0.8, 1.6),
    epsilon=st.floats(1.2, 5.0),
    kappa=st.floats(-0.2, 0.2),
)
def test_lossless_chiral_slab_conserves_power_at_every_order(
    engine, thickness, k0, epsilon, kappa
):
    x = np.array([thickness, k0, epsilon, kappa])
    value, gradient = engine.value_and_grad(lossless_slab_power, x)
    assert_allclose(value, 1, atol=1e-13)
    assert_allclose(gradient, 0, atol=1e-13)


def cluster_intensity(tr, x):
    # x holds two particle positions, then the observation point.
    a = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3 + 0.1j)
    b = tr.sphere_tmatrix(
        k0=1.2, lmax=2, radius=0.15, material=tr.Material(2.0, 1.0, 0.05)
    )
    system = tr.Cluster([a, b], positions=x[:6].reshape(2, 3))
    incident = tr.plane_wave([0.3, 0, 1], "positive_helicity", k0=1.2)
    return (abs(system.scatter(incident).efield(x[6:].reshape(1, 3))) ** 2).sum()


CLUSTER_GEOMETRY = np.array([0.0, 0.0, 0.0, 0.3, 0.2, 1.0, 0.2, 0.4, 2.0])


@pytest.mark.physics
@settings(max_examples=10)
@given(shift=st.tuples(*(st.floats(-1, 1) for _ in range(3))))
def test_scattered_intensity_is_invariant_under_joint_translation(engine, shift):
    # Moving particles and observer together only rephases the plane wave.
    reference, _ = engine.value_and_grad(cluster_intensity, CLUSTER_GEOMETRY)
    x = CLUSTER_GEOMETRY + np.tile(shift, 3)
    value, gradient = engine.value_and_grad(cluster_intensity, x)
    assert_allclose(value, reference, rtol=1e-12)
    assert_allclose(
        gradient.reshape(3, 3).sum(axis=0), 0, atol=1e-12 * np.abs(gradient).max()
    )


# Values and metadata against the NumPy namespace ----------------------------------


# Arguments before the points: G and F take each helicity label.
ARGUMENTS = {name: [()] for name in FIELDS} | {
    name: [(0,), (1,)] for name in ("gfield", "ffield")
}


def _fields(value, points):
    return {
        (name, *arguments): Engine.numpy(getattr(value, name)(*arguments, points))
        for name in FIELDS
        for arguments in ARGUMENTS[name]
    }


@pytest.mark.workflows
@pytest.mark.parametrize(
    "family,polarization,medium",
    [
        ("plane", "helicity", (1.2, 1.1, 0.05)),
        ("sphere", "helicity", (1.2, 1.1, 0.05)),
        ("sphere", "parity", (1.2, 1.1, 0)),
        ("cylinder", "helicity", (1.2, 1.1, 0.05)),
        ("cylinder", "parity", None),
    ],
)
def test_all_six_fields_match_normal_api(engine, family, polarization, medium):
    # A chiral medium weights F differently from G for each helicity label.
    tr = engine.tr
    points = [[1.2, 0.3, 0.7]]
    options = {} if medium is None else {"medium": tr.Material(*medium)}
    expected_options = {} if medium is None else {"medium": core.Material(*medium)}
    inc = tr.plane_wave([1.0, 0.0, 0.0], "positive_helicity", k0=1.2, **options)
    expected = core.plane_wave(
        [1.0, 0.0, 0.0], "positive_helicity", k0=1.2, **expected_options
    )
    actual = inc
    if family != "plane":
        make = "sphere_tmatrix" if family == "sphere" else "cylinder_tmatrix"
        shape = {"lmax": 1} if family == "sphere" else {"kz": 0.0, "mmax": 1}
        particle = dict(
            k0=1.2, radius=0.2, material=3 + 0.1j, polarization=polarization, **shape
        )
        actual = getattr(tr, make)(**particle, **options).scatter(inc)
        expected = getattr(core, make)(**particle, **expected_options).scatter(
            expected.with_polarization(polarization)
        )
    reference = _fields(expected, points)
    for name, value in _fields(actual, points).items():
        assert_allclose(value, reference[name], rtol=2e-12, atol=1e-14)
    with pytest.raises(ValueError, match="helicity label"):
        actual.gfield(2, points)


@pytest.mark.workflows
def test_derived_fields_in_a_chiral_exterior_match_normal_api(engine):
    # Nonzero kappa couples E and H in D and B; mu != 1 separates B from H.
    points = [[1.2, 0.3, 0.7], [-0.4, 0.9, -1.1]]

    def waves(api):
        medium = api.Material(1.5, 1.2, 0.1)
        incident = api.plane_wave(
            [0.3, 0.0, 1.0], "positive_helicity", k0=1.2, medium=medium
        )
        sphere = api.sphere_tmatrix(
            k0=1.2, lmax=2, radius=0.3, material=3 + 0.1j, medium=medium
        )
        return incident, sphere.scatter(incident)

    for actual, expected in zip(waves(engine.tr), waves(core), strict=True):
        for name in ["hfield", "dfield", "bfield", "gfield", "ffield"]:
            for args in [(0, points), (1, points)] if name[0] in "gf" else [(points,)]:
                assert_allclose(
                    Engine.numpy(getattr(actual, name)(*args)),
                    getattr(expected, name)(*args),
                    rtol=2e-12,
                    atol=1e-14,
                )


@pytest.mark.workflows
def test_interface_port_fields_match_normal_api(engine):
    tr = engine.tr
    ports = core.PlaneWavePorts.default([[0.0, 0.0]])
    points = [[0.1, 0.2, 0.7]]
    actual = tr.interface(
        basis=ports, k0=1.2, negative_medium=1.0, positive_medium=4.0
    ).scatter(negative=tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2))
    expected = core.interface(
        basis=ports, k0=1.2, negative_medium=1.0, positive_medium=4.0
    ).scatter(negative=core.plane_wave([0, 0, 1], "positive_helicity", k0=1.2))
    for side in ("positive", "negative"):
        reference = _fields(getattr(expected, side), points)
        for name, value in _fields(getattr(actual, side), points).items():
            assert_allclose(value, reference[name], rtol=2e-12, atol=1e-13)


def chiral_layers(api, ports):
    """Chiral and dielectric layers, and a chiral sphere lit in a chiral medium."""
    options = {"basis": ports, "k0": 1.2}
    chiral, medium = api.Material(2.0, 1.1, 0.1), api.Material(1.2, 1.1, 0.02)
    layers = api.multilayer_slab(
        thicknesses=[0.2, 0.3], materials=[chiral, 3.0], **options
    )
    sphere = api.sphere_tmatrix(
        k0=1.2, lmax=2, radius=0.2, material=api.Material(3, 1, 0.05), medium=medium
    )
    wave = api.plane_wave([0.3, 0, 1], "positive_helicity", k0=1.2, medium=medium)
    return options, chiral, layers, sphere, wave


@pytest.mark.workflows
def test_layers_raw_constructors_and_positive_side_match_normal_api(engine):
    tr = engine.tr
    ports = core.PlaneWavePorts.default([[0.1, 0.05]])
    options, chiral, layers, native, wave = chiral_layers(tr, ports)
    *_, expected, sphere, incident = chiral_layers(core, ports)
    assert_allclose(Engine.numpy(layers.array), expected.array, atol=1e-14)
    stacked = tr.stack(
        [
            tr.interface(negative_medium=1.0, positive_medium=chiral, **options),
            tr.propagation(distance=0.2, medium=chiral, **options),
            tr.interface(negative_medium=chiral, positive_medium=3.0, **options),
            tr.propagation(distance=0.3, medium=3.0, **options),
            tr.interface(negative_medium=3.0, positive_medium=1.0, **options),
        ]
    )
    assert_allclose(Engine.numpy(stacked.array), expected.array, atol=1e-14)
    raw = tr.smatrix(expected.array, **options)
    for network in (layers, raw):
        power = network.power([1.0, 0.3j], side="positive")
        reference = expected.power([1.0, 0.3j], side="positive")
        assert_allclose(Engine.numpy(power.transmission), reference.transmission)
        assert_allclose(Engine.numpy(power.reflection), reference.reflection)
    points = [[0.1, 0.2, -0.7]]
    assert_allclose(
        Engine.numpy(layers.scatter(positive=[1.0, 0.3j]).negative.efield(points)),
        expected.scatter(positive=[1.0, 0.3j]).negative.efield(points),
        atol=1e-14,
    )
    bands = layers.bands(0.5)
    reference = expected.bands(period=0.5)
    assert_allclose(Engine.numpy(bands.wavenumbers), reference.wavenumbers, atol=1e-13)
    assert_allclose(
        Engine.numpy(bands.eigenvectors), reference.eigenvectors, atol=1e-13
    )
    raw = tr.tmatrix(sphere.array, basis=sphere.basis, k0=1.2, medium=wave.medium)
    for particle in (native, raw):
        assert_allclose(Engine.numpy(particle.array), sphere.array, atol=1e-15)
        assert_allclose(
            Engine.numpy(particle.scatter(wave).efield(points)),
            sphere.scatter(incident).efield(points),
            atol=1e-14,
        )
        cross = particle.cross_sections(wave)
        assert_allclose(
            Engine.numpy(cross.extinction), sphere.cross_sections(incident).extinction
        )


@pytest.mark.workflows
def test_layer_blocks_follow_interleaved_port_groups(engine):
    # The thickness gradient is the interleaved_slab_thickness scenario.
    actual = engine.tr.slab(
        basis=INTERLEAVED_PORTS, k0=1.2, thickness=0.3, material=2.0 + 0.1j
    ).array
    expected = core.slab(
        basis=INTERLEAVED_PORTS, k0=1.2, thickness=0.3, material=2.0 + 0.1j
    )
    assert_allclose(Engine.numpy(actual), expected.array, rtol=1e-13, atol=1e-13)


@pytest.mark.workflows
def test_port_waves_drive_a_second_system_group_by_group(engine):
    # The thickness gradient is the port_wave_second_system scenario.
    tr = engine.tr
    two_groups = core.PlaneWavePorts(
        [(0.1, 0.0, 1), (0.0, 0.2, 0), (0.1, 0.0, 0), (0.0, 0.2, 1)]
    )
    wave, upper = port_systems(tr, 0.3, PORT_AMPLITUDES)
    expected_wave, expected_upper = port_systems(core, 0.3, PORT_AMPLITUDES)
    assert_allclose(
        Engine.numpy(upper.scatter(negative=wave).positive.coefficients),
        expected_upper.scatter(negative=expected_wave).positive.coefficients,
        rtol=1e-12,
        atol=1e-14,
    )
    assert_allclose(
        Engine.numpy(upper.power(wave).transmission),
        expected_upper.power(expected_wave).transmission,
        rtol=1e-12,
    )
    # A port absent from the second system may only carry zero amplitude.
    subset = tr.slab(basis=two_groups, k0=1.2, thickness=0.2, material=1.5)
    quiet, _ = port_systems(tr, 0.3, [0.0, 1.0, 0.3j, 0.0, 0.0, 0.0])
    reference, _ = port_systems(core, 0.3, [0.0, 1.0, 0.3j, 0.0, 0.0, 0.0])
    assert_allclose(
        Engine.numpy(subset.scatter(negative=quiet).negative.coefficients),
        core.slab(basis=two_groups, k0=1.2, thickness=0.2, material=1.5)
        .scatter(negative=reference)
        .negative.coefficients,
        rtol=1e-12,
        atol=1e-14,
    )
    with pytest.raises(Exception, match="one represented diffraction port"):
        subset.scatter(negative=wave).negative.coefficients.sum()


@pytest.mark.interface
def test_polarization_conversion_requires_complete_pairs(engine):
    tr = engine.tr
    basis = core.SphericalBasis.default(1)[:1]
    wave = tr.wave([1.0], basis=basis, k0=1.2)
    ports = core.PlaneWavePorts.default([0.1, 0.05])[:1]
    network = tr.slab(basis=ports, k0=1.2, thickness=0.2, material=2.0)
    port_wave = network.scatter(negative=[1.0]).positive
    for physical in (wave, network, port_wave):
        assert physical.with_polarization("helicity").polarization == "helicity"
        with pytest.raises(ValueError, match="both polarizations"):
            physical.with_polarization("parity")


@pytest.mark.interface
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("family", ["sphere", "cylinder", "cluster"])
def test_polarization_changes_equal_the_dense_operator(engine, seed, family):
    # The framework applies the change pair by pair; the dense changepoltype
    # matrix is the reference, in both directions and for permuted modes.
    tr = engine.tr
    rng = np.random.default_rng(seed)
    modes = (
        core.CylindricalBasis.default([0.0, 0.2], 2)
        if family == "cylinder"
        else core.SphericalBasis.default(
            2,
            nmax=2 if family == "cluster" else 1,
            positions=[[0, 0, 0], [0.6, 0.2, 0.3]] if family == "cluster" else None,
        )
    )
    basis = modes[rng.permutation(len(modes))]
    matrix = rng.normal(size=(len(basis),) * 2) + 1j * rng.normal(
        size=(len(basis),) * 2
    )
    for source, target in (("helicity", "parity"), ("parity", "helicity")):
        change = core.operators.changepoltype((target, source), basis=basis)
        tm = tr.tmatrix(matrix, basis=basis, k0=1.2, polarization=source)
        assert_allclose(
            Engine.numpy(tm.with_polarization(target).array),
            change @ matrix @ change.T,
            rtol=1e-14,
            atol=1e-15,
        )
        wave = tr.wave(matrix[0], basis=basis, k0=1.2, polarization=source)
        assert_allclose(
            Engine.numpy(wave.with_polarization(target).coefficients),
            change @ matrix[0],
            rtol=1e-14,
            atol=1e-15,
        )
    network = tr.slab(
        basis=INTERLEAVED_PORTS, k0=1.2, thickness=0.3, material=2.0 + 0.1j
    )
    change = core.operators.changepoltype("parity", basis=INTERLEAVED_PORTS)
    assert_allclose(
        Engine.numpy(network.with_polarization("parity").array),
        change @ Engine.numpy(network.array) @ change.T,
        rtol=1e-14,
        atol=1e-15,
    )


@pytest.mark.interface
def test_coincident_particles_raise_like_the_numpy_cluster(engine):
    tr = engine.tr
    particle = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)
    system = tr.Cluster([particle, particle], positions=[[0, 0, 0.5], [0, 0, 0.5]])
    wave = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    numpy_particle = core.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)
    with pytest.raises(ValueError, match="distinct positions"):
        core.Cluster([numpy_particle, numpy_particle], positions=[[0, 0, 0.5]] * 2)
    with pytest.raises(Exception, match="distinct positions"):
        Engine.numpy(system.solve().array)
    with pytest.raises(Exception, match="distinct positions"):
        Engine.numpy(system.scatter(wave).coefficients)


@pytest.mark.interface
def test_parity_requires_achiral_media_like_the_numpy_api(engine):
    tr = engine.tr
    chiral, achiral = tr.Material(1.0, 1.0, 0.1), tr.Material(1.0)
    ports = core.PlaneWavePorts.default([[0.1, 0.05]])
    message = "parity polarization requires an achiral embedding medium"
    tm = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0, medium=chiral)
    attempts = [
        lambda: (
            tr.interface(
                basis=ports, k0=1.2, negative_medium=achiral, positive_medium=chiral
            )
            .scatter(negative=[1.0, 0.0])
            .positive.with_polarization("parity")
            .coefficients
        ),
        lambda: tm.with_polarization("parity").array,
        lambda: (
            tr.sphere_tmatrix(
                k0=1.2,
                lmax=1,
                radius=0.2,
                material=3.0,
                medium=chiral,
                polarization="parity",
            ).array
        ),
        lambda: (
            tm.scatter(
                tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2, medium=chiral)
            )
            .with_polarization("parity")
            .coefficients
        ),
        # An interior layer counts, as in the NumPy slab.
        lambda: (
            tr.slab(
                basis=ports,
                k0=1.2,
                thickness=0.2,
                material=chiral,
                polarization="parity",
            ).array
        ),
        lambda: (
            tr.interface(
                basis=ports,
                k0=1.2,
                negative_medium=achiral,
                positive_medium=chiral,
                polarization="parity",
            ).array
        ),
        lambda: (
            tr.propagation(
                distance=0.3, basis=ports, k0=1.2, medium=chiral, polarization="parity"
            ).array
        ),
    ]
    for attempt in attempts:
        with pytest.raises(Exception, match=message):
            Engine.numpy(attempt())
    with pytest.raises(ValueError, match=message):
        core.slab(
            basis=ports,
            k0=1.2,
            thickness=0.2,
            material=core.Material(1.0, 1.0, 0.1),
            polarization="parity",
        )


@pytest.mark.gradients
@pytest.mark.parametrize("port_wave", [False, True])
def test_parity_chirality_check_follows_framework_values(engine, port_wave):
    # A chirality carrying gradients is checked when the framework evaluates it:
    # zero passes with its gradient, nonzero raises.
    ports = core.PlaneWavePorts.default([[0.1, 0.05]])

    def reflection(tr, kappa):
        if port_wave:
            sm = tr.interface(
                basis=ports,
                k0=1.2,
                negative_medium=1,
                positive_medium=tr.Material(2.0, 1.0, kappa),
            )
            wave = sm.scatter(negative=[1.0, 0.0]).positive.with_polarization("parity")
            return (abs(wave.coefficients) ** 2).sum()
        sm = tr.slab(
            basis=ports,
            k0=1.2,
            thickness=0.2,
            material=tr.Material(2.0, 1.0, kappa),
            polarization="parity",
        )
        return sm.power([1.0, 0.0]).reflection

    value, gradient = engine.value_and_grad(reflection, 0.0)
    assert np.isfinite(value) and np.isfinite(gradient)
    with pytest.raises(Exception, match="achiral"):
        engine.value_and_grad(reflection, 0.1)


@pytest.mark.interface
def test_order_ports_need_a_planar_lattice(engine):
    tr = engine.tr
    tm = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)
    # A 3x3 lattice solves on its xy sublattice; ports need the planar cell.
    response = tr.solve_periodic(
        tm, lattice=[[2.0, 0, 0], [0, 2.0, 0], [0, 0, 5.0]], kpar=[0.1, 0.05]
    )
    with pytest.raises(ValueError, match=r"\(2, 2\) xy lattice"):
        response.to_smatrix(core.PlaneWavePorts.default([[0.1, 0.05]]))


@pytest.mark.interface
def test_framework_physical_mismatches_raise(engine):
    tr = engine.tr
    tm = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)
    with pytest.raises(Exception, match="matching k0"):
        tm.scatter(tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.3)).efield(
            [[0.2, 0.3, 1.5]]
        )
    ports = core.PlaneWavePorts.default([[0.0, 0.0]])
    sm = tr.interface(basis=ports, k0=1.2, negative_medium=1.0, positive_medium=1.0)
    with pytest.raises(ValueError, match="away"):
        sm.scatter(negative=tr.plane_wave([0, 0, -1], "positive_helicity", k0=1.2))
    for api in (tr, core):
        with pytest.raises(ValueError, match="positive_helicity or negative_helicity"):
            api.plane_wave([0, 0, 1], "positive", k0=1.2)


@pytest.mark.interface
@settings(max_examples=10)
@given(
    epsilon=st.floats(1.2, 5.0),
    loss=st.floats(0.0, 0.3),
    mu=st.floats(0.8, 1.5),
    kappa=st.floats(-0.2, 0.2),
)
def test_every_material_spelling_builds_the_same_response(
    engine, epsilon, loss, mu, kappa
):
    # A root Material and an (epsilon, mu, kappa) tuple give the namespace
    # Material's three values; a bare value is epsilon.
    tr = engine.tr
    values, exterior = (epsilon + 1j * loss, mu, kappa), (1.1, 1.0, 0.02)

    def response(material, medium):
        particle = tr.sphere_tmatrix(
            k0=1.2, lmax=1, radius=0.2, material=material, medium=medium
        )
        return Engine.numpy(particle.array)

    expected = response(tr.Material(*values), tr.Material(*exterior))
    assert_array_equal(
        response(core.Material(*values), core.Material(*exterior)), expected
    )
    assert_array_equal(response(values, exterior), expected)
    assert_array_equal(
        response(values[0], exterior[0]),
        response(tr.Material(values[0]), tr.Material(exterior[0])),
    )


@pytest.mark.interface
def test_declared_record_output_is_checked_in_every_engine(engine):
    backend = engine.tr._backend
    value = backend.array([1.0, 2.0], complex_=True)

    def doubled(v):
        return np.concatenate((v, v)), lambda g: g[:2]

    with pytest.raises(Exception, match=r"doubled: declared complex .* \(2,\)"):
        backend.apply(doubled, (2,), value)
    with pytest.raises(Exception, match="declared real output"):
        backend.apply(lambda v: (v, lambda g: g), (2,), value, real=True)


@pytest.mark.interface
def test_public_types_and_supplied_singular_wave(engine):
    tr = engine.tr
    tm = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)
    assert isinstance(tm, tr.TMatrix)
    basis = core.SphericalBasis.default(1, positions=[[0.0, 0.0, 2.0]])
    coefficients = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=complex)
    source = tr.wave(coefficients, basis=basis, k0=1.2, kind="singular")
    assert source.kind == "singular"
    with pytest.raises(ValueError, match="outgoing multipole waves are 'singular'"):
        tr.wave(coefficients, basis=basis, k0=1.2, kind="outgoing")
    scattered = tm.scatter(source)
    assert isinstance(scattered, tr.Wave)
    expansion, _ = diff.expansion(tm.basis, basis, [1.2, 1.2], singular=True)
    expected = (
        np.asarray(core.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0).array)
        @ expansion
        @ coefficients
    )
    assert_allclose(Engine.numpy(scattered.coefficients), expected, rtol=2e-13)
    assert_allclose(tr.plane_wave([0, 0, 1], 1, k0=1.2).coefficients, [0, 1])


@pytest.mark.workflows
def test_propagation_cascade_and_parity(engine):
    tr = engine.tr
    ports = core.PlaneWavePorts.default([[0.1, 0.05]])
    options = dict(basis=ports, k0=1.2, polarization="parity")
    layers = [
        tr.interface(negative_medium=1.0, positive_medium=2.0, **options),
        tr.propagation(distance=0.3, medium=2.0, **options),
        tr.interface(negative_medium=2.0, positive_medium=1.0, **options),
    ]
    actual = tr.stack(layers)
    expected = core.slab(thickness=0.3, material=2.0, **options)
    assert actual.polarization == "parity"
    assert_allclose(Engine.numpy(actual.array), expected.array, rtol=2e-13, atol=1e-14)
    # Derived S-matrices and their port waves share the lowest layer's ports.
    helicity = actual.with_polarization("helicity")
    scattered = helicity.scatter(negative=[1.0, 0.0])
    assert actual.ports is helicity.ports is layers[0].ports
    assert scattered.positive.ports is scattered.negative.ports is actual.ports
    assert actual.basis is ports
    assert actual.modes == scattered.positive.modes == ((0, 1), (0, 0))


@pytest.mark.interface
def test_high_level_jax_rejects_reduced_precision_without_detaching():
    jax = pytest.importorskip("jax")
    from treams_rs import jax as tr

    with jax_x64(jax):
        operation = jax.jit(
            lambda radius: (
                tr.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3.0).array
            )
        )
        with pytest.raises(TypeError, match="float64"):
            operation(jax.numpy.asarray(0.2, dtype=jax.numpy.float32))
        with jax.enable_x64(False), pytest.raises(ValueError, match="jax_enable_x64"):
            tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3.0)


@pytest.mark.interface
def test_framework_constructors_pickle_by_reference(engine):
    # The shared constructors are bound methods of the namespace's `_api`; they
    # pickle through that namespace, while the backend holds an unpicklable module.
    names = [name for name in vars(Constructors) if not name.startswith("_")]
    assert len(names) == 12
    for name in names:
        constructor = getattr(engine.tr, name)
        assert pickle.loads(pickle.dumps(constructor)) == constructor
        restored = pickle.loads(pickle.dumps(functools.partial(constructor, k0=1.2)))
        assert (restored.func, restored.keywords) == (constructor, {"k0": 1.2})
    options = dict(lmax=1, radius=0.2, material=3.0)
    restored = pickle.loads(pickle.dumps(engine.tr.sphere_tmatrix))
    assert_array_equal(
        engine.numpy(restored(k0=1.2, **options).array),
        engine.numpy(engine.tr.sphere_tmatrix(k0=1.2, **options).array),
    )


@pytest.mark.interface
def test_framework_has_the_same_static_geometry_vocabulary(engine):
    for name in (
        "PlaneWavePorts",
        "PlaneWaveBasis",
        "SphericalBasis",
        "CylindricalBasis",
        "Lattice",
    ):
        assert getattr(engine.tr, name) is getattr(core, name)


@pytest.mark.interface
@pytest.mark.parametrize("name", ["jax", "torch"])
def test_framework_operations_are_shared_and_pickle_by_reference(name):
    # JAX and PyTorch bind the expert operations of one Operations instance,
    # `_ops`, which pickles through its namespace like `_api`.
    pytest.importorskip(name)
    tr = importlib.import_module(f"treams_rs.{name}")
    members = [member for member in vars(Operations) if not member.startswith("_")]
    assert members == ["solve", "interaction", "illuminate", "sphere", "bessel"]
    for operation in (getattr(tr, member) for member in members):
        assert isinstance(operation.__self__, Operations)
        assert operation.__self__ is tr.bessel.__self__
        assert operation.__self__.backend is tr.wave.__self__.backend
        assert pickle.loads(pickle.dumps(operation)) == operation
    scope = jax_x64() if name == "jax" else contextlib.nullcontext()
    with scope:
        restored = pickle.loads(pickle.dumps(tr.bessel))
        assert_array_equal(
            Engine.numpy(restored([0.3, 1.1], order=2, spherical=True)),
            Engine.numpy(tr.bessel([0.3, 1.1], order=2, spherical=True)),
        )


@pytest.mark.interface
def test_framework_namespaces_export_the_same_physical_names():
    # Classes, result tuples, the shared constructors, solve_periodic and stack
    # form one vocabulary; a name added to one adapter must appear in all three.
    pytest.importorskip("jax")
    pytest.importorskip("torch")
    namespaces = [
        importlib.import_module(f"treams_rs.{name}")
        for name in ("advect", "jax", "torch")
    ]
    constructors = {name for name in vars(Constructors) if not name.startswith("_")}
    functions = constructors | {"solve_periodic", "stack"}

    def physical(namespace):
        return {
            name for name in namespace.__all__ if name[0].isupper() or name in functions
        }

    advect_names, jax_names, torch_names = map(physical, namespaces)
    assert functions <= advect_names
    assert advect_names == jax_names == torch_names
    # JAX and PyTorch also share wrap; every adapter binds the expert operations.
    assert namespaces[1].__all__ == namespaces[2].__all__
    operations = [name for name in vars(Operations) if not name.startswith("_")]
    for namespace in namespaces:
        for name in operations:
            assert isinstance(getattr(namespace, name).__self__, Operations)
    # Every namespace exports the same classes, not per-adapter subclasses.
    for name in (name for name in advect_names if name[0].isupper()):
        assert all(getattr(n, name) is getattr(namespaces[0], name) for n in namespaces)
