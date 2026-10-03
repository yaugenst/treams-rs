"""Periodic response and radiation methods retain their ordinary public workflows."""

import functools

import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_transparent_api import engine as engine

import treams_rs as tr
from treams_rs.testing import check_gradient

pytestmark = pytest.mark.gradients


def _objective(api, x, *, operation, parameter):
    radius = x if parameter == "radius" else 0.2
    period = x + 0.9 if parameter == "period" else 1.1
    sphere = api.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3 + 0.1j)
    if operation == "ports":
        lattice, kpar = [[period, 0], [0, period]], [0, 0]
        basis = api.PlaneWavePorts.default([0, 0])
        response = api.solve_periodic(sphere, lattice=lattice, kpar=kpar)
        incident = api.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
        field = (
            response.scatter(incident)
            .in_basis(basis, kind="up")
            .efield([[0.1, 0.2, 1.4]])
        )
    else:
        response = api.solve_periodic(sphere, lattice=[[period]], kpar=[0])
        basis = api.CylindricalBasis.default([0], 1)
        incident = api.plane_wave([1, 0, 0], "positive_helicity", k0=1.2)
        if operation == "cylindrical_response":
            field = (
                response.to_cylindrical(basis)
                .scatter(incident)
                .efield([[0.7, 0.4, 0.1]])
            )
        else:
            field = (
                response.scatter(incident)
                .in_basis(basis, kind="singular")
                .efield([[0.7, 0.4, 0.1]])
            )
    return (abs(field) ** 2).sum()


@pytest.mark.parametrize(
    "operation", ["ports", "cylindrical_response", "cylindrical_wave"]
)
@pytest.mark.parametrize("parameter", ["radius", "period"])
def test_periodic_radiation_matches_numpy_and_finite_difference(
    engine, operation, parameter
):
    function = functools.partial(_objective, operation=operation, parameter=parameter)
    value, gradient = engine.value_and_grad(function, 0.2)
    step = 1e-6
    expected = (function(tr, 0.2 + step) - function(tr, 0.2 - step)) / (2 * step)
    assert_allclose(value, function(tr, 0.2), rtol=1e-11, atol=1e-14)
    assert_allclose(gradient, expected, rtol=2e-5, atol=1e-9)


@pytest.mark.physics
def test_explicit_diffraction_orders_match_a_fixed_basis():
    response = tr.solve_periodic(
        tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3),
        lattice=[[1.1, 0.1], [0.2, 1.0]],
        kpar=[0.1, 0.2],
    )
    orders = [[0, 0], [1, -1]]
    q = response.kpar + 2 * np.pi * np.array(orders) @ np.linalg.inv(response.lattice).T
    basis = tr.PlaneWavePorts([(*vector, pol) for vector in q for pol in (1, 0)])
    assert_allclose(
        response.to_smatrix(orders=orders).array, response.to_smatrix(basis).array
    )


@pytest.mark.parametrize("wrapper", ["response", "wave"])
def test_periodic_wrappers_accept_framework_values(engine, wrapper):
    particle = tr.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3 + 0.1j)
    lattice, kpar = np.eye(2) * 1.1, [0, 0]
    response = tr.solve_periodic(particle, lattice=lattice, kpar=kpar)
    incident = tr.plane_wave([0, 0, 1], "positive_helicity", k0=1.2)
    basis = tr.PlaneWavePorts.default([0, 0])

    def objective(api, x):
        if wrapper == "response":
            array = [[x * value.item() for value in row] for row in response.array]
            dynamic = api.PeriodicResponse(particle, array, lattice=lattice, kpar=kpar)
            periodic = dynamic.scatter(incident)
        else:
            coefficients = response.scatter(incident).coefficients
            local = api.Wave(
                [x * value.item() for value in coefficients],
                basis=particle.basis,
                k0=1.2,
                kind="singular",
            )
            periodic = api.PeriodicWave(local, lattice, kpar)
        field = periodic.in_basis(basis, kind="up").efield([[0.1, 0.2, 1.4]])
        return (abs(field) ** 2).sum()

    value, gradient = engine.value_and_grad(objective, 0.7)
    expected = objective(tr, 1.0)
    assert_allclose(value, 0.7**2 * expected, rtol=1e-12)
    assert_allclose(gradient, 1.4 * expected, rtol=1e-12)


@pytest.mark.parametrize(
    "method", ["coupling", "call", "solve", "factor", "illuminate"]
)
@pytest.mark.parametrize("constant_receiver", [False, True])
def test_latticeinteraction_methods_preserve_gradients(
    engine, method, constant_receiver
):
    def objective(api, x):
        radius = 0.2 if constant_receiver else x
        particle = api.sphere_tmatrix(k0=1.2, lmax=1, radius=radius, material=3 + 0.1j)
        interaction = particle.latticeinteraction
        lattice, kpar = [[x + 0.9, 0], [0, x + 0.9]], [0, 0]
        if method == "call":
            value = interaction(lattice, kpar)
        elif method == "factor":
            value = interaction.factor(lattice, kpar).solve(np.eye(6, dtype=complex))
        elif method == "illuminate":
            value = interaction.illuminate(np.eye(6, dtype=complex), lattice, kpar)
        else:
            value = getattr(interaction, method)(lattice, kpar)
        return (abs(value) ** 2).sum()

    value, gradient = engine.value_and_grad(objective, 0.2)
    step = 1e-6
    expected = (objective(tr, 0.2 + step) - objective(tr, 0.2 - step)) / (2 * step)
    assert_allclose(value, objective(tr, 0.2), rtol=1e-12)
    assert_allclose(gradient, expected, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize(
    "cell",
    [
        "lattice",
        "diagonal",
        "larger_diagonal",
        "larger_matrix",
        "wavevector",
        "moving_diagonal",
    ],
)
def test_periodic_solves_accept_lattice_metadata(engine, cell):
    def objective(api, x):
        radius = 0.2 if cell == "moving_diagonal" else x
        lattice, kpar = {
            "lattice": (tr.Lattice.square(0.9), [0, 0]),
            "diagonal": ([0.9, 0.95], [0, 0]),
            "larger_diagonal": ([0.9, 0.95, 1.1], [0, 0]),
            "larger_matrix": (
                [[0, 0, 1.1], [0.9, 0.05, 0], [0.03, 0.95, 0]],
                [0, 0],
            ),
            "wavevector": (np.eye(2) * 0.9, tr.WaveVector([0, 0])),
            "moving_diagonal": ([x + 0.7, x + 0.75], [0, 0]),
        }[cell]
        particle = api.sphere_tmatrix(k0=2.0, lmax=1, radius=radius, material=3 + 0.1j)
        response = api.solve_periodic(particle, lattice=lattice, kpar=kpar)
        incident = api.plane_wave([0, 0, 1], "positive_helicity", k0=2.0)
        power = response.to_smatrix(api.PlaneWavePorts.default([0, 0])).power(
            incident, side="negative"
        )
        coupled = particle.latticeinteraction.solve(lattice, kpar)
        return power.transmission + (abs(coupled) ** 2).sum()

    value, gradient = engine.value_and_grad(objective, 0.2)
    assert_allclose(value, objective(tr, 0.2), rtol=1e-12)
    check_gradient(
        functools.partial(objective, tr),
        lambda _: gradient,
        np.asarray(0.2),
        directions=(np.asarray(1.0),),
        step=1e-6,
        rtol=2e-5,
        atol=1e-8,
    )


@pytest.mark.parametrize("matrix", [False, True])
def test_sublattice_gradients_preserve_cell_shape_and_excluded_axes(engine, matrix):
    if matrix:
        point = np.array([[0, 0, 1.1], [0.9, 0.05, 0], [0.03, 0.95, 0]])
        # Preserve the xy sublattice while varying its rows and the unused z period.
        direction = np.array([[0, 0, 0.6], [0.3, -0.2, 0], [-0.1, 0.5, 0]])
    else:
        point = np.array([0.9, 0.95, 1.1])
        direction = np.array([0.3, 0.5, 0.6])

    def objective(api, lattice):
        particle = api.sphere_tmatrix(k0=2.0, lmax=1, radius=0.2, material=3 + 0.1j)
        response = api.solve_periodic(particle, lattice=lattice, kpar=[0, 0])
        incident = api.plane_wave([0, 0, 1], "positive_helicity", k0=2.0)
        return (
            response.to_smatrix(api.PlaneWavePorts.default([0, 0]))
            .power(incident, side="negative")
            .transmission
        )

    value, gradient = engine.value_and_grad(objective, point)
    assert_allclose(value, objective(tr, point), rtol=1e-12)
    assert gradient.shape == point.shape
    if matrix:
        assert_allclose(gradient[0], 0, atol=0)
        assert_allclose(gradient[:, 2], 0, atol=0)
    else:
        assert gradient[2] == 0
    check_gradient(
        functools.partial(objective, tr),
        lambda _: gradient,
        point,
        directions=(direction,),
        step=1e-6,
        rtol=2e-5,
        atol=1e-8,
    )


@pytest.mark.parametrize("spherical", [True, False])
@pytest.mark.parametrize(
    "period", ["lattice", "list", "moving_scalar", "moving_list", "moving_nested"]
)
def test_chain_solves_accept_every_period_form(engine, spherical, period):
    def objective(api, x):
        moving = period.startswith("moving")
        p, radius = (x + 0.9, 0.2) if moving else (1.1, x)
        if period == "lattice":
            lattice = tr.Lattice(p, "z" if spherical else "x")
        else:
            lattice = {"list": [p], "moving_list": [p], "moving_nested": [[p]]}.get(
                period, p
            )
        if spherical:
            particle = api.sphere_tmatrix(
                k0=1.2, lmax=1, radius=radius, material=3 + 0.1j
            )
            response = api.solve_periodic(particle, lattice=lattice, kpar=[0])
            incident = api.plane_wave([1, 0, 0], "positive_helicity", k0=1.2)
            basis = api.CylindricalBasis.default([0], 1)
            wave = response.scatter(incident).in_basis(basis, kind="singular")
            solved = wave.efield([[0.7, 0.4, 0.1]])
        else:
            particle = api.cylinder_tmatrix(
                k0=1.2, kz=[0.0], mmax=1, radius=radius, material=3 + 0.1j
            )
            solved = api.solve_periodic(particle, lattice=lattice, kpar=[0]).array
        coupled = particle.latticeinteraction.solve(lattice, [0])
        return (abs(solved) ** 2).sum() + (abs(coupled) ** 2).sum()

    value, gradient = engine.value_and_grad(objective, 0.2)
    assert_allclose(value, objective(tr, 0.2), rtol=1e-12)
    check_gradient(
        functools.partial(objective, tr),
        lambda _: gradient,
        np.asarray(0.2),
        directions=(np.asarray(1.0),),
        step=1e-6,
        rtol=2e-5,
        atol=1e-8,
    )
