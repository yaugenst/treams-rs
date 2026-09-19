"""Physical workflows retain metadata and preserve native physical invariants."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import _operators
from treams_rs._physics import (
    Cluster,
    cylinder_tmatrix,
    interface,
    multilayer_cylinder_tmatrix,
    multilayer_slab,
    multilayer_sphere_tmatrix,
    propagation,
    slab,
    solve_periodic,
    sphere_tmatrix,
    stack,
)
from treams_rs._tmatrix import _PeriodicInteraction


@settings(max_examples=12)
@given(radius=st.floats(0.1, 0.4), epsilon=st.floats(1.2, 5))
def test_sphere_typed_workflow_optical_theorem(radius, epsilon):
    sphere = sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=epsilon)
    incident = tr.plane_wave(
        direction=[0, 0, 1], polarization="positive_helicity", k0=1.3
    )
    scattered = sphere.scatter(incident)
    assert isinstance(scattered, tr.Wave)
    assert scattered.medium == sphere.medium
    assert not scattered.coefficients.flags.writeable
    assert_allclose(
        scattered.coefficients,
        sphere.array @ incident.in_basis(sphere.basis).coefficients,
    )
    cross = sphere.cross_sections(incident)
    assert_allclose(cross.scattering, cross.extinction, rtol=2e-11, atol=1e-14)
    assert_allclose(cross.absorption, 0, atol=1e-14)
    assert_allclose(
        sphere.average_cross_sections.scattering, cross.scattering, rtol=2e-11
    )
    assert_allclose(
        sphere.with_polarization("parity").with_polarization("helicity").array,
        sphere.array,
        atol=1e-14,
    )
    assert_allclose(sphere.in_basis(sphere.basis).array, sphere.array, atol=1e-14)
    dipoles = sphere.select(tr.SphericalBasis.default(1))
    assert len(dipoles.basis) == 6


@pytest.mark.parametrize("cylindrical", [False, True])
def test_wave_all_fields_and_batch_columns(cylindrical):
    matrix = (
        cylinder_tmatrix(k0=1.3, kz=0, mmax=2, radius=0.2, material=3)
        if cylindrical
        else sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
    )
    direction = [1, 0, 0] if cylindrical else [0, 0, 1]
    source = tr.plane_wave(
        direction=direction, polarization="positive_helicity", k0=1.3
    )
    coefficients = source.in_basis(matrix.basis).coefficients
    batch = matrix.scatter(np.column_stack([coefficients, 2j * coefficients]))
    points = np.array([[0.6, 0.4, 0.3], [0.8, -0.2, 0.1]])
    for field in ("efield", "hfield", "dfield", "bfield", "gfield", "ffield"):
        args = (1, points) if field in ("gfield", "ffield") else (points,)
        actual = getattr(batch, field)(*args)
        operator = getattr(_operators, field)(
            *args, basis=matrix.basis, k0=1.3, modetype="singular"
        )
        assert actual.shape == (2, 3, 2)
        assert_allclose(actual, operator @ batch.coefficients, atol=2e-13, rtol=2e-12)
        assert_allclose(actual[..., 1], 2j * actual[..., 0], atol=2e-13)
    if cylindrical:
        width = matrix.cross_widths(source)
        assert_allclose(width.scattering, width.extinction, atol=1e-13)
        assert_allclose(matrix.average_cross_widths.absorption, 0, atol=1e-13)
    with pytest.raises(ValueError, match="matching"):
        matrix.scatter(
            tr.plane_wave(direction=direction, polarization="positive_helicity", k0=2)
        )


@pytest.mark.parametrize("cylindrical", [False, True])
def test_cluster_requested_factor_and_dense_response_agree(cylindrical):
    make = cylinder_tmatrix if cylindrical else sphere_tmatrix
    options = {"kz": 0, "mmax": 1} if cylindrical else {"lmax": 1}
    particles = [
        make(k0=1.3, radius=radius, material=3, **options) for radius in (0.1, 0.2)
    ]
    cluster = Cluster(particles, positions=[[0, 0, 0], [0.9, 0.2, 0]])
    source = tr.plane_wave(
        direction=[1, 0, 0], polarization="positive_helicity", k0=1.3
    )
    solved = cluster.solve()
    direct = solved.scatter(source)
    assert_allclose(
        cluster.scatter(source).coefficients,
        direct.coefficients,
        rtol=1e-12,
        atol=1e-14,
    )
    assert_allclose(
        cluster.factor().scatter(source).coefficients,
        direct.coefficients,
        rtol=1e-12,
        atol=1e-14,
    )
    assert not hasattr(cluster, "array")
    points = [[0.4, 0.5, 0.3]]
    assert_allclose(
        cluster.scatter(source).efield(points), direct.efield(points), rtol=1e-12
    )
    periodic_options = (
        {"lattice": 2.0, "kpar": 0.0}
        if cylindrical
        else {"lattice": np.eye(2) * 2, "kpar": [0, 0]}
    )
    with pytest.raises(ValueError, match="twice"):
        solve_periodic(solved, **periodic_options)
    periodic = solve_periodic(cluster, **periodic_options)
    coupling = _operators.expandlattice(
        periodic_options["lattice"],
        periodic_options["kpar"],
        basis=cluster.basis,
        k0=cluster.k0,
    )
    local = np.zeros_like(periodic.array)
    offset = 0
    for particle in particles:
        size = len(particle.basis)
        local[offset : offset + size, offset : offset + size] = particle.array
        offset += size
    assert_allclose(
        (np.eye(len(local)) - local @ coupling) @ periodic.array, local, atol=1e-13
    )


def test_layer_materials_exclude_exterior_and_constructors_agree():
    sphere = multilayer_sphere_tmatrix(
        k0=1.3, lmax=2, radii=[0.1, 0.2], materials=[2, 3], medium=1.2
    )
    assert_allclose(
        sphere.array, tr.TMatrix.sphere(2, 1.3, [0.1, 0.2], [2, 3, 1.2]).array
    )
    cylinder = multilayer_cylinder_tmatrix(
        k0=1.3, kz=0.1, mmax=2, radii=[0.1, 0.2], materials=[2, 3], medium=1.2
    )
    assert_allclose(
        cylinder.array,
        tr.CylindricalTMatrix.cylinder(0.1, 2, 1.3, [0.1, 0.2], [2, 3, 1.2]).array,
    )


def test_wave_construction_and_representation_preserve_fields_and_origins():
    basis = tr.SphericalBasis.default(1)
    coefficients = np.arange(len(basis)) * (0.02 + 0.01j)
    wave = tr.Wave(
        coefficients,
        basis=basis,
        k0=1.3,
        medium=1.2,
        kind="outgoing",
        polarization="helicity",
    )
    assert wave.kind == "outgoing"
    assert wave.medium == tr.Material(1.2)
    points = [[0.5, 0.3, 0.4]]
    assert_allclose(
        wave.with_polarization("parity").efield(points), wave.efield(points), atol=1e-13
    )
    source = tr.plane_wave(
        direction=[0, 0, 1], polarization="positive_helicity", k0=1.3
    )
    assert_allclose(
        source.with_polarization("parity").efield(points),
        source.efield(points),
        atol=1e-13,
    )
    ports = tr.PlaneWavePorts.default([0, 0])
    for plane_basis in (ports, source.basis):
        converted = source.in_basis(plane_basis).in_basis(basis)
        assert converted.kind == "regular"
        assert_allclose(
            converted.coefficients, source.in_basis(basis).coefficients, atol=1e-13
        )
        with pytest.raises(ValueError, match="regular"):
            source.in_basis(plane_basis).in_basis(basis, kind="outgoing")
    matrix = sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
    with pytest.raises(ValueError, match="origins"):
        matrix.select(tr.SphericalBasis.default(1, positions=[[1, 0, 0]]))


@pytest.mark.parametrize("side", ["negative", "positive"])
def test_planar_named_outputs_conserve_power_and_evaluate_fields(side):
    basis = tr.PlaneWavePorts.default([0, 0])
    layer = slab(basis=basis, k0=1.3, thickness=0.4, material=3)
    source = tr.plane_wave(
        direction=[0, 0, 1 if side == "negative" else -1],
        polarization="positive_helicity",
        k0=1.3,
    )
    power = layer.power(source, side=side)
    assert_allclose(power.transmission + power.reflection, 1, atol=1e-12)
    assert_allclose(power.absorption, 0, atol=1e-12)
    outgoing = layer.scatter(**{side: source})
    typed_incident = source.in_basis(basis)
    assert_allclose(layer.power(typed_incident, side=side), power)
    for field, direction in ((outgoing.positive, "up"), (outgoing.negative, "down")):
        points = [[0.1, 0.2, 1 if direction == "up" else -1]]
        expected = (
            _operators.efield(points, basis=basis, k0=1.3, modetype=direction)
            @ field.coefficients
        )
        assert_allclose(field.efield(points), expected, atol=1e-13)
    assembled = stack(
        [
            interface(basis=basis, k0=1.3, negative_medium=1, positive_medium=3),
            propagation(distance=0.4, basis=basis, k0=1.3, medium=3),
            interface(basis=basis, k0=1.3, negative_medium=3, positive_medium=1),
        ]
    )
    assert_allclose(assembled.array, layer.array, atol=1e-13)
    assert_allclose(
        multilayer_slab(
            basis=basis, k0=1.3, thicknesses=[0.2, 0.2], materials=[3, 3]
        ).array,
        layer.array,
        atol=1e-13,
    )
    bands = layer.bands(period=0.4)
    transfer = layer.transfer_matrix()
    assert_allclose(
        transfer @ bands.eigenvectors,
        bands.eigenvectors * np.exp(1j * bands.wavenumbers * 0.4),
        atol=1e-12,
    )
    assert_allclose(layer.circular_dichroism(source).transmission, 0, atol=1e-12)


def test_periodic_spherical_ports_fields_and_no_second_solve(monkeypatch):
    sphere = sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
    lattice, kpar = np.eye(2) * 2, [0, 0]
    ports = tr.PlaneWavePorts.default([0, 0])
    response = solve_periodic(sphere, lattice=lattice, kpar=kpar)
    expected = tr.SMatrix._from_array(sphere, ports, lattice=lattice, kpar=kpar)

    def forbidden(*args, **kwargs):
        pytest.fail("conversion must not solve interactions a second time")

    monkeypatch.setattr(_PeriodicInteraction, "solve", forbidden)
    actual = response.to_smatrix(ports)
    assert_allclose(actual.array, expected.array, atol=1e-13)
    source = tr.plane_wave(
        direction=[0, 0, 1], polarization="positive_helicity", k0=1.3
    )
    wave = response.scatter(source)
    radiated = wave.in_basis(ports, kind="up")
    points = [[0.2, 0.3, 0.8]]
    total = actual.scatter(negative=source).positive.efield(points)
    assert_allclose(
        radiated.efield(points) + source.efield(points), total, rtol=1e-11, atol=1e-13
    )
    local_basis = tr.SphericalBasis.default(2, positions=[[0.3, 0.4, 0.5]])
    regular = wave.in_basis(local_basis, kind="regular")
    expected_regular = (
        _operators.expandlattice(
            lattice, kpar, basis=(local_basis, sphere.basis), k0=1.3
        )
        @ wave.coefficients
    )
    assert_allclose(regular.coefficients, expected_regular, atol=1e-13)
    assert np.linalg.norm(regular.efield([[0.3, 0.4, 0.5]])) > 0
    with pytest.raises(ValueError, match="Bloch"):
        response.scatter(
            tr.plane_wave(direction=[1, 0, 1], polarization="positive_helicity", k0=1.3)
        )
    mismatched = tr.plane_wave(
        direction=[1, 0, 1], polarization="positive_helicity", k0=1.3
    )
    wrong_ports = tr.PlaneWavePorts.default([1.3 / np.sqrt(2), 0])
    for basis in (wrong_ports, mismatched.basis):
        typed = mismatched.in_basis(basis)
        with pytest.raises(ValueError, match="Bloch"):
            response.scatter(typed)
        batch = tr.Wave(
            np.column_stack([typed.coefficients, 2j * typed.coefficients]),
            basis=basis,
            k0=1.3,
            kind=typed.kind,
        )
        with pytest.raises(ValueError, match="Bloch"):
            response.scatter(batch)
    assert_allclose(
        response.scatter(source.in_basis(ports)).coefficients,
        response.scatter(source).coefficients,
        atol=1e-13,
    )
    assert not hasattr(response, "cross_sections")


def test_periodic_cylindrical_and_spherical_to_cylindrical_paths(monkeypatch):
    sphere = sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
    cylindrical = tr.CylindricalBasis.default([0.1], 2)
    response = solve_periodic(sphere, lattice=2.0, kpar=0.1)
    expected = tr.CylindricalTMatrix._from_array(
        sphere, cylindrical, lattice=2.0, kpar=0.1
    )
    cylinder = cylinder_tmatrix(k0=1.3, kz=0, mmax=2, radius=0.2, material=3)
    ports = tr.PlaneWavePorts.default([0, 0], alignment="zx")
    cylinder_response = solve_periodic(cylinder, lattice=2.0, kpar=0.0)
    expected_ports = tr.SMatrix._from_array(cylinder, ports, lattice=2.0, kpar=0.0)

    def forbidden(*args, **kwargs):
        pytest.fail("conversion must not solve again")

    monkeypatch.setattr(_PeriodicInteraction, "solve", forbidden)
    assert_allclose(
        response.to_cylindrical(cylindrical).array, expected.array, atol=1e-13
    )
    assert_allclose(
        cylinder_response.to_smatrix(ports).array, expected_ports.array, atol=1e-13
    )
