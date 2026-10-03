"""Physical workflows retain metadata and preserve native physical invariants.

Checks the typed physics API (sphere_tmatrix, Cluster, slab, ...); test_sphere checks
the sphere optical theorem through the upstream-compatible TMatrix.sphere.
"""

import importlib

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

import treams_rs as tr
from treams_rs import (
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
from treams_rs import advect as ad

from _support import reciprocal

pytestmark = pytest.mark.workflows


@pytest.mark.physics
@settings(max_examples=12)
@given(radius=st.floats(0.1, 0.4), epsilon=st.floats(1.2, 5))
def test_sphere_typed_workflow_optical_theorem(radius, epsilon):
    sphere = sphere_tmatrix(k0=1.3, lmax=2, radius=radius, material=epsilon)
    incident = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.3)
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
    source = tr.plane_wave(direction=direction, pol="positive_helicity", k0=1.3)
    coefficients = source.in_basis(matrix.basis).coefficients
    batch = matrix.scatter(np.column_stack([coefficients, 2j * coefficients]))
    points = np.array([[0.6, 0.4, 0.3], [0.8, -0.2, 0.1]])
    for field in ("efield", "hfield", "dfield", "bfield", "gfield", "ffield"):
        # A response has no fields of its own; they belong to a scattered wave.
        with pytest.raises(AttributeError, match=r"response.scatter\(incident\)"):
            getattr(matrix, field)
        args = (1, points) if field in ("gfield", "ffield") else (points,)
        actual = getattr(batch, field)(*args)
        operator = getattr(tr.operators, field)(
            *args, basis=matrix.basis, k0=1.3, modetype="singular"
        )
        assert actual.shape == (2, 3, 2)
        assert_allclose(actual, operator @ batch.coefficients, atol=2e-13, rtol=2e-12)
        assert_allclose(actual[..., 1], 2j * actual[..., 0], atol=2e-13)
    if cylindrical:
        width = matrix.cross_widths(source)
        assert_allclose(width.scattering, width.extinction, atol=1e-13)
        # The family-neutral name gives the same widths, bit for bit.
        assert matrix.cross_sections(source) == width
        assert_allclose(matrix.average_cross_widths.absorption, 0, atol=1e-13)
    with pytest.raises(ValueError, match="matching"):
        matrix.scatter(
            tr.plane_wave(direction=direction, pol="positive_helicity", k0=2)
        )


@pytest.mark.parametrize("cylindrical", [False, True])
def test_cluster_requested_factor_and_dense_response_agree(cylindrical):
    make = cylinder_tmatrix if cylindrical else sphere_tmatrix
    options = {"kz": 0, "mmax": 1} if cylindrical else {"lmax": 1}
    particles = [
        make(k0=1.3, radius=radius, material=3, **options) for radius in (0.1, 0.2)
    ]
    cluster = Cluster(particles, positions=[[0, 0, 0], [0.9, 0.2, 0]])
    source = tr.plane_wave(direction=[1, 0, 0], pol="positive_helicity", k0=1.3)
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
    coupling = tr.operators.expandlattice(
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


@pytest.mark.interface
def test_wave_construction_and_representation_preserve_fields_and_origins():
    basis = tr.SphericalBasis.default(1)
    coefficients = np.arange(len(basis)) * (0.02 + 0.01j)
    wave = tr.Wave(
        coefficients,
        basis=basis,
        k0=1.3,
        medium=1.2,
        kind="singular",
        polarization="helicity",
    )
    assert wave.kind == "singular"
    assert wave.medium == tr.Material(1.2)
    points = [[0.5, 0.3, 0.4]]
    assert_allclose(
        wave.with_polarization("parity").efield(points), wave.efield(points), atol=1e-13
    )
    source = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.3)
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
            source.in_basis(plane_basis).in_basis(basis, kind="singular")
    matrix = sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
    with pytest.raises(ValueError, match="positions"):
        matrix.select(tr.SphericalBasis.default(1, positions=[[1, 0, 0]]))


@pytest.mark.physics
@pytest.mark.parametrize("side", ["negative", "positive"])
def test_planar_named_outputs_conserve_power_and_evaluate_fields(side):
    basis = tr.PlaneWavePorts.default([0, 0])
    layer = slab(basis=basis, k0=1.3, thickness=0.4, material=3)
    source = tr.plane_wave(
        direction=[0, 0, 1 if side == "negative" else -1],
        pol="positive_helicity",
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
            tr.operators.efield(points, basis=basis, k0=1.3, modetype=direction)
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


def _with_response(cell, response, array):
    """The periodic response of the same cell with another solved array."""
    return tr.PeriodicResponse(
        cell, array, lattice=response.lattice, kpar=response.kpar
    )


def test_periodic_spherical_ports_fields_and_no_second_solve():
    sphere = sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
    lattice, kpar = np.eye(2) * 2, [0, 0]
    ports = tr.PlaneWavePorts.default([0, 0])
    response = solve_periodic(sphere, lattice=lattice, kpar=kpar)
    actual = response.to_smatrix(ports)
    # The ports depend affinely on the stored response, so a conversion that
    # solved the cell again would not follow a rescaled array.
    empty = _with_response(sphere, response, np.zeros_like(response.array)).to_smatrix(
        ports
    )
    assert_array_equal(empty.array, tr.SMatrix.propagation(0, ports, 1.3).array)
    doubled = _with_response(sphere, response, 2 * response.array).to_smatrix(ports)
    assert_allclose(
        doubled.array - empty.array, 2 * (actual.array - empty.array), atol=1e-13
    )
    with pytest.raises(ValueError, match="diffraction orders"):
        response.to_smatrix(tr.PlaneWavePorts.default([0.4, 0]))
    source = tr.plane_wave(direction=[0, 0, 1], pol="positive_helicity", k0=1.3)
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
        tr.operators.expandlattice(
            lattice, kpar, basis=(local_basis, sphere.basis), k0=1.3
        )
        @ wave.coefficients
    )
    assert_allclose(regular.coefficients, expected_regular, atol=1e-13)
    assert np.linalg.norm(regular.efield([[0.3, 0.4, 0.5]])) > 0
    with pytest.raises(ValueError, match="Bloch"):
        response.scatter(
            tr.plane_wave(direction=[1, 0, 1], pol="positive_helicity", k0=1.3)
        )
    mismatched = tr.plane_wave(direction=[1, 0, 1], pol="positive_helicity", k0=1.3)
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


def test_periodic_cylindrical_and_spherical_to_cylindrical_paths():
    sphere = sphere_tmatrix(k0=1.3, lmax=2, radius=0.2, material=3)
    cylindrical = tr.CylindricalBasis.default([0.1], 2)
    response = solve_periodic(sphere, lattice=2.0, kpar=0.1)
    cylinder = cylinder_tmatrix(k0=1.3, kz=0, mmax=2, radius=0.2, material=3)
    ports = tr.PlaneWavePorts.default([0, 0], alignment="zx")
    cylinder_response = solve_periodic(cylinder, lattice=2.0, kpar=0.0)
    # Both conversions read the stored response: the cylindrical modes are
    # linear in it and the ports affine, so a second solve would show.
    doubled = _with_response(sphere, response, 2 * response.array)
    assert_array_equal(
        doubled.to_cylindrical(cylindrical).array,
        2 * response.to_cylindrical(cylindrical).array,
    )
    empty = _with_response(
        cylinder, cylinder_response, np.zeros_like(cylinder_response.array)
    ).to_smatrix(ports)
    assert_array_equal(empty.array, tr.SMatrix.propagation(0, ports, 1.3).array)
    doubled_ports = _with_response(
        cylinder, cylinder_response, 2 * cylinder_response.array
    ).to_smatrix(ports)
    assert_allclose(
        doubled_ports.array - empty.array,
        2 * (cylinder_response.to_smatrix(ports).array - empty.array),
        atol=1e-13,
    )
    with pytest.raises(ValueError, match="diffraction orders"):
        response.to_cylindrical(tr.CylindricalBasis.default([0.4], 2))
    with pytest.raises(ValueError, match="diffraction orders"):
        cylinder_response.to_smatrix(
            tr.PlaneWavePorts.default([0, 0.4], alignment="zx")
        )


# Physical invariants of the public constructors and workflows ---------------------


@st.composite
def materials(draw, *, chiral=True, lossy=True):
    """Passive materials; chiral only where the representation allows it."""
    epsilon = draw(st.floats(0.2, 8.0)) + 1j * (draw(st.floats(0, 0.3)) if lossy else 0)
    kappa = draw(st.floats(-0.1, 0.1)) if chiral else 0
    return tr.Material(epsilon, draw(st.floats(0.8, 2.0)), kappa)


@pytest.mark.physics
@pytest.mark.parametrize("polarization", ["helicity", "parity"])
@given(
    k0=st.floats(0.8, 1.5),
    radius=st.floats(0.25, 2.0),
    fraction=st.floats(0.2, 0.8),
    lmax=st.integers(1, 4),
    scale=st.floats(0.5, 2.0),
    material=materials(),
    medium=materials(lossy=False),
    embedded=materials(),
)
# Nearly index-matched weak scatterers, whose T-matrices carry rounding errors far
# above eps |T|.
@example(
    k0=1.287,
    radius=0.875,
    fraction=0.5,
    lmax=1,
    scale=1.06,
    material=tr.Material(3 + 1e-5j, 0.875),
    medium=tr.Material(3, 0.875),
    embedded=tr.Material(),
)
@example(
    k0=0.8,
    radius=0.875,
    fraction=0.5,
    lmax=1,
    scale=1.5,
    material=tr.Material(8, 0.8125),
    medium=tr.Material(7.875, 0.8125),
    embedded=tr.Material(),
)
def test_sphere_constructor_invariants(
    polarization, k0, radius, fraction, lmax, scale, material, medium, embedded
):
    # Parity rejects a chiral embedding; a chiral particle is allowed.
    if polarization == "parity":
        medium, embedded = (tr.Material(m.epsilon, m.mu) for m in (medium, embedded))
    options = dict(k0=k0, lmax=lmax, polarization=polarization)
    sphere = sphere_tmatrix(radius=radius, material=material, medium=medium, **options)
    # A particle made of its embedding does not scatter.
    assert_allclose(
        sphere_tmatrix(
            radius=radius, material=embedded, medium=embedded, **options
        ).array,
        0,
        atol=2e-13,
    )
    # Splitting a homogeneous sphere into two identical layers changes nothing.
    split = multilayer_sphere_tmatrix(
        radii=[fraction * radius, radius],
        materials=[material, material],
        medium=medium,
        **options,
    )
    assert_allclose(split.array, sphere.array, rtol=3e-11, atol=3e-12)
    # Maxwell scaling: k0 -> k0 / s and radius -> s radius leave the T-matrix
    # unchanged and multiply cross sections by s**2. The T-matrix of a nearly
    # index-matched particle is a small difference of Mie terms of unit size and
    # keeps their absolute rounding (up to about 25 eps at the widest draws).
    scaled = sphere_tmatrix(
        k0=k0 / scale,
        lmax=lmax,
        radius=radius * scale,
        material=material,
        medium=medium,
        polarization=polarization,
    )
    tolerance = 1e-12 * abs(sphere.array) + 2e-14
    assert_allclose(scaled.array, sphere.array, rtol=1e-12, atol=2e-14)

    def cross_sections(particle, wavenumber):
        incident = tr.plane_wave(
            [0.3, -0.2, 1], "positive_helicity", k0=wavenumber, medium=medium
        )
        return particle.cross_sections(incident.with_polarization(polarization))

    expected = cross_sections(sphere, k0)
    actual = cross_sections(scaled, k0 / scale)
    # Extinction -Re(a^H T a) / k^2 is linear in T and, for weak scatterers, far
    # below |a^H T a| / k^2: the T-matrix tolerance reaches it through the
    # incident coefficients a, over the entries that either matrix couples.
    incident = tr.plane_wave([0.3, -0.2, 1], "positive_helicity", k0=k0, medium=medium)
    coefficients = abs(incident.with_polarization(polarization).expand(sphere.basis))
    coupled = (sphere.array != 0) | (scaled.array != 0)
    extinction = coefficients @ (tolerance * coupled) @ coefficients
    assert_allclose(
        actual.extinction,
        scale**2 * expected.extinction,
        rtol=1e-12,
        atol=(scale / abs(sphere.ks).min()) ** 2 * extinction,
    )
    # Scattering is quadratic in T; weak scatterers scatter far below the area.
    area = np.pi * (scale * radius) ** 2
    assert_allclose(
        actual.scattering, scale**2 * expected.scattering, rtol=1e-12, atol=1e-16 * area
    )


@st.composite
def clusters(draw, *, polarization, lossy):
    """Up to three non-overlapping spheres (radii <= 0.25, spacing >= 0.6)."""
    particles, positions = [], [np.zeros(3)]
    for _ in range(draw(st.integers(1, 3))):
        loss = draw(st.floats(0.01, 0.2)) if lossy else 0
        material = tr.Material(
            draw(st.floats(1.5, 5.0)) + 1j * loss, 1.0, draw(st.floats(-0.1, 0.1))
        )
        particles.append(
            sphere_tmatrix(
                k0=1.3,
                lmax=draw(st.integers(1, 2)),
                radius=draw(st.floats(0.1, 0.25)),
                material=material,
                polarization=polarization,
            )
        )
        step = [draw(st.floats(-0.3, 0.3)), draw(st.floats(-0.3, 0.3)), 0.0]
        positions.append(positions[-1] + step + [0, 0, draw(st.floats(0.6, 1.0))])
    return particles, np.array(positions[:-1])


def _relative(expected):
    return dict(rtol=0, atol=1e-13 * np.abs(expected).max())


@pytest.mark.physics
@pytest.mark.parametrize("polarization", ["helicity", "parity"])
@settings(max_examples=15)
@given(
    data=st.data(),
    angles=st.tuples(st.floats(-3, 3), st.floats(0, 3), st.floats(-3, 3)),
    shift=st.tuples(*(st.floats(-2, 2) for _ in range(3))),
)
def test_cluster_reciprocity_rigid_motions_and_power(polarization, data, angles, shift):
    particles, positions = data.draw(clusters(polarization=polarization, lossy=True))
    local = Cluster(particles, positions=positions).solve()
    # Lorentz reciprocity of the solved multi-origin response and its expansion.
    array = np.asarray(local.array)
    assert_allclose(
        array,
        reciprocal(array, local.basis.modes, cylindrical=False),
        **_relative(array),
    )
    domain = tr.SphericalBasis.default(3)
    expanded = local.expand(domain)
    assert_allclose(
        expanded.array,
        reciprocal(expanded.array, domain.modes, cylindrical=False),
        **_relative(expanded.array),
    )
    # Rotating the geometry rotates the response about the global origin.
    rotation = Rotation.from_euler("ZYZ", angles).as_matrix()
    rotated = Cluster(particles, positions=positions @ rotation.T).solve()
    assert_allclose(
        rotated.expand(domain).array,
        expanded.rotate(*angles).array,
        **_relative(expanded.array),
    )
    # Cross sections do not depend on where the cluster is or how it and the
    # illumination are turned together; lossy particles absorb.
    direction = np.array([0.3, -0.2, 1.0])

    def cross_sections(response, direction):
        incident = tr.plane_wave(direction, "positive_helicity", k0=1.3)
        return response.cross_sections(incident.with_polarization(polarization))

    expected = cross_sections(local, direction)
    moved = Cluster(particles, positions=positions + shift).solve()
    for actual in (
        cross_sections(moved, direction),
        cross_sections(rotated, rotation @ direction),
    ):
        assert_allclose(actual, expected, rtol=1e-12)
    assert expected.extinction - expected.scattering > 0


@pytest.mark.physics
@pytest.mark.parametrize("polarization", ["helicity", "parity"])
@given(
    kz=st.floats(0.05, 0.6),
    epsilon=st.floats(1.5, 5.0),
    loss=st.floats(0, 0.2),
    kappa=st.floats(-0.1, 0.1),
    offset=st.tuples(st.floats(0.6, 1.2), st.floats(-0.6, 0.6)),
)
def test_cylinder_cluster_reciprocity_pairs_opposite_axial_wavenumbers(
    polarization, kz, epsilon, loss, kappa, offset
):
    # Reciprocity maps (kz, m) to (-kz, -m), so both signs of kz are present.
    material = tr.Material(epsilon + 1j * loss, 1.0, kappa)
    particles = [
        tr.CylindricalTMatrix.cylinder(
            [kz, -kz], mmax, 1.3, radius, [material, 1], poltype=polarization
        )
        for mmax, radius in [(1, 0.2), (2, 0.15)]
    ]
    positions = [[0.0, 0.0, 0.0], [*offset, 0.0]]
    local = Cluster(particles, positions=positions).solve()
    array = np.asarray(local.array)
    assert_allclose(
        array,
        reciprocal(array, local.basis.modes, cylindrical=True),
        **_relative(array),
    )


@pytest.mark.physics
@settings(max_examples=15)
@given(data=st.data())
def test_lossless_cluster_optical_theorem_in_both_representations(data):
    particles, positions = data.draw(clusters(polarization="helicity", lossy=False))
    direction = data.draw(
        st.tuples(*(st.floats(-1, 1) for _ in range(3))).filter(
            lambda v: np.linalg.norm(v) > 0.1
        )
    )
    incident = tr.plane_wave(direction, "positive_helicity", k0=1.3)
    solved = Cluster(particles, positions=positions).solve()
    cross = solved.cross_sections(incident)
    # Extinction -Re(a^H T a) / k^2 cancels for weak lossless scatterers, so its
    # rounding is a few ulps of |a|^T |T| |a| / k^2 rather than of the result (at
    # most 1.6 ulps over 400 random clusters, including this property's examples).
    coefficients = abs(incident.expand(solved.basis))
    floor = 16 * np.finfo(float).eps * (coefficients @ abs(solved.array) @ coefficients)
    atol = floor / abs(solved.ks).min() ** 2
    assert_allclose(cross.scattering, cross.extinction, rtol=1e-12, atol=atol)
    parity = Cluster(
        [particle.with_polarization("parity") for particle in particles],
        positions=positions,
    ).solve()
    assert_allclose(
        parity.cross_sections(incident.with_polarization("parity")),
        cross,
        rtol=1e-12,
        atol=atol,
    )


@pytest.mark.physics
@given(
    period=st.floats(1.0, 0.95 * 2 * np.pi / 1.3),
    fraction=st.floats(0.1, 0.3),
    epsilon=st.floats(1.5, 5.0),
    lmax=st.integers(1, 3),
)
def test_lossless_sphere_array_conserves_power(period, fraction, epsilon, lmax):
    # Below the first diffraction order only the zero order carries power.
    sphere = sphere_tmatrix(
        k0=1.3, lmax=lmax, radius=fraction * period, material=epsilon
    )
    network = solve_periodic(sphere, lattice=np.eye(2) * period, kpar=[0, 0])
    network = network.to_smatrix(tr.PlaneWavePorts.default([0, 0]))
    for z in (1, -1):
        for helicity in ("positive_helicity", "negative_helicity"):
            power = network.power(tr.plane_wave([0, 0, z], helicity, k0=1.3))
            assert power.transmission >= 0
            assert power.reflection >= 0
            assert_allclose(power.transmission + power.reflection, 1, atol=1e-12)


# Constructor validation -------------------------------------------------------------

_SPHERICAL = tr.SphericalBasis.default(1)
_PORTS = tr.PlaneWavePorts.default([[0.1, 0.2]])


def _sphere(**options):
    return sphere_tmatrix(
        **({"k0": 1, "lmax": 1, "radius": 0.1, "material": 2} | options)
    )


CONTRACTS = {
    "tmatrix-not-square": (
        lambda: tr.TMatrix(np.zeros((6, 5)), basis=_SPHERICAL, k0=1),
        ValueError,
        "finite",
    ),
    "tmatrix-not-finite": (
        lambda: tr.TMatrix(np.full((6, 6), np.nan), basis=_SPHERICAL, k0=1),
        ValueError,
        "finite",
    ),
    "tmatrix-basis-size": (
        lambda: tr.TMatrix(np.zeros((4, 4)), basis=_SPHERICAL, k0=1),
        ValueError,
        "dimension",
    ),
    "tmatrix-k0": (
        lambda: tr.TMatrix(np.zeros((6, 6)), basis=_SPHERICAL, k0=0.0),
        ValueError,
        "positive",
    ),
    "tmatrix-plane-basis": (
        lambda: tr.TMatrix(np.zeros((2, 2)), basis=_PORTS, k0=1),
        ValueError,
        "wave family",
    ),
    "tmatrix-parity-in-chiral-medium": (
        lambda: tr.TMatrix(
            np.zeros((6, 6)),
            basis=_SPHERICAL,
            k0=1,
            material=(2, 1, 0.1),
            poltype="parity",
        ),
        ValueError,
        "polarization",
    ),
    "wave-no-columns": (
        lambda: tr.Wave(np.zeros((6, 0)), basis=_SPHERICAL, k0=1),
        ValueError,
        "shape",
    ),
    "wave-length": (
        lambda: tr.Wave(np.zeros(5), basis=_SPHERICAL, k0=1),
        ValueError,
        "shape",
    ),
    "wave-mode-type": (
        lambda: tr.Wave(np.zeros(6), basis=_SPHERICAL, k0=1, modetype="up"),
        ValueError,
        "mode type",
    ),
    "smatrix-multipole-basis": (
        lambda: tr.SMatrix(np.zeros((2, 2, 6, 6)), basis=_SPHERICAL, k0=1),
        TypeError,
        "component plane-wave",
    ),
    "smatrix-shape": (
        lambda: tr.SMatrix(np.zeros((2, 2, 3, 3)), basis=_PORTS, k0=1),
        ValueError,
        "shape",
    ),
    "smatrix-three-media": (
        lambda: tr.SMatrix(
            np.zeros((2, 2, 2, 2)), basis=_PORTS, k0=1, material=(1, 2, 3)
        ),
        ValueError,
        "media",
    ),
    "block-multipole-basis": (
        lambda: tr.ScatteringBlock(np.zeros((6, 6)), basis=_SPHERICAL, k0=1),
        TypeError,
        "component plane-wave",
    ),
    "array-metadata-per-axis": (
        lambda: tr.operators.PhysicsArray(
            np.zeros((6, 6)), basis=_SPHERICAL, material=(1, 2, 3)
        ),
        ValueError,
        "axis",
    ),
    "plane-polarization-index": (
        lambda: tr.plane_wave([0, 0, 1], 2, k0=1),
        ValueError,
        "0 or 1",
    ),
    "plane-polarization-name": (
        lambda: tr.plane_wave([0, 0, 1], "circular", k0=1),
        ValueError,
        "helicity",
    ),
    "plane-zero-direction": (
        lambda: tr.plane_wave([0, 0, 0], 1, k0=1),
        ValueError,
        "nonzero",
    ),
    "cluster-positions": (
        lambda: Cluster([_sphere()], positions=[[0, 0]]),
        ValueError,
        "position",
    ),
    "cluster-mixed-k0": (
        lambda: Cluster(
            [_sphere(), _sphere(k0=1.1)], positions=[[0, 0, 0], [1, 0, 0]]
        ).solve(),
        ValueError,
        "matching k0",
    ),
    "slab-thickness": (
        lambda: slab(basis=_PORTS, k0=1, thickness=-0.1, material=2),
        ValueError,
        "thickness",
    ),
    "sphere-radius": (lambda: _sphere(radius=-0.1), ValueError, "increasing"),
    "sphere-lmax": (lambda: _sphere(lmax=0), ValueError, "lmax"),
    "polarization-name": (
        lambda: _sphere(polarization="linear"),
        ValueError,
        "helicity or parity",
    ),
}


@pytest.mark.interface
@pytest.mark.parametrize(
    "construct,error,match",
    [pytest.param(*case, id=name) for name, case in CONTRACTS.items()],
)
def test_constructor_contract(construct, error, match):
    with pytest.raises(error, match=match):
        construct()


_OUTGOING = (
    "kind must be 'regular' or 'singular'; outgoing multipole waves are 'singular'"
)


@pytest.mark.interface
def test_ambiguous_kind_and_repeated_treams_aliases_raise():
    """'outgoing' is no kind, and a keyword given with its treams alias raises."""
    coefficients = np.ones(len(_SPHERICAL))
    for construct in (
        lambda: tr.Wave(coefficients, basis=_SPHERICAL, k0=1, kind="outgoing"),
        lambda: tr.spherical_wave(1, 0, 1, k0=1, kind="outgoing"),
        lambda: tr.cylindrical_wave(0.1, 0, 1, k0=1, kind="outgoing"),
        lambda: tr.Wave(coefficients, basis=_SPHERICAL, k0=1).in_basis(
            _SPHERICAL, kind="outgoing"
        ),
    ):
        with pytest.raises(ValueError, match=_OUTGOING):
            construct()
    repeated = {
        "medium or material": dict(medium=1.5, material=1.5),
        "kind or modetype": dict(kind="regular", modetype="regular"),
        "polarization or poltype": dict(polarization="helicity", poltype="helicity"),
    }
    constructors = (
        (tr.Wave, (coefficients,), dict(basis=_SPHERICAL, k0=1)),
        (tr.spherical_wave, (1, 0, 1), dict(k0=1)),
        (tr.cylindrical_wave, (0.1, 0, 1), dict(k0=1)),
    )
    for names, options in repeated.items():
        for function, args, required in constructors:
            with pytest.raises(ValueError, match=f"specify {names}, not both"):
                function(*args, **required, **options)
    for names, options in (
        ("medium or material", dict(medium=1.5, material=1.5)),
        ("polarization or poltype", dict(polarization="parity", poltype="parity")),
    ):
        with pytest.raises(ValueError, match=f"specify {names}, not both"):
            tr.plane_wave([0, 0, 1], 1, k0=1, **options)
        with pytest.raises(ValueError, match=f"specify {names}, not both"):
            tr.plane_wave_angle(0.1, 0.2, 1, k0=1, **options)
    with pytest.raises(ValueError, match="specify direction or kvec, not both"):
        tr.plane_wave([0, 0, 1], 1, k0=1, direction=[0, 0, 1])
    # polarization names the convention; the plane-wave state is pol.
    with pytest.raises(ValueError, match="give the plane-wave state as pol"):
        tr.plane_wave(direction=[0, 0, 1], polarization=1, k0=1)
    with pytest.raises(ValueError, match="give the plane-wave state as pol"):
        tr.plane_wave(direction=[0, 0, 1], polarization=np.array([0, 1]), k0=1)
    with pytest.raises(TypeError, match="k0"):
        tr.plane_wave([0, 0, 1], 1)


@pytest.mark.interface
@settings(max_examples=20, deadline=None)
@given(
    epsilon=st.floats(1.0, 4.0),
    singular=st.booleans(),
    parity=st.booleans(),
    pol=st.sampled_from([0, 1]),
    theta=st.floats(0.0, 3.0),
)
def test_treams_aliases_mean_the_physics_keywords(
    epsilon, singular, parity, pol, theta
):
    """medium, kind, polarization and direction equal material, modetype, poltype, kvec."""
    convention = "parity" if parity else "helicity"
    kind = "singular" if singular else "regular"
    points = [[0.4, 0.3, 0.5]]
    physics = tr.spherical_wave(
        2, 1, pol, k0=1.3, medium=epsilon, kind=kind, polarization=convention
    )
    treams = tr.spherical_wave(
        2, 1, pol, k0=1.3, material=epsilon, modetype=kind, poltype=convention
    )
    assert (physics.medium, physics.kind, physics.polarization) == (
        treams.material,
        treams.modetype,
        treams.poltype,
    )
    assert_allclose(physics.efield(points), treams.efield(points), rtol=0, atol=0)
    direction = [np.sin(theta), 0.0, np.cos(theta)]
    plane = tr.plane_wave(
        direction=direction, pol=pol, k0=1.3, medium=epsilon, polarization=convention
    )
    for alias in (
        tr.plane_wave(direction, pol, k0=1.3, material=epsilon, poltype=convention),
        tr.plane_wave_angle(
            theta, 0.0, pol, k0=1.3, medium=epsilon, polarization=convention
        ),
    ):
        assert alias.poltype == plane.polarization
        assert_allclose(alias.coefficients, plane.coefficients, rtol=0, atol=0)
        assert_allclose(alias.kvecs, plane.kvecs, rtol=0, atol=1e-15)


@pytest.mark.interface
@pytest.mark.parametrize("family", ["sphere", "cylinder"])
def test_multipole_responses_reject_plane_permutations(family):
    response = (
        tr.sphere_tmatrix(k0=1.3, lmax=1, radius=0.2, material=3)
        if family == "sphere"
        else tr.cylinder_tmatrix(k0=1.3, kz=0, mmax=1, radius=0.2, material=3)
    )
    with pytest.raises(TypeError, match=r"plane-wave bases; use SMatrix\.permute"):
        response.permute(1)
    ports = tr.PlaneWavePorts.default([0.1, 0.2])
    network = tr.slab(basis=ports, k0=1.3, thickness=0.2, material=2)
    with pytest.raises(ValueError, match="at least one side"):
        network.scatter()


_LOSSLESS_CLUSTER = Cluster(
    [sphere_tmatrix(k0=1.3, lmax=2, radius=r, material=3) for r in (0.2, 0.25)],
    positions=[[0, 0, 0], [0.5, 0.3, 0.6]],
).solve()


@pytest.mark.physics
@settings(max_examples=15)
@given(
    theta=st.floats(0, np.pi),
    phi=st.floats(-np.pi, np.pi),
    pol=st.sampled_from(["positive_helicity", "negative_helicity"]),
)
def test_lossless_cluster_cross_sections_conserve_power(theta, phi, pol):
    # A lossless multi-origin response radiates what it extinguishes. Repeated
    # calls reuse the cached radiation overlap and match a fresh response.
    direction = [
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ]
    wave = tr.plane_wave(direction, pol, k0=1.3)
    cross = _LOSSLESS_CLUSTER.cross_sections(wave)
    assert cross.scattering > 0
    assert_allclose(cross.absorption, 0, atol=1e-12 * cross.extinction)
    fresh = tr.TMatrix(
        _LOSSLESS_CLUSTER.array, k0=1.3, basis=_LOSSLESS_CLUSTER.basis
    ).cross_sections(wave)
    assert_allclose(fresh, cross, rtol=1e-14, atol=0)


@pytest.mark.interface
def test_periodic_conversions_validate_ports_before_solving():
    """Periodic conversions reject ports off the diffraction orders, and the
    cylindrical conversion rejects a cylindrical unit cell."""
    sphere = sphere_tmatrix(k0=1.3, lmax=1, radius=0.2, material=3)
    off_order = tr.PlaneWavePorts.default([0.4, 0])
    with pytest.raises(ValueError, match="diffraction orders"):
        solve_periodic(sphere, lattice=np.eye(2) * 2, kpar=[0, 0]).to_smatrix(off_order)
    off_axis = tr.CylindricalBasis.default([0.4], 1)
    with pytest.raises(ValueError, match="diffraction orders"):
        solve_periodic(sphere, lattice=2.0, kpar=0.0).to_cylindrical(off_axis)
    cylinder = cylinder_tmatrix(k0=1.3, kz=0, mmax=1, radius=0.2, material=3)
    with pytest.raises(ValueError, match="spherical unit cell"):
        solve_periodic(cylinder, lattice=2.0, kpar=0.0).to_cylindrical(
            tr.CylindricalBasis.default([0], 1)
        )


@pytest.mark.interface
@pytest.mark.parametrize("framework", ["advect", "jax", "torch"])
def test_framework_results_are_the_root_classes(framework):
    pytest.importorskip(framework)
    namespace = importlib.import_module(f"treams_rs.{framework}")
    for name in ("CrossSections", "PowerBalance", "BandModes", "ScatteredPorts"):
        assert getattr(namespace, name) is getattr(tr, name)


@pytest.mark.interface
@settings(max_examples=20, deadline=None)
@given(
    epsilon=st.floats(1.0, 4.0),
    other=st.floats(1.0, 4.0),
    parity=st.booleans(),
    data=st.lists(
        st.complex_numbers(max_magnitude=1.0, allow_nan=False), min_size=4, max_size=4
    ),
)
def test_physics_constructors_take_medium_and_polarization(
    epsilon, other, parity, data
):
    """medium, polarization and the S-matrix sides equal material and poltype."""
    convention = "parity" if parity else "helicity"
    spherical = np.diag(np.resize(data, len(_SPHERICAL)))
    cylindrical_basis = tr.CylindricalBasis.default([0.1], 1)
    cylindrical = np.diag(np.resize(data, len(cylindrical_basis)))
    for cls, array, basis in (
        (tr.TMatrix, spherical, _SPHERICAL),
        (tr.CylindricalTMatrix, cylindrical, cylindrical_basis),
    ):
        physics = cls(
            array, k0=1.3, basis=basis, medium=epsilon, polarization=convention
        )
        treams = cls(array, k0=1.3, basis=basis, material=epsilon, poltype=convention)
        assert (physics.medium, physics.polarization) == (
            treams.medium,
            treams.polarization,
        )
        assert_array_equal(physics.array, treams.array)
        for names, options in (
            ("medium or material", dict(medium=epsilon, material=epsilon)),
            (
                "polarization or poltype",
                dict(polarization=convention, poltype=convention),
            ),
        ):
            with pytest.raises(ValueError, match=f"specify {names}, not both"):
                cls(array, k0=1.3, basis=basis, **options)
    physics = tr.PlaneWave(
        [0, 0, 1.3], [1, 0], k0=1.3, medium=1, polarization=convention
    )
    treams = tr.PlaneWave([0, 0, 1.3], [1, 0], k0=1.3, material=1, poltype=convention)
    assert physics.polarization == treams.polarization == convention
    assert_array_equal(physics.coefficients, treams.coefficients)
    blocks = np.zeros((2, 2, len(_PORTS), len(_PORTS)), complex)
    blocks[0, 0] = blocks[1, 1] = np.diag(np.resize(data, len(_PORTS)))
    sides = tr.SMatrix(
        blocks,
        k0=1.3,
        basis=_PORTS,
        positive_medium=epsilon,
        negative_medium=other,
        polarization="helicity",
    )
    pair = tr.SMatrix(
        blocks, k0=1.3, basis=_PORTS, material=(epsilon, other), poltype="helicity"
    )
    assert (sides.positive_medium, sides.negative_medium) == (
        pair.positive_medium,
        pair.negative_medium,
    )
    assert sides.positive_medium.epsilon == epsilon
    assert sides.negative_medium.epsilon == other
    assert_array_equal(sides.array, pair.array)
    with pytest.raises(ValueError, match="or material, not both"):
        tr.SMatrix(blocks, k0=1.3, basis=_PORTS, positive_medium=1, material=1)


@pytest.mark.interface
def test_diff_choice_keywords_and_their_aliases_agree():
    """metric, function and pol equal the kind and polarization aliases."""
    points = np.array([[0.4, 0.3, 0.5], [-0.2, 0.7, 0.1]])
    vectors = np.array([[1.0, 0.5j, 0.2], [0.3, -1.0, 0.4j]])
    operator = np.diag(np.linspace(0.2, 0.8, 6)).astype(complex)
    operator[0, 3] = operator[3, 0] = 0.1j
    polarizations = np.array([1, 0, 1, 0, 1, 0])
    pairs = (
        (
            lambda **k: tr.diff.tmatrix_metric(
                operator, polarizations=polarizations, **k
            )[0],
            "metric",
            "kind",
            "cd",
        ),
        (
            lambda **k: tr.diff.coordinates(points, **k)[0],
            "function",
            "kind",
            "car2sph",
        ),
        (
            lambda **k: tr.diff.vector_coordinates(vectors, points, **k)[0],
            "function",
            "kind",
            "car2sph",
        ),
        (
            lambda **k: tr.diff.vector_wave(
                1.2, 0.4, 0.3, function="vsw_rA", degree=1, order=0, **k
            )[0],
            "pol",
            "polarization",
            1,
        ),
    )
    for call, name, alias, value in pairs:
        assert_array_equal(call(**{name: value}), call(**{alias: value}))
        with pytest.raises(ValueError, match=f"specify {name} or {alias}, not both"):
            call(**{name: value, alias: value})
    for call, name, _, _ in pairs[:3]:
        with pytest.raises(TypeError, match=name):
            call()


@pytest.mark.interface
def test_advect_and_periodic_keywords_and_their_aliases_agree():
    """advect metric/function/pol and to_smatrix diffraction_orders equal the aliases."""
    points = np.array([[0.4, 0.3, 0.5], [-0.2, 0.7, 0.1]])
    vectors = np.array([[1.0, 0.5j, 0.2], [0.3, -1.0, 0.4j]])
    operator = np.diag(np.linspace(0.2, 0.8, 6)).astype(complex)
    operator[0, 3] = operator[3, 0] = 0.1j
    polarizations = np.array([1, 0, 1, 0, 1, 0])
    pairs = (
        (
            lambda **k: ad.tmatrix_metric(operator, polarizations=polarizations, **k),
            "metric",
            "kind",
            "db",
        ),
        (lambda **k: ad.coordinates(points, **k), "function", "kind", "car2cyl"),
        (
            lambda **k: ad.vector_coordinates(vectors, points, **k),
            "function",
            "kind",
            "car2cyl",
        ),
        (
            lambda **k: ad.vector_wave(
                1.2, 0.4, 0.3, function="vsw_rN", degree=2, order=1, **k
            ),
            "pol",
            "polarization",
            1,
        ),
    )
    for call, name, alias, value in pairs:
        assert_array_equal(
            np.asarray(call(**{name: value})), np.asarray(call(**{alias: value}))
        )
        with pytest.raises(ValueError, match=f"specify {name} or {alias}, not both"):
            call(**{name: value, alias: value})
    assert ad.smatrix_cd is ad.smatrix_circular_dichroism
    # The framework PeriodicResponse takes diffraction orders.
    tm = ad.sphere_tmatrix(k0=1.2, lmax=1, radius=0.2, material=3 + 0.1j)
    response = ad.solve_periodic(tm, lattice=[[2.0, 0.0], [0.0, 2.0]], kpar=[0.1, 0.05])
    named = response.to_smatrix(diffraction_orders=[[0, 0], [1, 0]])
    alias = response.to_smatrix(orders=[[0, 0], [1, 0]])
    assert_array_equal(np.asarray(named.array), np.asarray(alias.array))
    with pytest.raises(
        ValueError, match="specify diffraction_orders or orders, not both"
    ):
        response.to_smatrix(diffraction_orders=[[0, 0]], orders=[[0, 0]])
