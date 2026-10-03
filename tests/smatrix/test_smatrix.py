"""Planar S-matrices: slabs, stacks, interfaces and circular dichroism against upstream,
reciprocity, Poynting power and boundary continuity, and native and Advect pullbacks."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import (
    Material,
    PlaneWavePorts,
    SMatrix,
    coeffs,
    diff,
    plane_wave,
    poynting_avg_z,
)
from treams_rs import advect as ad
from treams_rs.operators import efield, hfield
from treams_rs.testing import check_gradient, check_pullback

from _support import (
    assert_unitary_ports,
    complex_arrays,
    complex_normal,
    oracle_smatrix_array,
    selecting,
)


@pytest.mark.interface
@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("direction", ["up", "down"])
def test_cd_reference_and_reordered_basis(poltype, direction):
    basis = PlaneWavePorts.default([[0.2, 0.1], [-0.3, 0.5]])
    ob = treams.PlaneWaveBasisByComp(basis.modes)
    materials = [1, (3 + 0.2j, 1.2 + 0.05j, 0.08j), 1]
    # A parity port basis remains valid with a chiral interior and achiral ports.
    value = SMatrix.slab(0.4, basis, 1.7, materials)
    oracle = treams.SMatrices.slab(0.4, ob, 1.7, materials)
    if poltype == "parity":
        value = value.changepoltype(poltype)
        oracle = oracle.changepoltype(poltype)
    incident = np.array([0.2 + 0.1j, 0.7, 0.3j, -0.2])
    expected = oracle.cd(treams.PhysicsArray(incident, modetype=direction))
    assert_allclose(value.cd(incident, modetype=direction), expected, atol=1e-12)
    order = [3, 0, 2, 1]
    reordered = SMatrix(
        value.array[:, :, order][:, :, :, order],
        k0=1.7,
        basis=PlaneWavePorts([basis.modes[i] for i in order]),
        poltype=poltype,
    )
    assert_allclose(
        reordered.cd(incident[order], modetype=direction), expected, atol=1e-12
    )


@pytest.mark.physics
@given(
    kappa=st.floats(-0.15, 0.15),
    thickness=st.floats(0.1, 1.0),
    scale=st.floats(0.2, 2.0),
)
def test_cd_helicity_swap_scaling_and_reflection_cd_vanishes(kappa, thickness, scale):
    basis = PlaneWavePorts.default([[0.2, 0.1]])
    layer = SMatrix.slab(thickness, basis, 1.7, [1, (3, 1.2, kappa), 1])
    positive, negative = np.array([1, 0]), np.array([0, 1])
    value = layer.cd(positive)
    assert_allclose(layer.cd(negative), -np.array(value), atol=1e-13)
    assert_allclose(layer.cd(positive * scale * (1 + 0.3j)), value, atol=1e-13)
    assert_allclose(value[1], 0, atol=1e-13)


@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("q", [[[0, 0]], [[0.2, 0.1], [2.8, 0.3], [-0.3, 0.5]]])
@pytest.mark.parametrize(
    "materials",
    [[1, 2.5, 1.3], [1, (3 + 0.1j, 1.2), 1], [(1.2, 1.1), (2.3, 0.9), (1.4, 0.8)]],
)
def test_slab_reference(poltype, q, materials):
    basis = PlaneWavePorts.default(q)
    oracle_basis = treams.PlaneWaveBasisByComp.default(q)
    value = SMatrix.slab(0.4, basis, 1.7, materials, poltype)
    # treams 0.4.5's parity interface uses a 2x2 mask for every basis size.
    expected = treams.SMatrices.slab(0.4, oracle_basis, 1.7, materials, "helicity")
    if poltype == "parity":
        expected = expected.changepoltype("parity")
    assert_allclose(value.array, oracle_smatrix_array(expected), rtol=2e-13, atol=2e-13)
    incident = np.zeros(len(basis), complex)
    incident[0] = 1
    assert_allclose(value.tr(incident), expected.tr(incident), rtol=2e-13, atol=2e-13)
    for mode in ("up", "down"):
        actual_vectors = basis.kvecs(1.7, Material(materials[1]), mode)
        expected_vectors = oracle_basis.kvecs(1.7, treams.Material(materials[1]), mode)
        assert_allclose(actual_vectors, expected_vectors, rtol=2e-14, atol=2e-14)


@pytest.mark.physics
@pytest.mark.reference
@given(
    alignment=st.sampled_from(["xy", "yz", "zx"]),
    epsilon=st.floats(1.2, 4),
    mu=st.floats(0.7, 1.3),
    chirality=st.floats(-0.2, 0.2),
    thickness=st.floats(0, 3),
    q=st.floats(0, 0.5),
)
def test_lossless_chiral_layer_flux_and_split(
    alignment, epsilon, mu, chirality, thickness, q
):
    components = [[q, 0.2], [-0.4, q]]
    basis = PlaneWavePorts.default(components, alignment)
    materials = [1, (epsilon, mu, chirality), 1]
    layer = SMatrix.slab(thickness, basis, 1.8, materials)
    split = SMatrix.slab(
        [thickness * 0.3, thickness * 0.7],
        basis,
        1.8,
        [1, materials[1], materials[1], 1],
    )
    assert_allclose(layer.array, split.array, rtol=2e-13, atol=2e-13)
    assert_unitary_ports(layer.array, atol=2e-13)
    incident = np.array([0.2 + 0.7j, 0.5, -0.1j, 0.3])
    for direction in ("up", "down"):
        assert_allclose(
            sum(layer.tr(incident, modetype=direction)), 1, rtol=2e-13, atol=2e-13
        )
    if alignment != "xy":
        return  # upstream treams builds slabs in the xy alignment only
    ob = treams.PlaneWaveBasisByComp.default(components)
    oracle = treams.SMatrices.slab(thickness, ob, 1.8, materials)
    assert_allclose(layer.array, oracle_smatrix_array(oracle), rtol=3e-13, atol=3e-13)
    for actual, expected in zip(
        poynting_avg_z(basis, 1.8, Material(materials[1])),
        treams.poynting_avg_z(ob, 1.8, treams.Material(materials[1])),
        strict=True,
    ):
        assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.physics
@settings(max_examples=20)
@given(
    q=st.tuples(st.floats(-0.6, 0.6), st.floats(-0.6, 0.6)),
    kappa=st.tuples(st.floats(-0.4, 0.4), st.floats(-0.4, 0.4)),
    thickness=st.tuples(st.floats(0.05, 1.0), st.floats(0.05, 1.0)),
    loss=st.floats(0, 0.3),
)
def test_transmittance_is_reciprocal_in_lossy_stacks(q, kappa, thickness, loss):
    # Reciprocity maps incidence from below at (q, helicity) onto incidence
    # from above at (-q, opposite helicity); a half turn about z, which leaves
    # isotropic layers unchanged, maps -q back to q. So the unpolarized
    # transmittance is the same from both sides despite loss, chirality and
    # different outer media. Achiral stacks keep it per parity polarization.
    # Reflectance generally differs between the sides.
    ports = PlaneWavePorts.default([q])
    layers = [(2.3 + loss * 1j, 1.1, kappa[0]), (3.1 + 0.1j, 0.9, kappa[1])]
    chiral = SMatrix.slab(list(thickness), ports, 1.6, [1.2, *layers, 1.7])
    up, down = (
        sum(chiral.tr(e, modetype=side)[0] for e in np.eye(2))
        for side in ("up", "down")
    )
    assert_allclose(up, down, rtol=1e-12, atol=1e-13)
    achiral = SMatrix.slab(
        list(thickness),
        ports,
        1.6,
        [1.2, *(layer[:2] for layer in layers), 1.7],
        "parity",
    )
    for e in np.eye(2):
        assert_allclose(
            achiral.tr(e, modetype="up")[0],
            achiral.tr(e, modetype="down")[0],
            rtol=1e-12,
            atol=1e-13,
        )


@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_stack_basis_order_and_propagation(poltype):
    modes = [(0.2, 0.1, 0), (2.8, 0.3, 1), (0.2, 0.1, 1), (2.8, 0.3, 0)]
    basis, ob = PlaneWavePorts(modes), treams.PlaneWaveBasisByComp(modes)
    value = SMatrix.propagation([0.1, -0.2, 0.3], basis, 1.8, 2, poltype)
    expected = treams.SMatrices.propagation([0.1, -0.2, 0.3], ob, 1.8, 2, poltype)
    assert_allclose(value.array, oracle_smatrix_array(expected), rtol=1e-14, atol=1e-14)
    direct = SMatrix.propagation([0.4, -0.8, 1.2], basis, 1.8, 2, poltype)
    assert_allclose(value.double(2).array, direct.array, rtol=1e-13, atol=1e-13)
    assert_allclose(
        SMatrix.stack([value] * 4).array, direct.array, rtol=1e-13, atol=1e-13
    )


@pytest.mark.gradients
@settings(max_examples=20)
@given(size=st.integers(1, 5), data=st.data())
def test_stack_native_pullback_and_identity(size, data):
    shape = (2, 2, size, size)
    # Entries of at most 0.1 bound |R1 R2| by 0.25, so the Redheffer inverse
    # (I - R1 R2)^-1 stays well conditioned and shrinking simplifies the blocks.
    # Independent entries keep the blocks asymmetric, so transposes show.
    blocks = complex_arrays(shape, max_magnitude=0.1, fill=st.nothing())
    values = [data.draw(blocks) for _ in range(2)]
    identity = np.zeros(shape, complex)
    identity[0, 0] = identity[1, 1] = np.eye(size)
    assert_allclose(diff.smatrix_add(values[0], identity)[0], values[0], atol=1e-14)
    assert_allclose(diff.smatrix_add(identity, values[0])[0], values[0], atol=1e-14)
    rng = np.random.default_rng(size)
    g = complex_normal(rng, shape)
    check_pullback(
        diff.smatrix_add,
        *values,
        directions=tuple(complex_normal(rng, shape) for _ in values),
        cotangents=g,
        step=1e-6,
        rtol=1e-7,
        atol=1e-8,
    )


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize("operation", ["fresnel", "propagation"])
def test_coefficient_pullbacks_all_complex_inputs(operation):
    rng = np.random.default_rng(18)
    if operation == "fresnel":
        values = [
            np.array([[1.2, 1.4], [1.9, 2.2]]) + 0.04j,
            np.array([[1.1, 1.3], [1.8, 2.1]]) + 0.09j,
            np.array([0.9, 0.7]) + 0.03j,
        ]
        function = diff.fresnel
        assert_allclose(
            function(*values)[0], treams.coeffs.fresnel(*values), rtol=2e-14, atol=2e-14
        )
    else:
        values = [
            np.array([[0.1, 0.2, 1.1 + 0.1j], [-0.2, 0.3, 0.8j]]),
            np.array([0.1, 0.2, 0.4]),
        ]
        function = diff.propagation_matrix
    result = function(*values)[0]
    g = complex_normal(rng, result.shape)
    check_pullback(
        function,
        *values,
        directions=tuple(
            complex_normal(rng, v.shape)
            if np.iscomplexobj(v)
            else rng.normal(size=v.shape)
            for v in values
        ),
        cotangents=g,
        step=1e-6,
        rtol=1e-7,
        atol=1e-8,
    )


@pytest.mark.gradients
@settings(max_examples=15)
@given(
    thickness=st.floats(0.1, 1.0), epsilon=st.floats(1.3, 4.0), q=st.floats(0.0, 0.5)
)
def test_advect_complete_slab(thickness, epsilon, q):
    def objective(d, eps, q):
        ks = 1.8 * anp.sqrt(anp.array([1.0 + 0j, eps, 1.0 + 0j]))
        kz = anp.sqrt(ks**2 - q**2)
        z = 1.8 / ks
        left = ad.fresnel(
            anp.stack([anp.repeat(ks[0], 2), anp.repeat(ks[1], 2)]),
            anp.stack([anp.repeat(kz[0], 2), anp.repeat(kz[1], 2)]),
            z[:2],
        )
        right = ad.fresnel(
            anp.stack([anp.repeat(ks[1], 2), anp.repeat(ks[2], 2)]),
            anp.stack([anp.repeat(kz[1], 2), anp.repeat(kz[2], 2)]),
            z[1:],
        )
        wave = anp.array([q, 0j, kz[1]])
        propagation = ad.propagation_matrix(
            anp.stack([wave, wave]), anp.array([0.0, 0.0, d])
        )
        result = ad.smatrix_add(ad.smatrix_add(left, propagation), right)
        return anp.sum(anp.real(result * anp.conj(result))) + 0.1 * anp.sum(
            anp.real(result)
        )

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2)),
        np.array(thickness),
        np.array(epsilon + 0.07j),
        np.array(q),
        directions=(np.array(0.1), np.array(0.2 + 0.03j), np.array(-0.1)),
        step=1e-5,
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.physics
@settings(max_examples=30)
@given(
    q=st.tuples(st.floats(-2.5, 2.5), st.floats(-2.5, 2.5)),
    epsilon=st.tuples(st.floats(-0.3, 0.3), st.floats(0, 0.3)),
    mu=st.floats(0.7, 1.4),
    kappa=st.floats(-0.2, 0.2),
    poltype=st.sampled_from(["helicity", "parity"]),
)
def test_fresnel_interfaces_match_the_native_layer_kernel(
    q, epsilon, mu, kappa, poltype
):
    # xy interfaces use the closed-form Fresnel blocks; a layer stack without
    # interior layers solves the same boundary in the native planar kernel. For
    # passive media both agree for propagating and evanescent wavevectors.
    basis = PlaneWavePorts.default([q, [0.0, 0.0], [q[1], -q[0]]])[::-1]
    lower = Material(1.4, 1.1, 0.05 if poltype == "helicity" else 0)
    upper = Material(
        complex(2.3 + epsilon[0], epsilon[1]),
        mu,
        kappa if poltype == "helicity" else 0,
    )
    interface = SMatrix.interface(basis, 1.3, [lower, upper], poltype)
    layers = SMatrix.slab([], basis, 1.3, [lower, upper], poltype)
    assert_allclose(interface.array, layers.array, rtol=1e-13, atol=1e-13)


@pytest.mark.physics
@settings(max_examples=20)
@given(
    alignment=st.sampled_from(["xy", "yz", "zx"]),
    poltype=st.sampled_from(["helicity", "parity"]),
    shifts=st.lists(st.floats(-0.5, 0.5), min_size=6, max_size=6),
    turns=st.tuples(st.integers(-4, 4), st.integers(-4, 4)),
)
def test_coordinate_changes_compose_and_match_propagation(
    alignment, poltype, shifts, turns
):
    # Translations form a group, shifting along the normal equals propagating
    # the exterior ports back and forth around the slab, and cyclic
    # permutations compose like their turn counts.
    basis = PlaneWavePorts.default([[0.2, 0.3], [-0.4, 0.1], [2.1, 0.3]], alignment)
    kappa = 0.05 if poltype == "helicity" else 0
    layer = SMatrix.slab(0.4, basis, 1.3, [1.2, (2.3, 1.1, kappa), 1.2], poltype)
    first, second = np.array(shifts[:3]), np.array(shifts[3:])
    assert_allclose(
        layer.translate(first).translate(second).array,
        layer.translate(first + second).array,
        rtol=1e-13,
        atol=1e-13,
    )
    d = shifts[0]
    normal = np.eye(3)[basis.normal_axis] * d
    around = SMatrix.stack(
        [
            SMatrix.propagation(-d, basis, 1.3, 1.2, poltype),
            layer,
            SMatrix.propagation(d, basis, 1.3, 1.2, poltype),
        ]
    )
    assert_allclose(layer.translate(normal).array, around.array, atol=1e-13)
    composed = layer.permute(turns[0]).permute(turns[1])
    direct = layer.permute(sum(turns))
    assert composed.basis == direct.basis
    assert_allclose(composed.array, direct.array, rtol=1e-13, atol=1e-13)


@pytest.mark.interface
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_empty_port_basis_gives_empty_layers(poltype):
    empty = PlaneWavePorts([])
    for network in (
        SMatrix.interface(empty, 1.3, [1, 2], poltype),
        SMatrix.slab([0.2, 0.3], empty, 1.3, [1, 2, 3, 1], poltype),
    ):
        assert network.array.shape == (2, 2, 0, 0)
        assert network.basis == empty


@pytest.mark.interface
def test_invalid_interfaces_and_metadata():
    basis = PlaneWavePorts.default([0.1, 0.2])
    lower = SMatrix.interface(basis, 1.5, [1, 2])
    with pytest.raises(ValueError, match="matching k0, medium"):
        lower.add(lower)
    with pytest.raises(ValueError, match="thickness"):
        SMatrix.slab(-1, basis, 1.5, [1, 2, 1])
    with pytest.raises(ValueError, match="polarization"):
        SMatrix.interface(basis, 1.5, [1, (2, 1, 0.1)], "parity")


@pytest.mark.workflows
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_plane_wave_slab_illumination(direction, poltype):
    basis = PlaneWavePorts.default([0.1, 0.2])
    slab = SMatrix.slab(0.4, basis, 1.7, [1, 2.3, 1], poltype)
    kz = direction * np.sqrt(1.7**2 - 0.1**2 - 0.2**2)
    wave = plane_wave([0.1, 0.2, kz], [0.3 + 0.2j, 0.7], k0=1.7, poltype=poltype)
    array = np.array([0.7, 0.3 + 0.2j])
    mode = "up" if direction > 0 else "down"
    assert_allclose(slab.tr(wave), slab.tr(array, modetype=mode), rtol=1e-13)
    assert_allclose(
        slab.illuminate(wave), slab.illuminate(array, modetype=mode), rtol=1e-13
    )


@pytest.mark.physics
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("material", [1, (2 + 0.2j, 1.1)])
def test_oriented_power_is_cartesian_poynting(alignment, poltype, material):
    basis = PlaneWavePorts.default([[0.2, 0.3]], alignment)
    incident = np.array([0.3 + 0.2j, -0.4j])
    reflected = np.array([0.1j, 0.7])
    electric = np.stack(
        [
            efield(
                [0, 0, 0],
                basis=basis,
                k0=1.3,
                material=material,
                poltype=poltype,
                modetype=side,
            )
            for side in ("up", "down")
        ]
    )
    magnetic = np.stack(
        [
            hfield(
                [0, 0, 0],
                basis=basis,
                k0=1.3,
                material=material,
                poltype=poltype,
                modetype=side,
            )
            for side in ("up", "down")
        ]
    )

    def flux(amplitudes):
        e = sum(electric[i] @ amplitudes[i] for i in range(2))
        h = sum(magnetic[i] @ amplitudes[i] for i in range(2))
        return 0.5 * np.cross(e, h.conj()).real[basis.normal_axis]

    array = np.zeros((2, 2, 2, 2), complex)
    array[0, 0] = array[1, 1] = np.eye(2)
    array[0, 1] = array[1, 0] = np.outer(reflected, incident.conj()) / np.vdot(
        incident, incident
    )
    stack = SMatrix(
        array, basis=basis, k0=1.3, material=(material, material), poltype=poltype
    )
    for t, side in enumerate(("up", "down")):
        source = np.zeros((2, 2), complex)
        reflection = source.copy()
        source[t] = incident
        reflection[1 - t] = reflected
        sign = 1 if t == 0 else -1
        incoming_flux = sign * (flux(source + reflection) - flux(reflection))
        expected = [
            sign * flux(source) / incoming_flux,
            -sign * flux(reflection) / incoming_flux,
        ]
        assert_allclose(
            stack.tr(incident, modetype=side), expected, rtol=2e-13, atol=2e-13
        )


@pytest.mark.physics
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
def test_oriented_propagation_phase_and_power(alignment):
    basis = PlaneWavePorts.default([[0.2, 0.3]], alignment)
    layer = SMatrix.propagation(0.4, basis, 1.3)
    expected = np.exp(0.4j * np.sqrt(1.3**2 - 0.2**2 - 0.3**2)) * np.eye(2)
    assert_allclose(layer[0, 0], expected, atol=1e-14)
    assert_allclose(layer[1, 1], expected, atol=1e-14)
    for side in ("up", "down"):
        assert_allclose(layer.tr([0.3 + 0.1j, 0.4], modetype=side), [1, 0], atol=1e-14)
    assert_allclose(
        layer.double().array, SMatrix.propagation(0.8, basis, 1.3).array, atol=1e-14
    )


@pytest.mark.physics
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("q", [[0, 0], [0.2, 0.3], [2.8, -0.1]])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_interface_cartesian_boundary_continuity(alignment, q, poltype):
    basis = PlaneWavePorts.default([q], alignment)
    materials = [
        Material(1.2 + 0.1j, 1.1, 0.07 if poltype == "helicity" else 0),
        Material(2.8 + 0.2j, 0.9),
    ]
    interface = SMatrix.interface(basis, 1.3, materials, poltype)
    inc_up, inc_down = np.array([0.2 + 0.1j, -0.4j]), np.array([0.7j, 0.3])
    out_up, out_down = interface.illuminate(inc_up, inc_down)
    tangential = ["xyz".index(a) for a in alignment]
    for field in (efield, hfield):

        def operator(medium, side, field=field):
            return field(
                [0, 0, 0],
                basis=basis,
                k0=1.3,
                material=medium,
                modetype=side,
                poltype=poltype,
            )

        lower = (
            operator(materials[0], "up") @ inc_up
            + operator(materials[0], "down") @ out_down
        )
        upper = (
            operator(materials[1], "up") @ out_up
            + operator(materials[1], "down") @ inc_down
        )
        assert_allclose(lower[tangential], upper[tangential], rtol=2e-12, atol=2e-12)


@pytest.mark.gradients
@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
def test_cartesian_interface_all_parameter_pullbacks(alignment):
    rng = np.random.default_rng(78)
    values = [
        np.array([[1.3 + 0.1j, 1.5 + 0.12j], [1.8 + 0.07j, 2.0 + 0.15j]]),
        np.array([0.9 + 0.03j, 0.7 + 0.04j]),
        np.array([0.2, 0.3]),
    ]
    directions = [
        rng.normal(size=v.shape) * 0.1 + (0.03j if np.iscomplexobj(v) else 0)
        for v in values
    ]

    def record(ks, zs, q):
        return diff.interface_coefficients(ks, zs, q, alignment=alignment)

    value = record(*values)[0]
    check_pullback(
        record,
        *values,
        directions=tuple(directions),
        cotangents=complex_normal(rng, value.shape),
        step=1e-5,
        rtol=3e-7,
        atol=2e-9,
    )
    if alignment == "xy":
        ks, zs, q = values
        expected = coeffs.fresnel(ks, np.sqrt(ks**2 - np.sum(q**2)), zs)
        assert_allclose(value, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.gradients
@pytest.mark.parametrize("alignment", ["yz", "zx"])
def test_advect_oriented_slab_gradient(alignment):
    def objective(epsilon, q, thickness):
        k = 1.3 * anp.sqrt(epsilon + 0.1j)
        ks = anp.stack([anp.array([1.3, 1.3]), anp.stack([k, k])])
        z = anp.stack([1 + 0j, 1.3 / k])
        first = ad.interface_coefficients(ks, z, q, alignment=alignment)
        second = ad.interface_coefficients(ks[::-1], z[::-1], q, alignment=alignment)
        normal = anp.sqrt(k**2 - anp.sum(q**2))
        vectors = anp.stack([anp.stack([q[0], q[1], normal])] * 2)
        propagation = ad.propagation_matrix(vectors, anp.stack([0.0, 0.0, thickness]))
        slab = ad.smatrix_add(ad.smatrix_add(first, propagation), second)
        reflected = slab[1, 0, :, 0]
        return anp.sum(anp.real(reflected * anp.conj(reflected)))

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2)),
        np.array(2.3),
        np.array([0.2, 0.3]),
        np.array(0.4),
        directions=(np.array(0.1), np.array([0.1, -0.2]), np.array(0.05)),
        step=1e-5,
        rtol=3e-7,
        atol=2e-9,
    )


@pytest.mark.gradients
def test_normal_interface_adjoint_has_no_azimuth_singularity():
    ks = np.array([[1.3 + 0.1j, 1.5 + 0.12j], [1.8 + 0.07j, 2.0 + 0.15j]])
    z = np.array([0.9 + 0.03j, 0.7 + 0.04j])
    value, context = diff.interface_coefficients(ks, z, [0, 0])
    rng = np.random.default_rng(80)
    g = complex_normal(rng, value.shape)
    assert_allclose(context.pullback(g)[2], 0, atol=0)
    assert_allclose(value, coeffs.fresnel(ks, ks, z), atol=2e-13)
    for direction in ([0.3, -0.2], [-0.2, 0.7]):
        nearby = diff.interface_coefficients(ks, z, np.array(direction) * 1e-4)[0]
        assert_allclose(nearby, value, atol=1e-8)
    check_pullback(
        selecting(lambda ks, z: diff.interface_coefficients(ks, z, [0, 0]), 0, 1),
        ks,
        z,
        directions=(np.full(ks.shape, 0.2 + 0.1j), np.full(z.shape, 0.2 + 0.1j)),
        cotangents=g,
        step=1e-5,
        rtol=3e-7,
        atol=1e-9,
    )
