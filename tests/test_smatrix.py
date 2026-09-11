import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from treams_rs import (
    Material,
    PlaneWaveBasisByComp,
    SMatrices,
    coeffs,
    diff,
    poynting_avg_z,
)

# treams calls ufunc(where=...) then explicitly zeroes excluded entries.
pytestmark = pytest.mark.filterwarnings(
    "ignore:'where' used without 'out'.*:UserWarning"
)


def oracle_array(value):
    return np.array([[np.asarray(value[i, j]) for j in range(2)] for i in range(2)])


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("q", [[[0, 0]], [[0.2, 0.1], [2.8, 0.3], [-0.3, 0.5]]])
@pytest.mark.parametrize(
    "materials",
    [[1, 2.5, 1.3], [1, (3 + 0.1j, 1.2), 1], [(1.2, 1.1), (2.3, 0.9), (1.4, 0.8)]],
)
def test_slab_reference(poltype, q, materials):
    basis = PlaneWaveBasisByComp.default(q)
    oracle_basis = treams.PlaneWaveBasisByComp.default(q)
    value = SMatrices.slab(0.4, basis, 1.7, materials, poltype)
    # treams 0.4.5's parity interface uses a 2x2 mask for every basis size.
    expected = treams.SMatrices.slab(0.4, oracle_basis, 1.7, materials, "helicity")
    if poltype == "parity":
        expected = expected.changepoltype("parity")
    assert_allclose(value.array, oracle_array(expected), rtol=2e-13, atol=2e-13)
    incident = np.zeros(len(basis), complex)
    incident[0] = 1
    assert_allclose(value.tr(incident), expected.tr(incident), rtol=2e-13, atol=2e-13)
    for mode in ("up", "down"):
        actual_vectors = basis.kvecs(1.7, Material(materials[1]), mode)
        expected_vectors = oracle_basis.kvecs(1.7, treams.Material(materials[1]), mode)
        assert_allclose(actual_vectors, expected_vectors, rtol=2e-14, atol=2e-14)


@settings(max_examples=30, deadline=None)
@given(
    epsilon=st.floats(1.2, 4),
    mu=st.floats(0.7, 1.3),
    chirality=st.floats(-0.2, 0.2),
    thickness=st.floats(0, 3),
    q=st.floats(0, 0.5),
)
def test_lossless_chiral_layer_flux_and_split(epsilon, mu, chirality, thickness, q):
    basis = PlaneWaveBasisByComp.default([[q, 0.2], [-0.4, q]])
    ob = treams.PlaneWaveBasisByComp.default([[q, 0.2], [-0.4, q]])
    materials = [1, (epsilon, mu, chirality), 1]
    layer = SMatrices.slab(thickness, basis, 1.8, materials)
    oracle = treams.SMatrices.slab(thickness, ob, 1.8, materials)
    assert_allclose(layer.array, oracle_array(oracle), rtol=3e-13, atol=3e-13)
    split = SMatrices.slab(
        [thickness * 0.3, thickness * 0.7],
        basis,
        1.8,
        [1, materials[1], materials[1], 1],
    )
    assert_allclose(layer.array, split.array, rtol=2e-13, atol=2e-13)
    incident = np.array([0.2 + 0.7j, 0.5, -0.1j, 0.3])
    for direction in ("up", "down"):
        assert_allclose(
            sum(layer.tr(incident, modetype=direction)), 1, rtol=2e-13, atol=2e-13
        )
    for actual, expected in zip(
        poynting_avg_z(basis, 1.8, Material(materials[1])),
        treams.poynting_avg_z(ob, 1.8, treams.Material(materials[1])),
        strict=True,
    ):
        assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_stack_basis_order_and_propagation(poltype):
    modes = [(0.2, 0.1, 0), (2.8, 0.3, 1), (0.2, 0.1, 1), (2.8, 0.3, 0)]
    basis, ob = PlaneWaveBasisByComp(modes), treams.PlaneWaveBasisByComp(modes)
    value = SMatrices.propagation([0.1, -0.2, 0.3], basis, 1.8, 2, poltype)
    expected = treams.SMatrices.propagation([0.1, -0.2, 0.3], ob, 1.8, 2, poltype)
    assert_allclose(value.array, oracle_array(expected), rtol=1e-14, atol=1e-14)
    direct = SMatrices.propagation([0.4, -0.8, 1.2], basis, 1.8, 2, poltype)
    assert_allclose(value.double(2).array, direct.array, rtol=1e-13, atol=1e-13)
    assert_allclose(
        SMatrices.stack([value] * 4).array, direct.array, rtol=1e-13, atol=1e-13
    )


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 10_000), size=st.integers(1, 5))
def test_stack_native_pullback_and_identity(seed, size):
    rng = np.random.default_rng(seed)
    shape = (2, 2, size, size)
    values = [
        (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * 0.07 for _ in range(2)
    ]
    identity = np.zeros(shape, complex)
    identity[0, 0] = identity[1, 1] = np.eye(size)
    assert_allclose(diff.smatrix_add(values[0], identity)[0], values[0], atol=1e-14)
    assert_allclose(diff.smatrix_add(identity, values[0])[0], values[0], atol=1e-14)
    g = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    directions = [rng.normal(size=shape) + 1j * rng.normal(size=shape) for _ in values]
    _, ctx = diff.smatrix_add(*values)
    gradients = ctx.pullback(g)
    h = 1e-6
    plus = diff.smatrix_add(
        *(v + h * d for v, d in zip(values, directions, strict=True))
    )[0]
    minus = diff.smatrix_add(
        *(v - h * d for v, d in zip(values, directions, strict=True))
    )[0]
    assert_allclose(
        sum(np.vdot(g, d).real for g, d in zip(gradients, directions, strict=True)),
        np.vdot(g, (plus - minus) / (2 * h)).real,
        rtol=1e-7,
        atol=1e-8,
    )
    with pytest.raises(ValueError, match="consumed"):
        ctx.pullback(g)


@pytest.mark.parametrize("operation", ["fresnel", "propagation"])
def test_coefficient_pullbacks_all_complex_inputs(operation):
    rng = np.random.default_rng(18)
    if operation == "fresnel":
        values = [
            np.array([[1.2, 1.4], [1.9, 2.2]]) + 0.04j,
            np.array([[1.1, 1.3], [1.8, 2.1]]) + 0.09j,
            np.array([0.9, 0.7]) + 0.03j,
        ]
        function = coeffs.fresnel_with_context
        assert_allclose(
            function(*values)[0], treams.coeffs.fresnel(*values), rtol=2e-14, atol=2e-14
        )
    else:
        values = [
            np.array([[0.1, 0.2, 1.1 + 0.1j], [-0.2, 0.3, 0.8j]]),
            np.array([0.1, 0.2, 0.4]),
        ]
        function = diff.propagation
    result, ctx = function(*values)
    g = rng.normal(size=result.shape) + 1j * rng.normal(size=result.shape)
    gradients = ctx.pullback(g)
    h = 1e-6
    for index, (value, gradient) in enumerate(zip(values, gradients, strict=True)):
        direction = rng.normal(size=value.shape)
        if np.iscomplexobj(value):
            direction = direction + 1j * rng.normal(size=value.shape)
        plus, minus = list(values), list(values)
        plus[index], minus[index] = value + h * direction, value - h * direction
        numerical = np.vdot(
            g, (function(*plus)[0] - function(*minus)[0]) / (2 * h)
        ).real
        assert_allclose(
            np.vdot(gradient, direction).real, numerical, rtol=1e-7, atol=1e-8
        )
    with pytest.raises(ValueError, match="consumed"):
        ctx.pullback(g)


@settings(max_examples=15, deadline=None)
@given(
    thickness=st.floats(0.1, 1.0), epsilon=st.floats(1.3, 4.0), q=st.floats(0.0, 0.5)
)
def test_advect_complete_slab(thickness, epsilon, q):
    import advect
    import advect.numpy as anp

    from treams_rs import advect as ad

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
        propagation = ad.propagation(anp.stack([wave, wave]), anp.array([0.0, 0.0, d]))
        result = ad.smatrix_add(ad.smatrix_add(left, propagation), right)
        return anp.sum(anp.real(result * anp.conj(result))) + 0.1 * anp.sum(
            anp.real(result)
        )

    args = [np.array(thickness), np.array(epsilon + 0.07j), np.array(q)]
    directions = [0.1, 0.2 + 0.03j, -0.1]
    gradients = advect.grad(objective, argnums=(0, 1, 2))(*args)
    h = 1e-5
    plus = [v + h * d for v, d in zip(args, directions, strict=True)]
    minus = [v - h * d for v, d in zip(args, directions, strict=True)]
    assert_allclose(
        sum(np.vdot(g, d).real for g, d in zip(gradients, directions, strict=True)),
        (objective(*plus) - objective(*minus)) / (2 * h),
        rtol=1e-6,
        atol=1e-8,
    )


def test_invalid_interfaces_and_metadata():
    basis = PlaneWaveBasisByComp.default([0.1, 0.2])
    lower = SMatrices.interface(basis, 1.5, [1, 2])
    with pytest.raises(ValueError, match="internal medium"):
        lower.add(lower)
    with pytest.raises(ValueError, match="thickness"):
        SMatrices.slab(-1, basis, 1.5, [1, 2, 1])
    with pytest.raises(ValueError, match="polarization"):
        SMatrices.interface(basis, 1.5, [1, (2, 1, 0.1)], "parity")


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_plane_wave_slab_illumination(direction, poltype):
    from treams_rs import plane_wave

    basis = PlaneWaveBasisByComp.default([0.1, 0.2])
    slab = SMatrices.slab(0.4, basis, 1.7, [1, 2.3, 1], poltype)
    kz = direction * np.sqrt(1.7**2 - 0.1**2 - 0.2**2)
    wave = plane_wave([0.1, 0.2, kz], [0.3 + 0.2j, 0.7], k0=1.7, poltype=poltype)
    array = np.array([0.7, 0.3 + 0.2j])
    mode = "up" if direction > 0 else "down"
    assert_allclose(slab.tr(wave), slab.tr(array, modetype=mode), rtol=1e-13)
    assert_allclose(
        slab.illuminate(wave), slab.illuminate(array, modetype=mode), rtol=1e-13
    )


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("material", [1, (2 + 0.2j, 1.1)])
def test_oriented_power_is_cartesian_poynting(alignment, poltype, material):
    from treams_rs import efield, hfield
    from treams_rs._smatrix import _power_forms

    # Equal transverse wavevectors let a direct field evaluation check all
    # polarization and counterpropagating interference terms independently.
    basis = PlaneWaveBasisByComp.default([[0.2, 0.3]], alignment)
    amplitudes = np.array([[0.3 + 0.2j, -0.4j], [0.1j, 0.7]])
    forms = _power_forms(basis, 1.3, material, poltype)
    electric = sum(
        efield(
            [0, 0, 0],
            basis=basis,
            k0=1.3,
            material=material,
            poltype=poltype,
            modetype=side,
        )
        @ amplitudes[i]
        for i, side in enumerate(("up", "down"))
    )
    magnetic = sum(
        hfield(
            [0, 0, 0],
            basis=basis,
            k0=1.3,
            material=material,
            poltype=poltype,
            modetype=side,
        )
        @ amplitudes[i]
        for i, side in enumerate(("up", "down"))
    )
    expected = 0.5 * np.cross(electric, magnetic.conj()).real[basis.normal_axis]
    actual = sum(
        np.vdot(amplitudes[i], forms[i, j] @ amplitudes[j])
        for i in range(2)
        for j in range(2)
    )
    assert_allclose(actual, expected, atol=2e-14)


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
def test_oriented_propagation_phase_and_power(alignment):
    basis = PlaneWaveBasisByComp.default([[0.2, 0.3]], alignment)
    layer = SMatrices.propagation(0.4, basis, 1.3)
    expected = np.exp(0.4j * np.sqrt(1.3**2 - 0.2**2 - 0.3**2)) * np.eye(2)
    assert_allclose(layer[0, 0], expected, atol=1e-14)
    assert_allclose(layer[1, 1], expected, atol=1e-14)
    for side in ("up", "down"):
        assert_allclose(layer.tr([0.3 + 0.1j, 0.4], modetype=side), [1, 0], atol=1e-14)
    assert_allclose(
        layer.double().array, SMatrices.propagation(0.8, basis, 1.3).array, atol=1e-14
    )


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@pytest.mark.parametrize("q", [[0, 0], [0.2, 0.3], [2.8, -0.1]])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_interface_cartesian_boundary_continuity(alignment, q, poltype):
    from treams_rs import efield, hfield

    basis = PlaneWaveBasisByComp.default([q], alignment)
    materials = [
        Material(1.2 + 0.1j, 1.1, 0.07 if poltype == "helicity" else 0),
        Material(2.8 + 0.2j, 0.9),
    ]
    interface = SMatrices.interface(basis, 1.3, materials, poltype)
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
    value, context = diff.interface(*values, alignment=alignment)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradients = context.pullback(g)
    for i in range(3):
        plus, minus = list(values), list(values)
        h = 1e-5
        plus[i], minus[i] = values[i] + h * directions[i], values[i] - h * directions[i]
        numeric = np.vdot(
            g,
            (
                diff.interface(*plus, alignment=alignment)[0]
                - diff.interface(*minus, alignment=alignment)[0]
            )
            / (2 * h),
        ).real
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real, numeric, rtol=3e-7, atol=2e-9
        )
    if alignment == "xy":
        ks, zs, q = values
        expected = coeffs.fresnel(ks, np.sqrt(ks**2 - np.sum(q**2)), zs)
        assert_allclose(value, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("alignment", ["xy", "yz", "zx"])
@given(epsilon=st.floats(1.2, 4), thickness=st.floats(0, 1), q=st.floats(0, 0.5))
@settings(max_examples=20, deadline=None)
def test_oriented_lossless_slab_and_split(alignment, epsilon, thickness, q):
    basis = PlaneWaveBasisByComp.default([[q, 0.2]], alignment)
    materials = [1, (epsilon, 1.1, 0.1), 1]
    slab = SMatrices.slab(thickness, basis, 1.3, materials)
    split = SMatrices.slab(
        [0.3 * thickness, 0.7 * thickness],
        basis,
        1.3,
        [1, materials[1], materials[1], 1],
    )
    assert_allclose(slab.array, split.array, rtol=2e-12, atol=2e-12)
    for side in ("up", "down"):
        assert_allclose(sum(slab.tr([0.3 + 0.1j, 0.7j], modetype=side)), 1, atol=2e-12)


@pytest.mark.parametrize("alignment", ["yz", "zx"])
def test_advect_oriented_slab_gradient(alignment):
    import advect
    import advect.numpy as anp

    from treams_rs import advect as ad

    def objective(epsilon, q, thickness):
        k = 1.3 * anp.sqrt(epsilon + 0.1j)
        ks = anp.stack([anp.array([1.3, 1.3]), anp.stack([k, k])])
        z = anp.stack([1 + 0j, 1.3 / k])
        first = ad.interface(ks, z, q, alignment=alignment)
        second = ad.interface(ks[::-1], z[::-1], q, alignment=alignment)
        normal = anp.sqrt(k**2 - anp.sum(q**2))
        vectors = anp.stack([anp.stack([q[0], q[1], normal])] * 2)
        propagation = ad.propagation(vectors, anp.stack([0.0, 0.0, thickness]))
        slab = ad.smatrix_add(ad.smatrix_add(first, propagation), second)
        reflected = slab[1, 0, :, 0]
        return anp.sum(anp.real(reflected * anp.conj(reflected)))

    values = [np.array(2.3), np.array([0.2, 0.3]), np.array(0.4)]
    gradients = advect.grad(objective, argnums=(0, 1, 2))(*values)
    directions = [0.1, np.array([0.1, -0.2]), 0.05]
    h = 1e-5
    for i in range(3):
        plus, minus = list(values), list(values)
        plus[i], minus[i] = values[i] + h * directions[i], values[i] - h * directions[i]
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real,
            (objective(*plus) - objective(*minus)) / (2 * h),
            rtol=3e-7,
            atol=2e-9,
        )


def test_normal_interface_adjoint_has_no_azimuth_singularity():
    ks = np.array([[1.3 + 0.1j, 1.5 + 0.12j], [1.8 + 0.07j, 2.0 + 0.15j]])
    z = np.array([0.9 + 0.03j, 0.7 + 0.04j])
    value, context = diff.interface(ks, z, [0, 0])
    rng = np.random.default_rng(80)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gk, gz, gq = context.pullback(g)
    assert_allclose(gq, 0, atol=0)
    assert_allclose(value, coeffs.fresnel(ks, ks, z), atol=2e-13)
    for direction in ([0.3, -0.2], [-0.2, 0.7]):
        nearby = diff.interface(ks, z, np.array(direction) * 1e-4)[0]
        assert_allclose(nearby, value, atol=1e-8)
    h = 1e-5
    for i, (v, gradient) in enumerate(zip([ks, z], [gk, gz], strict=True)):
        direction = np.full(v.shape, 0.2 + 0.1j)
        plus, minus = [ks, z], [ks, z]
        plus[i], minus[i] = v + h * direction, v - h * direction
        numeric = np.vdot(
            g,
            (diff.interface(*plus, [0, 0])[0] - diff.interface(*minus, [0, 0])[0])
            / (2 * h),
        ).real
        assert_allclose(
            np.vdot(gradient, direction).real, numeric, rtol=3e-7, atol=1e-9
        )
