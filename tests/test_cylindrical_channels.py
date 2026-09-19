import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_cylindrical_channels_reference(poltype):
    basis = tr.CylindricalBasis.default(
        [0.2, -0.3], 3, 2, [[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1]]
    )
    ports = tr.PlaneWavePorts.default(
        [[0.2, 0.1], [0.2, 2.5], [-0.3, -0.2], [-0.3, -2.4]], "zx"
    )
    material = tr.Material(1.4 + 0.1j, 1.2, 0.1 if poltype == "helicity" else 0)
    actual, _ = tr.diff.cylindrical_channels(
        basis, material.ks(1.3), ports.components, ports.pol, 1.7, poltype=poltype
    )
    for side, modetype in enumerate(("up", "down")):
        kx, ky, kz = ports.kvecs(1.3, material, modetype)
        # Upstream Cython accepts real stored components and complex normal components.
        kx, kz = kx.real, kz.real
        phase = np.exp(
            1j * (basis.positions[basis.pidx] @ np.column_stack([kx, ky, kz]).T)
        )
        incoming = (
            treams.pw.to_cw(
                basis.kz[:, None],
                basis.m[:, None],
                basis.pol[:, None],
                kx,
                ky,
                kz,
                ports.pol,
            )
            * phase
        )
        outgoing = (
            treams.cw.periodic_to_pw(
                kx,
                ky,
                kz,
                ports.pol,
                basis.kz[:, None],
                basis.m[:, None],
                basis.pol[:, None],
                1.7,
            )
            / phase
        )
        assert_allclose(actual[0, side], incoming, rtol=1e-12, atol=1e-12)
        assert_allclose(actual[1, side], outgoing, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_cylindrical_channel_all_continuous_pullbacks(poltype):
    basis = tr.CylindricalBasis.default([0.2, -0.3], 2, 2)
    origins = np.array([[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1]])
    ks = (
        np.array([1.3 + 0.1j, 1.4 + 0.15j])
        if poltype == "helicity"
        else np.full(2, 1.3 + 0.1j)
    )
    q = np.array([[0.2, 0.1], [0.2, 2.5], [-0.3, -0.2], [-0.3, -2.4]])
    values = [origins, ks, q, 1.7]
    directions = [
        np.full_like(origins, 0.1),
        np.full_like(ks, 0.1 + 0.03j),
        np.column_stack([np.zeros(4), [0.1, -0.2, 0.3, 0.2]]),
        0.07,
    ]

    def forward(values):
        return tr.diff.cylindrical_channels(
            type(basis)(basis.modes, values[0]),
            values[1],
            values[2],
            [0, 1, 1, 0],
            float(values[3]),
            poltype=poltype,
        )

    value, context = forward(values)
    rng = np.random.default_rng(41)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradients = context.pullback(g)
    assert_allclose(gradients[2][:, 0], 0, atol=0)
    h = 1e-5
    for i in range(4):
        plus, minus = list(values), list(values)
        plus[i], minus[i] = values[i] + h * directions[i], values[i] - h * directions[i]
        numeric = np.vdot(g, (forward(plus)[0] - forward(minus)[0]) / (2 * h)).real
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real, numeric, rtol=2e-8, atol=2e-8
        )


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("side", [0, 1])
def test_radiation_reconstructs_periodic_image_field(poltype, side):
    k = 1.3 + 0.2j
    period, bloch = 1.7, 0.2
    basis = tr.CylindricalBasis.default([0.2], 2, positions=[[0.1, 0.15, 0.2]])
    orders = np.arange(-8, 9)
    ports = tr.PlaneWavePorts.default(
        np.column_stack(
            [np.full(len(orders), 0.2), bloch + 2 * np.pi / period * orders]
        ),
        "zx",
    )
    channels, _ = tr.diff.cylindrical_channels(
        basis, [k, k], ports.components, ports.pol, period, poltype=poltype
    )
    coefficients = np.arange(len(basis)) * (0.03 + 0.02j)
    points = np.array(
        [
            [0.2, 2.0 if side == 0 else -2.0, 0.3],
            [0.7, 2.4 if side == 0 else -2.4, -0.1],
        ]
    )
    vectors = np.column_stack(
        ports.kvecs(1, tr.Material(k * k), "up" if side == 0 else "down")
    )
    actual = tr.diff.plane_field(
        channels[1, side].T @ coefficients, points, vectors, ports.pol, poltype=poltype
    )[0]
    images = np.arange(-90, 91)
    positions = basis.positions + np.column_stack(
        [images * period, np.zeros((len(images), 2))]
    )
    repeated = tr.CylindricalBasis.default([0.2], 2, len(images), positions)
    amplitudes = (np.exp(1j * bloch * period * images)[:, None] * coefficients).ravel()
    expected = tr.diff.field(
        amplitudes, points, repeated, [k, k], poltype=poltype, singular=True
    )[0]
    assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


@given(radius=st.floats(0.1, 0.3))
def test_cylindrical_array_lossless_power(radius):
    basis = tr.CylindricalBasis.default([0.2], 3)
    period, bloch, k0 = 1.7, 0.1, 1.3
    ports = tr.PlaneWavePorts.default(
        [
            [0.2, bloch],
            [0.2, bloch + 2 * np.pi / period],
            [0.2, bloch - 2 * np.pi / period],
        ],
        "zx",
    )
    local = tr.diff.cylinder([0.2], 3, k0, [radius], [4, 1])[0]
    coupling = tr.lattice.expansion_with_context(
        basis, basis, [k0, k0], [[period]], [bloch]
    )[0]
    response = tr.diff.interaction(local, coupling)[0]
    channels = tr.diff.cylindrical_channels(
        basis, [k0, k0], ports.components, ports.pol, period
    )[0]
    scattering = tr.diff.smatrix_from_array(response, channels)[0]
    # Only the zero diffraction order propagates; both polarizations have the same flux.
    power = np.sum(np.abs(scattering[0, 0, :2, 0]) ** 2) + np.sum(
        np.abs(scattering[1, 0, :2, 0]) ** 2
    )
    assert_allclose(power, 1, rtol=2e-10, atol=2e-10)


def test_advect_complete_cylindrical_array_reflectance_gradient():
    basis = tr.CylindricalBasis.default([0.2], 3)
    kz_labels = np.full(6, 0.2)
    polarizations = np.tile([1, 0], 3)

    def objective(radius, k0, period, bloch, origins):
        local = ad.cylinder([0.2], 3, k0, anp.reshape(radius, (1,)), [4 + 0.1j, 1])
        coupling = ad.lattice_expansion(
            origins,
            origins,
            anp.stack([k0, k0]),
            anp.reshape(bloch, (1,)),
            anp.reshape(period, (1, 1)),
            destination=basis,
            source=basis,
        )
        response = ad.interaction(local, coupling)
        kx = anp.repeat(bloch + 2 * np.pi / period * np.array([0, 1, -1]), 2)
        channels = ad.cylindrical_channels(
            origins,
            anp.stack([k0, k0]),
            kx,
            period,
            basis=basis,
            kz_labels=kz_labels,
            polarizations=polarizations,
        )
        reflected = ad.smatrix_from_array(response, channels)[1, 0, :2, 0]
        return anp.sum(anp.real(reflected * anp.conj(reflected)))

    values = [
        np.array(0.2),
        np.array(1.3),
        np.array(1.7),
        np.array(0.1),
        np.array([[0.1, 0.2, 0.3]]),
    ]
    directions = [0.03, 0.1, 0.07, 0.04, np.array([[0.03, 0.02, -0.04]])]
    gradients = advect.grad(objective, argnums=(0, 1, 2, 3, 4))(*values)
    assert_allclose(gradients[4], 0, atol=1e-10)
    h = 1e-5
    for i in range(4):
        plus, minus = list(values), list(values)
        plus[i], minus[i] = values[i] + h * directions[i], values[i] - h * directions[i]
        numeric = (objective(*plus) - objective(*minus)) / (2 * h)
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real, numeric, rtol=3e-6, atol=2e-9
        )


@pytest.mark.parametrize("fixed_q", [False, True])
def test_fixed_axial_labels_and_explicit_channel_threshold(fixed_q):
    basis = tr.CylindricalBasis.default([0.2], 2)
    value, context = tr.diff.cylindrical_channels(
        basis, [1, 1], [[0.3, 0.1]], [1], 1.7, fixed_q=fixed_q
    )
    assert_allclose(value, 0, atol=0)
    for gradient in context.pullback(np.ones_like(value)):
        assert_allclose(gradient, 0, atol=0)
    with pytest.raises(ValueError, match="threshold"):
        tr.diff.cylindrical_channels(basis, [1, 1], [[0, 1]], [1], 1.7)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("side", ["up", "down"])
def test_cylindrical_array_python_workflow(poltype, side):
    period, bloch, k0, kz = 1.7, 0.1, 1.3, 0.2
    cylinder = tr.CylindricalTMatrix.cylinder(
        [kz], 3, k0, [0.2], [4, 1], poltype=poltype
    )
    ports = tr.PlaneWavePorts.default(
        [
            [kz, bloch],
            [kz, bloch + 2 * np.pi / period],
            [kz, bloch - 2 * np.pi / period],
        ],
        "zx",
    )
    array = tr.SMatrix._from_array(cylinder, ports, lattice=period, kpar=bloch)
    response = cylinder.latticeinteraction.solve([[period]], [bloch])
    channels = tr.diff.cylindrical_channels(
        cylinder.basis,
        cylinder.ks,
        ports.components,
        ports.pol,
        period,
        poltype=poltype,
    )[0]
    assert_allclose(
        array.array, tr.diff.smatrix_from_array(response, channels)[0], atol=1e-14
    )
    incident = np.zeros(len(ports), complex)
    incident[:2] = [0.4 + 0.1j, 0.7]
    ky = (1 if side == "up" else -1) * np.sqrt(k0**2 - kz**2 - bloch**2)
    wave = tr.plane_wave([bloch, ky, kz], incident[:2][::-1], k0=k0, poltype=poltype)
    assert_allclose(array.tr(wave), array.tr(incident, modetype=side), atol=1e-13)
    assert_allclose(sum(array.tr(wave)), 1, atol=2e-10)
    spacer = tr.SMatrix.propagation(0.4, ports, k0, poltype=poltype)
    assert_allclose(array.add(spacer).tr(wave), array.tr(wave), atol=2e-10)
