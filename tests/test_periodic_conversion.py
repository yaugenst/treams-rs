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
@pytest.mark.filterwarnings("ignore:'where' used without 'out'.*:UserWarning")
def test_array_constructor_reference(poltype):
    period, bloch, k0 = 1.7, 0.2, 1.3
    material = (1.4 + 0.1j, 1.2, 0.07 if poltype == "helicity" else 0)
    kzs = bloch + 2 * np.pi / period * np.arange(-1, 2)
    basis = tr.CylindricalWaveBasis.default(kzs, 3)
    particle = tr.TMatrix.sphere(3, k0, 0.2, [3.1, material], poltype=poltype)
    actual = tr.TMatrixC.from_array(particle, basis, lattice=period, kpar=bloch)
    oracle = treams.TMatrix.sphere(3, k0, 0.2, [3.1, material], poltype=poltype)
    response = oracle.latticeinteraction.solve(period, bloch)
    expected = treams.TMatrixC.from_array(
        response, treams.CylindricalWaveBasis.default(kzs, 3)
    )
    assert_allclose(actual.array, expected, rtol=3e-10, atol=2e-11)
    assert actual.k0 == particle.k0
    assert actual.material == particle.material
    assert actual.poltype == poltype


@given(radius=st.floats(0.1, 0.3), scale=st.floats(0.6, 1.8))
def test_array_lossless_power_and_scale(radius, scale):
    period, bloch, k0 = 1.7, 0.2, 1.3
    particle = tr.TMatrix.sphere(3, k0, radius, [3, 1])
    basis = tr.CylindricalWaveBasis.default([bloch], 3)
    value = tr.TMatrixC.from_array(particle, basis, lattice=period, kpar=bloch).array
    scattering = np.eye(len(basis)) + 2 * value
    assert_allclose(scattering.conj().T @ scattering, np.eye(len(basis)), atol=2e-10)
    scaled = tr.TMatrixC.from_array(
        tr.TMatrix.sphere(3, k0 / scale, radius * scale, [3, 1]),
        tr.CylindricalWaveBasis.default([bloch / scale], 3),
        lattice=period * scale,
        kpar=bloch / scale,
    )
    assert_allclose(scaled.array, value, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("cylindrical", [False, True])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_expandlattice_same_family_and_plane_radiation(cylindrical, poltype):
    k0, material = 1.3, (1.4 + 0.1j, 1.2)
    if cylindrical:
        source = tr.CylindricalWaveBasis.default([0.2], 2)
        oracle_source = treams.CylindricalWaveBasis.default([0.2], 2)
        lattice, bloch = 1.7, 0.1
        q = [[0.2, bloch + n * 2 * np.pi / lattice] for n in [-1, 0, 1]]
        ports = tr.PlaneWaveBasisByComp.default(q, "zx")
        oracle_ports = treams.PlaneWaveBasisByComp.default(q, "zx")
    else:
        source = tr.SphericalWaveBasis.default(2)
        oracle_source = treams.SphericalWaveBasis.default(2)
        lattice, bloch = np.diag([1.7, 1.8]), [0.1, 0.2]
        ports = tr.PlaneWaveBasisByComp.diffr_orders(bloch, lattice, 4)
        oracle_ports = treams.PlaneWaveBasisByComp.default(ports.components[::2])
    common = dict(k0=k0, material=material, poltype=poltype)
    actual = tr.expandlattice(lattice, bloch, basis=source, **common)
    expected = treams.expandlattice(lattice, bloch, basis=oracle_source, **common)
    assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    for side in ("up", "down"):
        actual = tr.expandlattice(
            lattice, bloch, basis=(ports, source), modetype=(side, "singular"), **common
        )
        expected = treams.expandlattice(
            lattice, bloch, basis=(oracle_ports, oracle_source), modetype=side, **common
        )
        assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


def test_expandlattice_cylindrical_orders_and_wave_types():
    source = tr.SphericalWaveBasis.default(2)
    destination = tr.CylindricalWaveBasis.default([0.2, 0.2 + 2 * np.pi / 1.7], 2)
    for modetype in (None, "singular", ("singular", "singular")):
        actual = tr.expandlattice(
            1.7, 0.2, basis=(destination, source), k0=1.3, modetype=modetype
        )
        assert_allclose(
            actual, tr.diff.periodic_conversion(destination, source, [1.3, 1.3], 1.7)[0]
        )
    with pytest.raises(ValueError, match="diffraction orders"):
        tr.expandlattice(1.7, 0.3, basis=(destination, source), k0=1.3)
    with pytest.raises(ValueError, match="outgoing waves"):
        tr.expandlattice(
            1.7, 0.2, basis=(destination, source), k0=1.3, modetype="regular"
        )


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_periodic_conversion_reference(poltype):
    source = tr.SphericalWaveBasis.default(4)
    destination = tr.CylindricalWaveBasis.default([-0.3, 0.2, 2.7], 4)
    ks = (
        np.array([1.3 + 0.1j, 1.5 + 0.2j])
        if poltype == "helicity"
        else np.full(2, 1.3 + 0.1j)
    )
    actual = tr.diff.periodic_conversion(destination, source, ks, 1.7, poltype=poltype)[
        0
    ]
    expected = treams.sw.periodic_to_cw(
        destination.kz[:, None],
        destination.m[:, None],
        destination.pol[:, None],
        source.l,
        source.m,
        source.pol,
        ks[source.pol],
        1.7,
        poltype=poltype,
    )
    assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("coincident", [False, True])
def test_periodic_conversion_all_pullbacks(poltype, coincident):
    source = tr.SphericalWaveBasis.default(3, 2)
    destination = tr.CylindricalWaveBasis.default([-0.3, 0.2], 2, 2)
    rng = np.random.default_rng(89)
    to = np.zeros((2, 3)) if coincident else rng.normal(size=(2, 3)) * 0.1
    origin = np.zeros((2, 3)) if coincident else rng.normal(size=(2, 3)) * 0.1
    ks = (
        np.array([1.3 + 0.1j, 1.5 + 0.2j])
        if poltype == "helicity"
        else np.full(2, 1.3 + 0.1j)
    )
    values = [to, origin, ks, destination.kz, np.array(1.7)]
    directions = [
        rng.normal(size=v.shape) * 0.1 + (0.02j if np.iscomplexobj(v) else 0)
        for v in values
    ]
    if poltype == "parity":
        directions[2][:] = directions[2][0]

    def forward(values):
        modes = [
            (p, kz, m, pol)
            for (p, _, m, pol), kz in zip(destination.modes, values[3], strict=True)
        ]
        return tr.diff.periodic_conversion(
            type(destination)(modes, values[0]),
            type(source)(source.modes, values[1]),
            values[2],
            float(values[4]),
            poltype=poltype,
        )

    value, context = forward(values)
    cotangent = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradients = context.pullback(cotangent)
    assert_allclose(gradients[0].sum(axis=0) + gradients[1].sum(axis=0), 0, atol=2e-12)
    assert_allclose(
        gradients[4] * values[4], -np.vdot(cotangent, value).real, atol=2e-12
    )
    h = 1e-5
    for i in range(5):
        plus, minus = list(values), list(values)
        plus[i], minus[i] = values[i] + h * directions[i], values[i] - h * directions[i]
        numeric = np.vdot(
            cotangent, (forward(plus)[0] - forward(minus)[0]) / (2 * h)
        ).real
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real, numeric, rtol=3e-7, atol=2e-8
        )


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_periodic_conversion_reconstructs_image_sum(poltype):
    period, bloch = 1.7, 0.2
    source = tr.SphericalWaveBasis.default(2, positions=[[-0.1, 0.2, -0.2]])
    orders = np.arange(-8, 9)
    destination = tr.CylindricalWaveBasis.default(
        bloch + 2 * np.pi / period * orders, 20, positions=[[0.2, -0.1, 0.3]]
    )
    ks = (
        np.array([1.3 + 0.2j, 1.5 + 0.25j])
        if poltype == "helicity"
        else np.full(2, 1.3 + 0.2j)
    )
    coefficients = np.arange(len(source)) * (0.03 + 0.02j)
    converted = (
        tr.diff.periodic_conversion(destination, source, ks, period, poltype=poltype)[0]
        @ coefficients
    )
    points = np.array([[2.4, 0.4, 0.1], [-2.3, -0.4, -0.2]])
    actual = tr.diff.field(
        converted, points, destination, ks, poltype=poltype, singular=True
    )[0]
    images = np.arange(-90, 91)
    positions = source.positions + np.column_stack(
        [np.zeros((len(images), 2)), images * period]
    )
    repeated = tr.SphericalWaveBasis.default(2, len(images), positions)
    amplitudes = (np.exp(1j * bloch * period * images)[:, None] * coefficients).ravel()
    expected = tr.diff.field(
        amplitudes, points, repeated, ks, poltype=poltype, singular=True
    )[0]
    assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


def test_advect_periodic_sphere_to_moving_cylindrical_orders():
    source = tr.SphericalWaveBasis.default(2)
    orders = np.repeat(np.arange(-2, 3), 14)
    destination = tr.CylindricalWaveBasis.default(
        0.2 + 2 * np.pi / 1.7 * np.arange(-2, 3), 3
    )
    propagating = orders == 0

    def objective(radius, k0, period, bloch, origins):
        ks = anp.stack([k0, k0])
        local = ad.sphere(2, k0, anp.reshape(radius, (1,)), [3 + 0.1j, 1])
        coupling = ad.lattice_expansion(
            origins,
            origins,
            ks,
            anp.reshape(bloch, (1,)),
            anp.reshape(period, (1, 1)),
            destination=source,
            source=source,
        )
        response = ad.interaction(local, coupling)
        vector = anp.stack([anp.sqrt(k0**2 - bloch**2), 0.0, bloch])[None, :]
        incident = ad.plane_expansion(
            origins, vector, destination=source, polarizations=[1]
        )[:, 0]
        conversion = ad.periodic_conversion(
            destination.positions,
            origins,
            ks,
            bloch + 2 * np.pi / period * orders,
            period,
            destination=destination,
            source=source,
        )
        radiation = (conversion @ response @ incident)[propagating]
        return anp.sum(anp.real(radiation * anp.conj(radiation)))

    values = [
        np.array(0.2),
        np.array(1.3),
        np.array(1.7),
        np.array(0.2),
        np.array([[0.1, 0.05, -0.1]]),
    ]
    directions = [0.03, 0.1, 0.04, -0.02, np.array([[0.01, -0.02, 0.03]])]
    gradients = advect.grad(objective, argnums=(0, 1, 2, 3, 4))(*values)
    h = 1e-5
    for i in range(5):
        plus, minus = list(values), list(values)
        plus[i], minus[i] = values[i] + h * directions[i], values[i] - h * directions[i]
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real,
            (objective(*plus) - objective(*minus)) / (2 * h),
            rtol=3e-6,
            atol=2e-9,
        )
