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

pytestmark = pytest.mark.filterwarnings(
    "ignore:'where' used without 'out'.*:UserWarning"
)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("modetype", ["up", "down"])
def test_plane_expansion_reference(poltype, modetype):
    destination = tr.SphericalWaveBasis.default(
        4, 2, [[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1]]
    )
    source = tr.PlaneWaveBasisByComp.default([[0.2, 0.3], [2.5, -0.1]])
    material = (1.4 + 0.1j, 1.2, 0.1 if poltype == "helicity" else 0)
    actual = tr.expand(
        (destination, source),
        ("regular", modetype),
        k0=1.3,
        material=material,
        poltype=poltype,
    )
    expected = treams.expand(
        (
            treams.SphericalWaveBasis(destination.modes, destination.positions),
            treams.PlaneWaveBasisByComp(source.modes),
        ),
        ("regular", modetype),
        k0=1.3,
        material=material,
        poltype=poltype,
    )
    assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(kx=st.floats(0.1, 0.7), kz=st.floats(-0.8, 1.3))
def test_plane_expansion_pullback_and_field(poltype, kx, kz):
    basis = tr.SphericalWaveBasis.default(8)
    vectors = np.array([[kx, 0.2, kz + 0.1j], [1.5, -0.1, 0.2j]])
    origins = np.array([[0.1, -0.2, 0.3]])
    rng = np.random.default_rng(34)
    value, context = tr.diff.plane_expansion(
        type(basis)(basis.modes, origins), vectors, [0, 1], poltype=poltype
    )
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    go, gv = context.pullback(g)
    directions = [np.array([[0.2, -0.1, 0.3]]), np.full(vectors.shape, 0.1 + 0.03j)]
    h = 1e-5
    for i in range(2):

        def shifted(sign, i=i):
            return tr.diff.plane_expansion(
                type(basis)(
                    basis.modes,
                    origins + sign * h * directions[0] if i == 0 else origins,
                ),
                vectors + sign * h * directions[1] if i == 1 else vectors,
                [0, 1],
                poltype=poltype,
            )[0]

        numeric = np.vdot(g, (shifted(1) - shifted(-1)) / (2 * h)).real
        assert_allclose(
            np.vdot((go, gv)[i], directions[i]).real, numeric, rtol=2e-7, atol=2e-7
        )
    # Each plane has its own norm; reconstruct one at a time in an achiral medium.
    points = origins + np.array([[0.1, 0.1, 0.05], [-0.1, 0.05, 0.1]])
    for j in range(2):
        k = np.sqrt(np.sum(vectors[j] ** 2))
        actual = tr.diff.field(
            value[:, j],
            points,
            type(basis)(basis.modes, origins),
            [k, k],
            poltype=poltype,
        )[0]
        expected = tr.diff.plane_field(
            [1], points, vectors[j : j + 1], [j], poltype=poltype
        )[0]
        assert_allclose(actual, expected, rtol=1e-9, atol=2e-11)


def test_advect_complete_illumination_scattering_and_field_gradient():
    basis = tr.SphericalWaveBasis.default(3)
    points = np.array([[0.7, 0.4, 0.3], [-0.4, 0.6, 0.2]])

    def objective(angles, k0, radius, origins):
        theta, phi = angles
        vectors = (
            k0
            * anp.stack(
                [
                    anp.sin(theta) * anp.cos(phi),
                    anp.sin(theta) * anp.sin(phi),
                    anp.cos(theta),
                ]
            )[None, :]
        )
        incoming = ad.plane_expansion(
            origins, vectors, destination=basis, polarizations=[1]
        )[:, 0]
        scattering = ad.sphere(3, k0, anp.reshape(radius, (1,)), [3 + 0.1j, 1])
        scattered = ad.field(
            scattering @ incoming,
            points,
            origins,
            anp.stack([k0, k0]),
            basis=basis,
            singular=True,
        )
        incident = ad.plane_field([1], points, vectors, polarizations=[1])
        total = incident + scattered
        return anp.sum(anp.real(total * anp.conj(total)))

    values = [
        np.array([0.7, 0.3]),
        np.array(1.3),
        np.array(0.2),
        np.array([[0.1, -0.1, 0.05]]),
    ]
    directions = [np.array([0.2, -0.1]), 0.1, -0.03, np.array([[0.02, -0.04, 0.03]])]
    gradients = advect.grad(objective, argnums=(0, 1, 2, 3))(*values)
    h = 1e-5
    for i in range(4):
        plus, minus = list(values), list(values)
        plus[i] = values[i] + h * directions[i]
        minus[i] = values[i] - h * directions[i]
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real,
            (objective(*plus) - objective(*minus)) / (2 * h),
            rtol=3e-6,
            atol=2e-9,
        )


@pytest.mark.parametrize("z", [1.3, -1.3, 1.3 + 0.1j])
def test_axial_expansion_fixed_vectors(z):
    basis = tr.SphericalWaveBasis.default(3)
    vectors = [[0, 0, z]]
    origins = np.array([[0.2, -0.1, 0.3]])
    value, context = tr.diff.plane_expansion(
        type(basis)(basis.modes, origins), vectors, [1]
    )
    assert_allclose(value[np.abs(basis.m) != 1], 0, atol=0)
    with pytest.raises(ValueError, match="direction derivative"):
        context.pullback(np.ones_like(value))

    def objective(origins):
        value = ad.plane_expansion(
            origins, vectors, destination=basis, polarizations=[1], fixed_vectors=True
        )
        return anp.sum(anp.real(value * anp.conj(value)))

    actual = advect.grad(objective)(origins)
    assert_allclose(
        actual[0], [0, 0, -2 * np.imag(z) * objective(origins)], rtol=1e-12, atol=1e-11
    )


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("pol", [0, 1])
def test_complex_transverse_branch_reconstruction_and_gradient(poltype, pol):
    # The principal sqrt(1-cos(theta)^2) has the opposite sign from k_transverse/k.
    # Angular coefficients must use the same branch as Cartesian polarization.
    vector = np.array([[0.2 + 1j, 0.1 + 0.3j, 1.3 - 0.8j]])
    basis = tr.SphericalWaveBasis.default(10)
    points = np.array([[0, 0, 0], [0.1, 0.2, 0.05]])
    k = np.sqrt(np.sum(vector**2))
    value, context = tr.diff.plane_expansion(basis, vector, [pol], poltype=poltype)
    reconstructed = tr.diff.field(value[:, 0], points, basis, [k, k], poltype=poltype)[
        0
    ]
    expected = tr.diff.plane_field([1], points, vector, [pol], poltype=poltype)[0]
    assert_allclose(reconstructed, expected, rtol=1e-10, atol=1e-12)
    rng = np.random.default_rng(74)
    cotangent = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    _, gradient = context.pullback(cotangent)
    direction = np.array([[0.1 + 0.2j, -0.03 + 0.1j, 0.2 - 0.1j]])
    h = 1e-5
    plus = tr.diff.plane_expansion(
        basis, vector + h * direction, [pol], poltype=poltype
    )[0]
    minus = tr.diff.plane_expansion(
        basis, vector - h * direction, [pol], poltype=poltype
    )[0]
    assert_allclose(
        np.vdot(gradient, direction).real,
        np.vdot(cotangent, (plus - minus) / (2 * h)).real,
        rtol=1e-8,
        atol=1e-7,
    )


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("side", ["up", "down"])
def test_cylindrical_plane_expansion_reference(poltype, side):
    basis = tr.CylindricalWaveBasis.default(
        [0.2, -0.3], 4, 2, [[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1]]
    )
    ports = tr.PlaneWaveBasisByComp.default(
        [[0.2, 0.1], [0.2, 2.5], [-0.3, -0.4]], "zx"
    )
    material = tr.Material(1.4 + 0.1j, 1.2, 0.1 if poltype == "helicity" else 0)
    vectors = np.column_stack(ports.kvecs(1.3, material, side))
    value = tr.expand(
        (basis, ports), ("regular", side), k0=1.3, material=material, poltype=poltype
    )
    coefficients = treams.pw.to_cw(
        basis.kz[:, None],
        basis.m[:, None],
        basis.pol[:, None],
        vectors[:, 0].real,
        vectors[:, 1],
        vectors[:, 2].real,
        ports.pol,
    )
    expected = coefficients * np.exp(1j * basis.positions[basis.pidx] @ vectors.T)
    assert_allclose(value, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(kx=st.floats(0.1, 0.7), kz=st.floats(-0.8, 0.8))
def test_cylindrical_plane_expansion_pullback_and_field(poltype, kx, kz):
    origins = np.array([[0.1, -0.2, 0.3]])
    basis = tr.CylindricalWaveBasis.default([kz, -1.1], 10, positions=origins)
    vectors = np.array([[kx + 0.1j, 0.3 + 0.2j, kz], [1.5, -0.1j, -1.1]])
    value, context = tr.diff.plane_expansion(basis, vectors, [0, 1], poltype=poltype)
    rng = np.random.default_rng(75)
    g = rng.normal(size=value.shape) + 1j * rng.normal(size=value.shape)
    gradients = context.pullback(g)
    assert_allclose(gradients[1][:, 2], 0, atol=0)
    directions = [
        np.array([[0.2, -0.1, 0.3]]),
        np.array([[0.1 + 0.03j, -0.02j, 0], [0.2j, -0.1, 0]]),
    ]
    values = [origins, vectors]

    def forward(values):
        return tr.diff.plane_expansion(
            type(basis)(basis.modes, values[0]), values[1], [0, 1], poltype=poltype
        )[0]

    h = 1e-5
    for i in range(2):
        plus, minus = list(values), list(values)
        plus[i], minus[i] = values[i] + h * directions[i], values[i] - h * directions[i]
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real,
            np.vdot(g, (forward(plus) - forward(minus)) / (2 * h)).real,
            rtol=3e-7,
            atol=2e-7,
        )
    points = origins + np.array([[0.1, 0.1, 0.05], [-0.1, 0.05, 0.1]])
    for j in range(2):
        k = np.sqrt(np.sum(vectors[j] ** 2))
        actual = tr.diff.field(value[:, j], points, basis, [k, k], poltype=poltype)[0]
        expected = tr.diff.plane_field(
            [1], points, vectors[j : j + 1], [j], poltype=poltype
        )[0]
        assert_allclose(actual, expected, rtol=1e-10, atol=2e-12)


def test_advect_complete_cylindrical_illumination_and_scattering():
    basis = tr.CylindricalWaveBasis.default([0.2], 3)
    points = np.array([[0.7, 0.4, 0.3], [-0.4, 0.6, 0.2]])

    def objective(angle, k0, radius, origins):
        transverse = anp.sqrt(k0**2 - 0.2**2)
        vectors = anp.stack(
            [transverse * anp.cos(angle), transverse * anp.sin(angle), 0.2]
        )[None, :]
        incoming = ad.plane_expansion(
            origins, vectors, destination=basis, polarizations=[1]
        )[:, 0]
        scattering = ad.cylinder([0.2], 3, k0, anp.reshape(radius, (1,)), [3 + 0.1j, 1])
        scattered = ad.field(
            scattering @ incoming,
            points,
            origins,
            anp.stack([k0, k0]),
            basis=basis,
            singular=True,
        )
        total = scattered + ad.plane_field([1], points, vectors, polarizations=[1])
        return anp.sum(anp.real(total * anp.conj(total)))

    values = [
        np.array(0.7),
        np.array(1.3),
        np.array(0.2),
        np.array([[0.1, -0.1, 0.05]]),
    ]
    directions = [0.2, 0.1, -0.03, np.array([[0.02, -0.04, 0.03]])]
    gradients = advect.grad(objective, argnums=(0, 1, 2, 3))(*values)
    h = 1e-5
    for i in range(4):
        plus, minus = list(values), list(values)
        plus[i], minus[i] = values[i] + h * directions[i], values[i] - h * directions[i]
        assert_allclose(
            np.vdot(gradients[i], directions[i]).real,
            (objective(*plus) - objective(*minus)) / (2 * h),
            rtol=3e-6,
            atol=2e-9,
        )
