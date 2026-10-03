"""Plane-wave expansion into spherical and cylindrical waves: diff.plane_expansion
against upstream, complex transverse branches, field reconstruction and pullbacks."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
import treams
from hypothesis import example, given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import advect as ad
from treams_rs.testing import check_gradient, check_pullback

from _support import complex_normal, selecting


@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("modetype", ["up", "down"])
def test_plane_expansion_reference(poltype, modetype):
    destination = tr.SphericalBasis.default(4, 2, [[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1]])
    source = tr.PlaneWavePorts.default([[0.2, 0.3], [2.5, -0.1]])
    material = (1.4 + 0.1j, 1.2, 0.1 if poltype == "helicity" else 0)
    actual = tr.operators.expand(
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


@pytest.mark.gradients
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(kx=st.floats(0.1, 0.7), kz=st.floats(-0.8, 1.3))
@example(kx=0.119140625, kz=0.0)
def test_plane_expansion_pullback_and_field(poltype, kx, kz):
    basis = tr.SphericalBasis.default(8)
    vectors = np.array([[kx, 0.2, kz + 0.1j], [1.5, -0.1, 0.2j]])
    origins = np.array([[0.1, -0.2, 0.3]])
    rng = np.random.default_rng(34)
    value, context = tr.diff.plane_expansion(
        type(basis)(basis.modes, origins), vectors, [0, 1], poltype=poltype
    )
    g = complex_normal(rng, value.shape)
    go, gv = context.pullback(g)
    directions = [np.array([[0.2, -0.1, 0.3]]), np.full(vectors.shape, 0.1 + 0.03j)]
    # The degree-8 angular response has a large third derivative near small k.
    # A five-point stencil removes the observed O(h²) oracle error while keeping
    # h large enough to avoid cancellation; the analytic kernel is unchanged.
    h = 1e-4
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

        numeric = np.vdot(
            g, (-shifted(2) + 8 * shifted(1) - 8 * shifted(-1) + shifted(-2)) / (12 * h)
        ).real
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


@pytest.mark.gradients
def test_advect_complete_illumination_scattering_and_field_gradient():
    basis = tr.SphericalBasis.default(3)
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

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2, 3)),
        np.array([0.7, 0.3]),
        np.array(1.3),
        np.array(0.2),
        np.array([[0.1, -0.1, 0.05]]),
        directions=(
            np.array([0.2, -0.1]),
            np.array(0.1),
            np.array(-0.03),
            np.array([[0.02, -0.04, 0.03]]),
        ),
        step=1e-5,
        rtol=3e-6,
        atol=2e-9,
    )


@pytest.mark.gradients
@pytest.mark.interface
@pytest.mark.parametrize("z", [1.3, -1.3, 1.3 + 0.1j])
def test_axial_expansion_fixed_vectors(z):
    basis = tr.SphericalBasis.default(3)
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


@pytest.mark.physics
@pytest.mark.gradients
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("pol", [0, 1])
def test_complex_transverse_branch_reconstruction_and_gradient(poltype, pol):
    # The principal sqrt(1-cos(theta)^2) has the opposite sign from k_transverse/k.
    # Angular coefficients must use the same branch as Cartesian polarization.
    vector = np.array([[0.2 + 1j, 0.1 + 0.3j, 1.3 - 0.8j]])
    basis = tr.SphericalBasis.default(10)
    points = np.array([[0, 0, 0], [0.1, 0.2, 0.05]])
    k = np.sqrt(np.sum(vector**2))
    value = tr.diff.plane_expansion(basis, vector, [pol], poltype=poltype)[0]
    reconstructed = tr.diff.field(value[:, 0], points, basis, [k, k], poltype=poltype)[
        0
    ]
    expected = tr.diff.plane_field([1], points, vector, [pol], poltype=poltype)[0]
    assert_allclose(reconstructed, expected, rtol=1e-10, atol=1e-12)
    check_pullback(
        selecting(
            lambda v: tr.diff.plane_expansion(basis, v, [pol], poltype=poltype), 1
        ),
        vector,
        directions=(np.array([[0.1 + 0.2j, -0.03 + 0.1j, 0.2 - 0.1j]]),),
        cotangents=complex_normal(np.random.default_rng(74), value.shape),
        step=1e-5,
        rtol=1e-8,
        atol=1e-7,
    )


@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("side", ["up", "down"])
def test_cylindrical_plane_expansion_reference(poltype, side):
    basis = tr.CylindricalBasis.default(
        [0.2, -0.3], 4, 2, [[0.1, 0.2, 0.3], [-0.2, 0.1, -0.1]]
    )
    ports = tr.PlaneWavePorts.default([[0.2, 0.1], [0.2, 2.5], [-0.3, -0.4]], "zx")
    material = tr.Material(1.4 + 0.1j, 1.2, 0.1 if poltype == "helicity" else 0)
    vectors = np.column_stack(ports.kvecs(1.3, material, side))
    value = tr.operators.expand(
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


@pytest.mark.gradients
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@given(kx=st.floats(0.1, 0.7), kz=st.floats(-0.8, 0.8))
def test_cylindrical_plane_expansion_pullback_and_field(poltype, kx, kz):
    origins = np.array([[0.1, -0.2, 0.3]])
    basis = tr.CylindricalBasis.default([kz, -1.1], 10, positions=origins)
    vectors = np.array([[kx + 0.1j, 0.3 + 0.2j, kz], [1.5, -0.1j, -1.1]])

    def record(origins, vectors):
        return tr.diff.plane_expansion(
            type(basis)(basis.modes, origins), vectors, [0, 1], poltype=poltype
        )

    value, context = record(origins, vectors)
    g = complex_normal(np.random.default_rng(75), value.shape)
    # The axial wavenumber labels the basis: its column is not differentiated.
    assert_allclose(context.pullback(g)[1][:, 2], 0, atol=0)
    check_pullback(
        record,
        origins,
        vectors,
        directions=(
            np.array([[0.2, -0.1, 0.3]]),
            np.array([[0.1 + 0.03j, -0.02j, 0], [0.2j, -0.1, 0]]),
        ),
        cotangents=g,
        step=1e-5,
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


@pytest.mark.gradients
def test_advect_complete_cylindrical_illumination_and_scattering():
    basis = tr.CylindricalBasis.default([0.2], 3)
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

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2, 3)),
        np.array(0.7),
        np.array(1.3),
        np.array(0.2),
        np.array([[0.1, -0.1, 0.05]]),
        directions=(
            np.array(0.2),
            np.array(0.1),
            np.array(-0.03),
            np.array([[0.02, -0.04, 0.03]]),
        ),
        step=1e-5,
        rtol=3e-6,
        atol=2e-9,
    )
