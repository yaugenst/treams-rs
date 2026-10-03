"""EBCM surface integrals (ebcm.qmat): upstream references, the Mie limit, lossless
convergence with the radial area factor, zero contrast, and pullbacks."""

import advect
import advect.numpy as anp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from treams.ebcm import qmat as upstream_qmat

import treams_rs as tr
from treams_rs import advect as ad
from treams_rs.testing import check_gradient, check_pullback

from _support import complex_normal


def _surface(order=48):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    theta = (nodes + 1) * np.pi / 2
    return theta, weights * np.pi / 2


def _media():
    materials = [
        tr.Material((3.1 + 0.2j, 1.2 + 0.1j, 0.07)),
        tr.Material((1.3, 1.1, -0.02)),
    ]
    return np.array([m.ks(1.3) for m in materials]), np.array(
        [m.impedance for m in materials]
    )


@pytest.mark.reference
@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("deformation", [0, 0.23])
def test_surface_integral_reference(singular, deformation):
    basis = tr.SphericalBasis.default(2)
    out = tr.SphericalBasis(basis.modes[::3])
    source = tr.SphericalBasis(basis.modes[1::2])
    ks, zs = _media()

    def r(theta):
        return 0.3 * (1 + deformation * np.cos(theta) ** 2)

    def dr(theta):
        return -0.6 * deformation * np.cos(theta) * np.sin(theta)

    actual = tr.ebcm.qmat(
        r, dr, ks, zs, out, source, singular, radial_area_factor=False
    )
    expected = upstream_qmat(
        r,
        dr,
        ks,
        zs,
        (out.l, out.m, out.pol),
        (source.l, source.m, source.pol),
        singular,
    )
    assert_allclose(actual, expected, rtol=2e-11, atol=1e-12)
    assert_allclose(actual[out.m[:, None] != source.m], 0, atol=0)
    doubled = tr.ebcm.qmat(
        r, dr, ks, zs, out, source, singular, order=192, radial_area_factor=False
    )
    assert_allclose(doubled, actual, rtol=2e-11, atol=1e-12)


@pytest.mark.physics
@given(radius=st.floats(0.15, 0.45), epsilon=st.floats(2, 4))
def test_ebcm_recovers_mie_and_lossless_power(radius, epsilon):
    basis = tr.SphericalBasis.default(2)
    materials = [tr.Material((epsilon, 1.2, 0.08)), tr.Material(1)]
    ks = [m.ks(1.3) for m in materials]
    zs = [m.impedance for m in materials]
    q = tr.ebcm.qmat(lambda _: radius, lambda _: 0, ks, zs, basis, order=48)
    regular = tr.ebcm.qmat(
        lambda _: radius, lambda _: 0, ks, zs, basis, singular=False, order=48
    )
    matrix = -tr.diff.solve(q, regular)[0]
    assert_allclose(
        matrix, tr.TMatrix.sphere(2, 1.3, radius, materials).array, atol=1e-12
    )
    scattering = np.eye(len(basis)) + 2 * matrix
    assert_allclose(scattering.conj().T @ scattering, np.eye(len(basis)), atol=1e-12)


@pytest.mark.gradients
@pytest.mark.parametrize("singular", [True, False])
@pytest.mark.parametrize("radial_area_factor", [False, True])
def test_ebcm_all_native_pullbacks(singular, radial_area_factor):
    theta, weights = _surface(40)
    radii = 0.3 * (1 + 0.2 * np.cos(theta) ** 2)
    slopes = -0.12 * np.cos(theta) * np.sin(theta)
    ks, zs = _media()
    basis = tr.SphericalBasis.default(2)
    values = [radii, slopes, ks, zs]

    def forward(values):
        return tr.diff.ebcm_qmat(
            *values,
            theta=theta,
            weights=weights,
            destination=basis,
            singular=singular,
            radial_area_factor=radial_area_factor,
        )

    value, context = forward(values)
    rng = np.random.default_rng(84)
    g = complex_normal(rng, value.shape)
    directions = [rng.normal(size=x.shape) * 0.05 for x in values]
    directions[2] = directions[2] + 0.03j
    directions[3] = directions[3] - 0.02j
    gradients = context.pullback(g)
    check_pullback(
        lambda *inputs: forward(inputs),
        *values,
        directions=tuple(directions),
        cotangents=g,
        step=1e-5,
        rtol=2e-7,
        atol=1e-8,
    )
    contracted = np.vdot(g, value).real
    assert_allclose(
        np.dot(gradients[0], radii)
        + np.dot(gradients[1], slopes)
        - np.vdot(gradients[2], ks).real,
        (2 if radial_area_factor else 1) * contracted,
        atol=1e-10,
    )
    assert_allclose(np.vdot(gradients[3], zs).real, contracted, atol=1e-10)


@pytest.mark.gradients
def test_complete_deformed_particle_advect():
    basis = tr.SphericalBasis.default(2)
    theta, weights = _surface(48)

    def objective(radius, deformation, epsilon, kappa, k0):
        radii = radius * (1 + deformation * np.cos(theta) ** 2)
        slopes = -2 * radius * deformation * np.cos(theta) * np.sin(theta)
        mu = 1.2 + 0.1j
        n = anp.sqrt(epsilon * mu)
        ks = k0 * anp.stack([anp.stack([n - kappa, n + kappa]), np.ones(2)])
        zs = anp.stack([anp.sqrt(mu / epsilon), 1.0])
        q = ad.ebcm_qmat(
            radii, slopes, ks, zs, theta=theta, weights=weights, destination=basis
        )
        regular = ad.ebcm_qmat(
            radii,
            slopes,
            ks,
            zs,
            theta=theta,
            weights=weights,
            destination=basis,
            singular=False,
        )
        matrix = -ad.solve(q, regular)
        return anp.sum(anp.real(matrix * anp.conj(matrix)))

    values = [
        np.array(0.3),
        np.array(0.2),
        np.array(3.1 + 0.2j),
        np.array(0.07 + 0.01j),
        np.array(1.3),
    ]
    gradients = advect.grad(objective, argnums=(0, 1, 2, 3, 4))(*values)
    check_gradient(
        objective,
        lambda *_: gradients,
        *values,
        directions=tuple(
            np.asarray(d) for d in (0.04, -0.1, 0.2 + 0.03j, 0.02 - 0.03j, 0.1)
        ),
        step=1e-5,
        rtol=2e-7,
        atol=1e-10,
    )
    assert_allclose(gradients[0] * values[0] - gradients[4] * values[4], 0, atol=1e-11)


@pytest.mark.physics
@given(radius=st.floats(0.2, 0.5), deformation=st.floats(-0.3, 0.3))
def test_deformed_surface_zero_material_contrast(radius, deformation):
    basis = tr.SphericalBasis.default(2)
    medium = tr.Material((1.3, 1.1, 0.04))
    args = dict(
        r=lambda t: radius * (1 + deformation * np.cos(t) ** 2),
        dr=lambda t: -2 * radius * deformation * np.cos(t) * np.sin(t),
        ks=[medium.ks(1.3)] * 2,
        zs=[medium.impedance] * 2,
        out=basis,
        order=64,
    )
    q = tr.ebcm.qmat(**args)
    regular = tr.ebcm.qmat(**args, singular=False)
    assert_allclose(regular, 0, atol=1e-15)
    assert_allclose(tr.diff.solve(q, regular)[0], 0, atol=1e-14)


@pytest.mark.physics
def test_radial_area_factor_restores_lossless_convergence():
    args = dict(
        r=lambda t: 0.3 * (1 + 0.23 * np.cos(t) ** 2),
        dr=lambda t: -0.138 * np.cos(t) * np.sin(t),
        ks=1.3 * np.array([[np.sqrt(3.1) - 0.07, np.sqrt(3.1) + 0.07], [1, 1]]),
        zs=[1 / np.sqrt(3.1), 1],
    )
    errors = []
    for degree in (2, 4, 6):
        basis = tr.SphericalBasis.default(degree)
        q = tr.ebcm.qmat(**args, out=basis)
        regular = tr.ebcm.qmat(**args, out=basis, singular=False)
        scattering = np.eye(len(basis)) - 2 * tr.diff.solve(q, regular)[0]
        errors.append(
            np.max(abs(scattering.conj().T @ scattering - np.eye(len(basis))))
        )
    assert errors[1] < errors[0] * 0.1
    assert errors[2] < errors[1] * 0.1
    assert errors[-1] < 2e-8
    q = tr.ebcm.qmat(**args, out=basis, radial_area_factor=False)
    regular = tr.ebcm.qmat(**args, out=basis, singular=False, radial_area_factor=False)
    scattering = np.eye(len(basis)) - 2 * tr.diff.solve(q, regular)[0]
    assert np.max(abs(scattering.conj().T @ scattering - np.eye(len(basis)))) > 1e-3
    # The missing factor also creates scattering from a nonexistent interface.
    args["ks"] = np.full((2, 2), 1.3)
    args["zs"] = [1, 1]
    without = tr.ebcm.qmat(**args, out=basis, singular=False, radial_area_factor=False)
    corrected = tr.ebcm.qmat(**args, out=basis, singular=False)
    assert np.max(abs(without)) > 1e-4
    assert_allclose(corrected, 0, atol=1e-15)


@pytest.mark.interface
@pytest.mark.parametrize(
    "out,match",
    [
        (tr.SphericalBasis.default(1, 2), "one origin at zero"),
        (tr.SphericalBasis.default(1, positions=[0, 0, 0.1]), "one origin at zero"),
        (([1, 1], [0, 0.5], [0, 1]), "integer degree/order/polarization"),
        (([1, 1], [0, 0], [0, np.nan]), "integer degree/order/polarization"),
        (([1, 1], [0, 0]), "integer degree/order/polarization"),
    ],
)
def test_surface_integral_mode_labels(out, match):
    theta, weights = _surface(8)
    ks, zs = _media()
    radii = np.full_like(theta, 0.3)
    with pytest.raises(ValueError, match=match):
        tr.diff.ebcm_qmat(
            radii, 0 * radii, ks, zs, theta=theta, weights=weights, destination=out
        )
