"""Direct namespaces agree with matrix kernels and preserve NumPy calls."""

import numpy as np
import pytest
import treams
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import cw, pw, sw


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("singular", [False, True])
def test_spherical_translation_namespace(poltype, singular):
    labels = np.array(
        [
            (degree, m, p)
            for degree in range(1, 4)
            for m in range(-degree, degree + 1)
            for p in [1, 0]
        ]
    ).T
    for kr in [0, 1.3, 1.3 + 0.2j]:
        args = (*(a[:, None] for a in labels), *labels, kr, 0.7, 0.3)
        expected = treams.sw.translate(*args, poltype=poltype, singular=singular)
        actual = sw.translate(*args, poltype=poltype, singular=singular)
        assert_allclose(actual, expected, rtol=4e-11, atol=4e-11)
        out = np.empty_like(actual).T
        assert sw.translate(*args, poltype=poltype, singular=singular, out=out) is out
        assert_allclose(out, expected, rtol=4e-11, atol=4e-11)
    theta = np.linspace(0.2, 1.2, 2048)
    args = (3, -1, 0, 4, 1, 0, 1.3 + 0.2j, theta, 0.3)
    assert_allclose(
        sw.translate(*args, poltype=poltype, singular=singular),
        treams.sw.translate(*args, poltype=poltype, singular=singular),
        rtol=4e-11,
        atol=4e-11,
    )


def test_spherical_rotation_reference_and_selection():
    labels = np.array(
        [
            (degree, m, p)
            for degree in range(1, 4)
            for m in range(-degree, degree + 1)
            for p in [1, 0]
        ]
    ).T
    args = (*(a[:, None] for a in labels), *labels, 0.3, 0.7, -0.2)
    value = sw.rotate(*args)
    assert_allclose(value, treams.sw.rotate(*args), rtol=5e-12, atol=5e-12)
    assert_allclose(value.conj().T @ value, np.eye(value.shape[0]), atol=4e-13)


@pytest.mark.parametrize("singular", [False, True])
def test_cylindrical_rotation_translation_reference(singular):
    basis = treams.CylindricalWaveBasis.default([0.2, -0.3], 3)
    labels = (basis.kz, basis.m, basis.pol)
    prefix = (*(a[:, None] for a in labels), *labels)
    assert_allclose(cw.rotate(*prefix, 0.4), treams.cw.rotate(*prefix, 0.4), atol=3e-14)
    for kr, z in [(0, 0), (1.3, 0.2), (1.3 + 0.2j, 0.3)]:
        args = (*prefix, kr, 0.4, z)
        assert_allclose(
            cw.translate(*args, singular=singular),
            treams.cw.translate(*args, singular=singular),
            rtol=5e-12,
            atol=5e-12,
        )


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize(
    "vector", [[0.3, 0.4, 1.2], [0.3 + 0.1j, 0.4, 1.2 + 0.1j], [0, 0, 1], [0, 0, -1]]
)
def test_plane_multipole_coefficients_and_permutation(poltype, vector):
    basis = tr.SphericalWaveBasis.default(5)
    labels = (basis.l[:, None], basis.m[:, None], basis.pol[:, None])
    args = (*labels, *vector, np.array([0, 1]))
    assert_allclose(
        pw.to_sw(*args, poltype=poltype),
        treams.pw.to_sw(*args, poltype=poltype),
        rtol=4e-12,
        atol=4e-12,
    )
    p = np.array([0, 1])[:, None]
    q = np.array([0, 1])
    for inverse in [False, True]:
        actual = pw.permute_xyz(*vector, p, q, poltype=poltype, inverse=inverse)
        expected = treams.pw.permute_xyz(
            *vector, p, q, poltype=poltype, inverse=inverse
        )
        assert_allclose(actual, expected, rtol=4e-12, atol=4e-12)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_cylindrical_spherical_conversion(poltype):
    sb = tr.SphericalWaveBasis.default(4)
    cb = tr.CylindricalWaveBasis.default([0.2, -0.3], 3)
    args = (
        sb.l[:, None],
        sb.m[:, None],
        sb.pol[:, None],
        cb.kz,
        cb.m,
        cb.pol,
        1.3 + 0.2j,
    )
    assert_allclose(
        cw.to_sw(*args, poltype=poltype),
        treams.cw.to_sw(*args, poltype=poltype),
        rtol=5e-12,
        atol=5e-12,
    )


def test_plane_translation_and_cylindrical_expansion():
    x = np.linspace(-0.5, 0.8, 2048)
    args = (0.3, 0.4 + 0.1j, 1.2, x, 0.2, 0.3)
    expected = treams.pw.translate(*args)
    out = np.empty_like(expected)[::-1]
    assert pw.translate(*args, out=out) is out
    assert_allclose(out, expected, rtol=3e-13)
    cb = tr.CylindricalWaveBasis.default([0.2, -0.3], 4)
    args = (cb.kz, cb.m, cb.pol, 0.3, 0.4 + 0.1j, 0.2, 1)
    assert_allclose(pw.to_cw(*args), treams.pw.to_cw(*args), rtol=4e-12, atol=4e-12)


@given(phi=st.floats(-3, 3), psi=st.floats(-3, 3))
@settings(max_examples=25)
def test_cylindrical_rotation_group(phi, psi):
    basis = tr.CylindricalWaveBasis.default([0.2], 3)
    for args in (
        (basis.kz, basis.m, basis.pol, basis.kz, basis.m, basis.pol),
        (0.2, -3, 1, 0.2, -3, 1),
    ):
        assert_allclose(
            cw.rotate(*args, phi) * cw.rotate(*args, psi),
            cw.rotate(*args, phi + psi),
            rtol=4e-13,
            atol=4e-13,
        )
        expected = treams.cw.rotate(*args, phi)
        assert_allclose(cw.rotate(*args, phi), expected, atol=3e-14)
        out = np.empty_like(expected)
        assert cw.rotate(*args, phi, out=out) is out
        assert_allclose(out, expected, atol=3e-14)


@pytest.mark.filterwarnings(
    "ignore:`scipy.special.sph_harm` is deprecated:DeprecationWarning"
)
@pytest.mark.parametrize("family", ["sw", "cw"])
@pytest.mark.parametrize("dim", [1, 2])
def test_direct_periodic_namespace_rectangular_modes_and_origins(family, dim):
    if family == "sw":
        full = tr.SphericalWaveBasis.default(2)
        labels = (full.l, full.m, full.pol)
    else:
        full = tr.CylindricalWaveBasis.default([0.2, -0.3], 2)
        labels = (full.kz, full.m, full.pol)
    out = tuple(a[::3] for a in labels)
    incoming = tuple(a[1::2] for a in labels)
    cell = 1.7 if dim == 1 else np.diag([1.7, 1.8])
    bloch = 0.1 if dim == 1 else [0.1, 0.2]
    args = ([1.3, 1.4], bloch, cell, [0, 0, 0], out, incoming, [0.2, 0.1, 0.3])
    actual = getattr(tr, family).translate_periodic(*args, eta=0.7)
    expected = getattr(treams, family).translate_periodic(*args, eta=0.7)
    assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("kz", [1.2, -1.2, 1.2 + 0.2j, -1.2 - 0.2j, 1.2 - 0.2j])
def test_spherical_periodic_plane_radiation(poltype, kz):
    basis = tr.SphericalWaveBasis.default(4)
    args = (0.3, 0.4, kz, np.array([0, 1])[:, None], basis.l, basis.m, basis.pol, 2.3)
    assert_allclose(
        sw.periodic_to_pw(*args, poltype=poltype),
        treams.sw.periodic_to_pw(*args, poltype=poltype),
        rtol=4e-12,
        atol=4e-12,
    )


@pytest.mark.parametrize("ky", [0.4, -0.4, 0.4 + 0.1j, -0.4 - 0.1j, 0.4 - 0.1j, 0.3j])
def test_cylindrical_periodic_plane_radiation(ky):
    basis = tr.CylindricalWaveBasis.default([0.2, -0.3], 3)
    args = (0.3, ky, 0.2, np.array([0, 1])[:, None], basis.kz, basis.m, basis.pol, 2.3)
    assert_allclose(
        cw.periodic_to_pw(*args),
        treams.cw.periodic_to_pw(*args),
        rtol=5e-12,
        atol=5e-12,
    )


@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_spherical_periodic_cylindrical_radiation(poltype):
    sb = tr.SphericalWaveBasis.default(4)
    cb = tr.CylindricalWaveBasis.default([0.2, -0.3], 3)
    args = (
        cb.kz[:, None],
        cb.m[:, None],
        cb.pol[:, None],
        sb.l,
        sb.m,
        sb.pol,
        1.3 + 0.2j,
        1.7,
    )
    assert_allclose(
        sw.periodic_to_cw(*args, poltype=poltype),
        treams.sw.periodic_to_cw(*args, poltype=poltype),
        rtol=5e-12,
        atol=5e-12,
    )


@pytest.mark.parametrize("kz", [-1.0, 1.0])
def test_direct_conversion_at_cylindrical_cutoff(kz):
    sb = tr.SphericalWaveBasis.default(4)
    args = (sb.l, sb.m, sb.pol, kz, sb.m, sb.pol, 1)
    assert_allclose(cw.to_sw(*args), treams.cw.to_sw(*args), rtol=5e-12, atol=5e-12)


@given(
    x=st.floats(-3, 3),
    y=st.floats(-3, 3),
    kz=st.complex_numbers(
        min_magnitude=0.1, max_magnitude=3, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=30, deadline=None)
def test_scalar_plane_phase_group_and_ufunc_options(x, y, kz):
    args = (0.3, 1, kz, x, y, 0.4)
    value = pw.translate(*args)
    assert_allclose(value, treams.pw.translate(*args), rtol=3e-14, atol=1e-15)
    assert_allclose(value * pw.translate(0.3, 1, kz, -x, -y, -0.4), 1, rtol=4e-14)
    out = np.array(7 + 0j)
    assert pw.translate(*args, out, where=False) is out
    assert out == 7
    assert pw.translate(*args, out=out) is out
    assert_allclose(out, value, rtol=1e-15)
    assert_allclose(pw.translate(kx=0.3, ky=1, kz=kz, x=x, y=y, z=0.4), value)
