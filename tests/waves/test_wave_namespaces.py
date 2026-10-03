"""Direct namespaces agree with matrix kernels and preserve NumPy calls.

Checks the upstream-mirroring sw, cw, pw and misc namespaces; the same translation
references are checked through special in test_translation_coefficients and through
_native.cartesian_translation_jet and the operators in test_translation.
"""

from operator import attrgetter

import numpy as np
import pytest
import treams
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

import treams_rs as tr
from treams_rs import cw, misc, pw, sw

from _support import finite_complex


@pytest.mark.reference
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


@pytest.mark.interface
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize("size", [None, 1, 2, 5])
def test_singular_self_translation_validates_labels_in_every_layout(poltype, size):
    """The outgoing self coefficient is zero, but invalid labels always raise."""
    kr = 0.0 if size is None else np.zeros(size)
    varying = 1 if size is None else np.ones(size, dtype=int)
    for invalid in [
        (0, 0, 0, 1, 0, 0),  # degree 0
        (1, 0, 0, 1, 2, 0),  # |order| > degree
        (1, 0, 2, 1, 0, 0),  # polarization 2
        (varying, 0, 0, 0, 0, 0),  # broadcast labels, source degree 0
    ]:
        with pytest.raises(ValueError, match="invalid spherical mode"):
            sw.translate(*invalid, kr, 0.3, 0.2, poltype=poltype)
    for valid in [(1, 0, 0, 1, 0, 0), (varying, 0, 1, 2, -1, 1)]:
        value = sw.translate(*valid, kr, 0.3, 0.2, poltype=poltype)
        assert np.shape(value) == np.shape(kr)
        assert np.all(value == 0)


@pytest.mark.reference
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


@pytest.mark.reference
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


@pytest.mark.reference
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
@pytest.mark.parametrize(
    "vector", [[0.3, 0.4, 1.2], [0.3 + 0.1j, 0.4, 1.2 + 0.1j], [0, 0, 1], [0, 0, -1]]
)
def test_plane_multipole_coefficients_and_permutation(poltype, vector):
    basis = tr.SphericalBasis.default(5)
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


@pytest.mark.reference
def test_plane_translation_and_cylindrical_expansion():
    x = np.linspace(-0.5, 0.8, 2048)
    args = (0.3, 0.4 + 0.1j, 1.2, x, 0.2, 0.3)
    expected = treams.pw.translate(*args)
    out = np.empty_like(expected)[::-1]
    assert pw.translate(*args, out=out) is out
    assert_allclose(out, expected, rtol=3e-13)
    cb = tr.CylindricalBasis.default([0.2, -0.3], 4)
    args = (cb.kz, cb.m, cb.pol, 0.3, 0.4 + 0.1j, 0.2, 1)
    assert_allclose(pw.to_cw(*args), treams.pw.to_cw(*args), rtol=4e-12, atol=4e-12)


@pytest.mark.reference
@pytest.mark.physics
@given(phi=st.floats(-3, 3), psi=st.floats(-3, 3))
@settings(max_examples=25)
def test_cylindrical_rotation_group(phi, psi):
    basis = tr.CylindricalBasis.default([0.2], 3)
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


@pytest.mark.reference
@pytest.mark.parametrize("family", ["sw", "cw"])
@pytest.mark.parametrize("dim", [1, 2])
def test_direct_periodic_namespace_rectangular_modes_and_origins(family, dim):
    if family == "sw":
        full = tr.SphericalBasis.default(2)
        labels = (full.l, full.m, full.pol)
    else:
        full = tr.CylindricalBasis.default([0.2, -0.3], 2)
        labels = (full.kz, full.m, full.pol)
    out = tuple(a[::3] for a in labels)
    incoming = tuple(a[1::2] for a in labels)
    cell = 1.7 if dim == 1 else np.diag([1.7, 1.8])
    bloch = 0.1 if dim == 1 else [0.1, 0.2]
    args = ([1.3, 1.4], bloch, cell, [0, 0, 0], out, incoming, [0.2, 0.1, 0.3])
    actual = getattr(tr, family).translate_periodic(*args, eta=0.7)
    expected = getattr(treams, family).translate_periodic(*args, eta=0.7)
    assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)


def _oracle_cases():
    """(namespace function, arguments, poltype or None, tolerance) per case."""
    sb = tr.SphericalBasis.default(4)
    cb = tr.CylindricalBasis.default([0.2, -0.3], 3)
    pol = np.array([0, 1])[:, None]
    cases = []
    for poltype in ("helicity", "parity"):
        for kz in (1.2, -1.2, 1.2 + 0.2j, -1.2 - 0.2j, 1.2 - 0.2j):
            args = (0.3, 0.4, kz, pol, sb.l, sb.m, sb.pol, 2.3)
            cases.append(("sw.periodic_to_pw", args, poltype, 4e-12, f"kz={kz}"))
        args = (cb.kz[:, None], cb.m[:, None], cb.pol[:, None])
        args += (sb.l, sb.m, sb.pol, 1.3 + 0.2j, 1.7)
        cases.append(("sw.periodic_to_cw", args, poltype, 5e-12, ""))
        args = (sb.l[:, None], sb.m[:, None], sb.pol[:, None], cb.kz, cb.m, cb.pol)
        cases.append(("cw.to_sw", (*args, 1.3 + 0.2j), poltype, 5e-12, ""))
    for ky in (0.4, -0.4, 0.4 + 0.1j, -0.4 - 0.1j, 0.4 - 0.1j, 0.3j):
        args = (0.3, ky, 0.2, pol, cb.kz, cb.m, cb.pol, 2.3)
        cases.append(("cw.periodic_to_pw", args, None, 5e-12, f"ky={ky}"))
    # At the cylindrical cutoff kz = +-k the radial wavenumber vanishes.
    for kz in (-1.0, 1.0):
        args = (sb.l, sb.m, sb.pol, kz, sb.m, sb.pol, 1)
        cases.append(("cw.to_sw", args, None, 5e-12, f"cutoff-kz={kz}"))
    return [
        pytest.param(
            path, args, poltype, tol, id="-".join(filter(None, (path, poltype, label)))
        )
        for path, args, poltype, tol, label in cases
    ]


@pytest.mark.reference
@pytest.mark.parametrize("path,args,poltype,tol", _oracle_cases())
def test_direct_conversions_against_upstream(path, args, poltype, tol):
    options = {} if poltype is None else {"poltype": poltype}
    actual = attrgetter(path)(tr)(*args, **options)
    expected = attrgetter(path)(treams)(*args, **options)
    assert_allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.reference
def test_periodic_to_cw_takes_the_upstream_keywords():
    """m and pol label the cylindrical mode, mu and qol the spherical one."""
    sb = tr.SphericalBasis.default(2)
    cb = tr.CylindricalBasis.default([0.2, -0.3], 2)
    for poltype in ("helicity", "parity"):
        keywords = dict(
            kz=cb.kz[:, None],
            m=cb.m[:, None],
            pol=cb.pol[:, None],
            l=sb.l,
            mu=sb.m,
            qol=sb.pol,
            k=1.3 + 0.2j,
            area=1.7,
            poltype=poltype,
        )
        assert_allclose(
            sw.periodic_to_cw(**keywords),
            treams.sw.periodic_to_cw(**keywords),
            rtol=5e-12,
            atol=5e-12,
        )


@pytest.mark.reference
@pytest.mark.physics
@given(
    x=st.floats(-3, 3),
    y=st.floats(-3, 3),
    kz=finite_complex(min_magnitude=0.1, max_magnitude=3),
)
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


# Oracle-free symmetries of the polarized kernels ----------------------------------


def _spherical(lmax):
    basis = tr.SphericalBasis.default(lmax)
    return [basis.l, basis.m, basis.pol]


def _plane(kx, ky, kz):
    return [np.full(2, kx), np.full(2, ky), np.full(2, kz), np.array([0, 1])]


def _off_null_cone(*components):
    """Whether the algebraic square norm is not small against the Hermitian one."""
    algebraic = sum(c**2 for c in components)
    return abs(algebraic) > 1e-2 * sum(abs(c) ** 2 for c in components)


def _rows(labels):
    return [np.asarray(label)[:, None] for label in labels]


_CYLINDRICAL_BASIS = tr.CylindricalBasis.default([0.2, -0.3], 2)
CYLINDRICAL = [_CYLINDRICAL_BASIS.kz, _CYLINDRICAL_BASIS.m, _CYLINDRICAL_BASIS.pol]
wavenumbers = finite_complex(min_magnitude=0.2, max_magnitude=5).map(
    lambda k: complex(k.real, abs(k.imag))
)


@st.composite
def polarized_kernels(draw):
    """A kernel ``f(poltype)`` with its output and input mode labels."""
    name = draw(
        st.sampled_from(
            [
                "sw.translate",
                "pw.to_sw",
                "cw.to_sw",
                "sw.periodic_to_pw",
                "sw.periodic_to_cw",
                "pw.permute_xyz",
            ]
        )
    )
    spherical = _spherical(draw(st.integers(1, 3)))
    k = draw(wavenumbers)
    kx, ky = draw(st.floats(-2, 2)), draw(st.floats(-2, 2))
    plane = _plane(kx, ky, k)
    # Plane kernels reject wavevectors on the null cone kx² + ky² + kz² = 0, such
    # as (0, 1, 1j), and the permutation also a null destination transverse
    # vector, (kz, kx) or (ky, kz), by contract; keep the draw clear of them.
    if name in {"pw.to_sw", "sw.periodic_to_pw", "pw.permute_xyz"}:
        assume(_off_null_cone(kx, ky, k))
    if name == "pw.permute_xyz":
        assume(_off_null_cone(k, kx) and _off_null_cone(ky, k))
    if name == "sw.translate":
        singular = draw(st.booleans())
        angles = draw(st.floats(0.05, np.pi - 0.05)), draw(st.floats(-2, 2))
        return (
            lambda p: sw.translate(
                *_rows(spherical), *spherical, k, *angles, poltype=p, singular=singular
            ),
            spherical,
            spherical,
        )
    if name == "pw.to_sw":
        return (
            lambda p: pw.to_sw(*_rows(spherical), *plane, poltype=p),
            spherical,
            plane,
        )
    if name == "cw.to_sw":
        return (
            lambda p: cw.to_sw(*_rows(spherical), *CYLINDRICAL, k, poltype=p),
            spherical,
            CYLINDRICAL,
        )
    if name == "sw.periodic_to_pw":
        area = draw(st.floats(0.5, 5))
        return (
            lambda p: sw.periodic_to_pw(*_rows(plane), *spherical, area, poltype=p),
            plane,
            spherical,
        )
    if name == "sw.periodic_to_cw":
        period = draw(st.floats(0.5, 5))
        return (
            lambda p: sw.periodic_to_cw(
                *_rows(CYLINDRICAL), *spherical, k, period, poltype=p
            ),
            CYLINDRICAL,
            spherical,
        )
    inverse = draw(st.booleans())
    pol = np.array([0, 1])
    return (
        lambda p: pw.permute_xyz(
            kx, ky, k, pol[:, None], pol, poltype=p, inverse=inverse
        ),
        plane,
        plane,
    )


@pytest.mark.physics
@given(polarized_kernels())
def test_parity_kernels_are_the_basis_change_of_helicity_kernels(kernel):
    # Every helicity/parity pair of native kernels is related by the involution
    # B mapping helicity to parity modes: parity = B_out helicity B_in.
    function, out, into = kernel
    helicity = function("helicity")
    assert_allclose(
        function("parity"),
        misc.basischange(np.array(out)) @ helicity @ misc.basischange(np.array(into)),
        rtol=1e-12,
        atol=1e-13 * max(1, abs(helicity).max()),
    )


@pytest.mark.physics
@pytest.mark.parametrize("singular", [False, True])
@given(
    k=wavenumbers,
    theta=st.floats(0.05, np.pi - 0.05),
    phi=st.floats(-3, 3),
    lmax=st.integers(1, 3),
)
def test_spherical_translation_point_inversion(singular, k, theta, phi, lmax):
    # r -> -r multiplies a parity mode by (-1)^l for one parity and by
    # -(-1)^l for the other, so T(-r) = (-1)^(l + l') s T(r) with s = +1
    # between equal parities and -1 between different ones.
    labels = _spherical(lmax)
    degree, pol = labels[0], labels[2]

    def translation(theta, phi):
        return sw.translate(
            *_rows(labels), *labels, k, theta, phi, poltype="parity", singular=singular
        )

    value = translation(theta, phi)
    sign = (-1.0) ** np.add.outer(degree, degree)
    sign *= np.where(np.equal.outer(pol, pol), 1, -1)
    assert_allclose(
        translation(np.pi - theta, phi + np.pi),
        sign * value,
        rtol=0,
        atol=1e-13 * abs(value).max(),
    )


@pytest.mark.physics
@given(angles=st.tuples(st.floats(-3, 3), st.floats(-3, 3), st.floats(-3, 3)))
def test_spherical_rotation_inverse_euler_angles(angles):
    labels = _spherical(3)
    phi, theta, psi = angles
    rotation = sw.rotate(*_rows(labels), *labels, phi, theta, psi)
    inverse = sw.rotate(*_rows(labels), *labels, -psi, -theta, -phi)
    assert_allclose(rotation @ inverse, np.eye(len(labels[0])), rtol=0, atol=4e-13)
