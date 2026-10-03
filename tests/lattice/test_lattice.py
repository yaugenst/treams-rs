# ruff: noqa: E741 - conventional multipole degree l
"""Ewald sums: upstream references, independent direct sums, periodic coupling and
the public-path derivatives. Symmetries and exact identities are native properties.

The incomplete gamma and Kambe integrals of the Ewald sums are checked in
test_ewald_integrals.
"""

import itertools
from functools import partial

import advect
import advect.numpy as anp
import numpy as np
import pytest
import scipy.special as scipy_special
import treams
import treams.lattice as upstream_lattice
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose
from scipy.linalg import block_diag

import treams_rs as tr
from treams_rs import (
    CylindricalBasis,
    CylindricalTMatrix,
    SphericalBasis,
    TMatrix,
    diff,
    lattice,
)
from treams_rs import advect as ad
from treams_rs.testing import check_gradient, check_pullback

from _support import assert_one_use_context, complex_normal

#: (spherical, lattice dimension) of every Ewald family.
FAMILIES = [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2)]
#: Rows of a skewed cell; the leading ``dim`` x ``dim`` block is a lattice.
SKEW = np.array([[1.0, 0.0, 0.0], [0.2, 1.1, 0.0], [0.1, -0.15, 0.9]])


def _record(spherical, l, m, k, q, a, r, eta=0):
    """One lattice sum through the public recorded path; q sets the dimension."""
    dim = np.size(q)
    return diff.lattice_sum(
        dim,
        l,
        m,
        k,
        np.reshape(q, dim),
        np.reshape(a, (dim, dim)),
        np.asarray(r)[: 3 if spherical else 2],
        eta,
        spherical=spherical,
    )


def evaluate(spherical, l, m, k, q, a, r, eta=0):
    return complex(_record(spherical, l, m, k, q, a, r, eta)[0])


def derivatives(spherical, l, m, k, q, a, r, eta):
    """The sum and its complex derivatives in k, q, a and r from two pullbacks.

    Under the real pairing dL = Re(conj(g) df), the cotangent 1 returns
    Re(df/dx) for a real x and conj(df/dk) for the holomorphic k, and the
    cotangent 1j returns Im(df/dx).
    """
    value, real = _record(spherical, l, m, k, q, a, r, eta)
    _, imaginary = _record(spherical, l, m, k, q, a, r, eta)
    dk, *parts, _ = real.pullback(np.ones_like(value))
    _, *imaginary_parts, _ = imaginary.pullback(1j * np.ones_like(value))
    dq, da, dr = (x + 1j * y for x, y in zip(parts, imaginary_parts, strict=True))
    return complex(value), np.conj(dk), dq, da, dr


def reference(spherical, dim, l, m, k, q, a, r, eta=0):
    if spherical:
        if dim == 1:
            return upstream_lattice.lsumsw1d_shift(l, m, k, q[0], a[0, 0], r, eta)
        if dim == 2:
            return upstream_lattice.lsumsw2d_shift(l, m, k, q, a, r, eta)
        return upstream_lattice.lsumsw3d(l, m, k, q, a, r, eta)
    if dim == 1:
        return upstream_lattice.lsumcw1d_shift(m, k, q[0], a[0, 0], r[:2], eta)
    return upstream_lattice.lsumcw2d(m, k, q, a, r[:2], eta)


@pytest.mark.reference
@pytest.mark.parametrize("spherical,dim", FAMILIES)
@pytest.mark.parametrize("l,m", [(0, 0), (1, -1), (2, 0), (3, 1), (4, -2)])
@pytest.mark.parametrize("shift", [False, True])
@pytest.mark.parametrize("k", [2.1, 2.1 + 0.2j])
def test_lattice_reference(spherical, dim, l, m, shift, k):
    a = np.diag(np.linspace(1.4, 1.8, dim))
    if dim > 1:
        a[1, 0] = 0.2
    q = np.linspace(0.1, 0.3, dim)
    r = np.array([0.16, -0.11, 0.08] if shift else [0.0, 0.0, 0.0])
    if not spherical:
        r[2] = 0
    actual = evaluate(spherical, l, m, k, q, a, r)
    expected = reference(spherical, dim, l, m, k, q, a, r)
    assert_allclose(actual, expected, rtol=3e-8, atol=3e-9)


@pytest.mark.physics
@pytest.mark.parametrize("l", [0, 2, 4, 12])
@pytest.mark.parametrize(
    "dz", [1e-12, 1e-50, 1e-100, 1e-120, 1e-200, 5e-324, -1e-120, -0.0]
)
def test_tiny_out_of_plane_shift_is_continuous(l, dz):
    # The public 2D sums shifted dz off the plane tend to the in-plane sums: the
    # reciprocal parts take the series in (k dz eta)^2 of their reduced integrals,
    # where the Kambe integrals at eta = -i / (k dz eta) cancel (as upstream's do).
    a = np.diag([1.7, 1.8])
    in_plane = lattice.lsumsw(2, l, 0, 2.1, [0, 0], a, [0.6, 0.2, 0], 0)
    shifted = lattice.lsumsw(2, l, 0, 2.1, [0, 0], a, [0.6, 0.2, dz], 0)
    assert_allclose(shifted, in_plane, rtol=1e-12, atol=1e-12)


@pytest.mark.reference
@pytest.mark.parametrize("dz", [1e-7, 1e-6, 1e-5])
def test_out_of_plane_curvature_matches_high_precision(dz):
    # Near the plane the degree-4 sum of test_tiny_out_of_plane_shift_is_continuous
    # changes by dz^2 S''(0) / 2 (it is even in dz), with S''(0) from mpmath Ewald
    # sums at 70 digits and another split; the rest is rounding, 2e-15 of |S| = 10.
    second = -1.7376998538289372 + 584.322817411281j
    a = np.diag([1.7, 1.8])
    in_plane = lattice.lsumsw(2, 4, 0, 2.1, [0, 0], a, [0.6, 0.2, 0], 0)
    shifted = lattice.lsumsw(2, 4, 0, 2.1, [0, 0], a, [0.6, 0.2, dz], 0)
    assert_allclose(shifted - in_plane, 0.5 * dz**2 * second, rtol=0, atol=2e-14)


@pytest.mark.physics
@pytest.mark.parametrize("dz", [1e-8, 1e-12, 1e-50, 1e-100, 1e-150, 1e-200, 5e-324])
@pytest.mark.parametrize("k", [2.1, 2.1 + 0.3j])
def test_periodic_coupling_at_almost_equal_heights(dz, k):
    # A 2D array of pairs of particles at heights that differ by dz couples like pairs
    # at equal heights (2D spherical sums of degrees up to 6 shifted dz off the plane):
    # their part even in dz is the equal-height coupling to O(dz^2), the rest the
    # first-order change, 4.2 dz of the largest entry here.
    from treams_rs import sw

    basis = SphericalBasis.default(3)
    modes = (basis.l, basis.m, basis.pol)
    a = np.array([[1.7, 0.0], [0.3, 1.8]])

    def coupling(height):
        return sw.translate_periodic(
            k, [0.3, -0.2], a, [[0.6, 0.2, height]], modes, modes, [[0.0, 0.0, 0.0]]
        )

    equal, above, below = coupling(0.0), coupling(dz), coupling(-dz)
    scale = np.abs(equal).max()
    assert_allclose((above + below) / 2, equal, rtol=0, atol=1e-12 * scale)
    assert np.abs(above - equal).max() <= (1e-12 + 5 * dz) * scale


@pytest.mark.gradients
@pytest.mark.parametrize("dz", [1e-100, -1e-200, 5e-324, -0.0])
@pytest.mark.parametrize("spherical", [True, False])
def test_tiny_normal_shift_pullbacks_are_continuous(dz, spherical):
    # The public recorded path of 2D spherical sums shifted dz off the plane and of
    # 1D cylindrical sums shifted dz off the axis returns the value and pullbacks of
    # the sums on it: the first-order change dz dS is far below rounding.
    k, l, m = 2.1 + 0.2j, (6 if spherical else 0), (-3 if spherical else 5)
    q = [0.3, -0.2] if spherical else [0.4]
    a = np.array([[1.7, 0.0], [0.3, 1.8]]) if spherical else np.array([[1.3]])
    r = np.array([0.6, 0.2, 0.0] if spherical else [0.6, 0.0, 0.0])
    shifted = r.copy()
    shifted[2 if spherical else 1] = dz
    expected = derivatives(spherical, l, m, k, q, a, r, 0)
    for actual, wanted in zip(
        derivatives(spherical, l, m, k, q, a, shifted, 0), expected, strict=True
    ):
        assert_allclose(
            actual, wanted, rtol=1e-12, atol=1e-12 * max(abs(expected[0]), 1)
        )


@pytest.mark.physics
@pytest.mark.parametrize("dim,k,eta", [(3, 0.5j, -0.5j), (2, 0.15 + 0.03j, 0.6)])
@pytest.mark.parametrize("r", [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
def test_small_splits_reach_the_sum(dim, k, eta, r):
    # Splits with |k eta| far below the automatic one (k eta = 0.25 and 0.09) need more
    # real-space shells than the base limit.
    a = 1.4 * SKEW[:dim, :dim]
    q = [0.1, 0.2, -0.3][:dim]
    l, m = zip(*[(l, m) for l in range(4) for m in range(-l, l + 1)], strict=True)
    expected = lattice.lsumsw(dim, l, m, k, q, a, r, 0)
    assert_allclose(
        lattice.lsumsw(dim, l, m, k, q, a, r, eta),
        expected,
        rtol=0,
        atol=1e-11 * max(np.abs(expected).max(), 1),
    )


@pytest.mark.physics
def test_small_splits_fail_only_where_their_parts_cancel():
    # Below every automatic split the public sums fail with "Ewald split too small"
    # where their cancelling parts lose beyond use (k = 5 at the split 0.1), or keep
    # the automatic sum, their 40-digit erfc sum (a real-space part) or their Lerch
    # sum (1D on the axis); the real-space part alone, which no split-independent sum
    # verifies, fails where its shells settle at their limit.
    with pytest.raises(ValueError, match="Ewald split too small"):
        lattice.lsumsw(2, 2, 1, 5.0, [0.3, -0.2], np.eye(2), [0.31, 0.17, 0.22], 0.1)
    oblique = [[1.0, 0.0], [0.3, 1.1]]
    args = (2, 4, -2, 1 + 0.41j, [0.2, -0.3], oblique, [0.31, 0.17, 0.22])
    expected = lattice.lsumsw(*args, 0)
    assert_allclose(lattice.lsumsw(*args, 0.12), expected, rtol=1e-6)
    square = np.eye(2)
    real = lattice.realsumsw(
        2, 0, 0, 2.0, [0.3, -0.2], square, [0.31, 0.17, 0.22], 0.13
    )
    assert_allclose(real, -7689855444.561904 - 130185264027.05389j, rtol=1e-13)
    axis = lattice.lsumsw(1, 0, 0, 0.34, [0.25], [[1.0]], [0.0, 0.0, 0.1], 0.1383)
    assert_allclose(axis, 2.503032083509468 - 10.824290488894055j, rtol=1e-6)
    with pytest.raises(ValueError, match="did not converge"):
        lattice.realsumsw(
            2,
            7,
            -3,
            0.20034569239060623,
            [-1.7730209332374476, -0.4017536791446624],
            [[1.0, 0.0], [0.2, 1.1]],
            [0.3959688456884133, 0.38764428981495824, 0.4],
            0.1695451560326192,
        )


@pytest.mark.physics
@pytest.mark.parametrize("eta", [0.1136377564103017, 0.13])
def test_small_split_pullbacks_of_sums_that_vanish_in_the_plane(eta):
    # A 2D spherical sum in the lattice plane with l + m odd is zero there, but its z
    # derivative is not: below every automatic split the public pullbacks keep every
    # component within 1e-6 of |dS| + max(|S|, 1) s of the automatic split (s = |k| for
    # the shift and lattice vectors, 1 / |k| for k and q).
    k = 2.0725349916224642
    args = (
        True,
        5,
        2,
        k,
        [-1.8922460112572441, -2.2052357384605776],
        [[0.9854079277054311, 0.0], [-0.01517160952402613, 1.3355452517572264]],
        [-0.5101194666812942, -0.10290586266322108, 0.0],
    )
    expected = derivatives(*args, 0)
    actual = derivatives(*args, eta)
    assert actual[0] == expected[0] == 0
    for value, wanted, unit in zip(
        actual[1:], expected[1:], (1 / k, 1 / k, k, k), strict=True
    ):
        assert_allclose(value, wanted, rtol=1e-6, atol=1e-6 * unit)


@pytest.mark.physics
@settings(max_examples=60)
@given(
    family=st.sampled_from(FAMILIES),
    degree=st.integers(0, 3),
    order=st.integers(-3, 3),
    halves=st.lists(st.integers(-1, 1), min_size=3, max_size=3),
    wholes=st.lists(st.integers(-1, 1), min_size=3, max_size=3),
    k=st.floats(0.5, 5.0),
    split=st.sampled_from([0.0, 0.15, 0.2, 0.3, 0.2 * np.exp(0.25j)]),
    part=st.sampled_from(["lsum", "realsum", "recsum"]),
)
def test_sums_that_vanish_by_symmetry_are_exactly_zero(
    family, degree, order, halves, wholes, k, split, part
):
    # Odd degrees (orders for cylindrical waves) change sign under inversion, which maps
    # the images onto themselves at a zero Bloch vector half a lattice vector from a
    # lattice point: the sums and their Ewald parts are zero at every split.
    spherical, dim = family
    l = 2 * degree + 1
    m = max(-l, min(l, order)) if spherical else 2 * order + 1
    a = np.array([[1.0, 0.0, 0.0], [0.25, 1.25, 0.0], [-0.25, 0.25, 0.75]])[:dim, :dim]
    shift = (0.5 * np.array(halves[:dim]) + np.array(wholes[:dim])) @ a
    if spherical and dim == 1:
        r = [0.0, 0.0, shift[0]]
    else:
        r = list(shift) + [0.0] * (3 - dim)
    if part != "lsum" and split == 0.0:
        split = 0.2
    q = [0.0] * dim
    if spherical:
        value = getattr(lattice, part + "sw")(dim, l, m, k, q, a, r, split)
    else:
        value = getattr(lattice, part + "cw")(dim, m, k, q, a, r[:2], split)
    assert value == 0


@pytest.mark.reference
@pytest.mark.parametrize("spherical,dim", FAMILIES)
def test_absolutely_convergent_direct_sum(spherical, dim):
    a = np.diag([1.7] * dim)
    q = np.array([0.12] * dim)
    r = np.array([0.21, 0.13, -0.12 if spherical else 0.0])
    k, l, m = 2.0 + 1.2j, 2, -1
    total = _image_sum(spherical, dim, l, m, k, q, a, origin=True)(r)
    assert_allclose(evaluate(spherical, l, m, k, q, a, r), total, rtol=2e-9, atol=2e-10)


def _chain_spectral(k, kz, a, r):
    """The degree-0 chain sum from its spectral series, which converges fast off the
    axis: -i / (k sqrt(4 pi)) (i pi / a) sum_m H0(rho k_m) e^(-i q_m z), with
    q_m = kz + 2 pi m / a and k_m = sqrt(k^2 - q_m^2), Im k_m >= 0."""
    m = np.arange(-200, 201)
    q = kz + 2 * np.pi * m / a
    radial = np.sqrt((k**2 - q**2).astype(complex))
    radial = np.where(radial.imag < 0, -radial, radial)
    terms = scipy_special.hankel1(0, np.hypot(r[0], r[1]) * radial) * np.exp(
        -1j * q * r[2]
    )
    return -1j / (k * np.sqrt(4 * np.pi)) * (1j * np.pi / a) * terms.sum()


# Chains off the axis where the Ewald sums with the automatic split cancel by up to
# e^(w^2), w = k rho eta from 5.9 to 30 (k, kz, period, shift).
OFF_AXIS_CHAINS = [
    (1.0, 0.3, 1.7, [4.0, 0.0, 0.3]),
    (1.0, 0.3, 1.7, [4.25, 0.0, 0.3]),
    (6.0, 0.3, 1.7, [6.0, 0.0, 0.3]),
    (4.36, 1.1, 0.88, [-2.472, -1.380, 0.437]),
    (0.8, 0.3, 1.7, [5.5, 0.0, 0.3]),
    (12.0, 0.3, 1.7, [8.0, 0.0, 0.3]),
]


@pytest.mark.reference
@pytest.mark.parametrize("k,kz,a,r", OFF_AXIS_CHAINS)
def test_off_axis_chain_sums_match_the_spectral_series(k, kz, a, r):
    expected = _chain_spectral(k, kz, a, r)
    assert_allclose(
        lattice.lsumsw1d_shift(0, 0, k, kz, a, r),
        expected,
        rtol=0,
        atol=1e-13 * max(abs(expected), 1),
    )


@pytest.mark.gradients
@settings(max_examples=12)
@given(
    geometry=st.sampled_from(OFF_AXIS_CHAINS[:5]),
    degree=st.integers(0, 12),
    order=st.floats(-1, 1),
    loss=st.sampled_from([0.0, 0.3]),
)
def test_off_axis_chain_pullbacks(geometry, degree, order, loss):
    # Far off the axis the sums take their spectral series; the pullbacks of the
    # public recorded path are its exact derivatives: central differences in every
    # input, and the Euler identity of joint length scaling.
    k, kz, a, r = geometry
    k = k + 1j * loss
    l, m = degree, round(order * degree)
    value, dk, dq, da, dr = _chain_pullbacks(
        l, m, k, kz, a, r, 0, 1 + 0.2j * bool(loss)
    )
    euler = -k * dk - kz * dq[0] + np.dot(r, dr) + a * da[0, 0]
    assert_allclose(euler, 0, atol=1e-11 * (1 + abs(value) + abs(k * dk)))


def _chain_pullbacks(l, m, k, kz, a, r, eta, kdir):
    """The value and pullbacks of a 1D spherical sum, whose derivatives in k along
    kdir, in kz, a and r match central differences."""
    q, cell, r = np.array([kz]), np.array([[a]]), np.array(r)
    value, dk, dq, da, dr = derivatives(True, l, m, k, q, cell, r, eta)
    rdir = np.array([0.12, 0.09, -0.1])
    for kstep, qstep, astep, rstep, analytical in [
        (kdir, 0, 0, 0 * rdir, dk * kdir),
        (0, 0.13, 0, 0 * rdir, dq[0] * 0.13),
        (0, 0, 0.07, 0 * rdir, da[0, 0] * 0.07),
        (0, 0, 0, rdir, np.dot(dr, rdir)),
    ]:
        plus, minus = (
            evaluate(
                True,
                l,
                m,
                k + s * kstep,
                q + s * qstep,
                cell + s * astep,
                r + s * rstep,
                eta,
            )
            for s in (1e-5, -1e-5)
        )
        atol = 1e-8 * max(abs(value), 1)
        assert_allclose(analytical, (plus - minus) / 2e-5, rtol=1e-7, atol=atol)
    return value, dk, dq, da, dr


def _chain_direct(l, m, k, kz, a, r):
    """The 1D spherical sum over the images n a along z, absolutely convergent for
    Im k > 0: 42 / (a Im k) images on each side reach e^-42."""
    reach = int(42 / (a * k.imag)) + 10
    n = np.arange(-reach, reach + 1)
    shift = -np.asarray(r, dtype=float) - np.outer(n * a, [0.0, 0.0, 1.0])
    radius = np.linalg.norm(shift, axis=1)
    theta = np.arctan2(np.hypot(shift[:, 0], shift[:, 1]), shift[:, 2])
    phi = np.arctan2(shift[:, 1], shift[:, 0])
    terms = (
        np.sqrt(np.pi / (2 * k * radius))
        * scipy_special.hankel1(l + 0.5, k * radius)
        * scipy_special.sph_harm_y(l, m, theta, phi)
        * np.exp(1j * kz * n * a)
    )
    return complex(terms[np.argsort(-np.abs(n))].sum())


@pytest.mark.physics
@pytest.mark.parametrize(
    "k,kz",
    [
        (1.0, 0.3),
        (1.0 + 0.1j, 0.3),
        (1.0 - 0.1j, 0.0),
        (2.5 - 0.2j, 0.0),
        (1.3 + 1e-12j, 0.0),
    ],
)
def test_chain_sums_do_not_depend_on_a_complex_split(k, kz):
    # The sums take the root Im k_q >= 0 of every diffraction order at every split,
    # rotated to Im(k eta) > 0 or real (which puts kz = 0 of complex k on the cut of
    # the principal Kambe integrals), on both sides of w = |k| rho |eta| = 2.5.
    l, m = np.array([(l, m) for l in range(5) for m in range(-l, l + 1)]).T
    for x in (1.5, 2.5, 3.5, 5.0):
        r = [x, 0.4, 0.3]
        if k.imag < 0:
            # These formerly tested a gain-side continuation, outside the outgoing
            # sum's supported domain even where that continuation was split invariant.
            with pytest.raises(ValueError, match="gain media are unsupported"):
                lattice.lsumsw(1, l, m, k, kz, 1.7, r)
            continue
        values = np.array(
            [
                lattice.lsumsw(1, l, m, k, kz, 1.7, r, eta)
                for eta in (0, 0.72, 0.6 + 0.4j, 0.6 - 0.4j, 0.72 * abs(k) / k)
            ]
        )
        assert_allclose(
            values,
            np.broadcast_to(values[0], values.shape),
            rtol=0,
            atol=1e-12 * max(np.abs(values[0]).max(), 1),
            err_msg=f"x = {x}",
        )


# Lossy chains at a Bloch vector on the reciprocal lattice with a real split
# (l, m, k, kz, period, shift, split): the order q = 0 lies on the cut of the Kambe
# integrals (upstream treams is 0.36 and 0.19 of max(|S|, 1) off at the second and
# third).
LOSSY_CHAINS_AT_THE_ZONE_CENTRE = [
    (1, -1, 0.9145895718991697 + 0.3j, 0.0, 1.7, [-0.598, -0.435, 0.691], 0.9),
    (0, 0, 1.8912951319622482 + 0.3j, 0.0, 1.0, [0.665, 1.543, 0.41], 0.6),
    (4, 0, 1.3374062937855533 + 0.2j, 2 * np.pi / 1.7, 1.7, [-1.35, -1.37, 0.05], 0.6),
    (0, 0, 1.0 + 0.1j, 0.0, 1.7, [3.3, 0.0, 0.3], 0.6),
]


@pytest.mark.reference
@pytest.mark.parametrize("l,m,k,kz,a,r,eta", LOSSY_CHAINS_AT_THE_ZONE_CENTRE)
def test_lossy_chains_at_the_zone_centre(l, m, k, kz, a, r, eta):
    # Against the direct sum, with pullbacks that match central differences.
    expected = _chain_direct(l, m, k, kz, a, r)
    value = _chain_pullbacks(l, m, k, kz, a, r, eta, 1 + 0.2j)[0]
    assert_allclose(value, expected, rtol=0, atol=1e-12 * max(abs(expected), 1))


@pytest.mark.physics
@pytest.mark.parametrize("x", [1.0, 3.3])
def test_periodic_coupling_at_an_explicit_split(x):
    # Couplings of a lossy chain at the Bloch vector 0 take its sums at the order
    # q = 0, on the cut of real splits.
    from treams_rs import sw

    basis = SphericalBasis.default(2)
    modes = (basis.l, basis.m, basis.pol)
    geometry = ([[x, 0.0, 0.3]], modes, modes, [[0.0, 0.0, 0.0]])
    automatic = sw.translate_periodic(1.0 + 0.1j, 0.0, 1.7, *geometry)
    for eta in (0.6, 0.45, 0.6 + 0.2j):
        explicit = sw.translate_periodic(1.0 + 0.1j, 0.0, 1.7, *geometry, eta=eta)
        assert_allclose(
            explicit, automatic, rtol=0, atol=1e-12 * np.abs(automatic).max()
        )


@pytest.mark.interface
def test_chains_without_their_series_report_the_lost_accuracy():
    # With Re k <= 0 the chain sums have no spectral series. Here the automatic
    # split cancels by about e^(w^2) at w = 6.25; the sum fails and asks for a
    # smaller split, with which it matches the direct sum.
    r = [2.5, 0.0, 0.2]
    with pytest.raises(ValueError, match=r"lost its accuracy.*reduce the split"):
        lattice.lsumsw1d_shift(7, 1, 1j, 0.3, 1.0, r)
    expected = _chain_direct(7, 1, 1j, 0.3, 1.0, r)
    assert_allclose(
        lattice.lsumsw1d_shift(7, 1, 1j, 0.3, 1.0, r, -0.3j),
        expected,
        rtol=0,
        atol=1e-12 * max(abs(expected), 1),
    )


@pytest.mark.physics
@pytest.mark.parametrize("x", [4.0, 4.25])
@pytest.mark.parametrize("lmax", [4, 6, 10])
def test_periodic_coupling_off_the_axis(x, lmax):
    # Couplings along a chain between particles x off the axis take lattice sums of
    # degrees up to 2 lmax at w = 5.9 and 6.3 with the automatic split, where they
    # take their spectral series; they agree with the coupling from a table of sums
    # at the split 0.3 (w <= 1.3, where the Ewald sums keep their accuracy).
    from treams_rs import sw

    basis = SphericalBasis.default(lmax)
    modes = (basis.l, basis.m, basis.pol)
    geometry = ([[x, 0.0, 0.3]], modes, modes, [[0.0, 0.0, 0.0]])
    actual = sw.translate_periodic(1.0, 0.3, 1.7, *geometry)

    def table(dim, l, m, k, kpar, a, r, eta):
        return lattice.lsumsw(dim, l, m, k, kpar, a, r, 0.3)

    expected = sw.translate_periodic(1.0, 0.3, 1.7, *geometry, func=table)
    assert_allclose(
        actual, expected, rtol=0, atol=1e-12 * max(np.abs(expected).max(), 1)
    )


@pytest.mark.interface
def test_threshold_and_invalid_lattice():
    with pytest.raises(ValueError, match="threshold"):
        evaluate(False, 0, 0, 1, [1], [[1.5]], [0.1, 0.1, 0])
    with pytest.raises(ValueError, match="independent"):
        evaluate(True, 1, 0, 2, [0, 0], [[1, 1], [1, 1]], [0, 0, 0])


@pytest.mark.reference
@pytest.mark.parametrize(
    "dim,poltype",
    # test_periodic_callbacks compares the 3D parity coupling with the oracle.
    [(1, "helicity"), (1, "parity"), (2, "helicity"), (2, "parity"), (3, "helicity")],
)
def test_periodic_spherical_coupling_and_solve(dim, poltype):
    positions = np.array([[0, 0, 0], [0.31, 0.12, 0.21]])
    basis = SphericalBasis.default(2, positions=positions)
    ks = (
        np.array([1.9 + 0.08j, 2.1 + 0.09j])
        if poltype == "helicity"
        else np.array([2.0 + 0.08j] * 2)
    )
    a = 1.5 * SKEW[:dim, :dim]
    q = np.array([0.11] * dim)
    expected = treams.sw.translate_periodic(
        ks, q, a, positions, np.array(basis.modes).T, poltype=poltype
    )
    actual = diff.lattice_expansion(basis, basis, ks, q, a, poltype=poltype)[0]
    assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
    # Bloch periodicity: a reciprocal-lattice shift of q leaves the coupling,
    # including its position-dependent phases, unchanged.
    shift = np.array([1, -1, 1][:dim]) @ (2 * np.pi * np.linalg.inv(a).T)
    assert_allclose(
        diff.lattice_expansion(basis, basis, ks, q + shift, a, poltype=poltype)[0],
        actual,
        rtol=1e-12,
        atol=1e-12,
    )

    spheres = [
        TMatrix.sphere(1, 2, [0.1], [2, 1], poltype),
        TMatrix.sphere(2, 2, [0.12], [3, 1], poltype),
    ]
    modes = [
        (index, degree, order, pol)
        for index, sphere in enumerate(spheres)
        for _, degree, order, pol in sphere.basis
    ]
    local = TMatrix(
        block_diag(*(sphere.array for sphere in spheres)),
        basis=SphericalBasis(modes, positions),
        k0=2,
        poltype=poltype,
    )
    upstream = treams.TMatrix.cluster(
        [
            treams.TMatrix.sphere(1, 2, [0.1], [2, 1], poltype),
            treams.TMatrix.sphere(2, 2, [0.12], [3, 1], poltype),
        ],
        positions,
    )
    response = local.latticeinteraction.solve(a, q)
    assert_allclose(
        response, upstream.latticeinteraction.solve(a, q), rtol=3e-8, atol=1e-11
    )
    assert_allclose(
        local.latticeinteraction(a, q) @ response, local.array, rtol=3e-10, atol=1e-14
    )


@pytest.mark.reference
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("poltype", ["helicity", "parity"])
def test_periodic_cylindrical_coupling_and_solve(dim, poltype):
    positions = np.array([[0, 0, 0], [0.31, 0.12, 0.21]])
    basis = CylindricalBasis.default([0, 0.4], 2, positions=positions)
    ks = [1.9 + 0.08j, 2.1 + 0.09j] if poltype == "helicity" else [2.0 + 0.08j] * 2
    a = 1.5 * SKEW[:dim, :dim]
    q = np.array([0.11] * dim)
    oracle_a, oracle_q = (float(a[0, 0]), float(q[0])) if dim == 1 else (a, q)
    actual = diff.lattice_expansion(basis, basis, ks, q, a, poltype=poltype)[0]
    expected = treams.cw.translate_periodic(
        ks, oracle_q, oracle_a, positions, (basis.pidx, basis.kz, basis.m, basis.pol)
    )
    assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
    shift = np.array([1, -1][:dim]) @ (2 * np.pi * np.linalg.inv(a).T)
    assert_allclose(
        diff.lattice_expansion(basis, basis, ks, q + shift, a, poltype=poltype)[0],
        actual,
        rtol=1e-12,
        atol=1e-12,
    )

    local = CylindricalTMatrix.cylinder([0.4], 2, 2, [0.1], [2, 1], poltype)
    upstream = treams.TMatrixC.cylinder([0.4], 2, 2, [0.1], [2, 1])
    if poltype == "parity":
        upstream = upstream.changepoltype("parity")
    response = local.latticeinteraction.solve(a, q)
    assert_allclose(
        response,
        upstream.latticeinteraction.solve(oracle_a, oracle_q),
        rtol=3e-8,
        atol=1e-11,
    )
    assert_allclose(
        local.latticeinteraction(a, q) @ response, local.array, rtol=3e-10, atol=1e-14
    )


@pytest.mark.interface
@pytest.mark.reference
def test_public_lattice_mode_broadcasting():
    l = np.array([1, 2, 3])
    m = np.array([[-1], [0], [1]])
    actual = lattice.lsumsw2d(l, m, 2, [0.1, 0.2], [[1.5, 0], [0.1, 1.6]], [0.1, -0.2])
    assert_allclose(
        actual,
        upstream_lattice.lsumsw2d(
            l, m, 2, [0.1, 0.2], [[1.5, 0], [0.1, 1.6]], [0.1, -0.2], 0
        ),
        rtol=2e-8,
        atol=1e-9,
    )


@pytest.mark.gradients
@pytest.mark.reference
@pytest.mark.parametrize("spherical,dim", FAMILIES)
# (0, 0) and (2, 0) are covered natively by the Rust ewald_complete_derivative
# and ewald_derivative_identities properties for every family.
@pytest.mark.parametrize("l,m", [(1, -1), (3, 1)])
@pytest.mark.parametrize("origin", [True, False])
def test_all_continuous_lattice_derivatives(spherical, dim, l, m, origin):
    a = np.diag(np.linspace(1.5, 1.8, dim))
    if dim > 1:
        a[1, 0] = 0.17
    q = np.zeros(dim) if origin else np.linspace(0.1, 0.3, dim)
    r = np.zeros(3) if origin else np.array([0.21, -0.14, 0.17 if spherical else 0])
    k = 2.1 + 0.2j
    value, dk, dq, da, dr = derivatives(spherical, l, m, k, q, a, r, 1.2)
    coordinates = 3 if spherical else 2
    qdir = np.linspace(0.1, -0.2, dim)
    adir = np.arange(1, 1 + dim * dim).reshape(dim, dim) * 0.05
    rdir = np.array([0.12, 0.09, -0.1 if spherical else 0])
    directions = [
        (0.3 + 0.1j, np.zeros(dim), np.zeros_like(a), np.zeros(3), dk * (0.3 + 0.1j)),
        (0, qdir, np.zeros_like(a), np.zeros(3), np.dot(dq, qdir)),
        (0, np.zeros(dim), adir, np.zeros(3), np.sum(da * adir)),
    ]
    if not origin:
        directions.append(
            (0, np.zeros(dim), np.zeros_like(a), rdir, np.dot(dr, rdir[:coordinates]))
        )
    for kdir, qdir, adir, rdir, analytical in directions:
        h = 2e-5
        plus, minus = (
            evaluate(
                spherical,
                l,
                m,
                k + s * kdir,
                q + s * qdir,
                a + s * adir,
                r + s * rdir,
                1.2,
            )
            for s in (h, -h)
        )
        assert_allclose(analytical, (plus - minus) / (2 * h), rtol=2e-7, atol=2e-8)
    euler = -k * dk - np.dot(q, dq) + np.dot(r[:coordinates], dr) + np.sum(a * da)
    assert_allclose(euler, 0, atol=2e-9 * (1 + abs(value)))


@pytest.mark.gradients
@pytest.mark.parametrize(
    "spherical,dim,equal_wavenumbers",
    # Unequal wavenumbers stay covered in one and two dimensions; the 3D
    # spherical case with them is the slowest and adds no other code path.
    [
        (spherical, dim, equal)
        for spherical, dim in FAMILIES
        for equal in (False, True)
        if (spherical, dim, equal) != (True, 3, False)
    ],
)
def test_periodic_matrix_pullback(spherical, dim, equal_wavenumbers):
    basis = (
        SphericalBasis.default(1) if spherical else CylindricalBasis.default([0.3], 1)
    )
    ks = np.array([2.0 + 0.1j, (2.0 if equal_wavenumbers else 2.1) + 0.1j])
    q = np.linspace(0.1, 0.2, dim)
    a = np.diag([1.6] * dim)
    # Dynamic inputs in pullback order: positions, wavenumbers, Bloch vector, cell.
    parameters = [np.array([[0.21, 0.12, 0.15]]), np.zeros((1, 3)), ks, q, a]

    def record(destination, source, ks, q, a):
        return diff.lattice_expansion(
            type(basis)(basis.modes, destination),
            type(basis)(basis.modes, source),
            ks,
            q,
            a,
            eta=1.2,
        )

    value, context = record(*parameters)
    rng = np.random.default_rng(21)
    g = complex_normal(rng, value.shape)
    # After the rejected cotangents the context pulls back what the fresh
    # record, checked below, does.
    gradients = assert_one_use_context(
        context, g, record(*parameters)[1].pullback(g), rtol=1e-13, atol=1e-15
    )
    assert_allclose(gradients[0].sum(axis=0) + gradients[1].sum(axis=0), 0, atol=1e-12)
    directions = [rng.normal(size=x.shape) * 0.1 for x in parameters]
    directions[2] = directions[2] + 0.12j
    check_pullback(
        record,
        *parameters,
        directions=tuple(directions),
        cotangents=g,
        step=2e-5,
        rtol=2e-7,
        atol=2e-7,
    )


@pytest.mark.gradients
# Each example checks the three inputs separately (1 + 2 x 3 objectives).
@settings(max_examples=4)
@given(
    spherical=st.booleans(),
    pitch=st.floats(1.4, 1.9),
    radius=st.floats(0.1, 0.22),
    bloch=st.floats(0.0, 0.3),
)
def test_advect_periodic_complete_solve(spherical, pitch, radius, bloch):
    basis = (
        SphericalBasis.default(1) if spherical else CylindricalBasis.default([0.3], 1)
    )
    positions = np.array([[0.1, 0.03, 0.02]])

    def objective(radii, vectors, q):
        local = (
            ad.sphere(1, 2.0, radii, np.array([3.0 + 0.1j, 1.0]))
            if spherical
            else ad.cylinder(
                np.array([0.3]), 1, 2.0, radii, np.array([3.0 + 0.1j, 1.0])
            )
        )
        coupling = ad.lattice_expansion(
            positions,
            positions,
            np.array([2.0, 2.0]),
            q,
            vectors,
            destination=basis,
            source=basis,
        )
        effective = ad.interaction(local, coupling)
        return anp.sum(anp.real(effective * anp.conj(effective))) + 0.07 * anp.sum(
            anp.real(effective)
        )

    check_gradient(
        objective,
        advect.grad(objective, argnums=(0, 1, 2)),
        np.array([radius]),
        np.diag([pitch, pitch * 1.1]),
        np.array([bloch, 0.1]),
        directions=(
            np.array([0.2]),
            np.array([[0.1, 0.03], [-0.02, -0.1]]),
            np.array([0.07, -0.1]),
        ),
        step=1e-5,
        rtol=2e-6,
        atol=1e-10,
    )


def _image_sum(spherical, dim, l, m, k, q, a, cells=13, origin=False):
    """The images of a lattice point as a function of r, less the one at the origin
    unless ``origin``.

    The sums over ``cells`` cells in each direction converge absolutely for Im k > 0.
    """
    indices = np.array(list(itertools.product(range(-cells, cells + 1), repeat=dim)))
    if not origin:
        indices = indices[np.any(indices != 0, axis=1)]
    vectors = indices @ a
    embedded = np.zeros((len(vectors), 3))
    axes = [2] if spherical and dim == 1 else list(range(dim))
    embedded[:, axes] = vectors
    phase = np.exp(1j * (vectors @ q))

    def direct(r):
        shift = -embedded - r
        radius = np.linalg.norm(shift, axis=1)
        shift, radius, weights = (
            shift[radius > 0],
            radius[radius > 0],
            phase[radius > 0],
        )
        phi = np.arctan2(shift[:, 1], shift[:, 0])
        if spherical:
            theta = np.arctan2(np.hypot(shift[:, 0], shift[:, 1]), shift[:, 2])
            wave = (
                np.sqrt(np.pi / (2 * k * radius))
                * scipy_special.hankel1(l + 0.5, k * radius)
                * scipy_special.sph_harm_y(l, m, theta, phi)
            )
        else:
            wave = scipy_special.hankel1(m, k * radius) * np.exp(1j * m * phi)
        return np.sum(wave * weights)

    return direct


@pytest.mark.physics
@pytest.mark.reference
@pytest.mark.parametrize("period", [7.2, 12.8])
def test_large_cylindrical_cell_ewald_split_invariance(period):
    # Upstream's automatic split loses 0.00676 at this order/displacement.
    # The converged reference and our automatic split agree independently.

    k = np.sqrt(1.3**2 - 0.2**2)
    automatic = tr.lattice.lsumcw(1, -6, k, 0.1, period, [0.8, 0])
    expected = upstream_lattice.lsumcw1d(-6, k, 0.1, period, 0.8, 0.7)
    for eta in (0.4, 0.7, 1.0):
        actual = tr.lattice.lsumcw(1, -6, k, 0.1, period, [0.8, 0], eta=eta)
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
        assert_allclose(actual, automatic, rtol=1e-12, atol=1e-10)


# First diffraction threshold |kpar - 2 pi| of kpar = 0.3 at unit period.
THRESHOLD = abs(0.3 - 2 * np.pi)


@pytest.mark.parametrize(
    "k,kz",
    [
        (2.1, 0.3),
        (4.5 + 0.1j, 0.3),
        (THRESHOLD * (1 + 1e-8j), 0.3),
        (THRESHOLD / (1 + 1e-11), 0.3),
        (1.3 + 0.02j, 0.0),
        (2.1 + 1e-12j, 0.0),
    ],
)
@pytest.mark.physics
def test_ewald_split_invariance_off_the_lattice(k, kz):
    # Up to three wavelengths off the lattice, where the Kambe series cancels by about
    # e^(w^2), next to a threshold and at kz = 0, where the order q = 0 of complex k
    # lies on the cut of the Kambe integrals, every split gives the same sum.
    l, m = np.array([(l, m) for l in range(5) for m in range(-l, l + 1)]).T
    lsumcw, lsumsw = tr.lattice.lsumcw, tr.lattice.lsumsw
    sums = {}
    for offset in (0.3, 1.5, 3.0):
        sums[f"spherical 1d at {offset}"] = partial(
            lsumsw, 1, l, m, k, kz, 1.0, [0.6 * offset, 0.8 * offset, 0.05]
        )
        sums[f"cylindrical 1d at {offset}"] = partial(
            lsumcw, 1, np.arange(-4, 5), k, kz, 1.0, [0.1, offset]
        )
        sums[f"spherical 2d at {offset}"] = partial(
            lsumsw, 2, l, m, k, [kz, 0.0], np.eye(2), [0.1, 0.2, offset]
        )
    for name, lattice_sum in sums.items():
        values = np.array(
            [np.asarray(lattice_sum(eta=eta)) for eta in (0, 0.5, 0.8, 1.2)]
        )
        assert_allclose(
            values,
            np.broadcast_to(values[0], values.shape),
            rtol=0,
            atol=2e-13 * np.abs(values[0]).max(),
            err_msg=name,
        )


def _lattice_point_splits(spherical, k):
    """The automatic split, splits of modulus 0.7 rotated off 1 / k by +-0.2 and, for
    real and complex k, the real split 0.7; for imaginary k, imaginary splits with a
    zero real part of either sign instead (for spherical waves only those with
    Re(k eta) > 0, where their Ewald sums converge)."""
    rotated = [0.7 * abs(k) / k * np.exp(0.2j), 0.7 * abs(k) / k * np.exp(-0.2j)]
    if k.real != 0:
        return [0, 0.7, *rotated]
    imaginary = [complex(zero, sign * 0.7) for zero in (0.0, -0.0) for sign in (1, -1)]
    if spherical:
        imaginary = [eta for eta in imaginary if (k * eta).real > 0]
    return [0, *rotated, *imaginary]


@pytest.mark.gradients
@pytest.mark.physics
@pytest.mark.reference
@pytest.mark.parametrize(
    "spherical,dim,k",
    [
        (spherical, dim, k)
        for spherical, dim in [(True, 1), (True, 2), (True, 3), (False, 1), (False, 2)]
        for k in (-2.0 + 1.2j, 1.2j, complex(-0.0, 1.2), 2.0 + 1.2j)
        if not (spherical and k.real < 0)
    ],
)
def test_lattice_point_sums_and_pullbacks_match_the_image_sums(spherical, dim, k):
    # The degree-0 and degree-1 sums and the position derivatives of the latter at a
    # lattice point, where the self term supplies the regular image-sum derivative,
    # against the image sums at every split of _lattice_point_splits (and at 1.2).
    a = np.diag([1.7] * dim)
    q = np.linspace(0.1, 0.2, dim)
    splits = _lattice_point_splits(spherical, k) + ([1.2] if k == 2 + 1.2j else [])
    for eta in splits:
        value = derivatives(spherical, 0, 0, k, q, a, np.zeros(3), eta)[0]
        expected = _image_sum(spherical, dim, 0, 0, k, q, a)(np.zeros(3))
        assert_allclose(value, expected, rtol=2e-8, atol=1e-10, err_msg=eta)
        for m in (-1, 0, 1):
            direct = _image_sum(spherical, dim, 1, m, k, q, a)
            value, *_, dr = derivatives(spherical, 1, m, k, q, a, np.zeros(3), eta)
            assert_allclose(value, direct(np.zeros(3)), rtol=2e-8, atol=1e-10)
            for axis in range(3 if spherical else 2):
                step = np.eye(3)[axis] * 1e-5
                assert_allclose(
                    dr[axis],
                    (direct(step) - direct(-step)) / (2e-5),
                    rtol=2e-7,
                    atol=2e-9,
                    err_msg=(eta, m, axis),
                )


@pytest.mark.physics
@pytest.mark.parametrize("k", [-1.2 + 0.3j, 1.2 + 0.3j])
def test_periodic_cylinder_self_coupling_is_the_image_sum(k):
    # The self-cell block of a lattice of cylinders at one position holds the
    # degree-0 sum at the lattice point on its diagonal, also for Re k < 0, where a
    # self term on the wrong sheet is off by exactly -2.
    a, q = np.diag([1.7, 1.5]), np.array([0.3, 0.1])
    modes = (np.zeros(6), np.array([-1, 0, 1, -1, 0, 1]), np.array([0, 0, 0, 1, 1, 1]))
    coupling = tr.cw.translate_periodic(k, q, a, [[0.0, 0.0, 0.0]], modes)
    direct = _image_sum(False, 2, 0, 0, k, q, a, cells=100)(np.zeros(3))
    assert_allclose(np.diag(coupling), direct, rtol=1e-12)


@pytest.mark.physics
@pytest.mark.parametrize(
    "spherical,k",
    [
        (spherical, k)
        for spherical in (True, False)
        for k in (1.3, 2.9 + 0.2j, 1.3 + 1e-12j, 0.9j, -1.3 + 0.4j)
        if not (spherical and k.real < 0)
    ],
)
def test_plane_and_axis_sums_do_not_depend_on_the_split(spherical, k):
    # 2D spherical (1D cylindrical) sums take gamma functions and Kambe integrals of
    # half-integer degree for their reciprocal orders, on the sheet of the automatic
    # split at every split, of either sign for cylindrical sums, in and off the plane
    # (axis) and at the lattice point; spherical Ewald sums take Re k >= 0.
    l, m = np.array([(l, m) for l in range(4) for m in range(-l, l + 1)]).T
    splits = [0, 1.1 * abs(k) / k]
    splits += [0.7 * abs(k) / k * np.exp(1j * angle) for angle in (0.25, -0.25)]
    if k.real > 0:
        splits.append(0.7)
    if not spherical:
        splits += [-eta for eta in splits[1:]]
    if spherical:
        a, q = np.array([[1.7, 0.0], [0.4, 1.5]]), np.array([0.3, 0.1])
        points = ([0.3, 0.2, 0.0], [0.0, 0.0, 0.0], [0.7, -0.5, 0.0])
        points += ([0.3, 0.2, 0.05], [0.7, -0.5, -0.4])
        orders = np.arange(len(l))

        def lattice_sum(r, eta):
            return lattice.lsumsw(2, l, m, k, q, a, r, eta)
    else:
        a, q = np.array([[1.7]]), np.array([0.3])
        points = ([0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.6, 0.0, 0.0])
        points += ([0.3, 0.05, 0.0], [-0.6, -0.4, 0.0])
        orders = np.arange(-3, 4)

        def lattice_sum(r, eta):
            return lattice.lsumcw(1, orders, k, q[0], a[0, 0], r[:2], eta)

    for r in points:
        values = np.array([np.asarray(lattice_sum(r, eta)) for eta in splits])
        scale = max(np.abs(values[0]).max(), 1)
        assert_allclose(
            values,
            np.broadcast_to(values[0], values.shape),
            atol=1e-10 * scale,
            rtol=0,
            err_msg=r,
        )
        if k.imag >= 0.3:
            degrees, labels = (l, m) if spherical else (np.zeros_like(orders), orders)
            direct = [
                _image_sum(spherical, len(q), n, o, k, q, a, cells=60, origin=True)(
                    np.array(r)
                )
                for n, o in zip(degrees, labels, strict=True)
            ]
            assert_allclose(values[0], direct, atol=1e-10 * scale, rtol=0, err_msg=r)
